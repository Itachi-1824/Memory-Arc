"""Integration tests for memory evolution workflows."""

import pytest
import time
from pathlib import Path

from core.infinite.dynamic_memory_store import DynamicMemoryStore
from core.infinite.models import MemoryType
from tests.infinite.test_utils import generate_test_embedding


@pytest.mark.asyncio
async def test_end_to_end_memory_versioning(test_db_path: Path):
    """Test complete memory versioning workflow."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    
    # Step 1: Add initial memory
    v1_id = await store.add_memory(
        content="Initial fact",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        importance=7,
        embedding=generate_test_embedding("Initial fact")
    )
    
    # Verify initial state
    v1 = await store.document_store.get_memory(v1_id)
    assert v1.version == 1
    assert v1.parent_id is None
    
    # Step 2: Update the memory
    v2_id = await store.add_memory(
        content="Updated fact",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        importance=8,
        supersedes=v1_id,
        embedding=generate_test_embedding("Updated fact")
    )
    
    # Verify version chain
    v2 = await store.document_store.get_memory(v2_id)
    assert v2.version == 2
    assert v2.parent_id == v1_id
    
    # Step 3: Get current version
    current = await store.get_current_version(v1_id)
    assert current.id == v2_id
    assert current.content == "Updated fact"
    
    # Step 4: Get version history
    history = await store.get_version_history(v1_id)
    assert len(history) == 2
    assert history[0].id == v1_id
    assert history[1].id == v2_id
    
    # Step 5: Check temporal index
    events = await store.temporal_index.get_events(v1_id)
    assert len(events) >= 2  # created + superseded
    
    await store.close()


@pytest.mark.asyncio
async def test_apples_to_mangoes_scenario(test_db_path: Path):
    """Test the classic 'I like apples → I like mangoes' scenario."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    timestamps = []
    
    # Day 1: I like apples
    v1_id = await store.add_memory(
        content="I like apples",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        embedding=generate_test_embedding("I like apples")
    )
    timestamps.append(time.time())
    time.sleep(0.05)
    
    # Day 30: I like apples and oranges
    v2_id = await store.add_memory(
        content="I like apples and oranges",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        supersedes=v1_id,
        embedding=generate_test_embedding("I like apples and oranges")
    )
    timestamps.append(time.time())
    time.sleep(0.05)
    
    # Day 60: I like oranges
    v3_id = await store.add_memory(
        content="I like oranges",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        supersedes=v2_id,
        embedding=generate_test_embedding("I like oranges")
    )
    timestamps.append(time.time())
    time.sleep(0.05)
    
    # Day 90: I like mangoes
    v4_id = await store.add_memory(
        content="I like mangoes",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        supersedes=v3_id,
        embedding=generate_test_embedding("I like mangoes")
    )
    timestamps.append(time.time())
    
    # Test 1: Current preference
    current = await store.get_current_version(v1_id)
    assert current.content == "I like mangoes"
    
    # Test 2: Historical queries
    # What did I like on Day 1?
    memories_day1 = await store.query_at_time(
        query="fruit preference",
        timestamp=timestamps[0] + 0.01,
        context_id=context_id
    )
    assert any(m.content == "I like apples" for m in memories_day1)
    
    # What did I like on Day 60?
    memories_day60 = await store.query_at_time(
        query="fruit preference",
        timestamp=timestamps[2] + 0.01,
        context_id=context_id
    )
    assert any(m.content == "I like oranges" for m in memories_day60)
    
    # Test 3: Complete evolution
    evolution = await store.get_memory_evolution(v1_id)
    assert evolution["version_count"] == 4
    assert evolution["versions"][0]["content"] == "I like apples"
    assert evolution["versions"][3]["content"] == "I like mangoes"
    
    # Test 4: Version history shows progression
    history = await store.get_version_history(v1_id)
    assert len(history) == 4
    contents = [m.content for m in history]
    assert contents == [
        "I like apples",
        "I like apples and oranges",
        "I like oranges",
        "I like mangoes"
    ]
    
    await store.close()


@pytest.mark.asyncio
async def test_branching_version_scenario(test_db_path: Path):
    """Test branching version scenarios."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    
    # Create base memory
    base_id = await store.add_memory(
        content="I work in tech",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        embedding=generate_test_embedding("I work in tech")
    )
    
    # Branch 1: More specific
    branch1_id = await store.add_memory(
        content="I work as a software engineer",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        supersedes=base_id,
        embedding=generate_test_embedding("I work as a software engineer")
    )
    
    # Branch 2: Different direction
    branch2_id = await store.add_memory(
        content="I work in product management",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        supersedes=base_id,
        embedding=generate_test_embedding("I work in product management")
    )
    
    # Verify branching
    base_node = store.version_graph.get_node(base_id)
    assert len(base_node.children_ids) == 2
    assert store.version_graph.has_branching(base_id)
    
    # Get evolution info
    evolution = await store.get_memory_evolution(base_id)
    assert len(evolution["branches"]) > 0
    
    # Each branch should have its own current version
    current_branch1 = await store.get_current_version(branch1_id)
    current_branch2 = await store.get_current_version(branch2_id)
    
    assert current_branch1.id == branch1_id
    assert current_branch2.id == branch2_id
    assert current_branch1.content != current_branch2.content
    
    # Detect conflicts
    conflicts = store.version_graph.detect_conflicts(branch1_id)
    assert len(conflicts) > 0
    
    await store.close()


@pytest.mark.asyncio
async def test_historical_query_accuracy(test_db_path: Path):
    """Test accuracy of historical queries at various timestamps."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    
    # Create timeline of memories
    timeline = []
    
    # T0: Initial state
    t0 = time.time()
    mem1_id = await store.add_memory(
        content="Fact A",
        memory_type=MemoryType.FACT,
        context_id=context_id
    )
    timeline.append((t0, mem1_id, "Fact A"))
    time.sleep(0.05)
    
    # T1: Add another memory
    t1 = time.time()
    mem2_id = await store.add_memory(
        content="Fact B",
        memory_type=MemoryType.FACT,
        context_id=context_id
    )
    timeline.append((t1, mem2_id, "Fact B"))
    time.sleep(0.05)
    
    # T2: Update first memory
    t2 = time.time()
    mem1_v2_id = await store.add_memory(
        content="Fact A updated",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        supersedes=mem1_id
    )
    timeline.append((t2, mem1_v2_id, "Fact A updated"))
    time.sleep(0.05)
    
    # T3: Add third memory
    t3 = time.time()
    mem3_id = await store.add_memory(
        content="Fact C",
        memory_type=MemoryType.FACT,
        context_id=context_id
    )
    timeline.append((t3, mem3_id, "Fact C"))
    
    # Query at T0 + epsilon: should only have mem1
    memories_t0 = await store.query_at_time(
        query="",
        timestamp=t0 + 0.01,
        context_id=context_id
    )
    assert len(memories_t0) == 1
    assert any(m.content == "Fact A" for m in memories_t0)
    
    # Query at T1 + epsilon: should have mem1 and mem2
    memories_t1 = await store.query_at_time(
        query="",
        timestamp=t1 + 0.01,
        context_id=context_id
    )
    assert len(memories_t1) == 2
    
    # Query at T2 + epsilon: should have mem1_v2 and mem2 (mem1 superseded)
    memories_t2 = await store.query_at_time(
        query="",
        timestamp=t2 + 0.01,
        context_id=context_id
    )
    assert len(memories_t2) == 2
    assert any(m.content == "Fact A updated" for m in memories_t2)
    assert not any(m.content == "Fact A" for m in memories_t2)
    
    # Query at T3 + epsilon: should have all current versions
    memories_t3 = await store.query_at_time(
        query="",
        timestamp=t3 + 0.01,
        context_id=context_id
    )
    assert len(memories_t3) == 3
    
    await store.close()


@pytest.mark.asyncio
async def test_complex_evolution_chain(test_db_path: Path):
    """Test complex evolution with multiple updates and branches."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    
    # Create a complex evolution tree
    #       v1
    #      /  \
    #    v2    v3
    #    |
    #   v4
    
    v1_id = await store.add_memory(
        content="Root",
        memory_type=MemoryType.FACT,
        context_id=context_id
    )
    
    v2_id = await store.add_memory(
        content="Branch A - Step 1",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        supersedes=v1_id
    )
    
    v3_id = await store.add_memory(
        content="Branch B",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        supersedes=v1_id
    )
    
    v4_id = await store.add_memory(
        content="Branch A - Step 2",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        supersedes=v2_id
    )
    
    # Verify structure
    v1_node = store.version_graph.get_node(v1_id)
    assert len(v1_node.children_ids) == 2
    
    # Get descendants of v1
    descendants = store.version_graph.get_descendants(v1_id)
    assert len(descendants) == 3
    
    # Get ancestors of v4
    ancestors = store.version_graph.get_ancestors(v4_id)
    assert len(ancestors) == 2
    assert ancestors[0].memory_id == v1_id
    assert ancestors[1].memory_id == v2_id
    
    # Get latest in each branch
    latest_branch_a = await store.get_current_version(v2_id)
    latest_branch_b = await store.get_current_version(v3_id)
    
    assert latest_branch_a.id == v4_id
    assert latest_branch_b.id == v3_id
    
    await store.close()


@pytest.mark.asyncio
async def test_memory_evolution_with_metadata(test_db_path: Path):
    """Test memory evolution preserving and updating metadata."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    
    # Create memory with metadata
    v1_id = await store.add_memory(
        content="Original content",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        metadata={"source": "user_input", "confidence": 0.9}
    )
    
    # Update with new metadata
    v2_id = await store.add_memory(
        content="Updated content",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        supersedes=v1_id,
        metadata={"source": "correction", "confidence": 1.0, "verified": True}
    )
    
    # Verify metadata preserved in history
    history = await store.get_version_history(v1_id)
    assert history[0].metadata["source"] == "user_input"
    assert history[1].metadata["source"] == "correction"
    assert history[1].metadata["verified"] is True
    
    await store.close()
