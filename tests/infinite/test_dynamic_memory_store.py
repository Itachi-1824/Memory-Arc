"""Tests for DynamicMemoryStore."""

import pytest
import time
from pathlib import Path

from core.infinite.dynamic_memory_store import DynamicMemoryStore
from core.infinite.models import MemoryType
from tests.infinite.test_utils import generate_test_embedding


@pytest.mark.asyncio
async def test_dynamic_memory_store_initialization(test_db_path: Path):
    """Test DynamicMemoryStore initialization."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    assert store._initialized
    assert store.document_store._initialized
    assert store.temporal_index._initialized
    
    await store.close()


@pytest.mark.asyncio
async def test_add_memory_basic(test_db_path: Path):
    """Test adding a basic memory."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    memory_id = await store.add_memory(
        content="I like apples",
        memory_type=MemoryType.PREFERENCE,
        context_id="user_123",
        importance=5
    )
    
    assert memory_id is not None
    
    # Retrieve the memory
    memory = await store.document_store.get_memory(memory_id)
    assert memory is not None
    assert memory.content == "I like apples"
    assert memory.version == 1
    assert memory.parent_id is None
    
    await store.close()


@pytest.mark.asyncio
async def test_version_creation_and_linking(test_db_path: Path):
    """Test creating new versions and linking them."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    
    # Create initial memory
    mem1_id = await store.add_memory(
        content="I like apples",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        embedding=generate_test_embedding("I like apples")
    )
    
    # Create new version
    mem2_id = await store.add_memory(
        content="I like oranges",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        supersedes=mem1_id,
        embedding=generate_test_embedding("I like oranges")
    )
    
    # Verify version linking
    mem1 = await store.document_store.get_memory(mem1_id)
    mem2 = await store.document_store.get_memory(mem2_id)
    
    assert mem1.version == 1
    assert mem2.version == 2
    assert mem2.parent_id == mem1_id
    
    # Check version graph
    node1 = store.version_graph.get_node(mem1_id)
    node2 = store.version_graph.get_node(mem2_id)
    
    assert node1 is not None
    assert node2 is not None
    assert node2.parent_id == mem1_id
    assert mem2_id in node1.children_ids
    
    await store.close()


@pytest.mark.asyncio
async def test_get_current_version(test_db_path: Path):
    """Test retrieving the current version of a memory."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    
    # Create version chain
    mem1_id = await store.add_memory(
        content="I like apples",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id
    )
    
    mem2_id = await store.add_memory(
        content="I like oranges",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        supersedes=mem1_id
    )
    
    mem3_id = await store.add_memory(
        content="I like mangoes",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        supersedes=mem2_id
    )
    
    # Get current version from any point in chain
    current_from_v1 = await store.get_current_version(mem1_id)
    current_from_v2 = await store.get_current_version(mem2_id)
    current_from_v3 = await store.get_current_version(mem3_id)
    
    assert current_from_v1.id == mem3_id
    assert current_from_v2.id == mem3_id
    assert current_from_v3.id == mem3_id
    assert current_from_v3.content == "I like mangoes"
    
    await store.close()


@pytest.mark.asyncio
async def test_get_version_history(test_db_path: Path):
    """Test retrieving complete version history."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    
    # Create version chain
    mem1_id = await store.add_memory(
        content="Version 1",
        memory_type=MemoryType.FACT,
        context_id=context_id
    )
    
    mem2_id = await store.add_memory(
        content="Version 2",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        supersedes=mem1_id
    )
    
    mem3_id = await store.add_memory(
        content="Version 3",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        supersedes=mem2_id
    )
    
    # Get history
    history = await store.get_version_history(mem2_id)
    
    assert len(history) == 3
    assert history[0].id == mem1_id
    assert history[1].id == mem2_id
    assert history[2].id == mem3_id
    assert history[0].content == "Version 1"
    assert history[2].content == "Version 3"
    
    await store.close()


@pytest.mark.asyncio
async def test_temporal_queries(test_db_path: Path):
    """Test querying memories at various timestamps."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    base_time = time.time()
    
    # Create memories at different times
    mem1_id = await store.add_memory(
        content="Early memory",
        memory_type=MemoryType.FACT,
        context_id=context_id
    )
    
    # Wait a bit
    time.sleep(0.1)
    mid_time = time.time()
    
    mem2_id = await store.add_memory(
        content="Middle memory",
        memory_type=MemoryType.FACT,
        context_id=context_id
    )
    
    time.sleep(0.1)
    late_time = time.time()
    
    # Query at mid_time (should only get mem1)
    memories_at_mid = await store.query_at_time(
        query="",
        timestamp=mid_time,
        context_id=context_id
    )
    
    assert len(memories_at_mid) >= 1
    assert any(m.id == mem1_id for m in memories_at_mid)
    
    # Query at late_time (should get both)
    memories_at_late = await store.query_at_time(
        query="",
        timestamp=late_time,
        context_id=context_id
    )
    
    assert len(memories_at_late) >= 2
    
    await store.close()


@pytest.mark.asyncio
async def test_contradiction_detection(test_db_path: Path):
    """Test detecting contradictory memories."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    
    # Add contradictory memories
    mem1_id = await store.add_memory(
        content="I like Python",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        embedding=generate_test_embedding("I like Python")
    )
    
    mem2_id = await store.add_memory(
        content="I don't like Python",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        embedding=generate_test_embedding("I don't like Python")
    )
    
    # Detect contradictions
    contradictions = await store.detect_contradictions(context_id)
    
    # Should find at least one contradiction
    assert len(contradictions) > 0
    
    # Check that our memories are in the contradictions
    found = False
    for mem_a, mem_b, confidence in contradictions:
        if (mem_a.id == mem1_id and mem_b.id == mem2_id) or \
           (mem_a.id == mem2_id and mem_b.id == mem1_id):
            found = True
            assert confidence > 0.0
            break
    
    assert found, "Expected contradiction not found"
    
    await store.close()


@pytest.mark.asyncio
async def test_memory_evolution_scenario(test_db_path: Path):
    """Test the classic 'I like apples → I like mangoes' scenario."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    
    # Evolution: apples → apples and oranges → oranges → mangoes
    v1_id = await store.add_memory(
        content="I like apples",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        embedding=generate_test_embedding("I like apples")
    )
    
    v2_id = await store.add_memory(
        content="I like apples and oranges",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        supersedes=v1_id,
        embedding=generate_test_embedding("I like apples and oranges")
    )
    
    v3_id = await store.add_memory(
        content="I like oranges",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        supersedes=v2_id,
        embedding=generate_test_embedding("I like oranges")
    )
    
    v4_id = await store.add_memory(
        content="I like mangoes",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        supersedes=v3_id,
        embedding=generate_test_embedding("I like mangoes")
    )
    
    # Get current preference
    current = await store.get_current_version(v1_id)
    assert current.content == "I like mangoes"
    
    # Get evolution history
    evolution = await store.get_memory_evolution(v1_id)
    assert evolution["version_count"] == 4
    assert evolution["current_version"]["content"] == "I like mangoes"
    
    # Verify version chain
    history = await store.get_version_history(v1_id)
    assert len(history) == 4
    assert history[0].content == "I like apples"
    assert history[3].content == "I like mangoes"
    
    await store.close()


@pytest.mark.asyncio
async def test_concurrent_version_updates(test_db_path: Path):
    """Test handling concurrent updates (branching)."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    
    # Create base memory
    base_id = await store.add_memory(
        content="Base preference",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id
    )
    
    # Create two branches from the same parent
    branch1_id = await store.add_memory(
        content="Branch 1",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        supersedes=base_id
    )
    
    branch2_id = await store.add_memory(
        content="Branch 2",
        memory_type=MemoryType.PREFERENCE,
        context_id=context_id,
        supersedes=base_id
    )
    
    # Verify branching in version graph
    base_node = store.version_graph.get_node(base_id)
    assert len(base_node.children_ids) == 2
    assert branch1_id in base_node.children_ids
    assert branch2_id in base_node.children_ids
    
    # Check for branching
    has_branching = store.version_graph.has_branching(base_id)
    assert has_branching
    
    # Get evolution info
    evolution = await store.get_memory_evolution(base_id)
    assert len(evolution["branches"]) > 0
    
    await store.close()


@pytest.mark.asyncio
async def test_get_memory_evolution(test_db_path: Path):
    """Test getting complete evolution information."""
    store = DynamicMemoryStore(test_db_path)
    await store.initialize()
    
    context_id = "user_123"
    
    # Create simple chain
    v1_id = await store.add_memory(
        content="Version 1",
        memory_type=MemoryType.FACT,
        context_id=context_id
    )
    
    v2_id = await store.add_memory(
        content="Version 2",
        memory_type=MemoryType.FACT,
        context_id=context_id,
        supersedes=v1_id
    )
    
    # Get evolution
    evolution = await store.get_memory_evolution(v1_id)
    
    assert evolution["root_id"] == v1_id
    assert evolution["version_count"] == 2
    assert len(evolution["versions"]) == 2
    assert evolution["current_version"]["content"] == "Version 2"
    
    await store.close()
