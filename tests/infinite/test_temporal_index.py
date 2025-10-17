"""Tests for temporal indexing system."""

import pytest
import time
from pathlib import Path

from core.infinite.temporal_index import TemporalIndex


@pytest.mark.asyncio
async def test_temporal_index_initialization(test_db_path: Path):
    """Test temporal index initialization."""
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    assert temporal_index._initialized
    assert temporal_index._conn is not None
    
    await temporal_index.close()


@pytest.mark.asyncio
async def test_add_and_get_events(test_db_path: Path):
    """Test adding and retrieving temporal events."""
    # Initialize document store first to create schema
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    # Add events
    memory_id = "test_memory_1"
    timestamp1 = time.time()
    timestamp2 = timestamp1 + 10
    
    await temporal_index.add_event(memory_id, timestamp1, "created")
    await temporal_index.add_event(memory_id, timestamp2, "updated")
    
    # Get all events
    events = await temporal_index.get_events(memory_id)
    assert len(events) == 2
    assert events[0]["event_type"] == "created"
    assert events[1]["event_type"] == "updated"
    
    # Get filtered events
    created_events = await temporal_index.get_events(memory_id, "created")
    assert len(created_events) == 1
    assert created_events[0]["event_type"] == "created"
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_time_range_query(test_db_path: Path):
    """Test querying events within a time range."""
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    # Add events at different times
    base_time = time.time()
    memory_ids = ["mem1", "mem2", "mem3"]
    
    for i, mem_id in enumerate(memory_ids):
        timestamp = base_time + (i * 100)
        await temporal_index.add_event(mem_id, timestamp, "created")
    
    # Query middle range
    start_time = base_time + 50
    end_time = base_time + 250
    
    events = await temporal_index.query_by_time_range(start_time, end_time)
    assert len(events) == 2
    assert events[0]["memory_id"] == "mem2"
    assert events[1]["memory_id"] == "mem3"
    
    # Query with event type filter
    events_filtered = await temporal_index.query_by_time_range(
        start_time, end_time, event_type="created"
    )
    assert len(events_filtered) == 2
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_get_memories_at_time(test_db_path: Path):
    """Test getting memories that existed at a specific time."""
    from core.infinite.document_store import DocumentStore
    from core.infinite.models import Memory, MemoryType
    
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    # Create memories at different times
    base_time = time.time()
    context_id = "test_context"
    
    # Memory 1: created at base_time
    mem1 = Memory(
        id="mem1",
        context_id=context_id,
        content="First memory",
        memory_type=MemoryType.FACT,
        created_at=base_time
    )
    await doc_store.add_memory(mem1)
    
    # Memory 2: created at base_time + 100
    mem2 = Memory(
        id="mem2",
        context_id=context_id,
        content="Second memory",
        memory_type=MemoryType.FACT,
        created_at=base_time + 100
    )
    await doc_store.add_memory(mem2)
    
    # Memory 3: created at base_time + 200, superseded at base_time + 300
    mem3 = Memory(
        id="mem3",
        context_id=context_id,
        content="Third memory",
        memory_type=MemoryType.FACT,
        created_at=base_time + 200
    )
    await doc_store.add_memory(mem3)
    await temporal_index.add_event("mem3", base_time + 300, "superseded")
    
    # Query at base_time + 150: should get mem1 and mem2
    memories_at_150 = await temporal_index.get_memories_at_time(
        base_time + 150, context_id
    )
    assert len(memories_at_150) == 2
    assert "mem1" in memories_at_150
    assert "mem2" in memories_at_150
    
    # Query at base_time + 250: should get mem1, mem2, and mem3
    memories_at_250 = await temporal_index.get_memories_at_time(
        base_time + 250, context_id
    )
    assert len(memories_at_250) == 3
    
    # Query at base_time + 350: should get mem1 and mem2 (mem3 superseded)
    memories_at_350 = await temporal_index.get_memories_at_time(
        base_time + 350, context_id
    )
    assert len(memories_at_350) == 2
    assert "mem3" not in memories_at_350
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_get_latest_event_time(test_db_path: Path):
    """Test getting the latest event timestamp."""
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    memory_id = "test_memory"
    base_time = time.time()
    
    # Add multiple events
    await temporal_index.add_event(memory_id, base_time, "created")
    await temporal_index.add_event(memory_id, base_time + 100, "updated")
    await temporal_index.add_event(memory_id, base_time + 200, "updated")
    
    # Get latest event time
    latest = await temporal_index.get_latest_event_time(memory_id)
    assert latest is not None
    assert abs(latest - (base_time + 200)) < 1.0
    
    # Get latest for specific event type
    latest_created = await temporal_index.get_latest_event_time(memory_id, "created")
    assert latest_created is not None
    assert abs(latest_created - base_time) < 1.0
    
    # Non-existent memory
    latest_none = await temporal_index.get_latest_event_time("nonexistent")
    assert latest_none is None
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_edge_case_same_timestamp(test_db_path: Path):
    """Test handling of events with the same timestamp."""
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    # Add multiple events with same timestamp
    timestamp = time.time()
    await temporal_index.add_event("mem1", timestamp, "created")
    await temporal_index.add_event("mem2", timestamp, "created")
    await temporal_index.add_event("mem3", timestamp, "created")
    
    # Query at that exact time
    events = await temporal_index.query_by_time_range(timestamp, timestamp)
    assert len(events) == 3
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_edge_case_out_of_range(test_db_path: Path):
    """Test querying outside the range of stored events."""
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    # Add events
    base_time = time.time()
    await temporal_index.add_event("mem1", base_time, "created")
    
    # Query before any events
    events_before = await temporal_index.query_by_time_range(
        base_time - 1000, base_time - 500
    )
    assert len(events_before) == 0
    
    # Query after all events
    events_after = await temporal_index.query_by_time_range(
        base_time + 1000, base_time + 2000
    )
    assert len(events_after) == 0
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_timestamp_indexing_accuracy(test_db_path: Path):
    """Test timestamp indexing accuracy with precise timestamps."""
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    # Add events with precise timestamps
    base_time = time.time()
    timestamps = [
        base_time,
        base_time + 0.001,  # 1ms later
        base_time + 0.01,   # 10ms later
        base_time + 1.0,    # 1s later
        base_time + 60.0,   # 1min later
    ]
    
    for i, ts in enumerate(timestamps):
        await temporal_index.add_event(f"mem_{i}", ts, "created")
    
    # Verify each event is indexed with correct timestamp
    for i, expected_ts in enumerate(timestamps):
        events = await temporal_index.get_events(f"mem_{i}")
        assert len(events) == 1
        assert abs(events[0]["timestamp"] - expected_ts) < 0.0001  # Sub-millisecond accuracy
    
    # Verify chronological ordering
    all_events = await temporal_index.query_by_time_range(
        base_time - 1, base_time + 100
    )
    assert len(all_events) == 5
    for i in range(len(all_events) - 1):
        assert all_events[i]["timestamp"] <= all_events[i + 1]["timestamp"]
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_large_time_span_query_performance(test_db_path: Path):
    """Test query performance with large time spans and many events."""
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    # Add many events across a large time span (simulating years of data)
    base_time = time.time()
    num_events = 1000
    time_span = 365 * 24 * 3600  # 1 year in seconds
    
    # Add events distributed across the time span
    for i in range(num_events):
        timestamp = base_time + (i * time_span / num_events)
        await temporal_index.add_event(f"mem_{i}", timestamp, "created")
    
    # Query entire time span
    start = time.time()
    events = await temporal_index.query_by_time_range(
        base_time, base_time + time_span
    )
    query_time = time.time() - start
    
    assert len(events) == num_events
    assert query_time < 1.0  # Should complete in under 1 second
    
    # Query narrow time range within large span
    narrow_start = base_time + (time_span / 2)
    narrow_end = narrow_start + 3600  # 1 hour window
    
    start = time.time()
    narrow_events = await temporal_index.query_by_time_range(
        narrow_start, narrow_end
    )
    narrow_query_time = time.time() - start
    
    assert len(narrow_events) > 0
    assert narrow_query_time < 0.1  # Should be very fast
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_edge_case_boundary_timestamps(test_db_path: Path):
    """Test edge cases with boundary timestamps."""
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    base_time = time.time()
    
    # Add events at exact boundaries
    await temporal_index.add_event("mem1", base_time, "created")
    await temporal_index.add_event("mem2", base_time + 100, "created")
    await temporal_index.add_event("mem3", base_time + 200, "created")
    
    # Query with exact boundary matches (inclusive)
    events_exact_start = await temporal_index.query_by_time_range(
        base_time, base_time + 200
    )
    assert len(events_exact_start) == 3
    
    # Query with start boundary only
    events_start_only = await temporal_index.query_by_time_range(
        base_time + 100, base_time + 200
    )
    assert len(events_start_only) == 2
    assert events_start_only[0]["memory_id"] == "mem2"
    
    # Query with end boundary only
    events_end_only = await temporal_index.query_by_time_range(
        base_time, base_time + 100
    )
    assert len(events_end_only) == 2
    assert events_end_only[1]["memory_id"] == "mem2"
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_edge_case_zero_and_negative_timestamps(test_db_path: Path):
    """Test handling of zero and negative timestamps."""
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    # Add events with edge case timestamps
    await temporal_index.add_event("mem_zero", 0.0, "created")
    await temporal_index.add_event("mem_negative", -100.0, "created")
    await temporal_index.add_event("mem_positive", 100.0, "created")
    
    # Query including negative timestamps
    events = await temporal_index.query_by_time_range(-200.0, 200.0)
    assert len(events) == 3
    
    # Verify ordering with negative timestamps
    assert events[0]["memory_id"] == "mem_negative"
    assert events[1]["memory_id"] == "mem_zero"
    assert events[2]["memory_id"] == "mem_positive"
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_edge_case_inverted_time_range(test_db_path: Path):
    """Test behavior when start_time > end_time."""
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    base_time = time.time()
    await temporal_index.add_event("mem1", base_time, "created")
    
    # Query with inverted range (should return empty)
    events = await temporal_index.query_by_time_range(
        base_time + 100, base_time
    )
    assert len(events) == 0
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_multiple_event_types_same_memory(test_db_path: Path):
    """Test indexing multiple event types for the same memory."""
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    memory_id = "test_memory"
    base_time = time.time()
    
    # Add different event types
    await temporal_index.add_event(memory_id, base_time, "created")
    await temporal_index.add_event(memory_id, base_time + 10, "updated")
    await temporal_index.add_event(memory_id, base_time + 20, "updated")
    await temporal_index.add_event(memory_id, base_time + 30, "superseded")
    
    # Get all events
    all_events = await temporal_index.get_events(memory_id)
    assert len(all_events) == 4
    
    # Get specific event types
    created = await temporal_index.get_events(memory_id, "created")
    assert len(created) == 1
    
    updated = await temporal_index.get_events(memory_id, "updated")
    assert len(updated) == 2
    
    superseded = await temporal_index.get_events(memory_id, "superseded")
    assert len(superseded) == 1
    
    # Query by time range with event type filter
    updates_in_range = await temporal_index.query_by_time_range(
        base_time + 5, base_time + 25, event_type="updated"
    )
    assert len(updates_in_range) == 2
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_query_limit_enforcement(test_db_path: Path):
    """Test that query limit parameter is enforced."""
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    base_time = time.time()
    
    # Add 100 events
    for i in range(100):
        await temporal_index.add_event(f"mem_{i}", base_time + i, "created")
    
    # Query with limit
    events_limited = await temporal_index.query_by_time_range(
        base_time, base_time + 200, limit=10
    )
    assert len(events_limited) == 10
    
    # Query with higher limit
    events_50 = await temporal_index.query_by_time_range(
        base_time, base_time + 200, limit=50
    )
    assert len(events_50) == 50
    
    # Query with limit higher than available
    events_all = await temporal_index.query_by_time_range(
        base_time, base_time + 200, limit=200
    )
    assert len(events_all) == 100
    
    await temporal_index.close()
    await doc_store.close()


@pytest.mark.asyncio
async def test_very_large_timestamp_values(test_db_path: Path):
    """Test handling of very large timestamp values."""
    from core.infinite.document_store import DocumentStore
    doc_store = DocumentStore(test_db_path)
    await doc_store.initialize()
    
    temporal_index = TemporalIndex(test_db_path)
    await temporal_index.initialize()
    
    # Test with very large timestamps (far future)
    large_timestamp = 9999999999.0  # Year 2286
    await temporal_index.add_event("mem_future", large_timestamp, "created")
    
    # Verify retrieval
    events = await temporal_index.get_events("mem_future")
    assert len(events) == 1
    assert events[0]["timestamp"] == large_timestamp
    
    # Query with large range
    events_range = await temporal_index.query_by_time_range(
        large_timestamp - 1000, large_timestamp + 1000
    )
    assert len(events_range) == 1
    
    await temporal_index.close()
    await doc_store.close()
