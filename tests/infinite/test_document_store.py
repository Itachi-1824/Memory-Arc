"""Unit tests for DocumentStore."""

import pytest
import asyncio
import time
from pathlib import Path
import uuid
import sqlite3
from concurrent.futures import ThreadPoolExecutor

from core.infinite.document_store import DocumentStore
from core.infinite.models import Memory, MemoryType


@pytest.fixture
async def document_store(test_db_path):
    """Create and initialize a document store for testing."""
    store = DocumentStore(test_db_path, pool_size=3)
    await store.initialize()
    yield store
    await store.close()


def create_test_memory(
    content: str = "Test content",
    memory_type: MemoryType = MemoryType.CONVERSATION,
    context_id: str = "test_context",
    importance: int = 5,
    **kwargs
) -> Memory:
    """Helper to create test memory objects."""
    return Memory(
        id=kwargs.get("id", str(uuid.uuid4())),
        context_id=context_id,
        content=content,
        memory_type=memory_type,
        created_at=kwargs.get("created_at", time.time()),
        importance=importance,
        updated_at=kwargs.get("updated_at"),
        version=kwargs.get("version", 1),
        parent_id=kwargs.get("parent_id"),
        thread_id=kwargs.get("thread_id"),
        metadata=kwargs.get("metadata", {}),
    )


# Test Memory Insertion
@pytest.mark.asyncio
async def test_add_memory_success(document_store):
    """Test successful memory insertion."""
    memory = create_test_memory(content="Test memory insertion")
    
    result = await document_store.add_memory(memory)
    assert result is True
    
    # Verify memory was stored
    retrieved = await document_store.get_memory(memory.id)
    assert retrieved is not None
    assert retrieved.id == memory.id
    assert retrieved.content == memory.content
    assert retrieved.memory_type == memory.memory_type


@pytest.mark.asyncio
async def test_add_memory_with_metadata(document_store):
    """Test memory insertion with metadata."""
    metadata = {"key1": "value1", "key2": 123, "nested": {"data": True}}
    memory = create_test_memory(
        content="Memory with metadata",
        metadata=metadata
    )
    
    await document_store.add_memory(memory)
    retrieved = await document_store.get_memory(memory.id)
    
    assert retrieved.metadata == metadata


@pytest.mark.asyncio
async def test_add_duplicate_memory_fails(document_store):
    """Test that adding duplicate memory ID fails."""
    memory = create_test_memory(id="duplicate_id")
    
    result1 = await document_store.add_memory(memory)
    assert result1 is True
    
    # Try to add same ID again
    result2 = await document_store.add_memory(memory)
    assert result2 is False


# Test Memory Retrieval
@pytest.mark.asyncio
async def test_get_memory_not_found(document_store):
    """Test retrieving non-existent memory returns None."""
    result = await document_store.get_memory("nonexistent_id")
    assert result is None


@pytest.mark.asyncio
async def test_get_memory_with_all_fields(document_store):
    """Test retrieving memory with all fields populated."""
    memory = create_test_memory(
        content="Full memory",
        memory_type=MemoryType.FACT,
        context_id="ctx_123",
        thread_id="thread_456",
        importance=8,
        version=2,
        parent_id="parent_789",
        metadata={"tag": "important"}
    )
    
    await document_store.add_memory(memory)
    retrieved = await document_store.get_memory(memory.id)
    
    assert retrieved.context_id == "ctx_123"
    assert retrieved.thread_id == "thread_456"
    assert retrieved.importance == 8
    assert retrieved.version == 2
    assert retrieved.parent_id == "parent_789"
    assert retrieved.metadata == {"tag": "important"}


# Test Memory Update
@pytest.mark.asyncio
async def test_update_memory_success(document_store):
    """Test successful memory update."""
    memory = create_test_memory(content="Original content")
    await document_store.add_memory(memory)
    
    # Update memory
    memory.content = "Updated content"
    memory.importance = 9
    memory.updated_at = time.time()
    
    result = await document_store.update_memory(memory)
    assert result is True
    
    # Verify update
    retrieved = await document_store.get_memory(memory.id)
    assert retrieved.content == "Updated content"
    assert retrieved.importance == 9
    assert retrieved.updated_at is not None


@pytest.mark.asyncio
async def test_update_nonexistent_memory(document_store):
    """Test updating non-existent memory returns False."""
    memory = create_test_memory(id="nonexistent")
    result = await document_store.update_memory(memory)
    assert result is False


@pytest.mark.asyncio
async def test_update_memory_metadata(document_store):
    """Test updating memory metadata."""
    memory = create_test_memory(metadata={"version": 1})
    await document_store.add_memory(memory)
    
    memory.metadata = {"version": 2, "updated": True}
    memory.updated_at = time.time()
    await document_store.update_memory(memory)
    
    retrieved = await document_store.get_memory(memory.id)
    assert retrieved.metadata == {"version": 2, "updated": True}


# Test Memory Deletion
@pytest.mark.asyncio
async def test_delete_memory_success(document_store):
    """Test successful memory deletion."""
    memory = create_test_memory()
    await document_store.add_memory(memory)
    
    result = await document_store.delete_memory(memory.id)
    assert result is True
    
    # Verify deletion
    retrieved = await document_store.get_memory(memory.id)
    assert retrieved is None


@pytest.mark.asyncio
async def test_delete_nonexistent_memory(document_store):
    """Test deleting non-existent memory returns False."""
    result = await document_store.delete_memory("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_delete_memory_cascades(document_store):
    """Test that deleting memory also removes temporal index entries."""
    memory = create_test_memory()
    await document_store.add_memory(memory)
    
    # Update to create temporal entries
    memory.updated_at = time.time()
    await document_store.update_memory(memory)
    
    # Delete memory
    await document_store.delete_memory(memory.id)
    
    # Verify temporal entries are also deleted
    async with document_store._get_connection() as conn:
        def check():
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM temporal_index WHERE memory_id = ?",
                (memory.id,)
            )
            return cursor.fetchone()[0]
        
        count = await asyncio.to_thread(check)
        assert count == 0


# Test Query Memories
@pytest.mark.asyncio
async def test_query_memories_by_context(document_store):
    """Test querying memories by context ID."""
    mem1 = create_test_memory(context_id="ctx1", content="Memory 1")
    mem2 = create_test_memory(context_id="ctx1", content="Memory 2")
    mem3 = create_test_memory(context_id="ctx2", content="Memory 3")
    
    await document_store.add_memory(mem1)
    await document_store.add_memory(mem2)
    await document_store.add_memory(mem3)
    
    results = await document_store.query_memories(context_id="ctx1")
    assert len(results) == 2
    assert all(m.context_id == "ctx1" for m in results)


@pytest.mark.asyncio
async def test_query_memories_by_type(document_store):
    """Test querying memories by type."""
    mem1 = create_test_memory(memory_type=MemoryType.CONVERSATION)
    mem2 = create_test_memory(memory_type=MemoryType.FACT)
    mem3 = create_test_memory(memory_type=MemoryType.CONVERSATION)
    
    await document_store.add_memory(mem1)
    await document_store.add_memory(mem2)
    await document_store.add_memory(mem3)
    
    results = await document_store.query_memories(memory_type=MemoryType.CONVERSATION)
    assert len(results) == 2
    assert all(m.memory_type == MemoryType.CONVERSATION for m in results)


@pytest.mark.asyncio
async def test_query_memories_by_importance(document_store):
    """Test querying memories by minimum importance."""
    mem1 = create_test_memory(importance=3)
    mem2 = create_test_memory(importance=7)
    mem3 = create_test_memory(importance=5)
    
    await document_store.add_memory(mem1)
    await document_store.add_memory(mem2)
    await document_store.add_memory(mem3)
    
    results = await document_store.query_memories(min_importance=5)
    assert len(results) == 2
    assert all(m.importance >= 5 for m in results)


@pytest.mark.asyncio
async def test_query_memories_with_limit_offset(document_store):
    """Test querying memories with pagination."""
    for i in range(10):
        mem = create_test_memory(content=f"Memory {i}")
        await document_store.add_memory(mem)
    
    # Get first page
    page1 = await document_store.query_memories(limit=3, offset=0)
    assert len(page1) == 3
    
    # Get second page
    page2 = await document_store.query_memories(limit=3, offset=3)
    assert len(page2) == 3
    
    # Verify no overlap
    page1_ids = {m.id for m in page1}
    page2_ids = {m.id for m in page2}
    assert len(page1_ids & page2_ids) == 0


@pytest.mark.asyncio
async def test_query_memories_ordered_by_time(document_store):
    """Test that query results are ordered by creation time (newest first)."""
    base_time = time.time()
    mem1 = create_test_memory(content="First", created_at=base_time)
    mem2 = create_test_memory(content="Second", created_at=base_time + 1)
    mem3 = create_test_memory(content="Third", created_at=base_time + 2)
    
    await document_store.add_memory(mem1)
    await document_store.add_memory(mem2)
    await document_store.add_memory(mem3)
    
    results = await document_store.query_memories()
    assert len(results) == 3
    assert results[0].content == "Third"
    assert results[1].content == "Second"
    assert results[2].content == "First"


# Test Concurrent Access
@pytest.mark.asyncio
async def test_concurrent_reads(document_store):
    """Test concurrent read operations."""
    # Add test memories
    memories = [create_test_memory(content=f"Memory {i}") for i in range(10)]
    for mem in memories:
        await document_store.add_memory(mem)
    
    # Perform concurrent reads
    async def read_memory(mem_id):
        return await document_store.get_memory(mem_id)
    
    tasks = [read_memory(mem.id) for mem in memories]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 10
    assert all(r is not None for r in results)


@pytest.mark.asyncio
async def test_concurrent_writes(document_store):
    """Test concurrent write operations."""
    memories = [create_test_memory(content=f"Memory {i}") for i in range(20)]
    
    # Perform concurrent writes
    tasks = [document_store.add_memory(mem) for mem in memories]
    results = await asyncio.gather(*tasks)
    
    assert all(r is True for r in results)
    
    # Verify all memories were stored
    for mem in memories:
        retrieved = await document_store.get_memory(mem.id)
        assert retrieved is not None


@pytest.mark.asyncio
async def test_concurrent_mixed_operations(document_store):
    """Test concurrent mixed read/write/update operations."""
    # Add initial memories
    memories = [create_test_memory(content=f"Memory {i}") for i in range(10)]
    for mem in memories:
        await document_store.add_memory(mem)
    
    # Define mixed operations
    async def read_op(mem_id):
        return await document_store.get_memory(mem_id)
    
    async def update_op(memory):
        memory.content = f"Updated {memory.content}"
        memory.updated_at = time.time()
        return await document_store.update_memory(memory)
    
    async def write_op(memory):
        return await document_store.add_memory(memory)
    
    # Execute mixed operations concurrently
    tasks = []
    tasks.extend([read_op(mem.id) for mem in memories[:3]])
    tasks.extend([update_op(mem) for mem in memories[3:6]])
    tasks.extend([write_op(create_test_memory(content=f"New {i}")) for i in range(5)])
    
    results = await asyncio.gather(*tasks)
    assert len(results) == 11


# Test Transaction Rollback
@pytest.mark.asyncio
async def test_transaction_commit(document_store):
    """Test successful transaction commit."""
    memory = create_test_memory(content="Transaction test")
    
    async with document_store.transaction() as conn:
        def insert():
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (
                    id, context_id, content, memory_type, created_at, importance, version, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id, memory.context_id, memory.content,
                memory.memory_type.value, memory.created_at,
                memory.importance, memory.version, "{}"
            ))
        
        await asyncio.to_thread(insert)
    
    # Verify memory was committed
    retrieved = await document_store.get_memory(memory.id)
    assert retrieved is not None


@pytest.mark.asyncio
async def test_transaction_rollback_on_error(document_store):
    """Test transaction rollback on error."""
    memory = create_test_memory(content="Rollback test")
    
    try:
        async with document_store.transaction() as conn:
            def insert():
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO memories (
                        id, context_id, content, memory_type, created_at, importance, version, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id, memory.context_id, memory.content,
                    memory.memory_type.value, memory.created_at,
                    memory.importance, memory.version, "{}"
                ))
            
            await asyncio.to_thread(insert)
            # Force an error
            raise ValueError("Intentional error")
    except ValueError:
        pass
    
    # Verify memory was rolled back
    retrieved = await document_store.get_memory(memory.id)
    assert retrieved is None


@pytest.mark.asyncio
async def test_transaction_rollback_on_constraint_violation(document_store):
    """Test transaction rollback on constraint violation."""
    memory1 = create_test_memory(id="same_id", content="First")
    memory2 = create_test_memory(id="same_id", content="Second")
    
    # Add first memory
    await document_store.add_memory(memory1)
    
    # Try to add duplicate in transaction
    try:
        async with document_store.transaction() as conn:
            def insert():
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO memories (
                        id, context_id, content, memory_type, created_at, importance, version, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory2.id, memory2.context_id, memory2.content,
                    memory2.memory_type.value, memory2.created_at,
                    memory2.importance, memory2.version, "{}"
                ))
            
            await asyncio.to_thread(insert)
    except sqlite3.IntegrityError:
        pass
    
    # Verify original memory is unchanged
    retrieved = await document_store.get_memory(memory1.id)
    assert retrieved.content == "First"


# Test Query Performance with 10K+ Records
@pytest.mark.asyncio
async def test_query_performance_10k_records(document_store):
    """Test query performance with 10,000+ records."""
    num_records = 10000
    context_id = "perf_test"
    
    # Insert 10K records in batches
    batch_size = 100
    for batch_start in range(0, num_records, batch_size):
        memories = [
            create_test_memory(
                content=f"Performance test memory {i}",
                context_id=context_id,
                importance=i % 10
            )
            for i in range(batch_start, min(batch_start + batch_size, num_records))
        ]
        
        tasks = [document_store.add_memory(mem) for mem in memories]
        await asyncio.gather(*tasks)
    
    # Test query performance
    start_time = time.time()
    results = await document_store.query_memories(
        context_id=context_id,
        min_importance=5,
        limit=100
    )
    query_time = time.time() - start_time
    
    assert len(results) == 100
    assert query_time < 1.0  # Should complete in under 1 second
    assert all(m.importance >= 5 for m in results)


@pytest.mark.asyncio
async def test_retrieval_performance_10k_records(document_store):
    """Test individual memory retrieval performance with 10K+ records."""
    num_records = 10000
    
    # Insert records
    memories = []
    for i in range(num_records):
        mem = create_test_memory(content=f"Memory {i}")
        await document_store.add_memory(mem)
        if i % 1000 == 0:  # Save some IDs for testing
            memories.append(mem)
    
    # Test retrieval performance
    start_time = time.time()
    for mem in memories:
        retrieved = await document_store.get_memory(mem.id)
        assert retrieved is not None
    retrieval_time = time.time() - start_time
    
    # Should retrieve 10 memories quickly even with 10K total
    assert retrieval_time < 0.5


@pytest.mark.asyncio
async def test_index_effectiveness_large_dataset(document_store):
    """Test that indices improve query performance on large dataset."""
    num_records = 5000
    
    # Insert records with various attributes
    for i in range(num_records):
        mem = create_test_memory(
            content=f"Memory {i}",
            context_id=f"ctx_{i % 10}",
            memory_type=MemoryType.CONVERSATION if i % 2 == 0 else MemoryType.FACT,
            importance=i % 10,
            created_at=time.time() + i
        )
        await document_store.add_memory(mem)
    
    # Test indexed query performance
    start_time = time.time()
    results = await document_store.query_memories(
        context_id="ctx_5",
        memory_type=MemoryType.CONVERSATION,
        min_importance=5
    )
    query_time = time.time() - start_time
    
    assert len(results) > 0
    assert query_time < 0.5  # Indexed queries should be fast
