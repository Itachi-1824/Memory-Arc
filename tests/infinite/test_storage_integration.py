"""Integration tests for storage layer (SQLite, LMDB, Qdrant)."""

import pytest
import asyncio
import time
from pathlib import Path
import uuid

from core.infinite.document_store import DocumentStore
from core.infinite.embedding_cache import EmbeddingCache
from core.infinite.vector_store import VectorStore
from core.infinite.models import Memory, MemoryType


@pytest.fixture
async def storage_layer(temp_dir):
    """Create integrated storage layer with all components."""
    doc_store = DocumentStore(temp_dir / "test.db", pool_size=3)
    cache = EmbeddingCache(temp_dir / "cache", model_name="test_model")
    vector_store = VectorStore(path=temp_dir / "vector_db", embedding_dim=384)
    
    await doc_store.initialize()
    await cache.initialize()
    await vector_store.initialize()
    
    yield {
        "doc_store": doc_store,
        "cache": cache,
        "vector_store": vector_store,
    }
    
    await doc_store.close()
    await cache.close()
    await vector_store.close()


def create_test_memory(
    content: str = "Test content",
    memory_type: MemoryType = MemoryType.CONVERSATION,
    context_id: str = "test_context",
    importance: int = 5,
    embedding: list[float] | None = None,
    **kwargs
) -> Memory:
    """Helper to create test memory objects."""
    if embedding is None:
        # Create a simple embedding for testing
        embedding = [0.1] * 384
    
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
        embedding=embedding,
    )


# Test Data Flow Between Stores
@pytest.mark.asyncio
async def test_data_flow_across_all_stores(storage_layer):
    """Test data flows correctly between SQLite, LMDB, and Qdrant."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    # Create memory with embedding
    memory = create_test_memory(
        content="Integration test memory",
        memory_type=MemoryType.CONVERSATION,
        importance=7
    )
    
    # 1. Store in document store
    result = await doc_store.add_memory(memory)
    assert result is True
    
    # 2. Cache the embedding
    cache_result = await cache.put(memory.content, memory.embedding)
    assert cache_result is True
    
    # 3. Store in vector store
    vector_result = await vector_store.add_memory(memory)
    assert vector_result is True
    
    # Verify data in all stores
    # Document store
    retrieved_memory = await doc_store.get_memory(memory.id)
    assert retrieved_memory is not None
    assert retrieved_memory.content == memory.content
    
    # Embedding cache
    cached_embedding = await cache.get(memory.content)
    assert cached_embedding is not None
    assert cached_embedding == memory.embedding
    
    # Vector store
    search_results = await vector_store.search(
        query_vector=memory.embedding,
        context_id=memory.context_id,
        limit=5
    )
    assert len(search_results) > 0
    assert search_results[0][0] == memory.id


@pytest.mark.asyncio
async def test_batch_data_flow(storage_layer):
    """Test batch operations across all storage components."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    # Create multiple memories
    memories = [
        create_test_memory(
            content=f"Memory {i}",
            importance=i % 10,
            embedding=[0.1 * i] * 384
        )
        for i in range(20)
    ]
    
    # Batch add to document store
    doc_tasks = [doc_store.add_memory(mem) for mem in memories]
    doc_results = await asyncio.gather(*doc_tasks)
    assert all(doc_results)
    
    # Batch cache embeddings
    cache_tasks = [cache.put(mem.content, mem.embedding) for mem in memories]
    cache_results = await asyncio.gather(*cache_tasks)
    assert all(cache_results)
    
    # Batch add to vector store
    vector_count = await vector_store.add_memories_batch(memories)
    assert vector_count == 20
    
    # Verify all data is accessible
    for memory in memories[:5]:  # Check a sample
        # Document store
        doc_mem = await doc_store.get_memory(memory.id)
        assert doc_mem is not None
        
        # Cache
        cached = await cache.get(memory.content)
        assert cached is not None
        
        # Vector store (search should find it)
        results = await vector_store.search(
            query_vector=memory.embedding,
            limit=1
        )
        assert len(results) > 0


@pytest.mark.asyncio
async def test_cross_store_consistency(storage_layer):
    """Test data consistency across all storage components."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    memory = create_test_memory(
        content="Consistency test",
        memory_type=MemoryType.FACT,
        importance=8
    )
    
    # Add to all stores
    await doc_store.add_memory(memory)
    await cache.put(memory.content, memory.embedding)
    await vector_store.add_memory(memory)
    
    # Update memory
    memory.content = "Updated consistency test"
    memory.importance = 9
    memory.updated_at = time.time()
    new_embedding = [0.2] * 384
    memory.embedding = new_embedding
    
    # Update in all stores
    await doc_store.update_memory(memory)
    await cache.put(memory.content, new_embedding)
    await vector_store.add_memory(memory)  # Upsert
    
    # Verify consistency
    doc_mem = await doc_store.get_memory(memory.id)
    assert doc_mem.content == "Updated consistency test"
    assert doc_mem.importance == 9
    
    cached = await cache.get(memory.content)
    assert cached == new_embedding
    
    results = await vector_store.search(
        query_vector=new_embedding,
        memory_type=MemoryType.FACT,
        limit=1
    )
    assert len(results) > 0
    assert results[0][0] == memory.id


# Test Recovery from Storage Failures
@pytest.mark.asyncio
async def test_recovery_from_document_store_failure(storage_layer):
    """Test system continues when document store fails."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    memory = create_test_memory(content="Failure test")
    
    # Close document store to simulate failure
    await doc_store.close()
    
    # Cache and vector store should still work
    cache_result = await cache.put(memory.content, memory.embedding)
    assert cache_result is True
    
    vector_result = await vector_store.add_memory(memory)
    assert vector_result is True
    
    # Verify cache and vector store have the data
    cached = await cache.get(memory.content)
    assert cached is not None
    
    results = await vector_store.search(
        query_vector=memory.embedding,
        limit=1
    )
    assert len(results) > 0


@pytest.mark.asyncio
async def test_recovery_from_cache_failure(storage_layer):
    """Test system continues when cache fails."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    memory = create_test_memory(content="Cache failure test")
    
    # Close cache to simulate failure
    await cache.close()
    
    # Document store and vector store should still work
    doc_result = await doc_store.add_memory(memory)
    assert doc_result is True
    
    vector_result = await vector_store.add_memory(memory)
    assert vector_result is True
    
    # Verify data in other stores
    doc_mem = await doc_store.get_memory(memory.id)
    assert doc_mem is not None
    
    results = await vector_store.search(
        query_vector=memory.embedding,
        limit=1
    )
    assert len(results) > 0


@pytest.mark.asyncio
async def test_recovery_from_vector_store_failure(storage_layer):
    """Test system continues when vector store fails."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    memory = create_test_memory(content="Vector failure test")
    
    # Close vector store to simulate failure
    await vector_store.close()
    
    # Document store and cache should still work
    doc_result = await doc_store.add_memory(memory)
    assert doc_result is True
    
    cache_result = await cache.put(memory.content, memory.embedding)
    assert cache_result is True
    
    # Verify data in other stores
    doc_mem = await doc_store.get_memory(memory.id)
    assert doc_mem is not None
    
    cached = await cache.get(memory.content)
    assert cached is not None


@pytest.mark.asyncio
async def test_partial_failure_recovery(storage_layer):
    """Test recovery when some operations fail."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    memories = [
        create_test_memory(content=f"Memory {i}", embedding=[0.1 * i] * 384)
        for i in range(10)
    ]
    
    # Add first 5 successfully
    for mem in memories[:5]:
        await doc_store.add_memory(mem)
        await cache.put(mem.content, mem.embedding)
        await vector_store.add_memory(mem)
    
    # Simulate failure by closing cache
    await cache.close()
    
    # Try to add remaining 5 (cache will fail, others should succeed)
    for mem in memories[5:]:
        doc_result = await doc_store.add_memory(mem)
        assert doc_result is True
        
        # Cache will fail silently
        cache_result = await cache.put(mem.content, mem.embedding)
        # Don't assert - cache is closed
        
        vector_result = await vector_store.add_memory(mem)
        assert vector_result is True
    
    # Verify all 10 are in document store and vector store
    for mem in memories:
        doc_mem = await doc_store.get_memory(mem.id)
        assert doc_mem is not None
        
        results = await vector_store.search(
            query_vector=mem.embedding,
            limit=1
        )
        assert len(results) > 0


# Test Data Consistency Across Stores
@pytest.mark.asyncio
async def test_consistency_after_concurrent_operations(storage_layer):
    """Test data consistency after concurrent operations."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    # Create memories
    memories = [
        create_test_memory(
            content=f"Concurrent memory {i}",
            embedding=[0.1 * i] * 384
        )
        for i in range(30)
    ]
    
    # Concurrent operations across all stores
    async def add_to_all_stores(memory):
        await doc_store.add_memory(memory)
        await cache.put(memory.content, memory.embedding)
        await vector_store.add_memory(memory)
    
    tasks = [add_to_all_stores(mem) for mem in memories]
    await asyncio.gather(*tasks)
    
    # Verify consistency - all memories should be in all stores
    for memory in memories:
        # Document store
        doc_mem = await doc_store.get_memory(memory.id)
        assert doc_mem is not None
        assert doc_mem.content == memory.content
        
        # Cache
        cached = await cache.get(memory.content)
        assert cached is not None
        assert len(cached) == 384
        
        # Vector store
        results = await vector_store.search(
            query_vector=memory.embedding,
            limit=1
        )
        assert len(results) > 0


@pytest.mark.asyncio
async def test_consistency_with_updates(storage_layer):
    """Test consistency when updating across stores."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    memory = create_test_memory(content="Original content")
    
    # Initial add
    await doc_store.add_memory(memory)
    await cache.put(memory.content, memory.embedding)
    await vector_store.add_memory(memory)
    
    # Multiple updates
    for i in range(5):
        memory.content = f"Updated content {i}"
        memory.importance = 5 + i
        memory.updated_at = time.time()
        memory.embedding = [0.1 * (i + 1)] * 384
        
        await doc_store.update_memory(memory)
        await cache.put(memory.content, memory.embedding)
        await vector_store.add_memory(memory)  # Upsert
        
        # Small delay to ensure ordering
        await asyncio.sleep(0.01)
    
    # Verify final state is consistent
    doc_mem = await doc_store.get_memory(memory.id)
    assert doc_mem.content == "Updated content 4"
    assert doc_mem.importance == 9
    
    cached = await cache.get("Updated content 4")
    assert cached is not None
    
    results = await vector_store.search(
        query_vector=memory.embedding,
        limit=1
    )
    assert len(results) > 0
    assert results[0][0] == memory.id


@pytest.mark.asyncio
async def test_consistency_with_deletions(storage_layer):
    """Test consistency when deleting from stores."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    memories = [
        create_test_memory(content=f"Delete test {i}", embedding=[0.1 * i] * 384)
        for i in range(10)
    ]
    
    # Add all
    for mem in memories:
        await doc_store.add_memory(mem)
        await cache.put(mem.content, mem.embedding)
        await vector_store.add_memory(mem)
    
    # Delete half
    for mem in memories[:5]:
        await doc_store.delete_memory(mem.id)
        await cache.delete(mem.content)
        await vector_store.delete_memory(mem.id, mem.memory_type)
    
    # Verify deleted memories are gone from all stores
    for mem in memories[:5]:
        doc_mem = await doc_store.get_memory(mem.id)
        assert doc_mem is None
        
        cached = await cache.get(mem.content)
        assert cached is None
    
    # Verify remaining memories are still present
    for mem in memories[5:]:
        doc_mem = await doc_store.get_memory(mem.id)
        assert doc_mem is not None
        
        cached = await cache.get(mem.content)
        assert cached is not None


# Benchmark Storage Operations
@pytest.mark.asyncio
async def test_benchmark_write_performance(storage_layer):
    """Benchmark write performance across all stores."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    num_memories = 1000
    memories = [
        create_test_memory(
            content=f"Benchmark memory {i}",
            embedding=[0.001 * i] * 384
        )
        for i in range(num_memories)
    ]
    
    # Benchmark document store writes
    start = time.time()
    doc_tasks = [doc_store.add_memory(mem) for mem in memories]
    await asyncio.gather(*doc_tasks)
    doc_time = time.time() - start
    
    # Benchmark cache writes
    start = time.time()
    cache_tasks = [cache.put(mem.content, mem.embedding) for mem in memories]
    await asyncio.gather(*cache_tasks)
    cache_time = time.time() - start
    
    # Benchmark vector store writes (batch)
    start = time.time()
    await vector_store.add_memories_batch(memories)
    vector_time = time.time() - start
    
    # Performance assertions (generous thresholds for CI environments)
    assert doc_time < 20.0  # Should complete in under 20 seconds
    assert cache_time < 10.0  # Cache should be faster
    assert vector_time < 25.0  # Vector store batch should be reasonable
    
    print(f"\nWrite Performance (1000 memories):")
    print(f"  Document Store: {doc_time:.2f}s ({num_memories/doc_time:.0f} ops/s)")
    print(f"  Cache: {cache_time:.2f}s ({num_memories/cache_time:.0f} ops/s)")
    print(f"  Vector Store: {vector_time:.2f}s ({num_memories/vector_time:.0f} ops/s)")


@pytest.mark.asyncio
async def test_benchmark_read_performance(storage_layer):
    """Benchmark read performance across all stores."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    # Prepare data
    num_memories = 500
    memories = [
        create_test_memory(
            content=f"Read benchmark {i}",
            embedding=[0.001 * i] * 384
        )
        for i in range(num_memories)
    ]
    
    # Add data
    for mem in memories:
        await doc_store.add_memory(mem)
        await cache.put(mem.content, mem.embedding)
    await vector_store.add_memories_batch(memories)
    
    # Benchmark document store reads
    start = time.time()
    doc_tasks = [doc_store.get_memory(mem.id) for mem in memories]
    await asyncio.gather(*doc_tasks)
    doc_time = time.time() - start
    
    # Benchmark cache reads
    start = time.time()
    cache_tasks = [cache.get(mem.content) for mem in memories]
    await asyncio.gather(*cache_tasks)
    cache_time = time.time() - start
    
    # Benchmark vector store searches
    start = time.time()
    search_tasks = [
        vector_store.search(query_vector=mem.embedding, limit=5)
        for mem in memories[:100]  # Sample for vector search
    ]
    await asyncio.gather(*search_tasks)
    vector_time = time.time() - start
    
    # Performance assertions
    assert doc_time < 5.0
    assert cache_time < 1.0  # Cache should be very fast
    assert vector_time < 10.0
    
    print(f"\nRead Performance:")
    print(f"  Document Store (500 reads): {doc_time:.2f}s ({num_memories/doc_time:.0f} ops/s)")
    print(f"  Cache (500 reads): {cache_time:.2f}s ({num_memories/cache_time:.0f} ops/s)")
    print(f"  Vector Store (100 searches): {vector_time:.2f}s ({100/vector_time:.0f} ops/s)")


@pytest.mark.asyncio
async def test_benchmark_mixed_operations(storage_layer):
    """Benchmark mixed read/write operations."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    # Prepare initial data
    initial_memories = [
        create_test_memory(content=f"Initial {i}", embedding=[0.001 * i] * 384)
        for i in range(200)
    ]
    
    for mem in initial_memories:
        await doc_store.add_memory(mem)
        await cache.put(mem.content, mem.embedding)
    await vector_store.add_memories_batch(initial_memories)
    
    # Mixed operations
    async def mixed_operations():
        # Reads
        for mem in initial_memories[:50]:
            await doc_store.get_memory(mem.id)
            await cache.get(mem.content)
        
        # Writes
        new_memories = [
            create_test_memory(content=f"New {i}", embedding=[0.002 * i] * 384)
            for i in range(50)
        ]
        for mem in new_memories:
            await doc_store.add_memory(mem)
            await cache.put(mem.content, mem.embedding)
        
        # Searches
        for mem in initial_memories[:20]:
            await vector_store.search(query_vector=mem.embedding, limit=5)
    
    start = time.time()
    await mixed_operations()
    total_time = time.time() - start
    
    assert total_time < 10.0  # Should complete reasonably fast
    print(f"\nMixed Operations Time: {total_time:.2f}s")


@pytest.mark.asyncio
async def test_benchmark_large_dataset_query(storage_layer):
    """Benchmark query performance with large dataset."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    # Create large dataset
    num_memories = 5000
    context_id = "large_dataset"
    
    print(f"\nCreating {num_memories} memories...")
    memories = [
        create_test_memory(
            content=f"Large dataset memory {i}",
            context_id=context_id,
            importance=i % 10,
            embedding=[0.0001 * i] * 384
        )
        for i in range(num_memories)
    ]
    
    # Batch insert
    doc_tasks = [doc_store.add_memory(mem) for mem in memories]
    await asyncio.gather(*doc_tasks)
    await vector_store.add_memories_batch(memories)
    
    # Benchmark queries
    start = time.time()
    results = await doc_store.query_memories(
        context_id=context_id,
        min_importance=7,
        limit=100
    )
    query_time = time.time() - start
    
    assert len(results) == 100
    assert query_time < 1.0  # Should be fast with indices
    
    # Benchmark vector search
    query_vector = [0.5] * 384
    start = time.time()
    vector_results = await vector_store.search(
        query_vector=query_vector,
        context_id=context_id,
        limit=50
    )
    vector_search_time = time.time() - start
    
    assert len(vector_results) > 0
    assert vector_search_time < 2.0
    
    print(f"Query Performance (5000 memories):")
    print(f"  Document Store Query: {query_time:.3f}s")
    print(f"  Vector Search: {vector_search_time:.3f}s")


@pytest.mark.asyncio
async def test_storage_layer_scalability(storage_layer):
    """Test storage layer scales well with increasing data."""
    doc_store = storage_layer["doc_store"]
    cache = storage_layer["cache"]
    vector_store = storage_layer["vector_store"]
    
    # Test at different scales
    scales = [100, 500, 1000]
    results = {}
    
    for scale in scales:
        memories = [
            create_test_memory(
                content=f"Scale {scale} memory {i}",
                embedding=[0.0001 * i] * 384
            )
            for i in range(scale)
        ]
        
        # Measure write time
        start = time.time()
        doc_tasks = [doc_store.add_memory(mem) for mem in memories]
        await asyncio.gather(*doc_tasks)
        write_time = time.time() - start
        
        # Measure read time (sample)
        sample_size = min(50, scale)
        start = time.time()
        read_tasks = [doc_store.get_memory(mem.id) for mem in memories[:sample_size]]
        await asyncio.gather(*read_tasks)
        read_time = time.time() - start
        
        results[scale] = {
            "write_time": write_time,
            "read_time": read_time,
            "write_ops_per_sec": scale / write_time,
            "read_ops_per_sec": sample_size / read_time,
        }
    
    # Verify performance doesn't degrade significantly
    # Write performance should be relatively stable
    assert results[1000]["write_ops_per_sec"] > results[100]["write_ops_per_sec"] * 0.5
    
    print(f"\nScalability Results:")
    for scale, metrics in results.items():
        print(f"  {scale} memories:")
        print(f"    Write: {metrics['write_ops_per_sec']:.0f} ops/s")
        print(f"    Read: {metrics['read_ops_per_sec']:.0f} ops/s")
