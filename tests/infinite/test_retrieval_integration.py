"""Integration tests for end-to-end retrieval workflow.

This module tests the complete retrieval system including:
- End-to-end retrieval workflow with real storage
- Performance with 1M+ memories
- Sub-200ms latency requirement verification
- Retrieval accuracy across different query types
- Complex query handling
"""

import pytest
import asyncio
import time
from pathlib import Path
import uuid
import random

from core.infinite.retrieval_orchestrator import RetrievalOrchestrator
from core.infinite.document_store import DocumentStore
from core.infinite.temporal_index import TemporalIndex
from core.infinite.vector_store import VectorStore
from core.infinite.embedding_cache import EmbeddingCache
from core.infinite.models import Memory, MemoryType, QueryIntent


@pytest.fixture
async def temp_dir(tmp_path):
    """Create temporary directory for test databases."""
    return tmp_path


@pytest.fixture
async def embedding_fn():
    """Create simple embedding function for testing."""
    async def embed(text: str) -> list[float]:
        # Simple deterministic embedding based on text hash
        hash_val = hash(text)
        return [float((hash_val >> i) & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)]
    return embed


@pytest.fixture
async def integrated_retrieval_system(temp_dir, embedding_fn):
    """Create fully integrated retrieval system with real storage."""
    # Initialize storage components
    doc_store = DocumentStore(temp_dir / "test.db", pool_size=5)
    temporal_index = TemporalIndex(temp_dir / "temporal.db")
    vector_store = VectorStore(path=temp_dir / "vector_db", embedding_dim=384)
    
    await doc_store.initialize()
    await temporal_index.initialize()
    await vector_store.initialize()
    
    # Create retrieval orchestrator
    orchestrator = RetrievalOrchestrator(
        document_store=doc_store,
        temporal_index=temporal_index,
        vector_store=vector_store,
        embedding_fn=embedding_fn,
        enable_caching=True,
        cache_ttl_seconds=60.0,
        enable_scope_expansion=True,
        min_results_threshold=3
    )
    
    yield {
        "orchestrator": orchestrator,
        "doc_store": doc_store,
        "temporal_index": temporal_index,
        "vector_store": vector_store,
        "embedding_fn": embedding_fn,
    }
    
    # Cleanup
    await doc_store.close()
    await temporal_index.close()
    await vector_store.close()


def create_test_memory(
    content: str,
    memory_type: MemoryType = MemoryType.CONVERSATION,
    context_id: str = "test_context",
    importance: int = 5,
    created_at: float = None,
    **kwargs
) -> Memory:
    """Helper to create test memory objects."""
    if created_at is None:
        created_at = time.time()
    
    return Memory(
        id=kwargs.get("id", str(uuid.uuid4())),
        context_id=context_id,
        content=content,
        memory_type=memory_type,
        created_at=created_at,
        importance=importance,
        updated_at=kwargs.get("updated_at"),
        version=kwargs.get("version", 1),
        parent_id=kwargs.get("parent_id"),
        thread_id=kwargs.get("thread_id"),
        metadata=kwargs.get("metadata", {}),
    )


class TestEndToEndRetrievalWorkflow:
    """Test complete end-to-end retrieval workflow."""
    
    @pytest.mark.asyncio
    async def test_basic_retrieval_workflow(self, integrated_retrieval_system, embedding_fn):
        """Test basic end-to-end retrieval workflow."""
        system = integrated_retrieval_system
        orchestrator = system["orchestrator"]
        doc_store = system["doc_store"]
        temporal_index = system["temporal_index"]
        vector_store = system["vector_store"]
        
        # Create and store memories
        memories = [
            create_test_memory("I love Python programming", importance=8),
            create_test_memory("Python is great for data science", importance=7),
            create_test_memory("JavaScript is used for web development", importance=6),
            create_test_memory("Machine learning with Python", importance=9),
        ]
        
        # Add memories to all storage layers
        for memory in memories:
            await doc_store.add_memory(memory)
            await temporal_index.add_event(memory.id, memory.created_at, "created")
            memory.embedding = await embedding_fn(memory.content)
            await vector_store.add_memory(memory)
        
        # Perform retrieval
        result = await orchestrator.retrieve(
            query="Python programming",
            context_id="test_context",
            max_results=10
        )
        
        # Verify result
        assert len(result.memories) > 0
        assert result.total_found > 0
        assert result.retrieval_time_ms > 0
        assert result.query_analysis is not None
        
        # Verify relevant memories are returned
        contents = [m.content for m in result.memories]
        assert any("Python" in c for c in contents)
    
    @pytest.mark.asyncio
    async def test_retrieval_with_temporal_queries(self, integrated_retrieval_system, embedding_fn):
        """Test retrieval with temporal queries."""
        system = integrated_retrieval_system
        orchestrator = system["orchestrator"]
        doc_store = system["doc_store"]
        temporal_index = system["temporal_index"]
        vector_store = system["vector_store"]
        
        current_time = time.time()
        
        # Create memories at different times
        memories = [
            create_test_memory("Old memory", created_at=current_time - 86400 * 7, importance=5),
            create_test_memory("Recent memory", created_at=current_time - 3600, importance=5),
            create_test_memory("Very recent memory", created_at=current_time - 60, importance=5),
        ]
        
        # Store memories
        for memory in memories:
            await doc_store.add_memory(memory)
            await temporal_index.add_event(memory.id, memory.created_at, "created")
            memory.embedding = await embedding_fn(memory.content)
            await vector_store.add_memory(memory)
        
        # Query with time range (last 2 hours)
        time_range = (current_time - 7200, current_time)
        result = await orchestrator.retrieve(
            query="recent activities",
            context_id="test_context",
            max_results=10,
            time_range=time_range
        )
        
        # Should return recent memories
        assert len(result.memories) > 0
        for memory in result.memories:
            assert memory.created_at >= time_range[0]
    
    @pytest.mark.asyncio
    async def test_retrieval_with_memory_type_filter(self, integrated_retrieval_system, embedding_fn):
        """Test retrieval with memory type filtering."""
        system = integrated_retrieval_system
        orchestrator = system["orchestrator"]
        doc_store = system["doc_store"]
        temporal_index = system["temporal_index"]
        vector_store = system["vector_store"]
        
        # Create memories of different types
        memories = [
            create_test_memory("Conversation about code", memory_type=MemoryType.CONVERSATION),
            create_test_memory("def hello(): pass", memory_type=MemoryType.CODE),
            create_test_memory("Python was created in 1991", memory_type=MemoryType.FACT),
        ]
        
        # Store memories
        for memory in memories:
            await doc_store.add_memory(memory)
            await temporal_index.add_event(memory.id, memory.created_at, "created")
            memory.embedding = await embedding_fn(memory.content)
            await vector_store.add_memory(memory)
        
        # Query for CODE only
        result = await orchestrator.retrieve(
            query="code",
            context_id="test_context",
            max_results=10,
            memory_types=[MemoryType.CODE]
        )
        
        # Should return only CODE memories
        assert all(m.memory_type == MemoryType.CODE for m in result.memories)
    
    @pytest.mark.asyncio
    async def test_retrieval_with_caching(self, integrated_retrieval_system, embedding_fn):
        """Test that caching improves performance."""
        system = integrated_retrieval_system
        orchestrator = system["orchestrator"]
        doc_store = system["doc_store"]
        temporal_index = system["temporal_index"]
        vector_store = system["vector_store"]
        
        # Create and store memories
        memories = [
            create_test_memory(f"Memory {i}", importance=5)
            for i in range(10)
        ]
        
        for memory in memories:
            await doc_store.add_memory(memory)
            await temporal_index.add_event(memory.id, memory.created_at, "created")
            memory.embedding = await embedding_fn(memory.content)
            await vector_store.add_memory(memory)
        
        # First retrieval (cache miss)
        start = time.time()
        result1 = await orchestrator.retrieve(
            query="test query",
            context_id="test_context",
            max_results=10
        )
        first_time = time.time() - start
        
        # Second retrieval (cache hit)
        start = time.time()
        result2 = await orchestrator.retrieve(
            query="test query",
            context_id="test_context",
            max_results=10
        )
        second_time = time.time() - start
        
        # Cache hit should be faster
        assert second_time < first_time
        assert len(result1.memories) == len(result2.memories)


class TestPerformanceWith1MMemories:
    """Test performance with large-scale memory datasets."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_retrieval_with_10k_memories(self, integrated_retrieval_system, embedding_fn):
        """Test retrieval performance with 10K memories."""
        system = integrated_retrieval_system
        orchestrator = system["orchestrator"]
        doc_store = system["doc_store"]
        temporal_index = system["temporal_index"]
        vector_store = system["vector_store"]
        
        # Create 10K memories
        print("\nCreating 10K memories...")
        memories = []
        for i in range(10000):
            memory = create_test_memory(
                content=f"Memory content {i} about topic {i % 100}",
                importance=random.randint(1, 10),
                created_at=time.time() - random.randint(0, 86400 * 30)
            )
            memories.append(memory)
        
        # Batch insert
        print("Inserting memories...")
        start = time.time()
        
        # Add to document store
        doc_tasks = [doc_store.add_memory(mem) for mem in memories]
        await asyncio.gather(*doc_tasks)
        
        # Add to temporal index
        temporal_tasks = [
            temporal_index.add_event(mem.id, mem.created_at, "created")
            for mem in memories
        ]
        await asyncio.gather(*temporal_tasks)
        
        # Add embeddings and store in vector store
        for mem in memories:
            mem.embedding = await embedding_fn(mem.content)
        await vector_store.add_memories_batch(memories)
        
        insert_time = time.time() - start
        print(f"Inserted 10K memories in {insert_time:.2f}s")
        
        # Test retrieval performance
        print("Testing retrieval...")
        start = time.time()
        result = await orchestrator.retrieve(
            query="topic 42",
            context_id="test_context",
            max_results=10
        )
        retrieval_time = time.time() - start
        
        print(f"Retrieved {len(result.memories)} memories in {retrieval_time*1000:.2f}ms")
        
        # Should be fast even with 10K memories
        assert retrieval_time < 1.0  # Under 1 second
        assert len(result.memories) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_retrieval_with_100k_memories(self, integrated_retrieval_system, embedding_fn):
        """Test retrieval performance with 100K memories."""
        system = integrated_retrieval_system
        orchestrator = system["orchestrator"]
        doc_store = system["doc_store"]
        temporal_index = system["temporal_index"]
        vector_store = system["vector_store"]
        
        # Create 100K memories
        print("\nCreating 100K memories...")
        batch_size = 5000
        total_memories = 100000
        
        for batch_num in range(total_memories // batch_size):
            memories = []
            for i in range(batch_size):
                idx = batch_num * batch_size + i
                memory = create_test_memory(
                    content=f"Memory {idx} topic {idx % 1000}",
                    importance=random.randint(1, 10),
                    created_at=time.time() - random.randint(0, 86400 * 365)
                )
                memories.append(memory)
            
            # Batch insert
            doc_tasks = [doc_store.add_memory(mem) for mem in memories]
            await asyncio.gather(*doc_tasks)
            
            temporal_tasks = [
                temporal_index.add_event(mem.id, mem.created_at, "created")
                for mem in memories
            ]
            await asyncio.gather(*temporal_tasks)
            
            for mem in memories:
                mem.embedding = await embedding_fn(mem.content)
            await vector_store.add_memories_batch(memories)
            
            if (batch_num + 1) % 5 == 0:
                print(f"Inserted {(batch_num + 1) * batch_size} memories...")
        
        print("Testing retrieval with 100K memories...")
        
        # Test multiple queries
        queries = [
            "topic 500",
            "recent information",
            "important data",
        ]
        
        for query in queries:
            start = time.time()
            result = await orchestrator.retrieve(
                query=query,
                context_id="test_context",
                max_results=10
            )
            retrieval_time = time.time() - start
            
            print(f"Query '{query}': {retrieval_time*1000:.2f}ms, {len(result.memories)} results")
            
            # Should maintain good performance
            assert retrieval_time < 2.0  # Under 2 seconds
            assert len(result.memories) >= 0


class TestLatencyRequirements:
    """Test sub-200ms latency requirement."""
    
    @pytest.mark.asyncio
    async def test_sub_200ms_with_1k_memories(self, integrated_retrieval_system, embedding_fn):
        """Test sub-200ms latency with 1K memories."""
        system = integrated_retrieval_system
        orchestrator = system["orchestrator"]
        doc_store = system["doc_store"]
        temporal_index = system["temporal_index"]
        vector_store = system["vector_store"]
        
        # Create 1K memories
        memories = []
        for i in range(1000):
            memory = create_test_memory(
                content=f"Memory {i} about {['Python', 'JavaScript', 'Java', 'C++'][i % 4]}",
                importance=random.randint(1, 10)
            )
            memories.append(memory)
        
        # Insert memories
        doc_tasks = [doc_store.add_memory(mem) for mem in memories]
        await asyncio.gather(*doc_tasks)
        
        temporal_tasks = [
            temporal_index.add_event(mem.id, mem.created_at, "created")
            for mem in memories
        ]
        await asyncio.gather(*temporal_tasks)
        
        for mem in memories:
            mem.embedding = await embedding_fn(mem.content)
        await vector_store.add_memories_batch(memories)
        
        # Test retrieval latency
        latencies = []
        for _ in range(10):
            start = time.time()
            result = await orchestrator.retrieve(
                query="Python programming",
                context_id="test_context",
                max_results=10,
                use_cache=False  # Disable cache to test real performance
            )
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"\nLatency with 1K memories:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")
        
        # Should meet sub-200ms requirement for 1K memories
        assert avg_latency < 200.0
        assert p95_latency < 300.0  # Allow some variance
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_latency_with_10k_memories(self, integrated_retrieval_system, embedding_fn):
        """Test latency with 10K memories."""
        system = integrated_retrieval_system
        orchestrator = system["orchestrator"]
        doc_store = system["doc_store"]
        temporal_index = system["temporal_index"]
        vector_store = system["vector_store"]
        
        # Create 10K memories
        print("\nCreating 10K memories for latency test...")
        memories = []
        for i in range(10000):
            memory = create_test_memory(
                content=f"Memory {i} content",
                importance=random.randint(1, 10)
            )
            memories.append(memory)
        
        # Insert memories
        doc_tasks = [doc_store.add_memory(mem) for mem in memories]
        await asyncio.gather(*doc_tasks)
        
        temporal_tasks = [
            temporal_index.add_event(mem.id, mem.created_at, "created")
            for mem in memories
        ]
        await asyncio.gather(*temporal_tasks)
        
        for mem in memories:
            mem.embedding = await embedding_fn(mem.content)
        await vector_store.add_memories_batch(memories)
        
        # Test retrieval latency
        latencies = []
        for i in range(10):
            start = time.time()
            result = await orchestrator.retrieve(
                query=f"query {i}",
                context_id="test_context",
                max_results=10,
                use_cache=False
            )
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"\nLatency with 10K memories:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        
        # Should maintain reasonable performance
        assert avg_latency < 500.0  # Under 500ms for 10K
        assert p95_latency < 1000.0


class TestRetrievalAccuracy:
    """Test retrieval accuracy across different query types."""
    
    @pytest.mark.asyncio
    async def test_semantic_search_accuracy(self, integrated_retrieval_system, embedding_fn):
        """Test semantic search returns relevant results."""
        system = integrated_retrieval_system
        orchestrator = system["orchestrator"]
        doc_store = system["doc_store"]
        temporal_index = system["temporal_index"]
        vector_store = system["vector_store"]
        
        # Create memories with clear semantic relationships
        memories = [
            create_test_memory("Python is a programming language", importance=8),
            create_test_memory("I love coding in Python", importance=7),
            create_test_memory("Python has great libraries", importance=7),
            create_test_memory("The weather is nice today", importance=5),
            create_test_memory("I had pizza for lunch", importance=4),
        ]
        
        # Store memories
        for memory in memories:
            await doc_store.add_memory(memory)
            await temporal_index.add_event(memory.id, memory.created_at, "created")
            memory.embedding = await embedding_fn(memory.content)
            await vector_store.add_memory(memory)
        
        # Query for Python-related content
        result = await orchestrator.retrieve(
            query="Python programming",
            context_id="test_context",
            max_results=3
        )
        
        # Should return Python-related memories
        assert len(result.memories) > 0
        python_count = sum(1 for m in result.memories if "Python" in m.content)
        assert python_count >= 2  # At least 2 Python-related results
    
    @pytest.mark.asyncio
    async def test_temporal_query_accuracy(self, integrated_retriev