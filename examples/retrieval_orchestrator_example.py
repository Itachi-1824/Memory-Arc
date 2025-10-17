"""Example usage of RetrievalOrchestrator for intelligent memory retrieval.

This example demonstrates:
1. Setting up the retrieval orchestrator
2. Performing queries with different intents
3. Using caching for performance
4. Adaptive search scope expansion
5. Filtering by memory types and time ranges
"""

import asyncio
import time
from core.infinite.retrieval_orchestrator import RetrievalOrchestrator
from core.infinite.document_store import DocumentStore
from core.infinite.temporal_index import TemporalIndex
from core.infinite.vector_store import VectorStore
from core.infinite.code_change_store import CodeChangeStore
from core.infinite.models import Memory, MemoryType


async def simple_embedding_fn(text: str) -> list[float]:
    """Simple embedding function for demonstration."""
    # In production, use a real embedding model
    return [float(hash(text + str(i)) % 100) / 100.0 for i in range(384)]


async def main():
    """Main example function."""
    print("=== RetrievalOrchestrator Example ===\n")
    
    # Initialize storage components
    print("1. Initializing storage components...")
    document_store = DocumentStore(db_path=":memory:")
    await document_store.initialize()
    
    temporal_index = TemporalIndex(db_path=":memory:")
    await temporal_index.initialize()
    
    vector_store = VectorStore(
        collection_name="example_memories",
        host="localhost",
        port=6333
    )
    await vector_store.initialize()
    
    code_change_store = CodeChangeStore(db_path=":memory:")
    await code_change_store.initialize()
    
    # Initialize retrieval orchestrator
    print("2. Creating RetrievalOrchestrator...")
    orchestrator = RetrievalOrchestrator(
        document_store=document_store,
        temporal_index=temporal_index,
        vector_store=vector_store,
        code_change_store=code_change_store,
        embedding_fn=simple_embedding_fn,
        use_spacy=False,  # Set to True if spaCy is installed
        enable_caching=True,
        cache_ttl_seconds=300.0,
        enable_scope_expansion=True,
        min_results_threshold=3
    )
    
    # Add sample memories
    print("3. Adding sample memories...")
    current_time = time.time()
    context_id = "user_123"
    
    memories = [
        Memory(
            id="mem1",
            context_id=context_id,
            content="I prefer dark mode for my IDE",
            memory_type=MemoryType.PREFERENCE,
            created_at=current_time - 86400 * 7,  # 1 week ago
            importance=7
        ),
        Memory(
            id="mem2",
            context_id=context_id,
            content="Python is great for data science and machine learning",
            memory_type=MemoryType.CONVERSATION,
            created_at=current_time - 3600,  # 1 hour ago
            importance=5
        ),
        Memory(
            id="mem3",
            context_id=context_id,
            content="def authenticate(username, password): return check_credentials(username, password)",
            memory_type=MemoryType.CODE,
            created_at=current_time - 86400,  # 1 day ago
            importance=8
        ),
        Memory(
            id="mem4",
            context_id=context_id,
            content="I like working with TypeScript for frontend development",
            memory_type=MemoryType.PREFERENCE,
            created_at=current_time - 86400 * 3,  # 3 days ago
            importance=6
        ),
        Memory(
            id="mem5",
            context_id=context_id,
            content="The capital of France is Paris",
            memory_type=MemoryType.FACT,
            created_at=current_time - 86400 * 30,  # 30 days ago
            importance=4
        ),
    ]
    
    for memory in memories:
        await document_store.add_memory(memory)
        await temporal_index.add_event(
            memory_id=memory.id,
            timestamp=memory.created_at,
            event_type="created"
        )
        
        # Add to vector store
        embedding = await simple_embedding_fn(memory.content)
        await vector_store.add_memory(
            memory_id=memory.id,
            embedding=embedding,
            memory_type=memory.memory_type,
            context_id=memory.context_id,
            importance=memory.importance,
            created_at=memory.created_at
        )
    
    print(f"Added {len(memories)} memories\n")
    
    # Example 1: Basic retrieval
    print("4. Example 1: Basic retrieval")
    print("Query: 'What are my preferences?'")
    result = await orchestrator.retrieve(
        query="What are my preferences?",
        context_id=context_id,
        max_results=10
    )
    
    print(f"Found {len(result.memories)} memories in {result.retrieval_time_ms:.2f}ms")
    print(f"Query intent: {result.query_analysis.intent.value}")
    print(f"Total found: {result.total_found}")
    for i, memory in enumerate(result.memories[:3], 1):
        print(f"  {i}. [{memory.memory_type.value}] {memory.content[:60]}...")
    print()
    
    # Example 2: Temporal query
    print("5. Example 2: Temporal query")
    print("Query: 'What did I do yesterday?'")
    result = await orchestrator.retrieve(
        query="What did I do yesterday?",
        context_id=context_id,
        max_results=10
    )
    
    print(f"Found {len(result.memories)} memories in {result.retrieval_time_ms:.2f}ms")
    print(f"Query intent: {result.query_analysis.intent.value}")
    print(f"Temporal expressions: {result.query_analysis.temporal_expressions}")
    for i, memory in enumerate(result.memories[:3], 1):
        age_days = (current_time - memory.created_at) / 86400
        print(f"  {i}. [{memory.memory_type.value}] {memory.content[:60]}... ({age_days:.1f} days ago)")
    print()
    
    # Example 3: Code query
    print("6. Example 3: Code query")
    print("Query: 'Show me the authenticate function'")
    result = await orchestrator.retrieve(
        query="Show me the authenticate function",
        context_id=context_id,
        max_results=10
    )
    
    print(f"Found {len(result.memories)} memories in {result.retrieval_time_ms:.2f}ms")
    print(f"Query intent: {result.query_analysis.intent.value}")
    print(f"Code patterns: {result.query_analysis.code_patterns}")
    for i, memory in enumerate(result.memories[:3], 1):
        print(f"  {i}. [{memory.memory_type.value}] {memory.content[:60]}...")
    print()
    
    # Example 4: Cached retrieval
    print("7. Example 4: Cached retrieval (same query)")
    print("Query: 'What are my preferences?' (cached)")
    result = await orchestrator.retrieve(
        query="What are my preferences?",
        context_id=context_id,
        max_results=10
    )
    
    print(f"Found {len(result.memories)} memories in {result.retrieval_time_ms:.2f}ms (from cache)")
    print()
    
    # Example 5: Filter by memory type
    print("8. Example 5: Filter by memory type")
    print("Query: 'programming' with CODE filter")
    result = await orchestrator.retrieve(
        query="programming",
        context_id=context_id,
        max_results=10,
        memory_types=[MemoryType.CODE]
    )
    
    print(f"Found {len(result.memories)} CODE memories in {result.retrieval_time_ms:.2f}ms")
    for i, memory in enumerate(result.memories, 1):
        print(f"  {i}. {memory.content[:60]}...")
    print()
    
    # Example 6: Time range filter
    print("9. Example 6: Time range filter")
    print("Query: 'recent activities' (last 2 days)")
    time_range = (current_time - 86400 * 2, current_time)
    result = await orchestrator.retrieve(
        query="recent activities",
        context_id=context_id,
        max_results=10,
        time_range=time_range
    )
    
    print(f"Found {len(result.memories)} memories in {result.retrieval_time_ms:.2f}ms")
    for i, memory in enumerate(result.memories, 1):
        age_hours = (current_time - memory.created_at) / 3600
        print(f"  {i}. [{memory.memory_type.value}] {memory.content[:60]}... ({age_hours:.1f} hours ago)")
    print()
    
    # Example 7: Cache statistics
    print("10. Cache statistics")
    stats = orchestrator.get_cache_stats()
    print(f"Total cache entries: {stats['total_entries']}")
    print(f"Active entries: {stats['active_entries']}")
    print(f"Expired entries: {stats['expired_entries']}")
    print(f"Cache enabled: {stats['cache_enabled']}")
    print(f"TTL: {stats['ttl_seconds']} seconds")
    print()
    
    # Example 8: Clear cache
    print("11. Clearing cache")
    cleared = orchestrator.clear_cache()
    print(f"Cleared {cleared} cache entries")
    print()
    
    # Example 9: Query analysis
    print("12. Query analysis examples")
    queries = [
        "What is Python?",
        "Show me code from last week",
        "I prefer dark mode",
        "What did I do yesterday?",
    ]
    
    for query in queries:
        analysis = await orchestrator.analyze_query(query)
        print(f"Query: '{query}'")
        print(f"  Intent: {analysis.intent.value}")
        print(f"  Confidence: {analysis.confidence:.2f}")
        print(f"  Keywords: {analysis.keywords[:5]}")
        if analysis.temporal_expressions:
            print(f"  Temporal: {[expr for expr, _ in analysis.temporal_expressions]}")
        if analysis.code_patterns:
            print(f"  Code patterns: {analysis.code_patterns[:3]}")
        print()
    
    # Cleanup
    print("13. Cleaning up...")
    await document_store.close()
    await temporal_index.close()
    await vector_store.close()
    await code_change_store.close()
    
    print("Example completed!")


if __name__ == "__main__":
    asyncio.run(main())
