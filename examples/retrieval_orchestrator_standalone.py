"""Standalone example of RetrievalOrchestrator without external dependencies.

This example uses mock components to demonstrate the orchestrator's capabilities
without requiring Qdrant or other external services.
"""

import asyncio
import time
from typing import Optional
from unittest.mock import AsyncMock

from core.infinite.retrieval_orchestrator import RetrievalOrchestrator
from core.infinite.models import Memory, MemoryType


class MockDocumentStore:
    """Mock document store for standalone example."""
    
    def __init__(self):
        self.memories = {}
    
    async def initialize(self):
        pass
    
    async def add_memory(self, memory: Memory):
        self.memories[memory.id] = memory
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        return self.memories.get(memory_id)
    
    async def query_memories(
        self,
        context_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100
    ):
        results = [
            m for m in self.memories.values()
            if m.context_id == context_id
        ]
        if memory_type:
            results = [m for m in results if m.memory_type == memory_type]
        return results[:limit]
    
    async def close(self):
        pass


class MockTemporalIndex:
    """Mock temporal index for standalone example."""
    
    def __init__(self):
        self.events = []
    
    async def initialize(self):
        pass
    
    async def add_event(self, memory_id: str, timestamp: float, event_type: str):
        self.events.append({
            "memory_id": memory_id,
            "timestamp": timestamp,
            "event_type": event_type
        })
    
    async def query_by_time_range(
        self,
        start_time: float,
        end_time: float,
        event_type: str = "created",
        limit: int = 100
    ):
        results = [
            e for e in self.events
            if start_time <= e["timestamp"] <= end_time
            and e["event_type"] == event_type
        ]
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    async def close(self):
        pass


class MockVectorStore:
    """Mock vector store for standalone example."""
    
    def __init__(self):
        self.vectors = {}
    
    async def initialize(self):
        pass
    
    async def add_memory(
        self,
        memory_id: str,
        embedding: list[float],
        memory_type: MemoryType,
        context_id: str,
        importance: int,
        created_at: float
    ):
        self.vectors[memory_id] = {
            "embedding": embedding,
            "memory_type": memory_type,
            "context_id": context_id,
            "importance": importance,
            "created_at": created_at
        }
    
    async def search(
        self,
        query_vector: list[float],
        memory_type: Optional[MemoryType] = None,
        context_id: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.0
    ):
        # Simple mock: return all memories with random scores
        results = []
        for memory_id, data in self.vectors.items():
            if context_id and data["context_id"] != context_id:
                continue
            if memory_type and data["memory_type"] != memory_type:
                continue
            
            # Mock similarity score based on importance
            score = 0.5 + (data["importance"] / 20.0)
            if score >= min_score:
                results.append((memory_id, score))
        
        # Sort by score and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    async def close(self):
        pass


async def simple_embedding_fn(text: str) -> list[float]:
    """Simple embedding function for demonstration."""
    return [float(hash(text + str(i)) % 100) / 100.0 for i in range(384)]


async def main():
    """Main standalone example."""
    print("=== RetrievalOrchestrator Standalone Example ===\n")
    
    # Initialize mock storage components
    print("1. Initializing mock storage components...")
    document_store = MockDocumentStore()
    await document_store.initialize()
    
    temporal_index = MockTemporalIndex()
    await temporal_index.initialize()
    
    vector_store = MockVectorStore()
    await vector_store.initialize()
    
    # Initialize retrieval orchestrator
    print("2. Creating RetrievalOrchestrator...")
    orchestrator = RetrievalOrchestrator(
        document_store=document_store,
        temporal_index=temporal_index,
        vector_store=vector_store,
        embedding_fn=simple_embedding_fn,
        enable_caching=True,
        cache_ttl_seconds=60.0,
        enable_scope_expansion=True,
        min_results_threshold=2
    )
    
    print("   ✓ Orchestrator initialized with:")
    print("     - Query analyzer (rule-based, no AI)")
    print("     - Multi-strategy retrieval (semantic, temporal, full-text)")
    print("     - Adaptive ranker (multi-signal scoring)")
    print("     - Result composer (interleaving, breadcrumbs)")
    print("     - Query caching (60s TTL)")
    print("     - Adaptive scope expansion")
    print()
    
    # Add sample memories
    print("3. Adding sample memories...")
    current_time = time.time()
    context_id = "demo_user"
    
    memories = [
        Memory(
            id="mem1",
            context_id=context_id,
            content="I prefer using Python for backend development",
            memory_type=MemoryType.PREFERENCE,
            created_at=current_time - 86400 * 5,  # 5 days ago
            importance=8
        ),
        Memory(
            id="mem2",
            context_id=context_id,
            content="Machine learning is fascinating, especially neural networks",
            memory_type=MemoryType.CONVERSATION,
            created_at=current_time - 3600 * 2,  # 2 hours ago
            importance=6
        ),
        Memory(
            id="mem3",
            context_id=context_id,
            content="def calculate_total(items): return sum(item.price for item in items)",
            memory_type=MemoryType.CODE,
            created_at=current_time - 86400,  # 1 day ago
            importance=7
        ),
        Memory(
            id="mem4",
            context_id=context_id,
            content="I like dark mode for all my development tools",
            memory_type=MemoryType.PREFERENCE,
            created_at=current_time - 86400 * 10,  # 10 days ago
            importance=5
        ),
        Memory(
            id="mem5",
            context_id=context_id,
            content="The Eiffel Tower is located in Paris, France",
            memory_type=MemoryType.FACT,
            created_at=current_time - 86400 * 30,  # 30 days ago
            importance=4
        ),
        Memory(
            id="mem6",
            context_id=context_id,
            content="async def fetch_data(url): return await http_client.get(url)",
            memory_type=MemoryType.CODE,
            created_at=current_time - 86400 * 2,  # 2 days ago
            importance=9
        ),
    ]
    
    for memory in memories:
        await document_store.add_memory(memory)
        await temporal_index.add_event(
            memory_id=memory.id,
            timestamp=memory.created_at,
            event_type="created"
        )
        
        embedding = await simple_embedding_fn(memory.content)
        await vector_store.add_memory(
            memory_id=memory.id,
            embedding=embedding,
            memory_type=memory.memory_type,
            context_id=memory.context_id,
            importance=memory.importance,
            created_at=memory.created_at
        )
    
    print(f"   ✓ Added {len(memories)} memories")
    print()
    
    # Example queries
    examples = [
        {
            "title": "Preference Query",
            "query": "What do I prefer for development?",
            "description": "Should detect preference intent and return relevant preferences"
        },
        {
            "title": "Temporal Query",
            "query": "What happened yesterday?",
            "description": "Should detect temporal intent and filter by time"
        },
        {
            "title": "Code Query",
            "query": "Show me the calculate_total function",
            "description": "Should detect code intent and return code memories"
        },
        {
            "title": "Factual Query",
            "query": "Tell me about Paris",
            "description": "Should detect factual intent and return facts"
        },
        {
            "title": "Recent Query",
            "query": "What did I work on recently?",
            "description": "Should boost recent memories"
        },
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{3 + i}. Example {i}: {example['title']}")
        print(f"   Query: '{example['query']}'")
        print(f"   Expected: {example['description']}")
        
        result = await orchestrator.retrieve(
            query=example['query'],
            context_id=context_id,
            max_results=5
        )
        
        print(f"   Results:")
        print(f"     - Found: {len(result.memories)} memories")
        print(f"     - Time: {result.retrieval_time_ms:.2f}ms")
        print(f"     - Intent: {result.query_analysis.intent.value}")
        print(f"     - Confidence: {result.query_analysis.confidence:.2f}")
        
        if result.query_analysis.temporal_expressions:
            print(f"     - Temporal: {[expr for expr, _ in result.query_analysis.temporal_expressions]}")
        
        if result.query_analysis.code_patterns:
            print(f"     - Code patterns: {result.query_analysis.code_patterns}")
        
        print(f"   Top results:")
        for j, memory in enumerate(result.memories[:3], 1):
            age_hours = (current_time - memory.created_at) / 3600
            print(f"     {j}. [{memory.memory_type.value}] {memory.content[:50]}... ({age_hours:.1f}h ago)")
        
        print()
    
    # Demonstrate caching
    print(f"{3 + len(examples) + 1}. Caching Demonstration")
    print("   Running same query twice to show cache benefit...")
    
    query = "What do I prefer?"
    
    # First call
    start = time.time()
    result1 = await orchestrator.retrieve(query, context_id, max_results=5)
    time1 = (time.time() - start) * 1000
    
    # Second call (cached)
    start = time.time()
    result2 = await orchestrator.retrieve(query, context_id, max_results=5)
    time2 = (time.time() - start) * 1000
    
    print(f"   First call: {time1:.2f}ms")
    print(f"   Second call (cached): {time2:.2f}ms")
    print(f"   Speedup: {time1/time2:.1f}x faster")
    print()
    
    # Cache statistics
    print(f"{3 + len(examples) + 2}. Cache Statistics")
    stats = orchestrator.get_cache_stats()
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Active entries: {stats['active_entries']}")
    print(f"   Cache enabled: {stats['cache_enabled']}")
    print()
    
    # Query analysis demonstration
    print(f"{3 + len(examples) + 3}. Query Analysis Examples")
    test_queries = [
        "What is machine learning?",
        "Show me code from last week",
        "I prefer TypeScript",
        "What happened 2 days ago?",
    ]
    
    for query in test_queries:
        analysis = await orchestrator.analyze_query(query)
        print(f"   '{query}'")
        print(f"     → Intent: {analysis.intent.value} (confidence: {analysis.confidence:.2f})")
        if analysis.keywords:
            print(f"     → Keywords: {', '.join(analysis.keywords[:3])}")
        if analysis.temporal_expressions:
            print(f"     → Temporal: {[expr for expr, _ in analysis.temporal_expressions]}")
        if analysis.code_patterns:
            print(f"     → Code: {analysis.code_patterns[:2]}")
    
    print()
    print("=== Example completed successfully! ===")


if __name__ == "__main__":
    asyncio.run(main())
