"""Example usage of InfiniteContextEngine.

This example demonstrates:
1. Basic setup and initialization
2. Adding memories of different types
3. Retrieving memories with queries
4. Memory versioning and evolution
5. Temporal queries
6. Configuration options
7. Monitoring and metrics
"""

import asyncio
import time
from pathlib import Path

from core.infinite import (
    InfiniteContextEngine,
    InfiniteContextConfig,
    MemoryType,
    load_config_from_file,
    save_config_to_file,
)


def simple_embedding(text: str) -> list[float]:
    """Simple embedding function for demonstration."""
    # In production, use a real embedding model like OpenAI, Sentence Transformers, etc.
    hash_val = hash(text)
    return [float((hash_val >> i) & 0xFF) / 255.0 for i in range(1536)]


async def basic_usage_example():
    """Demonstrate basic usage of InfiniteContextEngine."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Create configuration
    config = InfiniteContextConfig(
        storage_path="./data/example_infinite_context",
        enable_caching=True,
        enable_code_tracking=False,
        use_spacy=False
    )
    
    # Initialize engine
    async with InfiniteContextEngine(config=config, embedding_fn=simple_embedding) as engine:
        # Add memories
        print("\n1. Adding memories...")
        
        memory_id1 = await engine.add_memory(
            content="Python is a high-level programming language",
            memory_type=MemoryType.FACT,
            context_id="user_123",
            importance=8
        )
        print(f"   Added fact: {memory_id1}")
        
        memory_id2 = await engine.add_memory(
            content="I love programming in Python",
            memory_type=MemoryType.CONVERSATION,
            context_id="user_123",
            importance=5
        )
        print(f"   Added conversation: {memory_id2}")
        
        memory_id3 = await engine.add_memory(
            content="I prefer dark mode for my IDE",
            memory_type=MemoryType.PREFERENCE,
            context_id="user_123",
            importance=6
        )
        print(f"   Added preference: {memory_id3}")
        
        # Retrieve memories
        print("\n2. Retrieving memories...")
        
        result = await engine.retrieve(
            query="Python programming",
            context_id="user_123",
            max_results=10
        )
        
        print(f"   Found {result.total_found} memories")
        print(f"   Query intent: {result.query_analysis.intent.value}")
        print(f"   Retrieval time: {result.retrieval_time_ms:.2f}ms")
        
        for i, memory in enumerate(result.memories[:3], 1):
            print(f"\n   Memory {i}:")
            print(f"     Type: {memory.memory_type.value}")
            print(f"     Content: {memory.content[:60]}...")
            print(f"     Importance: {memory.importance}")
        
        # Get metrics
        print("\n3. System metrics...")
        metrics = engine.get_metrics()
        print(f"   Total memories: {metrics.total_memories}")
        print(f"   Total queries: {metrics.total_queries}")
        print(f"   Avg query latency: {metrics.avg_query_latency_ms:.2f}ms")
        print(f"   Cache hit rate: {metrics.cache_hit_rate:.2%}")


async def memory_versioning_example():
    """Demonstrate memory versioning and evolution."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Memory Versioning")
    print("="*60)
    
    config = InfiniteContextConfig(
        storage_path="./data/example_versioning",
        enable_caching=True
    )
    
    async with InfiniteContextEngine(config=config, embedding_fn=simple_embedding) as engine:
        print("\n1. Creating initial preference...")
        
        # Initial preference
        pref_id1 = await engine.add_memory(
            content="I like apples",
            memory_type=MemoryType.PREFERENCE,
            context_id="user_456"
        )
        print(f"   Created: {pref_id1}")
        
        await asyncio.sleep(0.1)
        
        # Update preference (create new version)
        print("\n2. Updating preference...")
        pref_id2 = await engine.add_memory(
            content="I like mangoes",
            memory_type=MemoryType.PREFERENCE,
            context_id="user_456",
            supersedes=pref_id1  # This creates a new version
        )
        print(f"   Updated: {pref_id2}")
        
        # Get version history
        print("\n3. Retrieving version history...")
        history = await engine.get_version_history(pref_id2)
        
        print(f"   Found {len(history)} versions:")
        for i, version in enumerate(history, 1):
            print(f"     Version {i}: {version.content}")
            print(f"       Created: {time.ctime(version.created_at)}")
        
        # Detect contradictions
        print("\n4. Detecting contradictions...")
        contradictions = await engine.detect_contradictions(
            context_id="user_456"
        )
        
        if contradictions:
            print(f"   Found {len(contradictions)} contradictions")
            for mem1, mem2, similarity in contradictions[:3]:
                print(f"     - '{mem1.content}' vs '{mem2.content}' (similarity: {similarity:.2f})")
        else:
            print("   No contradictions detected")


async def temporal_queries_example():
    """Demonstrate temporal queries."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Temporal Queries")
    print("="*60)
    
    config = InfiniteContextConfig(
        storage_path="./data/example_temporal",
        enable_caching=True
    )
    
    async with InfiniteContextEngine(config=config, embedding_fn=simple_embedding) as engine:
        print("\n1. Adding time-stamped memories...")
        
        # Add memory at T1
        timestamp_t1 = time.time()
        await engine.add_memory(
            content="Event at time T1: Started project",
            memory_type=MemoryType.FACT,
            context_id="project_789"
        )
        print(f"   T1: {time.ctime(timestamp_t1)}")
        
        await asyncio.sleep(0.2)
        
        # Add memory at T2
        timestamp_t2 = time.time()
        await engine.add_memory(
            content="Event at time T2: Completed milestone 1",
            memory_type=MemoryType.FACT,
            context_id="project_789"
        )
        print(f"   T2: {time.ctime(timestamp_t2)}")
        
        await asyncio.sleep(0.2)
        
        # Add memory at T3
        timestamp_t3 = time.time()
        await engine.add_memory(
            content="Event at time T3: Completed milestone 2",
            memory_type=MemoryType.FACT,
            context_id="project_789"
        )
        print(f"   T3: {time.ctime(timestamp_t3)}")
        
        # Query at different times
        print("\n2. Querying at different times...")
        
        # Query at T2 (should see T1 and T2, not T3)
        result = await engine.query_at_time(
            query="milestone",
            timestamp=timestamp_t2 + 0.1,
            context_id="project_789"
        )
        
        print(f"\n   At T2 ({time.ctime(timestamp_t2)}):")
        print(f"   Found {len(result.memories)} memories")
        for memory in result.memories:
            print(f"     - {memory.content}")


async def configuration_example():
    """Demonstrate different configuration options."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Configuration Options")
    print("="*60)
    
    print("\n1. Using preset configurations...")
    
    # Minimal configuration
    print("\n   a) Minimal (low resources):")
    minimal_config = InfiniteContextConfig.minimal()
    minimal_config.storage_path = "./data/example_minimal"
    print(f"      Cache: {minimal_config.enable_caching}")
    print(f"      Batch size: {minimal_config.batch_size}")
    print(f"      Max concurrent: {minimal_config.max_concurrent_queries}")
    
    # Balanced configuration
    print("\n   b) Balanced (general use):")
    balanced_config = InfiniteContextConfig.balanced()
    balanced_config.storage_path = "./data/example_balanced"
    print(f"      Cache: {balanced_config.enable_caching}")
    print(f"      Cache size: {balanced_config.cache_max_size_gb}GB")
    print(f"      Batch size: {balanced_config.batch_size}")
    
    # Performance configuration
    print("\n   c) Performance (high throughput):")
    perf_config = InfiniteContextConfig.performance()
    perf_config.storage_path = "./data/example_performance"
    print(f"      Cache: {perf_config.enable_caching}")
    print(f"      Cache size: {perf_config.cache_max_size_gb}GB")
    print(f"      Use spaCy: {perf_config.use_spacy}")
    print(f"      Max concurrent: {perf_config.max_concurrent_queries}")
    
    # Save configuration to file
    print("\n2. Saving configuration to file...")
    config_path = Path("./data/example_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    save_config_to_file(balanced_config, config_path)
    print(f"   Saved to: {config_path}")
    
    # Load configuration from file
    print("\n3. Loading configuration from file...")
    loaded_config = load_config_from_file(config_path)
    print(f"   Loaded: {loaded_config.storage_path}")
    print(f"   Model: {loaded_config.model_name}")


async def monitoring_example():
    """Demonstrate monitoring and health checks."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Monitoring and Health Checks")
    print("="*60)
    
    config = InfiniteContextConfig(
        storage_path="./data/example_monitoring",
        enable_caching=True
    )
    
    async with InfiniteContextEngine(config=config, embedding_fn=simple_embedding) as engine:
        # Add some data
        print("\n1. Adding test data...")
        for i in range(50):
            await engine.add_memory(
                content=f"Test memory {i}",
                context_id="monitoring_test"
            )
        
        # Perform some queries
        print("\n2. Performing queries...")
        for i in range(20):
            await engine.retrieve(
                query=f"test {i}",
                context_id="monitoring_test"
            )
        
        # Check health status
        print("\n3. Health status:")
        health = engine.get_health_status()
        for component, status in health.items():
            status_symbol = "✅" if status == "healthy" else "⚠️"
            print(f"   {status_symbol} {component}: {status}")
        
        # Get detailed metrics
        print("\n4. Detailed metrics:")
        metrics = engine.get_metrics()
        print(f"   Total memories: {metrics.total_memories}")
        print(f"   Total queries: {metrics.total_queries}")
        print(f"   Avg query latency: {metrics.avg_query_latency_ms:.2f}ms")
        print(f"   P95 query latency: {metrics.p95_query_latency_ms:.2f}ms")
        print(f"   P99 query latency: {metrics.p99_query_latency_ms:.2f}ms")
        print(f"   Cache hits: {metrics.cache_hits}")
        print(f"   Cache misses: {metrics.cache_misses}")
        print(f"   Cache hit rate: {metrics.cache_hit_rate:.2%}")
        print(f"   Storage size: {metrics.storage_size_bytes / 1024 / 1024:.2f}MB")
        print(f"   Uptime: {metrics.uptime_seconds:.2f}s")


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("INFINITE CONTEXT ENGINE EXAMPLES")
    print("="*60)
    
    try:
        await basic_usage_example()
        await memory_versioning_example()
        await temporal_queries_example()
        await configuration_example()
        await monitoring_example()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
