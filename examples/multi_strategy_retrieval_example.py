"""Example demonstrating multi-strategy retrieval.

This example shows how to use different retrieval strategies
(semantic, temporal, full-text) and combine them with fusion.

Run this from the project root:
    python examples/multi_strategy_retrieval_example.py
"""

import asyncio
import time
from pathlib import Path
import tempfile
import shutil
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from infinite module to avoid core/__init__.py issues
from core.infinite.document_store import DocumentStore
from core.infinite.temporal_index import TemporalIndex
from core.infinite.vector_store import VectorStore
from core.infinite.models import Memory, MemoryType
from core.infinite.query_analyzer import QueryAnalyzer
from core.infinite.retrieval_strategies import (
    SemanticRetrievalStrategy,
    TemporalRetrievalStrategy,
    FullTextRetrievalStrategy,
    MultiStrategyRetrieval,
)


async def main():
    """Demonstrate multi-strategy retrieval."""
    # Create temporary directory for databases
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Initialize components
        db_path = temp_dir / "memories.db"
        document_store = DocumentStore(db_path)
        await document_store.initialize()
        
        temporal_index = TemporalIndex(db_path)
        await temporal_index.initialize()
        
        vector_store = VectorStore(path=temp_dir / "qdrant", embedding_dim=384)
        await vector_store.initialize()
        
        query_analyzer = QueryAnalyzer(use_spacy=False)
        
        print("\n=== Adding Sample Memories ===")
        
        # Add some sample memories
        current_time = time.time()
        memories = []
        
        # Recent preference
        mem1 = Memory(
            id="mem1",
            context_id="user_123",
            content="I prefer dark mode for coding because it reduces eye strain",
            memory_type=MemoryType.PREFERENCE,
            created_at=current_time - 3600,  # 1 hour ago
            importance=8,
            embedding=[0.1] * 384  # Dummy embedding
        )
        await document_store.add_memory(mem1)
        await vector_store.add_memory(mem1)
        memories.append(mem1)
        print(f"Added: {mem1.content[:50]}...")
        
        # Older preference
        mem2 = Memory(
            id="mem2",
            context_id="user_123",
            content="Python is my favorite programming language for data science",
            memory_type=MemoryType.PREFERENCE,
            created_at=current_time - 86400 * 7,  # 7 days ago
            importance=7,
            embedding=[0.2] * 384
        )
        await document_store.add_memory(mem2)
        await vector_store.add_memory(mem2)
        memories.append(mem2)
        print(f"Added: {mem2.content[:50]}...")
        
        # Code memory
        mem3 = Memory(
            id="mem3",
            context_id="user_123",
            content="def authenticate(username, password): return check_credentials(username, password)",
            memory_type=MemoryType.CODE,
            created_at=current_time - 86400 * 2,  # 2 days ago
            importance=9,
            embedding=[0.3] * 384
        )
        await document_store.add_memory(mem3)
        await vector_store.add_memory(mem3)
        memories.append(mem3)
        print(f"Added: {mem3.content[:50]}...")
        
        # Fact memory
        mem4 = Memory(
            id="mem4",
            context_id="user_123",
            content="The project deadline is next Friday and we need to finish testing",
            memory_type=MemoryType.FACT,
            created_at=current_time - 86400,  # 1 day ago
            importance=10,
            embedding=[0.4] * 384
        )
        await document_store.add_memory(mem4)
        await vector_store.add_memory(mem4)
        memories.append(mem4)
        print(f"Added: {mem4.content[:50]}...")
        
        print("\n=== Setting Up Retrieval Strategies ===")
        
        # Create individual strategies
        def dummy_embedding_fn(text):
            # In a real application, use a proper embedding model
            return [0.15] * 384
        
        semantic_strategy = SemanticRetrievalStrategy(
            vector_store=vector_store,
            document_store=document_store,
            embedding_fn=dummy_embedding_fn
        )
        print("✓ Semantic strategy initialized")
        
        temporal_strategy = TemporalRetrievalStrategy(
            temporal_index=temporal_index,
            document_store=document_store
        )
        print("✓ Temporal strategy initialized")
        
        fulltext_strategy = FullTextRetrievalStrategy(
            document_store=document_store
        )
        print("✓ Full-text strategy initialized")
        
        # Create multi-strategy retrieval
        multi_strategy = MultiStrategyRetrieval(
            semantic_strategy=semantic_strategy,
            temporal_strategy=temporal_strategy,
            fulltext_strategy=fulltext_strategy
        )
        print("✓ Multi-strategy retrieval initialized")
        
        print("\n=== Test Query 1: Semantic Search ===")
        query1 = "What are my coding preferences?"
        print(f"Query: {query1}")
        
        analysis1 = query_analyzer.analyze(query1)
        print(f"Intent: {analysis1.intent.value}")
        print(f"Keywords: {', '.join(analysis1.keywords)}")
        
        results1 = await multi_strategy.retrieve(
            query=query1,
            query_analysis=analysis1,
            context_id="user_123",
            limit=3
        )
        
        print(f"\nFound {len(results1)} results:")
        for i, result in enumerate(results1, 1):
            print(f"{i}. [{result.strategy}] Score: {result.score:.3f}")
            print(f"   {result.memory.content[:60]}...")
            if "strategies" in result.metadata:
                print(f"   Fused from: {', '.join(result.metadata['strategies'])}")
        
        print("\n=== Test Query 2: Temporal Search ===")
        query2 = "What happened in the last 2 days?"
        print(f"Query: {query2}")
        
        analysis2 = query_analyzer.analyze(query2)
        print(f"Intent: {analysis2.intent.value}")
        if analysis2.temporal_expressions:
            print(f"Temporal expressions: {len(analysis2.temporal_expressions)}")
        
        results2 = await multi_strategy.retrieve(
            query=query2,
            query_analysis=analysis2,
            context_id="user_123",
            limit=3
        )
        
        print(f"\nFound {len(results2)} results:")
        for i, result in enumerate(results2, 1):
            print(f"{i}. [{result.strategy}] Score: {result.score:.3f}")
            print(f"   {result.memory.content[:60]}...")
            age_hours = (current_time - result.memory.created_at) / 3600
            print(f"   Age: {age_hours:.1f} hours ago")
        
        print("\n=== Test Query 3: Full-Text Search ===")
        query3 = "project deadline testing"
        print(f"Query: {query3}")
        
        analysis3 = query_analyzer.analyze(query3)
        print(f"Intent: {analysis3.intent.value}")
        print(f"Keywords: {', '.join(analysis3.keywords)}")
        
        results3 = await multi_strategy.retrieve(
            query=query3,
            query_analysis=analysis3,
            context_id="user_123",
            limit=3
        )
        
        print(f"\nFound {len(results3)} results:")
        for i, result in enumerate(results3, 1):
            print(f"{i}. [{result.strategy}] Score: {result.score:.3f}")
            print(f"   {result.memory.content[:60]}...")
        
        print("\n=== Test Query 4: Multi-Strategy Fusion ===")
        query4 = "dark mode coding"
        print(f"Query: {query4}")
        
        analysis4 = query_analyzer.analyze(query4)
        print(f"Intent: {analysis4.intent.value}")
        
        results4 = await multi_strategy.retrieve(
            query=query4,
            query_analysis=analysis4,
            context_id="user_123",
            limit=3
        )
        
        print(f"\nFound {len(results4)} results:")
        for i, result in enumerate(results4, 1):
            print(f"{i}. [{result.strategy}] Score: {result.score:.3f}")
            print(f"   {result.memory.content[:60]}...")
            if "strategies" in result.metadata:
                print(f"   Fused from: {', '.join(result.metadata['strategies'])}")
                print(f"   Individual scores: {result.metadata['strategy_scores']}")
        
        print("\n=== Strategy Selection Demo ===")
        
        # Show which strategies are selected for different query types
        test_queries = [
            "What happened last week?",
            "Show me the login function",
            "What are my preferences?",
        ]
        
        for query in test_queries:
            analysis = query_analyzer.analyze(query)
            selected = multi_strategy._select_strategies(analysis)
            print(f"\nQuery: {query}")
            print(f"Intent: {analysis.intent.value}")
            print(f"Selected strategies: {', '.join(selected)}")
        
        # Cleanup
        await document_store.close()
        await temporal_index.close()
        await vector_store.close()
        
        print("\n=== Example Complete ===")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
