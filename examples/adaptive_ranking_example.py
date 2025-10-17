"""Example demonstrating adaptive ranking system for memory retrieval.

This example shows how the adaptive ranking system uses multi-signal scoring
to intelligently rank retrieved memories based on:
- Semantic similarity
- Recency (with time-sensitive boosting)
- Importance
- Redundancy penalization
"""

import asyncio
import time
from pathlib import Path
import tempfile
import shutil

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.infinite.retrieval_strategies import (
    AdaptiveRanker,
    SemanticRetrievalStrategy,
    TemporalRetrievalStrategy,
    FullTextRetrievalStrategy,
    MultiStrategyRetrieval,
    ScoredMemory,
)
from core.infinite.models import Memory, MemoryType, QueryAnalysis, QueryIntent
from core.infinite.document_store import DocumentStore
from core.infinite.temporal_index import TemporalIndex
from core.infinite.vector_store import VectorStore
from core.infinite.query_analyzer import QueryAnalyzer


async def main():
    """Demonstrate adaptive ranking system."""
    print("=" * 80)
    print("Adaptive Ranking System Example")
    print("=" * 80)
    
    # Create temporary directory for databases
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Initialize storage components
        print("\n1. Initializing storage components...")
        db_path = temp_dir / "example.db"
        
        document_store = DocumentStore(db_path)
        await document_store.initialize()
        
        temporal_index = TemporalIndex(db_path)
        await temporal_index.initialize()
        
        vector_store = VectorStore(path=temp_dir / "qdrant", embedding_dim=384)
        await vector_store.initialize()
        
        # Create sample memories with different characteristics
        print("\n2. Creating sample memories...")
        current_time = time.time()
        
        memories = [
            Memory(
                id="mem1",
                context_id="user_123",
                content="I prefer dark mode for coding because it reduces eye strain",
                memory_type=MemoryType.PREFERENCE,
                created_at=current_time - 3600,  # 1 hour ago
                importance=8,
                embedding=[0.1] * 384
            ),
            Memory(
                id="mem2",
                context_id="user_123",
                content="Python is my favorite programming language for data science",
                memory_type=MemoryType.PREFERENCE,
                created_at=current_time - 86400 * 30,  # 30 days ago
                importance=7,
                embedding=[0.2] * 384
            ),
            Memory(
                id="mem3",
                context_id="user_123",
                content="The project deadline is next Friday at 5 PM",
                memory_type=MemoryType.FACT,
                created_at=current_time - 86400,  # 1 day ago
                importance=10,
                embedding=[0.3] * 384
            ),
            Memory(
                id="mem4",
                context_id="user_123",
                content="I like using VS Code for development",
                memory_type=MemoryType.PREFERENCE,
                created_at=current_time - 86400 * 7,  # 7 days ago
                importance=6,
                embedding=[0.15] * 384  # Similar to mem1
            ),
            Memory(
                id="mem5",
                context_id="user_123",
                content="Meeting with the team scheduled for tomorrow at 10 AM",
                memory_type=MemoryType.FACT,
                created_at=current_time - 7200,  # 2 hours ago
                importance=9,
                embedding=[0.35] * 384
            ),
        ]
        
        # Store memories
        for memory in memories:
            await document_store.add_memory(memory)
            await vector_store.add_memory(memory)
        
        print(f"Created {len(memories)} memories with varying recency and importance")
        
        # Create retrieval strategies
        print("\n3. Setting up retrieval strategies...")
        
        def dummy_embedding_fn(text):
            """Simple embedding function for demo."""
            return [0.1] * 384
        
        semantic_strategy = SemanticRetrievalStrategy(
            vector_store=vector_store,
            document_store=document_store,
            embedding_fn=dummy_embedding_fn
        )
        
        temporal_strategy = TemporalRetrievalStrategy(
            temporal_index=temporal_index,
            document_store=document_store
        )
        
        fulltext_strategy = FullTextRetrievalStrategy(
            document_store=document_store
        )
        
        # Create adaptive ranker with custom weights
        print("\n4. Creating adaptive ranker...")
        ranker = AdaptiveRanker(
            recency_weight=0.3,
            importance_weight=0.2,
            semantic_weight=0.5,
            redundancy_threshold=0.7
        )
        
        # Create multi-strategy retrieval with adaptive ranking
        multi_strategy = MultiStrategyRetrieval(
            semantic_strategy=semantic_strategy,
            temporal_strategy=temporal_strategy,
            fulltext_strategy=fulltext_strategy,
            ranker=ranker
        )
        
        # Create query analyzer
        query_analyzer = QueryAnalyzer(use_spacy=False)
        
        # Example 1: Preference query (should boost recent preferences)
        print("\n" + "=" * 80)
        print("Example 1: Preference Query")
        print("=" * 80)
        
        query1 = "What are my coding preferences?"
        print(f"\nQuery: '{query1}'")
        
        query_analysis1 = query_analyzer.analyze(query1)
        print(f"Detected intent: {query_analysis1.intent.value}")
        
        results1 = await multi_strategy.retrieve(
            query=query1,
            query_analysis=query_analysis1,
            context_id="user_123",
            limit=5,
            use_adaptive_ranking=True
        )
        
        print(f"\nRetrieved {len(results1)} results with adaptive ranking:")
        for i, result in enumerate(results1, 1):
            print(f"\n{i}. {result.memory.content[:60]}...")
            print(f"   Final Score: {result.score:.3f}")
            if 'ranking' in result.metadata:
                ranking = result.metadata['ranking']
                print(f"   - Semantic: {ranking['semantic_score']:.3f}")
                print(f"   - Recency: {ranking['recency_score']:.3f}")
                print(f"   - Importance: {ranking['importance_score']:.3f}")
                if 'redundancy_penalty' in ranking:
                    print(f"   - Redundancy Penalty: {ranking['redundancy_penalty']:.3f}")
        
        # Example 2: Temporal query (should boost recency heavily)
        print("\n" + "=" * 80)
        print("Example 2: Temporal Query")
        print("=" * 80)
        
        query2 = "What happened yesterday?"
        print(f"\nQuery: '{query2}'")
        
        query_analysis2 = query_analyzer.analyze(query2)
        print(f"Detected intent: {query_analysis2.intent.value}")
        
        results2 = await multi_strategy.retrieve(
            query=query2,
            query_analysis=query_analysis2,
            context_id="user_123",
            limit=5,
            use_adaptive_ranking=True
        )
        
        print(f"\nRetrieved {len(results2)} results with adaptive ranking:")
        for i, result in enumerate(results2, 1):
            print(f"\n{i}. {result.memory.content[:60]}...")
            print(f"   Final Score: {result.score:.3f}")
            if 'ranking' in result.metadata:
                ranking = result.metadata['ranking']
                print(f"   - Semantic: {ranking['semantic_score']:.3f}")
                print(f"   - Recency: {ranking['recency_score']:.3f} (boosted for temporal query)")
                print(f"   - Importance: {ranking['importance_score']:.3f}")
        
        # Example 3: Factual query (should boost importance)
        print("\n" + "=" * 80)
        print("Example 3: Factual Query")
        print("=" * 80)
        
        query3 = "What are the important deadlines?"
        print(f"\nQuery: '{query3}'")
        
        query_analysis3 = query_analyzer.analyze(query3)
        print(f"Detected intent: {query_analysis3.intent.value}")
        
        results3 = await multi_strategy.retrieve(
            query=query3,
            query_analysis=query_analysis3,
            context_id="user_123",
            limit=5,
            use_adaptive_ranking=True
        )
        
        print(f"\nRetrieved {len(results3)} results with adaptive ranking:")
        for i, result in enumerate(results3, 1):
            print(f"\n{i}. {result.memory.content[:60]}...")
            print(f"   Final Score: {result.score:.3f}")
            print(f"   Importance Level: {result.memory.importance}/10")
            if 'ranking' in result.metadata:
                ranking = result.metadata['ranking']
                print(f"   - Semantic: {ranking['semantic_score']:.3f}")
                print(f"   - Recency: {ranking['recency_score']:.3f}")
                print(f"   - Importance: {ranking['importance_score']:.3f} (boosted for factual query)")
        
        # Example 4: Compare with and without adaptive ranking
        print("\n" + "=" * 80)
        print("Example 4: Comparison - With vs Without Adaptive Ranking")
        print("=" * 80)
        
        query4 = "Tell me about my preferences"
        print(f"\nQuery: '{query4}'")
        
        query_analysis4 = query_analyzer.analyze(query4)
        
        # Without adaptive ranking
        results_without = await multi_strategy.retrieve(
            query=query4,
            query_analysis=query_analysis4,
            context_id="user_123",
            limit=3,
            use_adaptive_ranking=False
        )
        
        print("\nWithout Adaptive Ranking:")
        for i, result in enumerate(results_without, 1):
            print(f"{i}. {result.memory.content[:50]}... (score: {result.score:.3f})")
        
        # With adaptive ranking
        results_with = await multi_strategy.retrieve(
            query=query4,
            query_analysis=query_analysis4,
            context_id="user_123",
            limit=3,
            use_adaptive_ranking=True
        )
        
        print("\nWith Adaptive Ranking:")
        for i, result in enumerate(results_with, 1):
            print(f"{i}. {result.memory.content[:50]}... (score: {result.score:.3f})")
            if 'ranking' in result.metadata:
                ranking = result.metadata['ranking']
                print(f"   Multi-signal: semantic={ranking['semantic_score']:.2f}, "
                      f"recency={ranking['recency_score']:.2f}, "
                      f"importance={ranking['importance_score']:.2f}")
        
        print("\n" + "=" * 80)
        print("Key Takeaways:")
        print("=" * 80)
        print("1. Adaptive ranking combines multiple signals (semantic, recency, importance)")
        print("2. Weights adapt based on query intent (temporal, code, preference, factual)")
        print("3. Recent memories get boosted for time-sensitive queries")
        print("4. Important memories get boosted for factual queries")
        print("5. Redundant results are penalized to promote diversity")
        print("6. The system provides transparency through ranking metadata")
        
    finally:
        # Cleanup
        await document_store.close()
        await temporal_index.close()
        await vector_store.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
