"""Standalone example demonstrating result composition.

This example shows how to use the ResultComposer to:
1. Interleave results by memory type
2. Group memories by type
3. Generate context breadcrumbs
4. Calculate confidence scores
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import uuid

# Direct imports bypassing core.__init__
from core.infinite.result_composer import ResultComposer
from core.infinite.retrieval_strategies import ScoredMemory
from core.infinite.models import Memory, MemoryType, QueryAnalysis, QueryIntent


def create_sample_memories():
    """Create sample scored memories for demonstration."""
    current_time = time.time()
    
    memories = [
        # Code memories
        ScoredMemory(
            memory=Memory(
                id=str(uuid.uuid4()),
                context_id="demo",
                content="def authenticate(username, password):\n    return check_credentials(username, password)",
                memory_type=MemoryType.CODE,
                created_at=current_time - 3600,
                importance=9
            ),
            score=0.95,
            strategy="semantic",
            metadata={"similarity": 0.95}
        ),
        ScoredMemory(
            memory=Memory(
                id=str(uuid.uuid4()),
                context_id="demo",
                content="class UserAuth:\n    def __init__(self):\n        self.users = {}",
                memory_type=MemoryType.CODE,
                created_at=current_time - 1800,
                importance=8
            ),
            score=0.88,
            strategy="fused",
            metadata={
                "strategies": ["semantic", "structural"],
                "strategy_scores": {"semantic": 0.85, "structural": 0.90}
            }
        ),
        
        # Conversation memories
        ScoredMemory(
            memory=Memory(
                id=str(uuid.uuid4()),
                context_id="demo",
                content="We discussed implementing authentication with JWT tokens",
                memory_type=MemoryType.CONVERSATION,
                created_at=current_time - 86400,
                importance=8
            ),
            score=0.85,
            strategy="temporal",
            metadata={"timestamp": current_time - 86400}
        ),
        ScoredMemory(
            memory=Memory(
                id=str(uuid.uuid4()),
                context_id="demo",
                content="User mentioned preferring OAuth2 for third-party authentication",
                memory_type=MemoryType.CONVERSATION,
                created_at=current_time - 172800,
                importance=7
            ),
            score=0.72,
            strategy="semantic",
            metadata={"similarity": 0.72}
        ),
        
        # Preference memory
        ScoredMemory(
            memory=Memory(
                id=str(uuid.uuid4()),
                context_id="demo",
                content="I prefer using bcrypt for password hashing",
                memory_type=MemoryType.PREFERENCE,
                created_at=current_time - 7200,
                importance=7
            ),
            score=0.75,
            strategy="semantic",
            metadata={"similarity": 0.75}
        ),
        
        # Fact memories
        ScoredMemory(
            memory=Memory(
                id=str(uuid.uuid4()),
                context_id="demo",
                content="JWT tokens consist of header, payload, and signature",
                memory_type=MemoryType.FACT,
                created_at=current_time - 259200,
                importance=6
            ),
            score=0.68,
            strategy="fulltext",
            metadata={"keyword_matches": 3}
        ),
        ScoredMemory(
            memory=Memory(
                id=str(uuid.uuid4()),
                context_id="demo",
                content="OAuth2 is an authorization framework",
                memory_type=MemoryType.FACT,
                created_at=current_time - 345600,
                importance=5
            ),
            score=0.55,
            strategy="fulltext",
            metadata={"keyword_matches": 2}
        ),
    ]
    
    return memories


def example_basic_composition():
    """Example 1: Basic result composition with interleaving."""
    print("=" * 80)
    print("Example 1: Basic Result Composition")
    print("=" * 80)
    
    # Create sample data
    memories = create_sample_memories()
    query_analysis = QueryAnalysis(
        intent=QueryIntent.CODE,
        entities=[("function", "authenticate")],
        code_patterns=["authenticate", "password"],
        keywords=["authentication", "password"],
        confidence=0.9
    )
    
    # Create composer with default settings
    composer = ResultComposer(
        interleave_by_type=True,
        include_breadcrumbs=True
    )
    
    # Compose results
    result = composer.compose(
        scored_memories=memories,
        query_analysis=query_analysis,
        retrieval_time_ms=45.5,
        limit=10
    )
    
    print(f"\nTotal memories found: {result.total_found}")
    print(f"Memories returned: {len(result.memories)}")
    print(f"Retrieval time: {result.retrieval_time_ms}ms")
    print(f"Overall confidence: {result.metadata['overall_confidence']:.2f}")
    print(f"Composition strategy: {result.metadata['composition_strategy']}")
    
    print("\n--- Memory Groups ---")
    for group in result.metadata['memory_groups']:
        print(f"  {group['type']}: {group['count']} memories, avg score: {group['avg_score']:.2f}")
    
    print("\n--- Retrieved Memories (Interleaved) ---")
    for i, memory in enumerate(result.memories, 1):
        print(f"{i}. [{memory.memory_type.value}] {memory.content[:60]}...")
        # Find corresponding breadcrumb
        breadcrumb = next(
            (bc for bc in result.metadata['breadcrumbs'] if bc['memory_id'] == memory.id),
            None
        )
        if breadcrumb:
            print(f"   Confidence: {breadcrumb['confidence']:.2f}")
            print(f"   Reasoning: {breadcrumb['reasoning']}")
            print(f"   Strategies: {', '.join(breadcrumb['strategies'])}")
        print()


def example_filtered_composition():
    """Example 2: Composition with confidence filtering."""
    print("=" * 80)
    print("Example 2: Filtered Composition (min confidence 0.7)")
    print("=" * 80)
    
    memories = create_sample_memories()
    query_analysis = QueryAnalysis(
        intent=QueryIntent.MIXED,
        keywords=["authentication"],
        confidence=0.8
    )
    
    # Create composer with confidence threshold
    composer = ResultComposer(
        interleave_by_type=True,
        min_confidence=0.7,
        include_breadcrumbs=False
    )
    
    result = composer.compose(
        scored_memories=memories,
        query_analysis=query_analysis,
        retrieval_time_ms=38.2
    )
    
    print(f"\nTotal memories found: {result.total_found}")
    print(f"Memories after filtering: {len(result.memories)}")
    print(f"Filtered out: {result.metadata['filtered_count']}")
    
    print("\n--- Filtered Results ---")
    for memory in result.memories:
        # Find original score
        original = next(m for m in memories if m.memory.id == memory.id)
        print(f"[{memory.memory_type.value}] Score: {original.score:.2f} - {memory.content[:50]}...")


def example_ranked_composition():
    """Example 3: Pure ranking without interleaving."""
    print("=" * 80)
    print("Example 3: Ranked Composition (no interleaving)")
    print("=" * 80)
    
    memories = create_sample_memories()
    query_analysis = QueryAnalysis(
        intent=QueryIntent.CODE,
        code_patterns=["authenticate"],
        confidence=0.85
    )
    
    # Create composer without interleaving
    composer = ResultComposer(
        interleave_by_type=False,
        include_breadcrumbs=True
    )
    
    result = composer.compose(
        scored_memories=memories,
        query_analysis=query_analysis,
        retrieval_time_ms=41.8,
        limit=5
    )
    
    print(f"\nTop {len(result.memories)} results (ranked by score):")
    
    for i, memory in enumerate(result.memories, 1):
        # Find original score
        original = next(m for m in memories if m.memory.id == memory.id)
        breadcrumb = next(
            bc for bc in result.metadata['breadcrumbs'] if bc['memory_id'] == memory.id
        )
        
        print(f"\n{i}. Score: {original.score:.2f} | Confidence: {breadcrumb['confidence']:.2f}")
        print(f"   Type: {memory.memory_type.value}")
        print(f"   Content: {memory.content[:70]}...")
        print(f"   Reasoning: {breadcrumb['reasoning']}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("RESULT COMPOSER DEMONSTRATION")
    print("=" * 80 + "\n")
    
    examples = [
        example_basic_composition,
        example_filtered_composition,
        example_ranked_composition,
    ]
    
    for example in examples:
        example()
        print("\n")


if __name__ == "__main__":
    main()
