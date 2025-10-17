"""Standalone example demonstrating priority-based chunk selection."""

import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Import only what we need directly
class MemoryType(Enum):
    """Types of memories that can be stored."""
    CONVERSATION = "conversation"
    CODE = "code"
    FACT = "fact"
    SUMMARY = "summary"
    PREFERENCE = "preference"
    DOCUMENT = "document"


class BoundaryType(Enum):
    """Types of content boundaries for chunking."""
    PARAGRAPH = "paragraph"
    FUNCTION = "function"
    CLASS = "class"
    SECTION = "section"
    SENTENCE = "sentence"


@dataclass
class Memory:
    """Enhanced memory entry with versioning support."""
    id: str
    context_id: str
    content: str
    memory_type: MemoryType
    created_at: float
    importance: int = 5
    updated_at: float | None = None
    version: int = 1
    parent_id: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class Chunk:
    """A chunk of content with metadata."""
    id: str
    content: str
    chunk_index: int
    total_chunks: int
    token_count: int
    relevance_score: float = 0.0
    start_pos: int = 0
    end_pos: int = 0
    boundary_type: BoundaryType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Now import the chunk selector directly from the module file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "chunk_selector",
    Path(__file__).parent.parent / "core" / "infinite" / "chunk_selector.py"
)
chunk_selector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunk_selector_module)
ChunkSelector = chunk_selector_module.ChunkSelector


def main():
    """Demonstrate chunk selection with various strategies."""
    
    print("=" * 70)
    print("Priority-Based Chunk Selection Example")
    print("=" * 70)
    
    # Create sample chunks with different characteristics
    current_time = time.time()
    
    chunks = [
        # Recent, low importance, somewhat relevant
        Chunk(
            id="chunk_1",
            content="Python is a popular programming language for data science",
            chunk_index=0,
            total_chunks=5,
            token_count=12,
            boundary_type=BoundaryType.PARAGRAPH,
            metadata={
                "importance": 4,
                "timestamp": current_time - 3600,  # 1 hour ago
                "source": "recent_article"
            }
        ),
        # Old, high importance, very relevant
        Chunk(
            id="chunk_2",
            content="Machine learning algorithms require careful data preprocessing",
            chunk_index=1,
            total_chunks=5,
            token_count=10,
            boundary_type=BoundaryType.PARAGRAPH,
            metadata={
                "importance": 9,
                "timestamp": current_time - (365 * 24 * 3600),  # 1 year ago
                "source": "important_paper"
            }
        ),
        # Medium age, medium importance, not relevant
        Chunk(
            id="chunk_3",
            content="Cooking pasta requires boiling water and salt",
            chunk_index=2,
            total_chunks=5,
            token_count=9,
            boundary_type=BoundaryType.SENTENCE,
            metadata={
                "importance": 5,
                "timestamp": current_time - (7 * 24 * 3600),  # 1 week ago
                "source": "recipe_book"
            }
        ),
        # Recent, high importance, very relevant
        Chunk(
            id="chunk_4",
            content="Deep learning models excel at pattern recognition tasks",
            chunk_index=3,
            total_chunks=5,
            token_count=10,
            boundary_type=BoundaryType.PARAGRAPH,
            metadata={
                "importance": 8,
                "timestamp": current_time - 7200,  # 2 hours ago
                "source": "latest_research"
            }
        ),
        # Old, low importance, not relevant
        Chunk(
            id="chunk_5",
            content="The weather forecast predicts rain tomorrow",
            chunk_index=4,
            total_chunks=5,
            token_count=8,
            boundary_type=BoundaryType.SENTENCE,
            metadata={
                "importance": 3,
                "timestamp": current_time - (30 * 24 * 3600),  # 1 month ago
                "source": "weather_report"
            }
        ),
    ]
    
    # Example 1: Balanced selection
    print("\n1. BALANCED SELECTION (equal weights)")
    print("-" * 70)
    
    selector_balanced = ChunkSelector(
        relevance_weight=0.33,
        importance_weight=0.33,
        recency_weight=0.34
    )
    
    query = "machine learning and data science"
    scored_chunks = selector_balanced.select_chunks(
        chunks,
        query=query,
        current_time=current_time
    )
    
    print(f"Query: '{query}'")
    print(f"\nTop chunks (balanced scoring):")
    for i, sc in enumerate(scored_chunks, 1):
        print(f"\n{i}. {sc.chunk.id} - Final Score: {sc.final_score:.3f}")
        print(f"   Content: {sc.chunk.content[:60]}...")
        print(f"   Relevance: {sc.relevance_score:.3f} | "
              f"Importance: {sc.importance_score:.3f} | "
              f"Recency: {sc.recency_score:.3f}")
        print(f"   Source: {sc.chunk.metadata.get('source', 'unknown')}")
    
    # Example 2: Relevance-focused selection
    print("\n\n2. RELEVANCE-FOCUSED SELECTION")
    print("-" * 70)
    
    selector_relevance = ChunkSelector(
        relevance_weight=0.8,
        importance_weight=0.1,
        recency_weight=0.1
    )
    
    scored_chunks = selector_relevance.select_chunks(
        chunks,
        query=query,
        current_time=current_time,
        max_chunks=3
    )
    
    print(f"Query: '{query}'")
    print(f"\nTop 3 chunks (relevance-focused):")
    for i, sc in enumerate(scored_chunks, 1):
        print(f"\n{i}. {sc.chunk.id} - Final Score: {sc.final_score:.3f}")
        print(f"   Content: {sc.chunk.content[:60]}...")
        print(f"   Relevance: {sc.relevance_score:.3f} (primary factor)")
    
    # Example 3: Recency-focused selection
    print("\n\n3. RECENCY-FOCUSED SELECTION")
    print("-" * 70)
    
    selector_recency = ChunkSelector(
        relevance_weight=0.1,
        importance_weight=0.1,
        recency_weight=0.8,
        recency_decay_hours=168.0  # 1 week decay
    )
    
    scored_chunks = selector_recency.select_chunks(
        chunks,
        current_time=current_time,
        max_chunks=3
    )
    
    print("Top 3 chunks (recency-focused):")
    for i, sc in enumerate(scored_chunks, 1):
        age_hours = (current_time - sc.chunk.metadata["timestamp"]) / 3600
        print(f"\n{i}. {sc.chunk.id} - Final Score: {sc.final_score:.3f}")
        print(f"   Content: {sc.chunk.content[:60]}...")
        print(f"   Age: {age_hours:.1f} hours | Recency: {sc.recency_score:.3f}")
    
    # Example 4: Importance-focused selection
    print("\n\n4. IMPORTANCE-FOCUSED SELECTION")
    print("-" * 70)
    
    selector_importance = ChunkSelector(
        relevance_weight=0.1,
        importance_weight=0.8,
        recency_weight=0.1
    )
    
    scored_chunks = selector_importance.select_chunks(
        chunks,
        current_time=current_time,
        max_chunks=3
    )
    
    print("Top 3 chunks (importance-focused):")
    for i, sc in enumerate(scored_chunks, 1):
        print(f"\n{i}. {sc.chunk.id} - Final Score: {sc.final_score:.3f}")
        print(f"   Content: {sc.chunk.content[:60]}...")
        print(f"   Importance: {sc.importance_score:.3f} (primary factor)")
    
    # Example 5: Using minimum score threshold
    print("\n\n5. SELECTION WITH MINIMUM SCORE THRESHOLD")
    print("-" * 70)
    
    scored_chunks = selector_balanced.select_chunks(
        chunks,
        query=query,
        current_time=current_time,
        min_score=0.5
    )
    
    print(f"Query: '{query}'")
    print(f"Minimum score threshold: 0.5")
    print(f"\nChunks meeting threshold: {len(scored_chunks)}")
    for i, sc in enumerate(scored_chunks, 1):
        print(f"\n{i}. {sc.chunk.id} - Final Score: {sc.final_score:.3f}")
        print(f"   Content: {sc.chunk.content[:60]}...")
    
    # Example 6: Custom recency decay
    print("\n\n6. CUSTOM RECENCY DECAY")
    print("-" * 70)
    
    # Fast decay (24 hours)
    selector_fast_decay = ChunkSelector(
        relevance_weight=0.2,
        importance_weight=0.2,
        recency_weight=0.6,
        recency_decay_hours=24.0
    )
    
    # Slow decay (1 year)
    selector_slow_decay = ChunkSelector(
        relevance_weight=0.2,
        importance_weight=0.2,
        recency_weight=0.6,
        recency_decay_hours=8760.0
    )
    
    test_chunk = chunks[1]  # 1 year old chunk
    
    fast_score = selector_fast_decay.compute_recency_score(test_chunk, current_time=current_time)
    slow_score = selector_slow_decay.compute_recency_score(test_chunk, current_time=current_time)
    
    print(f"Chunk age: 1 year")
    print(f"Fast decay (24h): {fast_score:.4f}")
    print(f"Slow decay (1y): {slow_score:.4f}")
    print(f"\nFast decay penalizes old content more heavily")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
