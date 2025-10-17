"""Standalone example of ChunkManager - can be run independently."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

try:
    from core.infinite.chunk_manager import create_chunk_manager
    from core.infinite.models import Memory, MemoryType
    from core.infinite.chunk_formatter import FormatType
except ImportError as e:
    print(f"Import error: {e}")
    print("\nPlease ensure you're running from the project root directory:")
    print("  python examples/chunk_manager_standalone.py")
    sys.exit(1)


def main():
    """Demonstrate ChunkManager capabilities."""
    
    print("=" * 70)
    print("ChunkManager Standalone Demo")
    print("=" * 70)
    
    # Create a chunk manager for GPT-4
    print("\n1. Creating ChunkManager for GPT-4...")
    manager = create_chunk_manager(
        model_name="gpt-4",
        max_tokens=8000,
        overlap_tokens=100
    )
    print("   ✓ ChunkManager created")
    
    # Example 1: Process a document
    print("\n2. Processing a markdown document...")
    document = """
# Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## Supervised Learning

In supervised learning, the model learns from labeled training data. Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

## Unsupervised Learning

Unsupervised learning works with unlabeled data to find patterns. Examples include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Deep Learning

Deep learning uses neural networks with multiple layers to learn complex patterns.
It has revolutionized fields like computer vision and natural language processing.

## Conclusion

Machine learning continues to advance rapidly, with new techniques emerging regularly.
"""
    
    # Chunk and format the document
    formatted_chunks = manager.process_and_format(
        document,
        content_type="markdown",
        format_type=FormatType.MARKDOWN,
        return_single_string=False
    )
    
    print(f"   ✓ Created {len(formatted_chunks)} chunks")
    print(f"   ✓ Total tokens: {sum(fc.token_count for fc in formatted_chunks)}")
    
    # Show first chunk
    if formatted_chunks:
        print("\n   First chunk preview:")
        print("   " + "-" * 66)
        preview = formatted_chunks[0].content[:200].replace('\n', '\n   ')
        print(f"   {preview}...")
        print("   " + "-" * 66)
    
    # Example 2: Process memories with query-based selection
    print("\n3. Processing memories with query-based selection...")
    
    memories = [
        Memory(
            id="m1",
            context_id="user_001",
            content="User is learning Python for data analysis.",
            memory_type=MemoryType.FACT,
            created_at=time.time() - 7200,  # 2 hours ago
            importance=8
        ),
        Memory(
            id="m2",
            context_id="user_001",
            content="User asked about pandas DataFrame operations.",
            memory_type=MemoryType.CONVERSATION,
            created_at=time.time() - 3600,  # 1 hour ago
            importance=7
        ),
        Memory(
            id="m3",
            context_id="user_001",
            content="User prefers examples with real-world datasets.",
            memory_type=MemoryType.PREFERENCE,
            created_at=time.time() - 1800,  # 30 minutes ago
            importance=6
        ),
        Memory(
            id="m4",
            context_id="user_001",
            content="User completed tutorial on data visualization.",
            memory_type=MemoryType.FACT,
            created_at=time.time() - 900,  # 15 minutes ago
            importance=5
        )
    ]
    
    # Process with query
    result = manager.process_and_format(
        memories,
        query="Python pandas data analysis",
        format_type=FormatType.JSON,
        max_chunks=3
    )
    
    print(f"   ✓ Selected {len(result)} most relevant chunks from {len(memories)} memories")
    
    # Example 3: Stream large content
    print("\n4. Streaming large content...")
    
    large_content = "\n\n".join([
        f"## Chapter {i}\n\nContent for chapter {i}. " + " ".join(["word"] * 30)
        for i in range(15)
    ])
    
    chunk_count = 0
    total_tokens = 0
    
    for formatted_chunk in manager.stream_chunks(
        large_content,
        content_type="markdown",
        max_chunks=5
    ):
        chunk_count += 1
        total_tokens += formatted_chunk.token_count
    
    print(f"   ✓ Streamed {chunk_count} chunks")
    print(f"   ✓ Total tokens streamed: {total_tokens}")
    
    # Example 4: Token budget management
    print("\n5. Managing token budget...")
    
    # Create content that exceeds budget
    large_text = " ".join(["word"] * 1000)
    
    # Process with strict budget
    budget_result = manager.process_and_format(
        large_text,
        max_tokens=300,
        return_single_string=False
    )
    
    actual_tokens = sum(fc.token_count for fc in budget_result)
    
    print(f"   ✓ Budget limit: 300 tokens")
    print(f"   ✓ Actual usage: {actual_tokens} tokens")
    print(f"   ✓ Within budget: {actual_tokens <= 300}")
    
    # Example 5: Code chunking
    print("\n6. Chunking code with syntax preservation...")
    
    code = """
class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.processed = False
    
    def clean_data(self):
        # Remove null values
        self.data = [x for x in self.data if x is not None]
        return self
    
    def transform_data(self):
        # Apply transformations
        self.data = [x * 2 for x in self.data]
        return self
    
    def get_result(self):
        if not self.processed:
            self.clean_data().transform_data()
            self.processed = True
        return self.data
"""
    
    code_chunks = manager.chunk_content(code, content_type="code")
    
    print(f"   ✓ Created {len(code_chunks)} code chunks")
    print(f"   ✓ Preserved function boundaries")
    
    # Show statistics
    print("\n7. ChunkManager Statistics:")
    stats = manager.get_stats()
    print(f"   Model: {stats['model_name']}")
    print(f"   Max tokens: {stats['max_tokens']}")
    print(f"   Overlap tokens: {stats['overlap_tokens']}")
    print(f"   Cached chunks: {stats['cached_chunks']}")
    print(f"   Scoring weights:")
    print(f"     - Relevance: {stats['weights']['relevance']:.2f}")
    print(f"     - Importance: {stats['weights']['importance']:.2f}")
    print(f"     - Recency: {stats['weights']['recency']:.2f}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    
    # Summary
    print("\nKey Features Demonstrated:")
    print("  ✓ Semantic chunking with natural boundaries")
    print("  ✓ Query-based chunk selection")
    print("  ✓ Streaming for large content")
    print("  ✓ Token budget management")
    print("  ✓ Code syntax preservation")
    print("  ✓ Multiple output formats (Markdown, JSON)")
    print("  ✓ Priority-based ranking (relevance + importance + recency)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
