"""Example usage of ChunkManager for intelligent content chunking and formatting."""

import time
from core.infinite.chunk_manager import create_chunk_manager, ChunkManagerConfig
from core.infinite.models import Memory, MemoryType
from core.infinite.chunk_formatter import FormatType


def example_basic_text_chunking():
    """Example: Basic text chunking."""
    print("=" * 60)
    print("Example 1: Basic Text Chunking")
    print("=" * 60)
    
    # Create chunk manager
    manager = create_chunk_manager(model_name="gpt-4")
    
    # Sample text content
    content = """
# Introduction to Python

Python is a high-level programming language known for its simplicity and readability.

## Key Features

Python offers several key features that make it popular:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Large community support

## Use Cases

Python is widely used in:
1. Web development
2. Data science and machine learning
3. Automation and scripting
4. Scientific computing

## Conclusion

Python continues to be one of the most popular programming languages in the world.
"""
    
    # Chunk the content
    chunks = manager.chunk_content(content, content_type="markdown")
    
    print(f"\nCreated {len(chunks)} chunks")
    for chunk in chunks:
        print(f"\nChunk {chunk.chunk_index + 1}/{chunk.total_chunks}:")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Preview: {chunk.content[:100]}...")


def example_code_chunking():
    """Example: Code chunking with syntax preservation."""
    print("\n" + "=" * 60)
    print("Example 2: Code Chunking")
    print("=" * 60)
    
    manager = create_chunk_manager(model_name="gpt-4")
    
    code = """
def calculate_fibonacci(n):
    '''Calculate nth Fibonacci number.'''
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def factorial(n):
    '''Calculate factorial of n.'''
    if n <= 1:
        return 1
    return n * factorial(n-1)

class MathOperations:
    '''Collection of mathematical operations.'''
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(('add', a, b, result))
        return result
    
    def multiply(self, a, b):
        result = a * b
        self.history.append(('multiply', a, b, result))
        return result
"""
    
    # Chunk and format as markdown with syntax highlighting
    formatted_chunks = manager.process_and_format(
        code,
        content_type="code",
        format_type=FormatType.MARKDOWN
    )
    
    print(f"\nCreated {len(formatted_chunks)} formatted chunks")
    for fc in formatted_chunks:
        print(f"\n{fc.content[:200]}...")


def example_memory_chunking():
    """Example: Chunking Memory objects."""
    print("\n" + "=" * 60)
    print("Example 3: Memory Chunking")
    print("=" * 60)
    
    manager = create_chunk_manager(model_name="gpt-4")
    
    # Create sample memories
    memories = [
        Memory(
            id="mem1",
            context_id="user_123",
            content="User prefers dark mode for the interface.",
            memory_type=MemoryType.PREFERENCE,
            created_at=time.time() - 86400,  # 1 day ago
            importance=7
        ),
        Memory(
            id="mem2",
            context_id="user_123",
            content="User is working on a Python data science project.",
            memory_type=MemoryType.FACT,
            created_at=time.time() - 3600,  # 1 hour ago
            importance=8
        ),
        Memory(
            id="mem3",
            context_id="user_123",
            content="User asked about pandas DataFrame operations.",
            memory_type=MemoryType.CONVERSATION,
            created_at=time.time() - 300,  # 5 minutes ago
            importance=6
        )
    ]
    
    # Chunk and format
    result = manager.process_and_format(
        memories,
        format_type=FormatType.JSON,
        return_single_string=False
    )
    
    print(f"\nCreated {len(result)} chunks from {len(memories)} memories")
    for i, fc in enumerate(result):
        print(f"\nChunk {i + 1}:")
        print(f"  Format: {fc.format_type.value}")
        print(f"  Tokens: {fc.token_count}")


def example_query_based_selection():
    """Example: Query-based chunk selection."""
    print("\n" + "=" * 60)
    print("Example 4: Query-Based Selection")
    print("=" * 60)
    
    manager = create_chunk_manager(model_name="gpt-4")
    
    content = """
Python is excellent for data science.

JavaScript is great for web development.

Python has powerful libraries like pandas and numpy.

JavaScript frameworks include React and Vue.

Python is also used in machine learning with TensorFlow.

JavaScript can run on both client and server with Node.js.
"""
    
    # Chunk the content
    chunks = manager.chunk_content(content)
    
    # Select chunks relevant to "Python data science"
    selected = manager.select_chunks(
        chunks,
        query="Python data science",
        max_chunks=3
    )
    
    print(f"\nSelected {len(selected)} most relevant chunks:")
    for sc in selected:
        print(f"\nScore: {sc.final_score:.3f}")
        print(f"  Relevance: {sc.relevance_score:.3f}")
        print(f"  Content: {sc.chunk.content[:80]}...")


def example_streaming_chunks():
    """Example: Streaming chunks for large content."""
    print("\n" + "=" * 60)
    print("Example 5: Streaming Chunks")
    print("=" * 60)
    
    manager = create_chunk_manager(model_name="gpt-4")
    
    # Generate large content
    sections = [
        f"## Section {i}\n\nThis is the content for section {i}. "
        f"It contains important information about topic {i}."
        for i in range(10)
    ]
    content = "\n\n".join(sections)
    
    print("\nStreaming chunks...")
    chunk_count = 0
    for formatted_chunk in manager.stream_chunks(
        content,
        content_type="markdown",
        format_type=FormatType.MARKDOWN,
        max_chunks=5
    ):
        chunk_count += 1
        print(f"\nReceived chunk {chunk_count}:")
        print(f"  Tokens: {formatted_chunk.token_count}")
        print(f"  Preview: {formatted_chunk.content[:100]}...")
    
    print(f"\nTotal chunks streamed: {chunk_count}")


def example_token_budget():
    """Example: Working with token budgets."""
    print("\n" + "=" * 60)
    print("Example 6: Token Budget Management")
    print("=" * 60)
    
    manager = create_chunk_manager(model_name="gpt-4", max_tokens=8000)
    
    # Create content that would exceed budget
    large_content = "\n\n".join([
        f"Paragraph {i}: " + " ".join(["word"] * 50)
        for i in range(20)
    ])
    
    # Process with token limit
    result = manager.process_and_format(
        large_content,
        max_tokens=500,  # Strict budget
        return_single_string=False
    )
    
    total_tokens = sum(fc.token_count for fc in result)
    
    print(f"\nProcessed content within budget:")
    print(f"  Chunks created: {len(result)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Budget limit: 500")
    print(f"  Within budget: {total_tokens <= 500}")


def example_chunk_navigation():
    """Example: Navigating through chunks."""
    print("\n" + "=" * 60)
    print("Example 7: Chunk Navigation")
    print("=" * 60)
    
    manager = create_chunk_manager(model_name="gpt-4")
    
    content = "\n\n".join([f"Section {i} content." for i in range(5)])
    
    # Create chunks
    chunks = manager.chunk_content(content)
    
    if len(chunks) > 1:
        print(f"\nCreated {len(chunks)} chunks")
        
        # Navigate forward
        current = chunks[0]
        print(f"\nStarting at chunk {current.chunk_index}")
        
        next_chunk = manager.get_next_chunk(current.id, direction="forward")
        if next_chunk:
            print(f"Navigated forward to chunk {next_chunk.chunk_index}")
        
        # Navigate backward
        if len(chunks) > 2:
            current = chunks[2]
            prev_chunk = manager.get_next_chunk(current.id, direction="backward")
            if prev_chunk:
                print(f"Navigated backward from chunk {current.chunk_index} to {prev_chunk.chunk_index}")


def example_custom_configuration():
    """Example: Custom ChunkManager configuration."""
    print("\n" + "=" * 60)
    print("Example 8: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = ChunkManagerConfig(
        model_name="claude-3-opus",
        max_tokens=200000,
        overlap_tokens=200,
        relevance_weight=0.6,  # Prioritize relevance
        importance_weight=0.3,
        recency_weight=0.1,
        include_metadata=True,
        compress_repetitive=True,
        preserve_structure=True
    )
    
    from core.infinite.chunk_manager import ChunkManager
    manager = ChunkManager(config)
    
    # Get stats
    stats = manager.get_stats()
    
    print("\nChunkManager Configuration:")
    print(f"  Model: {stats['model_name']}")
    print(f"  Max tokens: {stats['max_tokens']}")
    print(f"  Overlap: {stats['overlap_tokens']}")
    print(f"  Weights:")
    print(f"    Relevance: {stats['weights']['relevance']:.2f}")
    print(f"    Importance: {stats['weights']['importance']:.2f}")
    print(f"    Recency: {stats['weights']['recency']:.2f}")


def example_multiple_formats():
    """Example: Formatting chunks in different formats."""
    print("\n" + "=" * 60)
    print("Example 9: Multiple Output Formats")
    print("=" * 60)
    
    manager = create_chunk_manager(model_name="gpt-4")
    
    content = "This is sample content for formatting demonstration."
    chunks = manager.chunk_content(content)
    
    if chunks:
        chunk = chunks[0]
        
        # Format as Markdown
        md_formatted = manager.format_chunk(chunk, format_type=FormatType.MARKDOWN)
        print("\nMarkdown format:")
        print(md_formatted.content[:150])
        
        # Format as JSON
        json_formatted = manager.format_chunk(chunk, format_type=FormatType.JSON)
        print("\nJSON format:")
        print(json_formatted.content[:150])
        
        # Format as plain text
        plain_formatted = manager.format_chunk(chunk, format_type=FormatType.PLAIN)
        print("\nPlain text format:")
        print(plain_formatted.content[:150])


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ChunkManager Examples")
    print("=" * 60)
    
    example_basic_text_chunking()
    example_code_chunking()
    example_memory_chunking()
    example_query_based_selection()
    example_streaming_chunks()
    example_token_budget()
    example_chunk_navigation()
    example_custom_configuration()
    example_multiple_formats()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
