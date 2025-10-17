"""
Standalone example of ChunkFormatter usage.
This example demonstrates the formatter without requiring the full memory system.
"""

import json
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


# Minimal models for demonstration
class MemoryType(Enum):
    CONVERSATION = "conversation"
    CODE = "code"
    FACT = "fact"


class BoundaryType(Enum):
    PARAGRAPH = "paragraph"
    FUNCTION = "function"
    CLASS = "class"


@dataclass
class Memory:
    id: str
    context_id: str
    content: str
    memory_type: MemoryType
    created_at: float
    importance: int = 5


@dataclass
class Chunk:
    id: str
    content: str
    chunk_index: int
    total_chunks: int
    token_count: int
    relevance_score: float = 0.0
    boundary_type: BoundaryType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def example_basic_formatting():
    """Demonstrate basic chunk formatting."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Chunk Formatting")
    print("=" * 70)
    
    # Create a sample chunk
    chunk = Chunk(
        id="chunk_1",
        content="This is a sample chunk of text that demonstrates the formatting capabilities. "
                "It contains multiple sentences and shows how metadata is included.",
        chunk_index=0,
        total_chunks=3,
        token_count=25,
        relevance_score=0.85,
        boundary_type=BoundaryType.PARAGRAPH,
        metadata={"source": "example"}
    )
    
    print("\nChunk created:")
    print(f"  ID: {chunk.id}")
    print(f"  Index: {chunk.chunk_index + 1}/{chunk.total_chunks}")
    print(f"  Tokens: {chunk.token_count}")
    print(f"  Relevance: {chunk.relevance_score}")
    print(f"  Content: {chunk.content[:50]}...")


def example_code_chunk():
    """Demonstrate code chunk creation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Code Chunk")
    print("=" * 70)
    
    code_content = """def calculate_fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"""
    
    chunk = Chunk(
        id="chunk_code",
        content=code_content,
        chunk_index=0,
        total_chunks=1,
        token_count=50,
        relevance_score=0.95,
        boundary_type=BoundaryType.FUNCTION,
        metadata={"language": "python", "file": "fibonacci.py"}
    )
    
    memory = Memory(
        id="mem_code",
        context_id="ctx_1",
        content=code_content,
        memory_type=MemoryType.CODE,
        created_at=1697500000.0,
        importance=8
    )
    
    print("\nCode chunk created:")
    print(f"  Language: {chunk.metadata.get('language')}")
    print(f"  File: {chunk.metadata.get('file')}")
    print(f"  Memory Type: {memory.memory_type.value}")
    print(f"  Content:\n{chunk.content}")


def example_json_structure():
    """Demonstrate JSON structure for chunks."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: JSON Structure")
    print("=" * 70)
    
    chunk = Chunk(
        id="chunk_json",
        content="This chunk demonstrates JSON formatting.",
        chunk_index=1,
        total_chunks=3,
        token_count=15,
        relevance_score=0.75
    )
    
    # Simulate JSON output
    json_output = {
        "content": chunk.content,
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "metadata": {
            "chunk_id": chunk.id,
            "token_count": chunk.token_count,
            "relevance_score": chunk.relevance_score
        },
        "navigation": {
            "has_previous": chunk.chunk_index > 0,
            "has_next": chunk.chunk_index < chunk.total_chunks - 1,
            "progress": f"{chunk.chunk_index + 1}/{chunk.total_chunks}",
            "percentage": round((chunk.chunk_index + 1) / chunk.total_chunks * 100, 1)
        }
    }
    
    print("\nJSON structure:")
    print(json.dumps(json_output, indent=2))


def example_markdown_format():
    """Demonstrate Markdown formatting."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Markdown Format")
    print("=" * 70)
    
    chunk = Chunk(
        id="chunk_md",
        content="This is formatted as Markdown with metadata headers.",
        chunk_index=0,
        total_chunks=2,
        token_count=12,
        relevance_score=0.88
    )
    
    # Simulate Markdown output
    markdown = f"""---
chunk_id: {chunk.id}
chunk_index: {chunk.chunk_index + 1}
total_chunks: {chunk.total_chunks}
token_count: {chunk.token_count}
relevance_score: {chunk.relevance_score}
---

**Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}** (50.0%) | Next chunk available →

{chunk.content}"""
    
    print("\nMarkdown output:")
    print(markdown)


def example_code_with_syntax():
    """Demonstrate code with syntax highlighting markers."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Code with Syntax Highlighting")
    print("=" * 70)
    
    python_code = """def greet(name):
    return f"Hello, {name}!"

result = greet("World")
print(result)"""
    
    chunk = Chunk(
        id="chunk_syntax",
        content=python_code,
        chunk_index=0,
        total_chunks=1,
        token_count=20,
        relevance_score=0.92,
        boundary_type=BoundaryType.FUNCTION
    )
    
    # Simulate Markdown with syntax highlighting
    markdown = f"""```python
{chunk.content}
```"""
    
    print("\nCode with syntax highlighting:")
    print(markdown)


def example_navigation_info():
    """Demonstrate navigation information."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Navigation Information")
    print("=" * 70)
    
    chunks = [
        Chunk(id="c1", content="First", chunk_index=0, total_chunks=3, token_count=5, relevance_score=0.8),
        Chunk(id="c2", content="Second", chunk_index=1, total_chunks=3, token_count=5, relevance_score=0.8),
        Chunk(id="c3", content="Third", chunk_index=2, total_chunks=3, token_count=5, relevance_score=0.8),
    ]
    
    for chunk in chunks:
        nav_info = {
            "has_previous": chunk.chunk_index > 0,
            "has_next": chunk.chunk_index < chunk.total_chunks - 1,
            "progress": f"{chunk.chunk_index + 1}/{chunk.total_chunks}",
            "percentage": round((chunk.chunk_index + 1) / chunk.total_chunks * 100, 1)
        }
        
        print(f"\nChunk {chunk.id}:")
        print(f"  Progress: {nav_info['progress']} ({nav_info['percentage']}%)")
        print(f"  Has Previous: {nav_info['has_previous']}")
        print(f"  Has Next: {nav_info['has_next']}")


def example_compression():
    """Demonstrate content compression concept."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Content Compression")
    print("=" * 70)
    
    repetitive_content = """Line 1: Important
repeated line
repeated line
repeated line
repeated line
Line 2: More info"""
    
    compressed_content = """Line 1: Important
repeated line
[... repeated 3 more times ...]
Line 2: More info"""
    
    print("\nOriginal content:")
    print(repetitive_content)
    print("\nCompressed content:")
    print(compressed_content)
    print("\nCompression saved ~60 characters while maintaining meaning")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CHUNK FORMATTER EXAMPLES (Standalone)")
    print("=" * 70)
    print("\nThese examples demonstrate the ChunkFormatter concepts")
    print("without requiring the full memory system installation.")
    
    example_basic_formatting()
    example_code_chunk()
    example_json_structure()
    example_markdown_format()
    example_code_with_syntax()
    example_navigation_info()
    example_compression()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nTo use the actual ChunkFormatter implementation:")
    print("  from core.infinite.chunk_formatter import ChunkFormatter")
    print("  formatter = ChunkFormatter(model_name='gpt-4')")
    print("  result = formatter.format_chunk(chunk, format_type='markdown')")


if __name__ == "__main__":
    main()
