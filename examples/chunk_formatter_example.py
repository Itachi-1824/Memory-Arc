"""
Example usage of ChunkFormatter for model-specific formatting.

Run this from the project root:
    python -m examples.chunk_formatter_example
"""

from core.infinite.chunk_formatter import ChunkFormatter, FormatType, create_formatter
from core.infinite.models import Chunk, Memory, MemoryType, BoundaryType


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
    
    # Create formatter
    formatter = ChunkFormatter(model_name="gpt-4")
    
    # Format as Markdown
    print("\n--- Markdown Format ---")
    result = formatter.format_chunk(chunk, format_type=FormatType.MARKDOWN)
    print(result.content)
    print(f"\nEstimated tokens: {result.token_count}")


def example_code_formatting():
    """Demonstrate code chunk formatting with syntax highlighting."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Code Chunk Formatting")
    print("=" * 70)
    
    # Create a code chunk
    code_content = """def calculate_fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Example usage
result = calculate_fibonacci(10)
print(f"Fibonacci(10) = {result}")"""
    
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
    
    # Create memory for code
    memory = Memory(
        id="mem_code",
        context_id="ctx_1",
        content=code_content,
        memory_type=MemoryType.CODE,
        created_at=1697500000.0,
        importance=8
    )
    
    # Format as Markdown with syntax highlighting
    formatter = ChunkFormatter(model_name="gpt-4")
    print("\n--- Markdown Format with Syntax Highlighting ---")
    result = formatter.format_chunk(chunk, format_type=FormatType.MARKDOWN, memory=memory)
    print(result.content)


def example_json_formatting():
    """Demonstrate JSON formatting."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: JSON Format")
    print("=" * 70)
    
    chunk = Chunk(
        id="chunk_json",
        content="This chunk will be formatted as JSON for structured processing.",
        chunk_index=1,
        total_chunks=3,
        token_count=15,
        relevance_score=0.75
    )
    
    memory = Memory(
        id="mem_1",
        context_id="ctx_1",
        content="Sample memory",
        memory_type=MemoryType.CONVERSATION,
        created_at=1697500000.0,
        importance=6
    )
    
    formatter = ChunkFormatter(model_name="gpt-4")
    result = formatter.format_chunk(chunk, format_type=FormatType.JSON, memory=memory)
    print(result.content)


def example_xml_formatting():
    """Demonstrate XML formatting."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: XML Format")
    print("=" * 70)
    
    chunk = Chunk(
        id="chunk_xml",
        content="This chunk demonstrates XML formatting with proper escaping of special characters like <, >, and &.",
        chunk_index=2,
        total_chunks=3,
        token_count=20,
        relevance_score=0.80
    )
    
    formatter = ChunkFormatter(model_name="gpt-4")
    result = formatter.format_chunk(chunk, format_type=FormatType.XML)
    print(result.content)


def example_model_specific_optimization():
    """Demonstrate model-specific optimizations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Model-Specific Optimizations")
    print("=" * 70)
    
    code_content = """def process_data(data):
    # Process the data
    result = []
    for item in data:
        result.append(item * 2)
    return result"""
    
    chunk = Chunk(
        id="chunk_opt",
        content=code_content,
        chunk_index=0,
        total_chunks=1,
        token_count=30,
        relevance_score=0.90
    )
    
    memory = Memory(
        id="mem_opt",
        context_id="ctx_1",
        content=code_content,
        memory_type=MemoryType.CODE,
        created_at=1697500000.0,
        importance=7
    )
    
    # Format for different models
    print("\n--- GPT-4 Optimization ---")
    gpt_formatter = ChunkFormatter(model_name="gpt-4")
    gpt_result = gpt_formatter.format_chunk(chunk, format_type=FormatType.MARKDOWN, memory=memory)
    print(gpt_result.content[:200] + "...")
    
    print("\n--- Claude-3 Optimization ---")
    claude_formatter = ChunkFormatter(model_name="claude-3")
    claude_result = claude_formatter.format_chunk(chunk, format_type=FormatType.MARKDOWN, memory=memory)
    print(claude_result.content[:200] + "...")
    
    print("\n--- Llama-3 Optimization ---")
    llama_formatter = ChunkFormatter(model_name="llama-3")
    llama_result = llama_formatter.format_chunk(chunk, format_type=FormatType.MARKDOWN, memory=memory)
    print(llama_result.content[:200] + "...")


def example_compression():
    """Demonstrate repetitive content compression."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Repetitive Content Compression")
    print("=" * 70)
    
    # Create content with repetition
    repetitive_content = """Line 1: Important information
Line 2: Some data
repeated line
repeated line
repeated line
repeated line
repeated line
Line 3: More important information"""
    
    chunk = Chunk(
        id="chunk_compress",
        content=repetitive_content,
        chunk_index=0,
        total_chunks=1,
        token_count=40,
        relevance_score=0.70
    )
    
    print("\n--- Without Compression ---")
    formatter_no_compress = ChunkFormatter(compress_repetitive=False)
    result_no_compress = formatter_no_compress.format_chunk(chunk, format_type=FormatType.PLAIN)
    print(result_no_compress.content)
    
    print("\n--- With Compression ---")
    formatter_compress = ChunkFormatter(compress_repetitive=True)
    result_compress = formatter_compress.format_chunk(chunk, format_type=FormatType.PLAIN)
    print(result_compress.content)


def example_multiple_chunks():
    """Demonstrate formatting multiple chunks together."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Multiple Chunks Formatting")
    print("=" * 70)
    
    chunks = [
        Chunk(
            id="chunk_1",
            content="First chunk of a longer document. This contains the introduction.",
            chunk_index=0,
            total_chunks=3,
            token_count=15,
            relevance_score=0.85
        ),
        Chunk(
            id="chunk_2",
            content="Second chunk with the main content. This is where the details are.",
            chunk_index=1,
            total_chunks=3,
            token_count=18,
            relevance_score=0.90
        ),
        Chunk(
            id="chunk_3",
            content="Third and final chunk. This contains the conclusion.",
            chunk_index=2,
            total_chunks=3,
            token_count=12,
            relevance_score=0.80
        )
    ]
    
    formatter = ChunkFormatter(model_name="gpt-4")
    result = formatter.format_multiple_chunks(chunks, format_type=FormatType.MARKDOWN)
    print(result)


def example_navigation_info():
    """Demonstrate navigation information in chunks."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Navigation Information")
    print("=" * 70)
    
    # First chunk
    print("\n--- First Chunk (has next) ---")
    chunk_first = Chunk(
        id="chunk_1",
        content="This is the first chunk.",
        chunk_index=0,
        total_chunks=3,
        token_count=10,
        relevance_score=0.85
    )
    formatter = ChunkFormatter()
    result = formatter.format_chunk(chunk_first, format_type=FormatType.MARKDOWN)
    print(result.content)
    
    # Middle chunk
    print("\n--- Middle Chunk (has previous and next) ---")
    chunk_middle = Chunk(
        id="chunk_2",
        content="This is the middle chunk.",
        chunk_index=1,
        total_chunks=3,
        token_count=10,
        relevance_score=0.85
    )
    result = formatter.format_chunk(chunk_middle, format_type=FormatType.MARKDOWN)
    print(result.content)
    
    # Last chunk
    print("\n--- Last Chunk (has previous) ---")
    chunk_last = Chunk(
        id="chunk_3",
        content="This is the last chunk.",
        chunk_index=2,
        total_chunks=3,
        token_count=10,
        relevance_score=0.85
    )
    result = formatter.format_chunk(chunk_last, format_type=FormatType.MARKDOWN)
    print(result.content)


def example_language_detection():
    """Demonstrate automatic language detection for code."""
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Automatic Language Detection")
    print("=" * 70)
    
    code_samples = {
        "Python": "def hello():\n    print('Hello')\n    return True",
        "JavaScript": "function hello() {\n    const msg = 'Hello';\n    console.log(msg);\n}",
        "TypeScript": "function hello(): string {\n    const msg: string = 'Hello';\n    return msg;\n}",
        "Java": "public class Hello {\n    private void greet() {\n        System.out.println('Hello');\n    }\n}",
        "Go": "package main\nimport \"fmt\"\nfunc main() {\n    fmt.Println(\"Hello\")\n}",
        "Rust": "fn main() {\n    let msg = \"Hello\";\n    println!(\"{}\", msg);\n}"
    }
    
    formatter = ChunkFormatter()
    
    for expected_lang, code in code_samples.items():
        detected = formatter._detect_language(code)
        print(f"{expected_lang}: detected as '{detected}'")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CHUNK FORMATTER EXAMPLES")
    print("=" * 70)
    
    example_basic_formatting()
    example_code_formatting()
    example_json_formatting()
    example_xml_formatting()
    example_model_specific_optimization()
    example_compression()
    example_multiple_chunks()
    example_navigation_info()
    example_language_detection()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
