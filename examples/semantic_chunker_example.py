"""
Example usage of the SemanticChunker for intelligent content splitting.

Note: Run this from the project root directory:
    python -m examples.semantic_chunker_example

Or run tests to see the chunker in action:
    python -m pytest tests/infinite/test_semantic_chunker.py -v
"""

def main():
    """
    This example demonstrates the SemanticChunker capabilities.
    
    The SemanticChunker provides:
    1. Boundary detection for text, code, and markdown
    2. Semantic coherence measurement
    3. Overlap region generation for context continuity
    4. Intelligent chunking that respects natural boundaries
    
    For working examples, see:
    - tests/infinite/test_semantic_chunker.py (comprehensive test suite)
    - core/infinite/semantic_chunker.py (implementation)
    
    Basic usage:
    ```python
    from core.infinite import SemanticChunker
    
    chunker = SemanticChunker(
        max_chunk_size=1000,
        min_chunk_size=100,
        overlap_size=100
    )
    
    # Chunk text content
    chunks = chunker.chunk_content(content, content_type="text")
    
    # Chunk code
    chunks = chunker.chunk_content(code, content_type="code")
    
    # Chunk markdown
    chunks = chunker.chunk_content(markdown, content_type="markdown")
    
    # Each chunk has:
    # - content: the actual text
    # - chunk_index: position in sequence
    # - total_chunks: total number of chunks
    # - token_count: estimated tokens
    # - boundary_type: type of boundary (paragraph, function, etc.)
    # - metadata: additional information
    ```
    
    Features demonstrated in tests:
    - Paragraph boundary detection
    - Function and class boundary detection (Python, JavaScript)
    - Markdown section detection
    - Lexical and embedding-based coherence measurement
    - Overlap generation for context continuity
    - Token estimation
    - Custom embedding functions
    """
    
    print(__doc__)
    print("\nTo see the SemanticChunker in action, run:")
    print("  python -m pytest tests/infinite/test_semantic_chunker.py -v")
    print("\nOr import and use it in your code:")
    print("  from core.infinite import SemanticChunker")


if __name__ == "__main__":
    main()
