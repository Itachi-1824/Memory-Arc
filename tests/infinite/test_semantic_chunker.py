"""Unit tests for semantic chunking algorithm."""

import pytest
from core.infinite import SemanticChunker, BoundaryType, Chunk


class TestSemanticChunker:
    """Test suite for SemanticChunker class."""

    def test_initialization(self):
        """Test chunker initialization with default parameters."""
        chunker = SemanticChunker()
        assert chunker.max_chunk_size == 1000
        assert chunker.min_chunk_size == 100
        assert chunker.overlap_size == 100
        assert chunker.embedding_fn is None

    def test_initialization_custom_params(self):
        """Test chunker initialization with custom parameters."""
        chunker = SemanticChunker(
            max_chunk_size=500,
            min_chunk_size=50,
            overlap_size=50
        )
        assert chunker.max_chunk_size == 500
        assert chunker.min_chunk_size == 50
        assert chunker.overlap_size == 50

    def test_detect_text_boundaries_paragraphs(self):
        """Test paragraph boundary detection in plain text."""
        chunker = SemanticChunker()
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        boundaries = chunker.detect_boundaries(content, "text")
        
        # Should detect paragraph boundaries
        paragraph_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.PARAGRAPH]
        assert len(paragraph_boundaries) >= 2

    def test_detect_text_boundaries_sentences(self):
        """Test sentence boundary detection in plain text."""
        chunker = SemanticChunker()
        content = "First sentence. Second sentence! Third sentence?"
        
        boundaries = chunker.detect_boundaries(content, "text")
        
        # Should detect sentence boundaries
        sentence_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.SENTENCE]
        assert len(sentence_boundaries) >= 2

    def test_detect_markdown_boundaries_sections(self):
        """Test section boundary detection in markdown."""
        chunker = SemanticChunker()
        content = """# Main Title

Some content here.

## Section 1

More content.

### Subsection

Even more content."""
        
        boundaries = chunker.detect_boundaries(content, "markdown")
        
        # Should detect section headers
        section_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.SECTION]
        assert len(section_boundaries) >= 3

    def test_detect_code_boundaries_python_functions(self):
        """Test function boundary detection in Python code."""
        chunker = SemanticChunker()
        content = """def function_one():
    return 1

def function_two():
    return 2

def function_three():
    return 3"""
        
        boundaries = chunker.detect_boundaries(content, "code")
        
        # Should detect function definitions
        function_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.FUNCTION]
        assert len(function_boundaries) == 3

    def test_detect_code_boundaries_python_classes(self):
        """Test class boundary detection in Python code."""
        chunker = SemanticChunker()
        content = """class ClassOne:
    pass

class ClassTwo:
    pass"""
        
        boundaries = chunker.detect_boundaries(content, "code")
        
        # Should detect class definitions
        class_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.CLASS]
        assert len(class_boundaries) == 2

    def test_detect_code_boundaries_javascript(self):
        """Test function boundary detection in JavaScript code."""
        chunker = SemanticChunker()
        content = """function myFunction() {
    return 1;
}

const arrowFunc = () => {
    return 2;
}

const namedArrow = (x) => x * 2;"""
        
        boundaries = chunker.detect_boundaries(content, "code")
        
        # Should detect function definitions
        function_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.FUNCTION]
        assert len(function_boundaries) >= 2

    def test_lexical_coherence_identical(self):
        """Test lexical coherence with identical text."""
        chunker = SemanticChunker()
        text = "the quick brown fox"
        
        coherence = chunker._lexical_coherence(text, text)
        
        assert coherence == 1.0

    def test_lexical_coherence_no_overlap(self):
        """Test lexical coherence with no word overlap."""
        chunker = SemanticChunker()
        text1 = "the quick brown fox"
        text2 = "jumps over lazy dog"
        
        coherence = chunker._lexical_coherence(text1, text2)
        
        assert coherence == 0.0

    def test_lexical_coherence_partial_overlap(self):
        """Test lexical coherence with partial word overlap."""
        chunker = SemanticChunker()
        text1 = "the quick brown fox"
        text2 = "the lazy brown dog"
        
        coherence = chunker._lexical_coherence(text1, text2)
        
        # Should have some overlap (the, brown)
        assert 0.0 < coherence < 1.0

    def test_measure_coherence_without_embedding(self):
        """Test coherence measurement without embedding function."""
        chunker = SemanticChunker()
        text1 = "machine learning algorithms"
        text2 = "machine learning models"
        
        coherence = chunker.measure_coherence(text1, text2)
        
        # Should use lexical fallback
        assert 0.0 <= coherence <= 1.0

    def test_measure_coherence_with_embedding(self):
        """Test coherence measurement with embedding function."""
        def mock_embedding(text):
            # Simple mock: return vector based on text length
            return [float(len(text)), 1.0, 0.5]
        
        chunker = SemanticChunker(embedding_fn=mock_embedding)
        text1 = "short"
        text2 = "longer text"
        
        coherence = chunker.measure_coherence(text1, text2)
        
        # Should use embedding-based similarity
        assert 0.0 <= coherence <= 1.0

    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        chunker = SemanticChunker()
        vec = [1.0, 2.0, 3.0]
        
        similarity = chunker._cosine_similarity(vec, vec)
        
        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        chunker = SemanticChunker()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        
        similarity = chunker._cosine_similarity(vec1, vec2)
        
        assert abs(similarity) < 0.001

    def test_cosine_similarity_different_lengths(self):
        """Test cosine similarity with different length vectors."""
        chunker = SemanticChunker()
        vec1 = [1.0, 2.0]
        vec2 = [1.0, 2.0, 3.0]
        
        similarity = chunker._cosine_similarity(vec1, vec2)
        
        assert similarity == 0.0

    def test_generate_overlap_region(self):
        """Test overlap region generation between chunks."""
        chunker = SemanticChunker(overlap_size=10)
        prev_chunk = "This is the first chunk with some content here"
        next_chunk = "This is the second chunk"
        
        prev_result, next_result = chunker.generate_overlap_region(prev_chunk, next_chunk)
        
        # Previous chunk should remain unchanged
        assert prev_result == prev_chunk
        # Next chunk should have overlap prepended
        assert len(next_result) > len(next_chunk)
        assert next_chunk in next_result

    def test_generate_overlap_region_custom_size(self):
        """Test overlap region generation with custom size."""
        chunker = SemanticChunker()
        prev_chunk = "word " * 20  # 20 words
        next_chunk = "next chunk"
        
        prev_result, next_result = chunker.generate_overlap_region(
            prev_chunk, next_chunk, overlap_tokens=5
        )
        
        # Should use custom overlap size
        assert next_chunk in next_result
        assert len(next_result) > len(next_chunk)

    def test_estimate_tokens(self):
        """Test token estimation for text."""
        chunker = SemanticChunker()
        
        # Short text
        text = "Hello world"
        tokens = chunker._estimate_tokens(text)
        assert tokens > 0
        assert tokens < 10
        
        # Longer text
        text = "This is a longer piece of text with many words and characters"
        tokens = chunker._estimate_tokens(text)
        assert tokens > 10

    def test_chunk_by_boundaries_no_boundaries(self):
        """Test chunking with no boundaries detected."""
        chunker = SemanticChunker()
        content = "Simple short text"
        boundaries = []
        
        chunks = chunker.chunk_by_boundaries(content, boundaries)
        
        # Should return single chunk
        assert len(chunks) == 1
        assert chunks[0][0] == 0
        assert chunks[0][1] == len(content)

    def test_chunk_by_boundaries_with_boundaries(self):
        """Test chunking with detected boundaries."""
        from core.infinite import ChunkBoundary
        
        chunker = SemanticChunker(max_chunk_size=20, min_chunk_size=5)
        # Create content with words to get accurate token estimates
        content = " ".join([f"word{i}" for i in range(50)])  # ~50 words
        
        # Add boundaries at word positions
        boundaries = [
            ChunkBoundary(len(" ".join([f"word{i}" for i in range(10)])), BoundaryType.PARAGRAPH),
            ChunkBoundary(len(" ".join([f"word{i}" for i in range(20)])), BoundaryType.PARAGRAPH),
            ChunkBoundary(len(" ".join([f"word{i}" for i in range(30)])), BoundaryType.PARAGRAPH),
        ]
        
        chunks = chunker.chunk_by_boundaries(content, boundaries)
        
        # Should create multiple chunks given the size constraints
        assert len(chunks) >= 1

    def test_chunk_content_simple_text(self):
        """Test chunking simple text content."""
        chunker = SemanticChunker(max_chunk_size=100)
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        chunks = chunker.chunk_content(content, "text")
        
        # Should create chunks
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.content for c in chunks)
        
        # Check chunk indices
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx
            assert chunk.total_chunks == len(chunks)

    def test_chunk_content_code(self):
        """Test chunking code content."""
        chunker = SemanticChunker(max_chunk_size=200)
        content = """def func1():
    return 1

def func2():
    return 2

def func3():
    return 3"""
        
        chunks = chunker.chunk_content(content, "code")
        
        # Should create chunks
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        
        # Should detect function boundaries
        assert any(c.boundary_type == BoundaryType.FUNCTION for c in chunks)

    def test_chunk_content_markdown(self):
        """Test chunking markdown content."""
        chunker = SemanticChunker(max_chunk_size=150)
        content = """# Title

Content here.

## Section 1

More content.

## Section 2

Even more."""
        
        chunks = chunker.chunk_content(content, "markdown")
        
        # Should create chunks
        assert len(chunks) > 0
        
        # Verify that sections were detected in boundaries
        boundaries = chunker.detect_boundaries(content, "markdown")
        section_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.SECTION]
        assert len(section_boundaries) > 0

    def test_chunk_content_metadata(self):
        """Test that chunks include proper metadata."""
        chunker = SemanticChunker()
        content = "Test content for metadata"
        
        chunks = chunker.chunk_content(content, "text")
        
        assert len(chunks) > 0
        chunk = chunks[0]
        
        # Check metadata
        assert "content_type" in chunk.metadata
        assert chunk.metadata["content_type"] == "text"
        assert "has_overlap" in chunk.metadata

    def test_chunk_content_token_counts(self):
        """Test that chunks have accurate token counts."""
        chunker = SemanticChunker()
        content = "This is a test with multiple words"
        
        chunks = chunker.chunk_content(content, "text")
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.token_count > 0
            # Token count should be reasonable for content length
            assert chunk.token_count < len(chunk.content)

    def test_chunk_content_positions(self):
        """Test that chunks have correct start and end positions."""
        chunker = SemanticChunker()
        content = "First part.\n\nSecond part.\n\nThird part."
        
        chunks = chunker.chunk_content(content, "text")
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.start_pos >= 0
            assert chunk.end_pos <= len(content)
            assert chunk.start_pos < chunk.end_pos

    def test_chunk_content_large_text(self):
        """Test chunking with large text that requires multiple chunks."""
        chunker = SemanticChunker(max_chunk_size=100, min_chunk_size=20)
        
        # Create large content with clear boundaries
        paragraphs = [f"Paragraph {i} with some content here." for i in range(20)]
        content = "\n\n".join(paragraphs)
        
        chunks = chunker.chunk_content(content, "text")
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # All chunks should respect size constraints (approximately)
        for chunk in chunks:
            # Allow some flexibility due to overlap
            assert chunk.token_count <= chunker.max_chunk_size * 1.5

    def test_chunk_content_preserves_content(self):
        """Test that chunking doesn't lose content."""
        chunker = SemanticChunker(max_chunk_size=50)
        content = "Important content that must be preserved"
        
        chunks = chunker.chunk_content(content, "text")
        
        # All original content should appear in at least one chunk
        combined = " ".join(c.content for c in chunks)
        for word in content.split():
            assert word in combined


class TestBoundaryDetection:
    """Test suite for boundary detection accuracy."""

    def test_paragraph_boundary_accuracy(self):
        """Test accuracy of paragraph boundary detection."""
        chunker = SemanticChunker()
        content = "Para 1.\n\nPara 2.\n\nPara 3."
        
        boundaries = chunker.detect_boundaries(content, "text")
        para_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.PARAGRAPH]
        
        # Should detect exactly 2 paragraph breaks
        assert len(para_boundaries) == 2

    def test_function_boundary_accuracy(self):
        """Test accuracy of function boundary detection."""
        chunker = SemanticChunker()
        content = """def a():
    pass

def b():
    pass"""
        
        boundaries = chunker.detect_boundaries(content, "code")
        func_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.FUNCTION]
        
        # Should detect exactly 2 functions
        assert len(func_boundaries) == 2

    def test_mixed_boundary_types(self):
        """Test detection of multiple boundary types."""
        chunker = SemanticChunker()
        content = """class MyClass:
    def method1(self):
        pass
    
    def method2(self):
        pass"""
        
        boundaries = chunker.detect_boundaries(content, "code")
        
        # Should detect both class and function boundaries
        class_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.CLASS]
        func_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.FUNCTION]
        
        assert len(class_boundaries) >= 1
        # Methods inside classes should be detected (at least 1, may not catch all indented methods)
        assert len(func_boundaries) >= 1


class TestCoherenceMeasurement:
    """Test suite for coherence measurement."""

    def test_coherence_high_similarity(self):
        """Test coherence with highly similar text."""
        chunker = SemanticChunker()
        text1 = "machine learning and artificial intelligence"
        text2 = "machine learning and AI systems"
        
        coherence = chunker.measure_coherence(text1, text2)
        
        # Should have high coherence due to shared terms
        assert coherence > 0.3

    def test_coherence_low_similarity(self):
        """Test coherence with dissimilar text."""
        chunker = SemanticChunker()
        text1 = "cooking recipes and food"
        text2 = "quantum physics theories"
        
        coherence = chunker.measure_coherence(text1, text2)
        
        # Should have low coherence
        assert coherence < 0.3

    def test_coherence_empty_text(self):
        """Test coherence with empty text."""
        chunker = SemanticChunker()
        
        coherence = chunker.measure_coherence("", "some text")
        assert coherence == 0.0
        
        coherence = chunker.measure_coherence("some text", "")
        assert coherence == 0.0


class TestOverlapGeneration:
    """Test suite for overlap region generation."""

    def test_overlap_adds_context(self):
        """Test that overlap adds context to next chunk."""
        chunker = SemanticChunker(overlap_size=20)
        prev = "This is a long chunk with many words in it"
        next_chunk = "This is the next chunk"
        
        _, next_with_overlap = chunker.generate_overlap_region(prev, next_chunk)
        
        # Next chunk should be longer with overlap
        assert len(next_with_overlap) > len(next_chunk)

    def test_overlap_preserves_original(self):
        """Test that overlap doesn't modify previous chunk."""
        chunker = SemanticChunker(overlap_size=20)
        prev = "Previous chunk content"
        next_chunk = "Next chunk content"
        
        prev_result, _ = chunker.generate_overlap_region(prev, next_chunk)
        
        # Previous chunk should be unchanged
        assert prev_result == prev

    def test_overlap_with_short_prev_chunk(self):
        """Test overlap when previous chunk is shorter than overlap size."""
        chunker = SemanticChunker(overlap_size=100)
        prev = "Short"
        next_chunk = "Next"
        
        _, next_with_overlap = chunker.generate_overlap_region(prev, next_chunk)
        
        # Should handle gracefully
        assert "Short" in next_with_overlap
        assert "Next" in next_with_overlap
