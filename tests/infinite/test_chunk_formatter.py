"""Tests for chunk formatter with model-specific optimizations."""

import json
import pytest
from core.infinite.chunk_formatter import (
    ChunkFormatter,
    FormatType,
    FormattedChunk,
    create_formatter
)
from core.infinite.models import Chunk, Memory, MemoryType, BoundaryType


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing."""
    return Chunk(
        id="chunk_1",
        content="This is a test chunk with some content.\nIt has multiple lines.",
        chunk_index=0,
        total_chunks=3,
        token_count=15,
        relevance_score=0.85,
        start_pos=0,
        end_pos=100,
        boundary_type=BoundaryType.PARAGRAPH,
        metadata={"test_key": "test_value"}
    )


@pytest.fixture
def sample_code_chunk():
    """Create a sample code chunk for testing."""
    code_content = """def hello_world():
    print("Hello, World!")
    return True"""
    
    return Chunk(
        id="chunk_code",
        content=code_content,
        chunk_index=0,
        total_chunks=1,
        token_count=20,
        relevance_score=0.9,
        boundary_type=BoundaryType.FUNCTION,
        metadata={"language": "python"}
    )


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        id="mem_1",
        context_id="ctx_1",
        content="Test memory content",
        memory_type=MemoryType.CONVERSATION,
        created_at=1697500000.0,
        importance=7
    )


@pytest.fixture
def sample_code_memory():
    """Create a sample code memory for testing."""
    return Memory(
        id="mem_code",
        context_id="ctx_1",
        content="def test(): pass",
        memory_type=MemoryType.CODE,
        created_at=1697500000.0,
        importance=8
    )


class TestChunkFormatter:
    """Test suite for ChunkFormatter."""

    def test_initialization(self):
        """Test formatter initialization."""
        formatter = ChunkFormatter(model_name="gpt-4")
        assert formatter.model_name == "gpt-4"
        assert formatter.include_metadata is True
        assert formatter.compress_repetitive is True

    def test_initialization_custom_settings(self):
        """Test formatter with custom settings."""
        formatter = ChunkFormatter(
            model_name="claude-3",
            include_metadata=False,
            compress_repetitive=False
        )
        assert formatter.model_name == "claude-3"
        assert formatter.include_metadata is False
        assert formatter.compress_repetitive is False

    def test_format_chunk_markdown(self, sample_chunk, sample_memory):
        """Test formatting chunk as Markdown."""
        formatter = ChunkFormatter()
        result = formatter.format_chunk(
            sample_chunk,
            format_type=FormatType.MARKDOWN,
            memory=sample_memory
        )
        
        assert isinstance(result, FormattedChunk)
        assert result.format_type == FormatType.MARKDOWN
        assert "---" in result.content  # Metadata header
        assert "chunk_index: 1" in result.content  # 1-indexed
        assert "total_chunks: 3" in result.content
        assert sample_chunk.content in result.content

    def test_format_chunk_json(self, sample_chunk, sample_memory):
        """Test formatting chunk as JSON."""
        formatter = ChunkFormatter()
        result = formatter.format_chunk(
            sample_chunk,
            format_type=FormatType.JSON,
            memory=sample_memory
        )
        
        assert isinstance(result, FormattedChunk)
        assert result.format_type == FormatType.JSON
        
        # Parse JSON to verify structure
        data = json.loads(result.content)
        assert "content" in data
        assert "chunk_index" in data
        assert "total_chunks" in data
        assert "metadata" in data
        assert "navigation" in data
        assert data["chunk_index"] == 0
        assert data["total_chunks"] == 3

    def test_format_chunk_xml(self, sample_chunk, sample_memory):
        """Test formatting chunk as XML."""
        formatter = ChunkFormatter()
        result = formatter.format_chunk(
            sample_chunk,
            format_type=FormatType.XML,
            memory=sample_memory
        )
        
        assert isinstance(result, FormattedChunk)
        assert result.format_type == FormatType.XML
        assert "<chunk>" in result.content
        assert "</chunk>" in result.content
        assert "<metadata>" in result.content
        assert "<navigation>" in result.content
        assert "<content>" in result.content
        assert "<![CDATA[" in result.content

    def test_format_chunk_plain(self, sample_chunk, sample_memory):
        """Test formatting chunk as plain text."""
        formatter = ChunkFormatter()
        result = formatter.format_chunk(
            sample_chunk,
            format_type=FormatType.PLAIN,
            memory=sample_memory
        )
        
        assert isinstance(result, FormattedChunk)
        assert result.format_type == FormatType.PLAIN
        assert "=" * 60 in result.content  # Header separator
        assert sample_chunk.content in result.content

    def test_format_code_chunk_markdown(self, sample_code_chunk, sample_code_memory):
        """Test formatting code chunk with syntax highlighting."""
        formatter = ChunkFormatter()
        result = formatter.format_chunk(
            sample_code_chunk,
            format_type=FormatType.MARKDOWN,
            memory=sample_code_memory
        )
        
        assert "```python" in result.content
        assert "def hello_world():" in result.content
        assert "```" in result.content

    def test_format_code_chunk_json(self, sample_code_chunk, sample_code_memory):
        """Test formatting code chunk as JSON with syntax info."""
        formatter = ChunkFormatter()
        result = formatter.format_chunk(
            sample_code_chunk,
            format_type=FormatType.JSON,
            memory=sample_code_memory
        )
        
        data = json.loads(result.content)
        assert "syntax" in data
        assert data["syntax"] == "python"
        assert data["type"] == "code"

    def test_format_without_metadata(self, sample_chunk):
        """Test formatting without metadata headers."""
        formatter = ChunkFormatter(include_metadata=False)
        result = formatter.format_chunk(
            sample_chunk,
            format_type=FormatType.MARKDOWN
        )
        
        # Should not have metadata header
        assert "---" not in result.content or result.content.count("---") < 2

    def test_format_without_navigation(self, sample_chunk):
        """Test formatting without navigation info."""
        formatter = ChunkFormatter()
        result = formatter.format_chunk(
            sample_chunk,
            format_type=FormatType.MARKDOWN,
            include_navigation=False
        )
        
        # Should not have navigation info
        assert "Chunk" not in result.content or "available" not in result.content

    def test_detect_language_python(self):
        """Test Python language detection."""
        formatter = ChunkFormatter()
        
        python_code = "def test():\n    import os\n    return True"
        assert formatter._detect_language(python_code) == "python"

    def test_detect_language_javascript(self):
        """Test JavaScript language detection."""
        formatter = ChunkFormatter()
        
        js_code = "function test() {\n    const x = 10;\n    return x;\n}"
        assert formatter._detect_language(js_code) == "javascript"

    def test_detect_language_typescript(self):
        """Test TypeScript language detection."""
        formatter = ChunkFormatter()
        
        ts_code = "function test(): string {\n    const x: number = 10;\n    return 'test';\n}"
        assert formatter._detect_language(ts_code) == "typescript"

    def test_detect_language_java(self):
        """Test Java language detection."""
        formatter = ChunkFormatter()
        
        java_code = "public class Test {\n    private void method() {}\n}"
        assert formatter._detect_language(java_code) == "java"

    def test_detect_language_cpp(self):
        """Test C++ language detection."""
        formatter = ChunkFormatter()
        
        cpp_code = "#include <iostream>\nusing namespace std;\nint main() {}"
        assert formatter._detect_language(cpp_code) == "cpp"

    def test_detect_language_go(self):
        """Test Go language detection."""
        formatter = ChunkFormatter()
        
        go_code = "package main\nimport \"fmt\"\nfunc main() {}"
        assert formatter._detect_language(go_code) == "go"

    def test_detect_language_rust(self):
        """Test Rust language detection."""
        formatter = ChunkFormatter()
        
        rust_code = "fn main() {\n    let mut x = 5;\n}"
        assert formatter._detect_language(rust_code) == "rust"

    def test_compress_repetitive_content(self):
        """Test compression of repetitive content."""
        formatter = ChunkFormatter(compress_repetitive=True)
        
        repetitive_content = "line 1\n" + "repeated line\n" * 5 + "line 2"
        compressed = formatter._compress_repetitive_content(repetitive_content)
        
        assert "repeated" in compressed
        assert compressed.count("repeated line") == 1
        assert "repeated 4 more times" in compressed

    def test_no_compression_when_disabled(self):
        """Test that compression is skipped when disabled."""
        formatter = ChunkFormatter(compress_repetitive=False)
        
        repetitive_content = "line 1\n" + "repeated line\n" * 5 + "line 2"
        result = formatter.format_chunk(
            Chunk(
                id="test",
                content=repetitive_content,
                chunk_index=0,
                total_chunks=1,
                token_count=10
            ),
            format_type=FormatType.PLAIN
        )
        
        # Should contain all repetitions
        assert result.content.count("repeated line") == 5

    def test_model_optimization_gpt(self, sample_chunk):
        """Test GPT-specific optimizations."""
        formatter = ChunkFormatter(model_name="gpt-4")
        optimized = formatter._apply_model_optimizations(sample_chunk.content, None)
        
        # GPT optimization should not change content significantly
        assert optimized == sample_chunk.content

    def test_model_optimization_claude(self, sample_code_chunk, sample_code_memory):
        """Test Claude-specific optimizations."""
        formatter = ChunkFormatter(model_name="claude-3")
        optimized = formatter._apply_model_optimizations(
            sample_code_chunk.content,
            sample_code_memory
        )
        
        # Claude optimization adds context markers for code
        assert "[Code snippet]" in optimized

    def test_model_optimization_llama(self):
        """Test Llama-specific optimizations."""
        formatter = ChunkFormatter(model_name="llama-3")
        
        content_with_whitespace = "line 1\n\n\n\nline 2\n\n\nline 3"
        optimized = formatter._apply_model_optimizations(content_with_whitespace, None)
        
        # Llama optimization should reduce excessive whitespace
        assert optimized.count("\n\n\n") == 0

    def test_navigation_info_first_chunk(self):
        """Test navigation info for first chunk."""
        formatter = ChunkFormatter()
        chunk = Chunk(
            id="chunk_0",
            content="First chunk",
            chunk_index=0,
            total_chunks=3,
            token_count=5
        )
        
        nav = formatter._build_navigation_info(chunk)
        assert nav["has_previous"] is False
        assert nav["has_next"] is True
        assert nav["progress"] == "1/3"
        assert nav["percentage"] == 33.3

    def test_navigation_info_middle_chunk(self):
        """Test navigation info for middle chunk."""
        formatter = ChunkFormatter()
        chunk = Chunk(
            id="chunk_1",
            content="Middle chunk",
            chunk_index=1,
            total_chunks=3,
            token_count=5
        )
        
        nav = formatter._build_navigation_info(chunk)
        assert nav["has_previous"] is True
        assert nav["has_next"] is True
        assert nav["progress"] == "2/3"
        assert nav["percentage"] == 66.7

    def test_navigation_info_last_chunk(self):
        """Test navigation info for last chunk."""
        formatter = ChunkFormatter()
        chunk = Chunk(
            id="chunk_2",
            content="Last chunk",
            chunk_index=2,
            total_chunks=3,
            token_count=5
        )
        
        nav = formatter._build_navigation_info(chunk)
        assert nav["has_previous"] is True
        assert nav["has_next"] is False
        assert nav["progress"] == "3/3"
        assert nav["percentage"] == 100.0

    def test_format_multiple_chunks(self, sample_chunk):
        """Test formatting multiple chunks."""
        formatter = ChunkFormatter()
        
        chunks = [
            sample_chunk,
            Chunk(
                id="chunk_2",
                content="Second chunk",
                chunk_index=1,
                total_chunks=3,
                token_count=10
            ),
            Chunk(
                id="chunk_3",
                content="Third chunk",
                chunk_index=2,
                total_chunks=3,
                token_count=10
            )
        ]
        
        result = formatter.format_multiple_chunks(
            chunks,
            format_type=FormatType.MARKDOWN
        )
        
        assert "---" in result  # Separator
        assert "First chunk" in result or sample_chunk.content in result
        assert "Second chunk" in result
        assert "Third chunk" in result

    def test_format_multiple_chunks_with_memories(self, sample_chunk, sample_memory):
        """Test formatting multiple chunks with associated memories."""
        formatter = ChunkFormatter()
        
        chunks = [sample_chunk]
        memories = [sample_memory]
        
        result = formatter.format_multiple_chunks(
            chunks,
            format_type=FormatType.JSON,
            memories=memories
        )
        
        # Should be valid JSON
        data = json.loads(result)
        assert "metadata" in data

    def test_xml_escaping(self):
        """Test XML special character escaping."""
        formatter = ChunkFormatter()
        
        text_with_special = "Test <tag> & \"quotes\" & 'apostrophe'"
        escaped = formatter._escape_xml(text_with_special)
        
        assert "&lt;" in escaped
        assert "&gt;" in escaped
        assert "&amp;" in escaped
        assert "&quot;" in escaped
        assert "&apos;" in escaped

    def test_metadata_includes_chunk_info(self, sample_chunk, sample_memory):
        """Test that metadata includes all relevant chunk information."""
        formatter = ChunkFormatter()
        metadata = formatter._build_metadata_dict(sample_chunk, sample_memory)
        
        assert "chunk_id" in metadata
        assert "chunk_index" in metadata
        assert "total_chunks" in metadata
        assert "token_count" in metadata
        assert "relevance_score" in metadata
        assert "memory_type" in metadata
        assert "importance" in metadata
        assert metadata["chunk_index"] == 1  # 1-indexed

    def test_metadata_includes_boundary_type(self, sample_chunk):
        """Test that metadata includes boundary type."""
        formatter = ChunkFormatter()
        metadata = formatter._build_metadata_dict(sample_chunk, None)
        
        assert "boundary_type" in metadata
        assert metadata["boundary_type"] == "paragraph"

    def test_create_formatter_factory(self):
        """Test factory function for creating formatters."""
        formatter = create_formatter(model_name="gpt-4")
        assert isinstance(formatter, ChunkFormatter)
        assert formatter.model_name == "gpt-4"

    def test_format_type_string_conversion(self, sample_chunk):
        """Test that format type can be specified as string."""
        formatter = ChunkFormatter()
        
        # Should accept string format type
        result = formatter.format_chunk(sample_chunk, format_type="json")
        assert result.format_type == FormatType.JSON

    def test_formatted_chunk_token_estimation(self, sample_chunk):
        """Test that formatted chunk includes token count estimation."""
        formatter = ChunkFormatter()
        result = formatter.format_chunk(sample_chunk, format_type=FormatType.PLAIN)
        
        assert result.token_count > 0
        assert isinstance(result.token_count, int)

    def test_formatted_chunk_metadata(self, sample_chunk):
        """Test that formatted chunk includes metadata."""
        formatter = ChunkFormatter()
        result = formatter.format_chunk(sample_chunk, format_type=FormatType.PLAIN)
        
        assert "chunk_id" in result.metadata
        assert "chunk_index" in result.metadata
        assert "total_chunks" in result.metadata
        assert "original_tokens" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
