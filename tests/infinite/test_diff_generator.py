"""Tests for multi-level diff generation."""

import pytest

from core.infinite.diff_generator import DiffGenerator, Diff


class TestDiff:
    """Test Diff dataclass."""
    
    def test_diff_creation(self):
        """Test creating a Diff."""
        diff = Diff(
            level="line",
            content="test content"
        )
        
        assert diff.level == "line"
        assert diff.content == "test content"
        assert not diff.compressed
        assert diff.compression_ratio == 1.0
    
    def test_get_content_uncompressed(self):
        """Test getting uncompressed content."""
        diff = Diff(level="char", content="test")
        assert diff.get_content() == "test"
    
    def test_compression_and_decompression(self):
        """Test compressing and decompressing diff content."""
        import zstandard as zstd
        
        original = "test content " * 100
        compressor = zstd.ZstdCompressor()
        compressed_bytes = compressor.compress(original.encode('utf-8'))
        compressed_str = compressed_bytes.decode('latin-1')
        
        diff = Diff(
            level="char",
            content=compressed_str,
            compressed=True
        )
        
        decompressed = diff.decompress()
        assert decompressed == original
        assert diff.get_content() == original


class TestDiffGenerator:
    """Test DiffGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a DiffGenerator instance."""
        return DiffGenerator(
            compression_threshold=100,
            compression_level=3
        )
    
    def test_generator_initialization(self, generator):
        """Test creating a DiffGenerator."""
        assert generator.compression_threshold == 100
        assert generator.compression_level == 3
    
    def test_char_diff_simple(self, generator):
        """Test character-level diff for simple changes."""
        before = "hello world"
        after = "hello python"
        
        diff = generator.generate_char_diff(before, after, compress=False)
        
        assert diff.level == "char"
        assert not diff.compressed
        assert "hello" in diff.content
        assert "-" in diff.content  # Deletion marker
        assert "+" in diff.content  # Addition marker
    
    def test_char_diff_insertion(self, generator):
        """Test character-level diff for insertion."""
        before = "hello"
        after = "hello world"
        
        diff = generator.generate_char_diff(before, after, compress=False)
        
        assert diff.level == "char"
        assert "+" in diff.content
        assert "world" in diff.content
    
    def test_char_diff_deletion(self, generator):
        """Test character-level diff for deletion."""
        before = "hello world"
        after = "hello"
        
        diff = generator.generate_char_diff(before, after, compress=False)
        
        assert diff.level == "char"
        assert "-" in diff.content
        assert "world" in diff.content
    
    def test_char_diff_no_change(self, generator):
        """Test character-level diff when content is identical."""
        content = "hello world"
        
        diff = generator.generate_char_diff(content, content, compress=False)
        
        assert diff.level == "char"
        assert "=" in diff.content
        assert "hello world" in diff.content
    
    def test_line_diff_simple(self, generator):
        """Test line-level diff for simple changes."""
        before = "line 1\nline 2\nline 3"
        after = "line 1\nmodified line 2\nline 3"
        
        diff = generator.generate_line_diff(before, after, compress=False)
        
        assert diff.level == "line"
        assert not diff.compressed
        assert "-" in diff.content
        assert "+" in diff.content
    
    def test_line_diff_insertion(self, generator):
        """Test line-level diff for line insertion."""
        before = "line 1\nline 3"
        after = "line 1\nline 2\nline 3"
        
        diff = generator.generate_line_diff(before, after, compress=False)
        
        assert diff.level == "line"
        assert "+line 2" in diff.content
    
    def test_line_diff_deletion(self, generator):
        """Test line-level diff for line deletion."""
        before = "line 1\nline 2\nline 3"
        after = "line 1\nline 3"
        
        diff = generator.generate_line_diff(before, after, compress=False)
        
        assert diff.level == "line"
        assert "-line 2" in diff.content
    
    def test_unified_diff_format(self, generator):
        """Test unified diff format generation."""
        before = "line 1\nline 2\nline 3"
        after = "line 1\nmodified line 2\nline 3"
        
        diff = generator.generate_unified_diff(
            before, after,
            before_name="test.py",
            after_name="test.py",
            compress=False
        )
        
        assert diff.level == "unified"
        assert "---" in diff.content or "+++" in diff.content or "@@" in diff.content
    
    def test_unified_diff_context_lines(self, generator):
        """Test unified diff with custom context lines."""
        before = "line 1\nline 2\nline 3\nline 4\nline 5"
        after = "line 1\nline 2\nmodified\nline 4\nline 5"
        
        diff = generator.generate_unified_diff(
            before, after,
            context_lines=1,
            compress=False
        )
        
        assert diff.level == "unified"
        # Should have context lines around the change
        assert "@@" in diff.content
    
    def test_compression_threshold(self):
        """Test that compression is applied when threshold is exceeded."""
        generator = DiffGenerator(compression_threshold=10)
        
        before = "a" * 100
        after = "b" * 100
        
        diff = generator.generate_char_diff(before, after, compress=True)
        
        # Should be compressed since content > 10 bytes
        assert diff.compressed
        assert diff.compression_ratio < 1.0
    
    def test_no_compression_below_threshold(self):
        """Test that compression is not applied below threshold."""
        generator = DiffGenerator(compression_threshold=1000)
        
        before = "hello"
        after = "world"
        
        diff = generator.generate_char_diff(before, after, compress=True)
        
        # Should not be compressed since content < 1000 bytes
        assert not diff.compressed
    
    def test_generate_all_diffs(self, generator):
        """Test generating all diff levels at once."""
        before = "line 1\nline 2\nline 3"
        after = "line 1\nmodified line 2\nline 3"
        
        diffs = generator.generate_all_diffs(
            before, after,
            before_name="test.py",
            after_name="test.py",
            compress=False
        )
        
        assert "char" in diffs
        assert "line" in diffs
        assert "unified" in diffs
        
        assert diffs["char"].level == "char"
        assert diffs["line"].level == "line"
        assert diffs["unified"].level == "unified"
    
    def test_reconstruct_from_char_diff(self, generator):
        """Test reconstructing content from character-level diff."""
        before = "hello world"
        after = "hello python"
        
        diff = generator.generate_char_diff(before, after, compress=False)
        reconstructed = generator.reconstruct_from_char_diff(before, diff)
        
        assert reconstructed == after
    
    def test_reconstruct_from_char_diff_insertion(self, generator):
        """Test reconstruction with insertion."""
        before = "hello"
        after = "hello world"
        
        diff = generator.generate_char_diff(before, after, compress=False)
        reconstructed = generator.reconstruct_from_char_diff(before, diff)
        
        assert reconstructed == after
    
    def test_reconstruct_from_char_diff_deletion(self, generator):
        """Test reconstruction with deletion."""
        before = "hello world"
        after = "hello"
        
        diff = generator.generate_char_diff(before, after, compress=False)
        reconstructed = generator.reconstruct_from_char_diff(before, diff)
        
        assert reconstructed == after
    
    def test_reconstruct_from_char_diff_complex(self, generator):
        """Test reconstruction with complex changes."""
        before = "The quick brown fox jumps over the lazy dog"
        after = "The fast brown cat jumps over the sleepy dog"
        
        diff = generator.generate_char_diff(before, after, compress=False)
        reconstructed = generator.reconstruct_from_char_diff(before, diff)
        
        assert reconstructed == after
    
    def test_reconstruct_from_line_diff(self, generator):
        """Test reconstructing content from line-level diff."""
        before = "line 1\nline 2\nline 3"
        after = "line 1\nmodified line 2\nline 3"
        
        diff = generator.generate_line_diff(before, after, compress=False)
        reconstructed = generator.reconstruct_from_line_diff(before, diff)
        
        assert reconstructed == after
    
    def test_reconstruct_from_line_diff_multiline(self, generator):
        """Test reconstruction with multiple line changes."""
        before = "line 1\nline 2\nline 3\nline 4"
        after = "line 1\nmodified 2\nmodified 3\nline 4"
        
        diff = generator.generate_line_diff(before, after, compress=False)
        reconstructed = generator.reconstruct_from_line_diff(before, diff)
        
        assert reconstructed == after
    
    def test_reconstruct_from_compressed_diff(self, generator):
        """Test reconstruction from compressed diff."""
        before = "line " * 100
        after = "modified " * 100
        
        # Generate compressed diff
        diff = generator.generate_char_diff(before, after, compress=True)
        assert diff.compressed
        
        # Should still reconstruct correctly
        reconstructed = generator.reconstruct_from_char_diff(before, diff)
        assert reconstructed == after
    
    def test_reconstruct_wrong_diff_level(self, generator):
        """Test that reconstruction fails with wrong diff level."""
        before = "hello"
        after = "world"
        
        line_diff = generator.generate_line_diff(before, after, compress=False)
        
        with pytest.raises(ValueError, match="Expected char diff"):
            generator.reconstruct_from_char_diff(before, line_diff)
    
    def test_large_diff_compression(self, generator):
        """Test compression with large diffs."""
        # Create large content
        before_lines = ["line {}\n".format(i) for i in range(1000)]
        before = ''.join(before_lines)
        
        after_lines = ["modified line {}\n".format(i) for i in range(1000)]
        after = ''.join(after_lines)
        
        diff = generator.generate_char_diff(before, after, compress=True)
        
        # Should be compressed
        assert diff.compressed
        
        # Should still reconstruct correctly
        reconstructed = generator.reconstruct_from_char_diff(before, diff)
        assert reconstructed == after
    
    def test_empty_content(self, generator):
        """Test diff generation with empty content."""
        before = ""
        after = "new content"
        
        diff = generator.generate_char_diff(before, after, compress=False)
        reconstructed = generator.reconstruct_from_char_diff(before, diff)
        
        assert reconstructed == after
    
    def test_code_diff_example(self, generator):
        """Test diff generation with actual code."""
        before = """def hello():
    print('world')
    return True"""
        
        after = """def hello(name):
    print(f'Hello {name}')
    return True"""
        
        # Generate all diffs
        diffs = generator.generate_all_diffs(before, after, compress=False)
        
        # Verify char diff reconstruction
        reconstructed = generator.reconstruct_from_char_diff(before, diffs["char"])
        assert reconstructed == after
        
        # Verify line diff reconstruction
        reconstructed = generator.reconstruct_from_line_diff(before, diffs["line"])
        assert reconstructed == after
    
    def test_whitespace_changes(self, generator):
        """Test diff generation with whitespace changes."""
        before = "hello world"
        after = "hello  world"  # Extra space
        
        diff = generator.generate_char_diff(before, after, compress=False)
        reconstructed = generator.reconstruct_from_char_diff(before, diff)
        
        assert reconstructed == after
        assert len(reconstructed) == len(after)
    
    def test_newline_handling(self, generator):
        """Test diff generation with different newline styles."""
        before = "line1\nline2\nline3"
        after = "line1\nmodified\nline3"
        
        diff = generator.generate_line_diff(before, after, compress=False)
        reconstructed = generator.reconstruct_from_line_diff(before, diff)
        
        assert "modified" in reconstructed
