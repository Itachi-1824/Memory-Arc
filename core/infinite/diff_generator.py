"""Multi-level diff generation for code changes."""

import difflib
import zstandard as zstd
from dataclasses import dataclass
from typing import Literal


DiffLevel = Literal["char", "line", "unified"]


@dataclass
class Diff:
    """Represents a diff at a specific level."""
    level: DiffLevel
    content: str
    compressed: bool = False
    compression_ratio: float = 1.0
    
    def decompress(self) -> str:
        """Decompress the diff content if compressed."""
        if not self.compressed:
            return self.content
        
        dctx = zstd.ZstdDecompressor()
        decompressed_bytes = dctx.decompress(self.content.encode('latin-1'))
        return decompressed_bytes.decode('utf-8')
    
    def get_content(self) -> str:
        """Get the diff content (decompressing if necessary)."""
        return self.decompress() if self.compressed else self.content


class DiffGenerator:
    """
    Generate multi-level diffs for code changes.
    
    Supports:
    - Character-level diffs (precise changes)
    - Line-level diffs (traditional diff)
    - Unified diff format (standard patch format)
    - Automatic compression for large diffs
    """
    
    def __init__(
        self,
        compression_threshold: int = 1000,
        compression_level: int = 3
    ):
        """
        Initialize diff generator.
        
        Args:
            compression_threshold: Compress diffs larger than this (bytes)
            compression_level: zstd compression level (1-22, higher = better compression)
        """
        self.compression_threshold = compression_threshold
        self.compression_level = compression_level
        self._compressor = zstd.ZstdCompressor(level=compression_level)
        self._decompressor = zstd.ZstdDecompressor()
    
    def generate_char_diff(
        self,
        before: str,
        after: str,
        compress: bool = True
    ) -> Diff:
        """
        Generate character-level diff.
        
        Args:
            before: Original content
            after: Modified content
            compress: Whether to compress the diff
            
        Returns:
            Diff object with character-level changes
        """
        # Use SequenceMatcher for character-level diff
        matcher = difflib.SequenceMatcher(None, before, after)
        
        # Build diff representation
        diff_parts = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Keep equal parts as-is (for context)
                diff_parts.append(f"={before[i1:i2]}")
            elif tag == 'delete':
                diff_parts.append(f"-{before[i1:i2]}")
            elif tag == 'insert':
                diff_parts.append(f"+{after[j1:j2]}")
            elif tag == 'replace':
                diff_parts.append(f"-{before[i1:i2]}")
                diff_parts.append(f"+{after[j1:j2]}")
        
        diff_content = ''.join(diff_parts)
        
        # Compress if needed
        should_compress = compress and len(diff_content) > self.compression_threshold
        if should_compress:
            compressed_bytes = self._compressor.compress(diff_content.encode('utf-8'))
            compressed_content = compressed_bytes.decode('latin-1')
            compression_ratio = len(compressed_content) / len(diff_content)
            
            return Diff(
                level="char",
                content=compressed_content,
                compressed=True,
                compression_ratio=compression_ratio
            )
        
        return Diff(level="char", content=diff_content)
    
    def generate_line_diff(
        self,
        before: str,
        after: str,
        compress: bool = True
    ) -> Diff:
        """
        Generate line-level diff.
        
        Args:
            before: Original content
            after: Modified content
            compress: Whether to compress the diff
            
        Returns:
            Diff object with line-level changes
        """
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        
        # Use SequenceMatcher for line-level diff
        matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
        
        # Build diff representation
        diff_parts = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                diff_parts.extend(f" {line}" for line in before_lines[i1:i2])
            elif tag == 'delete':
                diff_parts.extend(f"-{line}" for line in before_lines[i1:i2])
            elif tag == 'insert':
                diff_parts.extend(f"+{line}" for line in after_lines[j1:j2])
            elif tag == 'replace':
                diff_parts.extend(f"-{line}" for line in before_lines[i1:i2])
                diff_parts.extend(f"+{line}" for line in after_lines[j1:j2])
        
        diff_content = ''.join(diff_parts)
        
        # Compress if needed
        should_compress = compress and len(diff_content) > self.compression_threshold
        if should_compress:
            compressed_bytes = self._compressor.compress(diff_content.encode('utf-8'))
            compressed_content = compressed_bytes.decode('latin-1')
            compression_ratio = len(compressed_content) / len(diff_content)
            
            return Diff(
                level="line",
                content=compressed_content,
                compressed=True,
                compression_ratio=compression_ratio
            )
        
        return Diff(level="line", content=diff_content)
    
    def generate_unified_diff(
        self,
        before: str,
        after: str,
        before_name: str = "before",
        after_name: str = "after",
        context_lines: int = 3,
        compress: bool = True
    ) -> Diff:
        """
        Generate unified diff format (standard patch format).
        
        Args:
            before: Original content
            after: Modified content
            before_name: Name for before file
            after_name: Name for after file
            context_lines: Number of context lines
            compress: Whether to compress the diff
            
        Returns:
            Diff object in unified format
        """
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        
        # Generate unified diff
        diff_lines = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=before_name,
            tofile=after_name,
            n=context_lines
        )
        
        diff_content = ''.join(diff_lines)
        
        # Compress if needed
        should_compress = compress and len(diff_content) > self.compression_threshold
        if should_compress:
            compressed_bytes = self._compressor.compress(diff_content.encode('utf-8'))
            compressed_content = compressed_bytes.decode('latin-1')
            compression_ratio = len(compressed_content) / len(diff_content)
            
            return Diff(
                level="unified",
                content=compressed_content,
                compressed=True,
                compression_ratio=compression_ratio
            )
        
        return Diff(level="unified", content=diff_content)
    
    def generate_all_diffs(
        self,
        before: str,
        after: str,
        before_name: str = "before",
        after_name: str = "after",
        compress: bool = True
    ) -> dict[DiffLevel, Diff]:
        """
        Generate all diff levels at once.
        
        Args:
            before: Original content
            after: Modified content
            before_name: Name for before file
            after_name: Name for after file
            compress: Whether to compress diffs
            
        Returns:
            Dictionary mapping diff level to Diff object
        """
        return {
            "char": self.generate_char_diff(before, after, compress),
            "line": self.generate_line_diff(before, after, compress),
            "unified": self.generate_unified_diff(
                before, after, before_name, after_name, compress=compress
            )
        }
    
    def reconstruct_from_char_diff(
        self,
        before: str,
        diff: Diff
    ) -> str:
        """
        Reconstruct content from character-level diff.
        
        Args:
            before: Original content
            diff: Character-level diff
            
        Returns:
            Reconstructed content
        """
        if diff.level != "char":
            raise ValueError(f"Expected char diff, got {diff.level}")
        
        diff_content = diff.get_content()
        
        # Parse diff and reconstruct
        result = []
        i = 0
        before_pos = 0
        
        while i < len(diff_content):
            op = diff_content[i]
            i += 1
            
            # Find the end of this operation
            end = i
            while end < len(diff_content) and diff_content[end] not in '=+-':
                end += 1
            
            content = diff_content[i:end]
            
            if op == '=':
                # Equal content - verify it matches
                expected = before[before_pos:before_pos + len(content)]
                if expected != content:
                    # Fallback: use the diff content
                    result.append(content)
                else:
                    result.append(content)
                before_pos += len(content)
            elif op == '-':
                # Deleted content - skip in before
                before_pos += len(content)
            elif op == '+':
                # Inserted content - add to result
                result.append(content)
            
            i = end
        
        return ''.join(result)
    
    def reconstruct_from_line_diff(
        self,
        before: str,
        diff: Diff
    ) -> str:
        """
        Reconstruct content from line-level diff.
        
        Args:
            before: Original content
            diff: Line-level diff
            
        Returns:
            Reconstructed content
        """
        if diff.level != "line":
            raise ValueError(f"Expected line diff, got {diff.level}")
        
        diff_content = diff.get_content()
        diff_lines = diff_content.splitlines(keepends=True)
        
        result = []
        for line in diff_lines:
            if not line:
                continue
            
            op = line[0]
            content = line[1:]
            
            if op == ' ' or op == '=':
                # Equal line
                result.append(content)
            elif op == '+':
                # Added line
                result.append(content)
            # Skip deleted lines (op == '-')
        
        return ''.join(result)
    
    def apply_unified_diff(
        self,
        before: str,
        diff: Diff
    ) -> str:
        """
        Apply unified diff to reconstruct content.
        
        Args:
            before: Original content
            diff: Unified diff
            
        Returns:
            Reconstructed content
        """
        if diff.level != "unified":
            raise ValueError(f"Expected unified diff, got {diff.level}")
        
        # For unified diff, we'll use a simple line-based approach
        # In production, you might want to use a proper patch library
        diff_content = diff.get_content()
        diff_lines = diff_content.splitlines()
        
        before_lines = before.splitlines()
        result = []
        
        # Skip header lines
        i = 0
        while i < len(diff_lines) and not diff_lines[i].startswith('@@'):
            i += 1
        
        before_idx = 0
        
        while i < len(diff_lines):
            line = diff_lines[i]
            
            if line.startswith('@@'):
                # Parse hunk header
                # Format: @@ -start,count +start,count @@
                parts = line.split()
                if len(parts) >= 2:
                    before_part = parts[1]  # -start,count
                    if ',' in before_part:
                        start = int(before_part.split(',')[0][1:])
                        before_idx = start - 1
                i += 1
                continue
            
            if line.startswith('-'):
                # Deleted line - skip
                before_idx += 1
            elif line.startswith('+'):
                # Added line
                result.append(line[1:])
            elif line.startswith(' '):
                # Context line
                result.append(line[1:])
                before_idx += 1
            else:
                # Unknown line type, treat as context
                result.append(line)
            
            i += 1
        
        return '\n'.join(result)
