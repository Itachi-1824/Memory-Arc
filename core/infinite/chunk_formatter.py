"""Model-specific formatting for chunks with optimizations and metadata."""

import json
from typing import Any, Literal
from dataclasses import dataclass
from enum import Enum

from .models import Chunk, Memory, MemoryType


class FormatType(Enum):
    """Supported output formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    PLAIN = "plain"
    XML = "xml"


@dataclass
class FormattedChunk:
    """A formatted chunk ready for model consumption."""
    content: str
    format_type: FormatType
    token_count: int
    metadata: dict[str, Any]


class ChunkFormatter:
    """
    Model-specific formatting for chunks.
    
    Features:
    - Multiple output formats (JSON, Markdown, plain text, XML)
    - Model-specific optimizations for token efficiency
    - Metadata headers for context awareness
    - Syntax highlighting markers for code
    - Compression of repetitive information
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        include_metadata: bool = True,
        compress_repetitive: bool = True
    ):
        """
        Initialize chunk formatter.
        
        Args:
            model_name: Name of the target model for optimizations
            include_metadata: Whether to include metadata headers
            compress_repetitive: Whether to compress repetitive content
        """
        self.model_name = model_name.lower()
        self.include_metadata = include_metadata
        self.compress_repetitive = compress_repetitive

    def format_chunk(
        self,
        chunk: Chunk,
        format_type: FormatType | str = FormatType.MARKDOWN,
        memory: Memory | None = None,
        include_navigation: bool = True
    ) -> FormattedChunk:
        """
        Format a chunk for model consumption.
        
        Args:
            chunk: The chunk to format
            format_type: Output format type
            memory: Optional associated memory for additional context
            include_navigation: Whether to include navigation hints
            
        Returns:
            FormattedChunk with formatted content
        """
        if isinstance(format_type, str):
            format_type = FormatType(format_type.lower())
        
        # Apply model-specific optimizations
        content = self._apply_model_optimizations(chunk.content, memory)
        
        # Compress repetitive content if enabled
        if self.compress_repetitive:
            content = self._compress_repetitive_content(content)
        
        # Format based on type
        if format_type == FormatType.JSON:
            formatted = self._format_json(chunk, content, memory, include_navigation)
        elif format_type == FormatType.MARKDOWN:
            formatted = self._format_markdown(chunk, content, memory, include_navigation)
        elif format_type == FormatType.XML:
            formatted = self._format_xml(chunk, content, memory, include_navigation)
        else:  # PLAIN
            formatted = self._format_plain(chunk, content, memory, include_navigation)
        
        # Estimate token count (rough approximation)
        token_count = len(formatted.split()) // 0.75
        
        return FormattedChunk(
            content=formatted,
            format_type=format_type,
            token_count=int(token_count),
            metadata={
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "original_tokens": chunk.token_count
            }
        )

    def _format_json(
        self,
        chunk: Chunk,
        content: str,
        memory: Memory | None,
        include_navigation: bool
    ) -> str:
        """Format chunk as JSON."""
        data = {
            "content": content,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks
        }
        
        if self.include_metadata:
            data["metadata"] = self._build_metadata_dict(chunk, memory)
        
        if include_navigation:
            data["navigation"] = self._build_navigation_info(chunk)
        
        # Add syntax highlighting info for code
        if memory and memory.memory_type == MemoryType.CODE:
            data["syntax"] = self._detect_language(content)
            data["type"] = "code"
        
        return json.dumps(data, indent=2)

    def _format_markdown(
        self,
        chunk: Chunk,
        content: str,
        memory: Memory | None,
        include_navigation: bool
    ) -> str:
        """Format chunk as Markdown."""
        parts = []
        
        # Add metadata header
        if self.include_metadata:
            parts.append(self._build_metadata_header_markdown(chunk, memory))
        
        # Add navigation info
        if include_navigation:
            parts.append(self._build_navigation_markdown(chunk))
        
        # Add content with appropriate formatting
        if memory and memory.memory_type == MemoryType.CODE:
            # Code block with syntax highlighting
            language = self._detect_language(content)
            parts.append(f"```{language}\n{content}\n```")
        else:
            parts.append(content)
        
        return "\n\n".join(parts)

    def _format_xml(
        self,
        chunk: Chunk,
        content: str,
        memory: Memory | None,
        include_navigation: bool
    ) -> str:
        """Format chunk as XML."""
        parts = ['<chunk>']
        
        # Add metadata
        if self.include_metadata:
            parts.append('  <metadata>')
            metadata = self._build_metadata_dict(chunk, memory)
            for key, value in metadata.items():
                parts.append(f'    <{key}>{self._escape_xml(str(value))}</{key}>')
            parts.append('  </metadata>')
        
        # Add navigation
        if include_navigation:
            nav = self._build_navigation_info(chunk)
            parts.append('  <navigation>')
            for key, value in nav.items():
                parts.append(f'    <{key}>{value}</{key}>')
            parts.append('  </navigation>')
        
        # Add content
        if memory and memory.memory_type == MemoryType.CODE:
            language = self._detect_language(content)
            parts.append(f'  <content type="code" language="{language}">')
            parts.append(f'    <![CDATA[{content}]]>')
            parts.append('  </content>')
        else:
            parts.append('  <content>')
            parts.append(f'    <![CDATA[{content}]]>')
            parts.append('  </content>')
        
        parts.append('</chunk>')
        return '\n'.join(parts)

    def _format_plain(
        self,
        chunk: Chunk,
        content: str,
        memory: Memory | None,
        include_navigation: bool
    ) -> str:
        """Format chunk as plain text."""
        parts = []
        
        # Add metadata header
        if self.include_metadata:
            parts.append(self._build_metadata_header_plain(chunk, memory))
        
        # Add navigation info
        if include_navigation:
            parts.append(self._build_navigation_plain(chunk))
        
        # Add content
        parts.append(content)
        
        return "\n\n".join(parts)

    def _build_metadata_dict(self, chunk: Chunk, memory: Memory | None) -> dict[str, Any]:
        """Build metadata dictionary."""
        metadata = {
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index + 1,  # 1-indexed for display
            "total_chunks": chunk.total_chunks,
            "token_count": chunk.token_count,
            "relevance_score": round(chunk.relevance_score, 3)
        }
        
        if memory:
            metadata["memory_type"] = memory.memory_type.value
            metadata["importance"] = memory.importance
            metadata["created_at"] = memory.created_at
        
        if chunk.boundary_type:
            metadata["boundary_type"] = chunk.boundary_type.value
        
        # Add custom metadata from chunk
        if chunk.metadata:
            metadata.update(chunk.metadata)
        
        return metadata

    def _build_metadata_header_markdown(self, chunk: Chunk, memory: Memory | None) -> str:
        """Build metadata header in Markdown format."""
        lines = ["---"]
        metadata = self._build_metadata_dict(chunk, memory)
        
        for key, value in metadata.items():
            lines.append(f"{key}: {value}")
        
        lines.append("---")
        return "\n".join(lines)

    def _build_metadata_header_plain(self, chunk: Chunk, memory: Memory | None) -> str:
        """Build metadata header in plain text format."""
        lines = ["=" * 60]
        metadata = self._build_metadata_dict(chunk, memory)
        
        for key, value in metadata.items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)

    def _build_navigation_info(self, chunk: Chunk) -> dict[str, Any]:
        """Build navigation information."""
        return {
            "has_previous": chunk.chunk_index > 0,
            "has_next": chunk.chunk_index < chunk.total_chunks - 1,
            "progress": f"{chunk.chunk_index + 1}/{chunk.total_chunks}",
            "percentage": round((chunk.chunk_index + 1) / chunk.total_chunks * 100, 1)
        }

    def _build_navigation_markdown(self, chunk: Chunk) -> str:
        """Build navigation info in Markdown format."""
        nav = self._build_navigation_info(chunk)
        
        parts = [f"**Chunk {nav['progress']}** ({nav['percentage']}%)"]
        
        if nav["has_previous"]:
            parts.append("← Previous chunk available")
        if nav["has_next"]:
            parts.append("Next chunk available →")
        
        return " | ".join(parts)

    def _build_navigation_plain(self, chunk: Chunk) -> str:
        """Build navigation info in plain text format."""
        nav = self._build_navigation_info(chunk)
        
        parts = [f"Chunk {nav['progress']} ({nav['percentage']}%)"]
        
        if nav["has_previous"]:
            parts.append("[Previous available]")
        if nav["has_next"]:
            parts.append("[Next available]")
        
        return " ".join(parts)

    def _detect_language(self, content: str) -> str:
        """
        Detect programming language from content.
        
        Args:
            content: Code content to analyze
            
        Returns:
            Language identifier (e.g., 'python', 'javascript')
        """
        # Simple heuristic-based detection with priority order
        # Check more specific patterns first to avoid false positives
        
        # Go indicators (check before Python due to 'import')
        if any(keyword in content for keyword in ['package main', 'package ', 'func main', 'fmt.Print']):
            return "go"
        
        # Rust indicators (check before JavaScript due to 'let')
        if any(keyword in content for keyword in ['fn main', 'fn ', 'let mut', 'impl ', 'pub fn']):
            return "rust"
        
        # Java indicators (check before Python due to 'class')
        if any(keyword in content for keyword in ['public class', 'private class', 'System.out', 'void main']):
            return "java"
        if 'public ' in content and 'private ' in content and 'void ' in content:
            return "java"
        
        # C/C++ indicators
        if any(keyword in content for keyword in ['#include', 'int main', 'printf']):
            if 'std::' in content or 'namespace' in content or 'using namespace' in content:
                return "cpp"
            return "c"
        
        # TypeScript indicators (check before JavaScript)
        if any(keyword in content for keyword in ['interface ', ': string', ': number', ': boolean']):
            return "typescript"
        
        # JavaScript indicators
        if any(keyword in content for keyword in ['function ', 'const ', 'let ', 'var ', '=>', 'console.log']):
            return "javascript"
        
        # Python indicators (check last as it has common keywords)
        if any(keyword in content for keyword in ['def ', 'import ', 'class ', '__init__', 'self.', 'print(']):
            return "python"
        
        # Default
        return "text"

    def _apply_model_optimizations(self, content: str, memory: Memory | None) -> str:
        """
        Apply model-specific optimizations for token efficiency.
        
        Args:
            content: Content to optimize
            memory: Optional memory for context
            
        Returns:
            Optimized content
        """
        # GPT models: prefer structured format, handle long contexts well
        if "gpt" in self.model_name:
            return self._optimize_for_gpt(content, memory)
        
        # Claude models: prefer natural language, excellent with long contexts
        elif "claude" in self.model_name:
            return self._optimize_for_claude(content, memory)
        
        # Llama models: prefer concise format, limited context
        elif "llama" in self.model_name:
            return self._optimize_for_llama(content, memory)
        
        # Default: no optimization
        return content

    def _optimize_for_gpt(self, content: str, memory: Memory | None) -> str:
        """Optimize content for GPT models."""
        # GPT models handle structured content well
        # No major changes needed
        return content

    def _optimize_for_claude(self, content: str, memory: Memory | None) -> str:
        """Optimize content for Claude models."""
        # Claude prefers natural language flow
        # Add context markers for better understanding
        if memory and memory.memory_type == MemoryType.CODE:
            # Add brief description for code
            return f"[Code snippet]\n{content}"
        return content

    def _optimize_for_llama(self, content: str, memory: Memory | None) -> str:
        """Optimize content for Llama models."""
        # Llama models benefit from more concise format
        # Remove excessive whitespace
        lines = [line.rstrip() for line in content.split('\n')]
        # Remove multiple consecutive blank lines
        optimized_lines = []
        prev_blank = False
        for line in lines:
            if not line.strip():
                if not prev_blank:
                    optimized_lines.append(line)
                prev_blank = True
            else:
                optimized_lines.append(line)
                prev_blank = False
        
        return '\n'.join(optimized_lines)

    def _compress_repetitive_content(self, content: str) -> str:
        """
        Compress repetitive information while maintaining semantic completeness.
        
        Args:
            content: Content to compress
            
        Returns:
            Compressed content
        """
        # Detect and compress repeated patterns
        lines = content.split('\n')
        compressed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for repeated lines
            repeat_count = 1
            j = i + 1
            while j < len(lines) and lines[j] == line and line.strip():
                repeat_count += 1
                j += 1
            
            if repeat_count >= 3:
                # Compress repeated lines
                compressed_lines.append(line)
                compressed_lines.append(f"[... repeated {repeat_count - 1} more times ...]")
                i = j
            else:
                compressed_lines.append(line)
                i += 1
        
        return '\n'.join(compressed_lines)

    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&apos;'))

    def format_multiple_chunks(
        self,
        chunks: list[Chunk],
        format_type: FormatType | str = FormatType.MARKDOWN,
        memories: list[Memory] | None = None,
        separator: str = "\n\n---\n\n"
    ) -> str:
        """
        Format multiple chunks into a single output.
        
        Args:
            chunks: List of chunks to format
            format_type: Output format type
            memories: Optional list of associated memories
            separator: Separator between chunks
            
        Returns:
            Formatted string containing all chunks
        """
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            memory = memories[i] if memories and i < len(memories) else None
            formatted = self.format_chunk(chunk, format_type, memory)
            formatted_chunks.append(formatted.content)
        
        return separator.join(formatted_chunks)


def create_formatter(
    model_name: str = "gpt-4",
    include_metadata: bool = True,
    compress_repetitive: bool = True
) -> ChunkFormatter:
    """
    Factory function to create a chunk formatter.
    
    Args:
        model_name: Name of the target model
        include_metadata: Whether to include metadata headers
        compress_repetitive: Whether to compress repetitive content
        
    Returns:
        ChunkFormatter instance
    """
    return ChunkFormatter(model_name, include_metadata, compress_repetitive)
