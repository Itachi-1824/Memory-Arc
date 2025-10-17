"""Semantic chunking algorithm for intelligent content splitting."""

import re
from typing import Any, Callable

from .models import Chunk, ChunkBoundary, BoundaryType


class SemanticChunker:
    """
    Semantic chunking algorithm that splits content at natural boundaries.
    
    Features:
    - Detects paragraphs, functions, classes, sections
    - Measures semantic coherence using embeddings
    - Generates overlap regions for context continuity
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap_size: int = 100,
        embedding_fn: Callable | None = None
    ):
        """
        Initialize semantic chunker.
        
        Args:
            max_chunk_size: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk
            overlap_size: Number of tokens to overlap between chunks
            embedding_fn: Optional function to compute embeddings for coherence
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.embedding_fn = embedding_fn

    def detect_boundaries(self, content: str, content_type: str = "text") -> list[ChunkBoundary]:
        """
        Detect natural boundaries in content.
        
        Args:
            content: The text content to analyze
            content_type: Type of content ('text', 'code', 'markdown')
            
        Returns:
            List of detected boundaries with positions and types
        """
        boundaries = []
        
        if content_type == "code":
            boundaries.extend(self._detect_code_boundaries(content))
        elif content_type == "markdown":
            boundaries.extend(self._detect_markdown_boundaries(content))
        else:
            boundaries.extend(self._detect_text_boundaries(content))
        
        # Sort by position
        boundaries.sort(key=lambda b: b.position)
        return boundaries

    def _detect_text_boundaries(self, content: str) -> list[ChunkBoundary]:
        """Detect boundaries in plain text."""
        boundaries = []
        
        # Detect paragraph boundaries (double newline)
        for match in re.finditer(r'\n\s*\n', content):
            boundaries.append(ChunkBoundary(
                position=match.end(),
                boundary_type=BoundaryType.PARAGRAPH,
                confidence=1.0
            ))
        
        # Detect sentence boundaries as fallback
        for match in re.finditer(r'[.!?]\s+', content):
            boundaries.append(ChunkBoundary(
                position=match.end(),
                boundary_type=BoundaryType.SENTENCE,
                confidence=0.7
            ))
        
        return boundaries

    def _detect_markdown_boundaries(self, content: str) -> list[ChunkBoundary]:
        """Detect boundaries in markdown content."""
        boundaries = []
        
        # Detect section headers
        for match in re.finditer(r'^#{1,6}\s+.+$', content, re.MULTILINE):
            boundaries.append(ChunkBoundary(
                position=match.start(),
                boundary_type=BoundaryType.SECTION,
                confidence=1.0
            ))
        
        # Detect paragraph boundaries
        for match in re.finditer(r'\n\s*\n', content):
            boundaries.append(ChunkBoundary(
                position=match.end(),
                boundary_type=BoundaryType.PARAGRAPH,
                confidence=0.9
            ))
        
        return boundaries

    def _detect_code_boundaries(self, content: str) -> list[ChunkBoundary]:
        """Detect boundaries in code content."""
        boundaries = []
        
        # Detect function definitions (Python-style, including indented methods)
        for match in re.finditer(r'^\s*def\s+\w+\s*\(', content, re.MULTILINE):
            boundaries.append(ChunkBoundary(
                position=match.start(),
                boundary_type=BoundaryType.FUNCTION,
                confidence=1.0
            ))
        
        # Detect class definitions (Python-style)
        for match in re.finditer(r'^class\s+\w+', content, re.MULTILINE):
            boundaries.append(ChunkBoundary(
                position=match.start(),
                boundary_type=BoundaryType.CLASS,
                confidence=1.0
            ))
        
        # Detect JavaScript/TypeScript functions
        for match in re.finditer(r'function\s+\w+\s*\(', content):
            boundaries.append(ChunkBoundary(
                position=match.start(),
                boundary_type=BoundaryType.FUNCTION,
                confidence=1.0
            ))
        
        # Detect arrow functions
        for match in re.finditer(r'const\s+\w+\s*=\s*\([^)]*\)\s*=>', content):
            boundaries.append(ChunkBoundary(
                position=match.start(),
                boundary_type=BoundaryType.FUNCTION,
                confidence=0.9
            ))
        
        return boundaries

    def measure_coherence(self, text1: str, text2: str) -> float:
        """
        Measure semantic coherence between two text segments.
        
        Args:
            text1: First text segment
            text2: Second text segment
            
        Returns:
            Coherence score between 0 and 1
        """
        if not self.embedding_fn:
            # Fallback: simple lexical overlap
            return self._lexical_coherence(text1, text2)
        
        # Use embeddings for semantic similarity
        try:
            emb1 = self.embedding_fn(text1)
            emb2 = self.embedding_fn(text2)
            return self._cosine_similarity(emb1, emb2)
        except Exception:
            # Fallback on error
            return self._lexical_coherence(text1, text2)

    def _lexical_coherence(self, text1: str, text2: str) -> float:
        """Calculate coherence based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)

    def generate_overlap_region(
        self,
        prev_chunk: str,
        next_chunk: str,
        overlap_tokens: int | None = None
    ) -> tuple[str, str]:
        """
        Generate overlap regions between consecutive chunks.
        
        Args:
            prev_chunk: Previous chunk content
            next_chunk: Next chunk content
            overlap_tokens: Number of tokens to overlap (uses self.overlap_size if None)
            
        Returns:
            Tuple of (prev_chunk_with_overlap, next_chunk_with_overlap)
        """
        if overlap_tokens is None:
            overlap_tokens = self.overlap_size
        
        # Simple word-based approximation (1 token ≈ 0.75 words)
        overlap_words = int(overlap_tokens * 0.75)
        
        # Extract overlap from end of prev_chunk
        prev_words = prev_chunk.split()
        if len(prev_words) > overlap_words:
            overlap_text = ' '.join(prev_words[-overlap_words:])
        else:
            overlap_text = prev_chunk
        
        # Add overlap to start of next_chunk
        next_with_overlap = overlap_text + " " + next_chunk
        
        return prev_chunk, next_with_overlap

    def chunk_by_boundaries(
        self,
        content: str,
        boundaries: list[ChunkBoundary],
        token_estimator: Callable | None = None
    ) -> list[tuple[int, int, BoundaryType]]:
        """
        Split content into chunks based on detected boundaries.
        
        Args:
            content: Content to chunk
            boundaries: Detected boundaries
            token_estimator: Function to estimate token count
            
        Returns:
            List of (start_pos, end_pos, boundary_type) tuples
        """
        if not boundaries:
            # No boundaries, return whole content as one chunk
            return [(0, len(content), BoundaryType.PARAGRAPH)]
        
        if token_estimator is None:
            token_estimator = self._estimate_tokens
        
        chunks = []
        current_start = 0
        current_tokens = 0
        
        for i, boundary in enumerate(boundaries):
            segment = content[current_start:boundary.position]
            segment_tokens = token_estimator(segment)
            
            # Check if adding this segment would exceed max size
            if current_tokens + segment_tokens > self.max_chunk_size and current_tokens >= self.min_chunk_size:
                # Create chunk up to previous boundary
                chunks.append((current_start, boundary.position, boundary.boundary_type))
                current_start = boundary.position
                current_tokens = 0
            else:
                current_tokens += segment_tokens
        
        # Add final chunk
        if current_start < len(content):
            # Find the boundary type for the last chunk
            last_boundary_type = boundaries[-1].boundary_type if boundaries else BoundaryType.PARAGRAPH
            chunks.append((current_start, len(content), last_boundary_type))
        
        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Simple approximation: 1 token ≈ 4 characters or 0.75 words.
        """
        word_count = len(text.split())
        char_count = len(text)
        
        # Use average of both methods
        token_estimate = (word_count / 0.75 + char_count / 4) / 2
        return int(token_estimate)

    def chunk_content(
        self,
        content: str,
        content_type: str = "text",
        token_estimator: Callable | None = None
    ) -> list[Chunk]:
        """
        Main method to chunk content semantically.
        
        Args:
            content: Content to chunk
            content_type: Type of content ('text', 'code', 'markdown')
            token_estimator: Optional function to estimate tokens
            
        Returns:
            List of Chunk objects
        """
        # Detect boundaries
        boundaries = self.detect_boundaries(content, content_type)
        
        # Split into chunks
        chunk_positions = self.chunk_by_boundaries(content, boundaries, token_estimator)
        
        # Create Chunk objects
        chunks = []
        total_chunks = len(chunk_positions)
        
        for idx, (start, end, boundary_type) in enumerate(chunk_positions):
            chunk_content = content[start:end]
            
            # Apply overlap if not the last chunk
            if idx < total_chunks - 1:
                next_start = chunk_positions[idx + 1][0]
                next_content = content[next_start:min(next_start + 200, len(content))]
                _, chunk_content = self.generate_overlap_region(chunk_content, next_content)
            
            token_count = token_estimator(chunk_content) if token_estimator else self._estimate_tokens(chunk_content)
            
            chunk = Chunk(
                id=f"chunk_{idx}",
                content=chunk_content,
                chunk_index=idx,
                total_chunks=total_chunks,
                token_count=token_count,
                start_pos=start,
                end_pos=end,
                boundary_type=boundary_type,
                metadata={
                    "content_type": content_type,
                    "has_overlap": idx < total_chunks - 1
                }
            )
            chunks.append(chunk)
        
        return chunks
