"""ChunkManager: Orchestrates chunking, selection, and formatting for infinite context."""

from typing import Any, Literal, Iterator
from dataclasses import dataclass

from .models import Chunk, Memory, MemoryType
from .semantic_chunker import SemanticChunker
from .token_counter import TokenCounter, get_model_context_window
from .chunk_selector import ChunkSelector, ScoredChunk
from .chunk_formatter import ChunkFormatter, FormatType, FormattedChunk


@dataclass
class ChunkManagerConfig:
    """Configuration for ChunkManager."""
    model_name: str = "gpt-4"
    max_tokens: int | None = None  # Auto-detect from model if None
    overlap_tokens: int = 100
    relevance_weight: float = 0.5
    importance_weight: float = 0.3
    recency_weight: float = 0.2
    include_metadata: bool = True
    compress_repetitive: bool = True
    preserve_structure: bool = True


class ChunkManager:
    """
    Orchestrates chunking, selection, and formatting for infinite context.
    
    Integrates:
    - SemanticChunker: Splits content at natural boundaries
    - TokenCounter: Accurate token counting for models
    - ChunkSelector: Priority-based chunk selection
    - ChunkFormatter: Model-specific formatting
    
    Features:
    - Intelligent content chunking with semantic boundaries
    - Priority-based chunk selection (relevance + importance + recency)
    - Model-specific formatting and optimizations
    - Streaming chunk generation for large contexts
    - Navigation between chunks
    """

    def __init__(
        self,
        config: ChunkManagerConfig | None = None,
        embedding_fn: Any = None
    ):
        """
        Initialize ChunkManager.
        
        Args:
            config: Configuration for chunk manager
            embedding_fn: Optional function to compute embeddings
        """
        self.config = config or ChunkManagerConfig()
        self.embedding_fn = embedding_fn
        
        # Auto-detect max tokens if not specified
        if self.config.max_tokens is None:
            self.config.max_tokens = get_model_context_window(self.config.model_name)
        
        # Initialize components
        self.token_counter = TokenCounter(self.config.model_name)
        
        self.semantic_chunker = SemanticChunker(
            max_chunk_size=self.config.max_tokens // 2,  # Leave room for prompt/response
            min_chunk_size=100,
            overlap_size=self.config.overlap_tokens,
            embedding_fn=embedding_fn
        )
        
        self.chunk_selector = ChunkSelector(
            relevance_weight=self.config.relevance_weight,
            importance_weight=self.config.importance_weight,
            recency_weight=self.config.recency_weight,
            embedding_fn=embedding_fn
        )
        
        self.chunk_formatter = ChunkFormatter(
            model_name=self.config.model_name,
            include_metadata=self.config.include_metadata,
            compress_repetitive=self.config.compress_repetitive
        )
        
        # Cache for chunk navigation
        self._chunk_cache: dict[str, Chunk] = {}

    def chunk_content(
        self,
        content: str | list[Memory],
        content_type: str = "text",
        query: str | None = None,
        preserve_structure: bool | None = None
    ) -> list[Chunk]:
        """
        Split content into model-appropriate chunks.
        
        Args:
            content: Content to chunk (string or list of Memory objects)
            content_type: Type of content ('text', 'code', 'markdown', 'conversation')
            query: Optional query for relevance ranking
            preserve_structure: Whether to preserve structural boundaries (uses config if None)
            
        Returns:
            List of Chunk objects
        """
        if preserve_structure is None:
            preserve_structure = self.config.preserve_structure
        
        # Handle Memory objects
        if isinstance(content, list):
            return self._chunk_memories(content, query)
        
        # Handle string content
        token_estimator = self.token_counter.count_tokens
        
        chunks = self.semantic_chunker.chunk_content(
            content=content,
            content_type=content_type,
            token_estimator=token_estimator
        )
        
        # Cache chunks for navigation
        for chunk in chunks:
            self._chunk_cache[chunk.id] = chunk
        
        return chunks

    def _chunk_memories(
        self,
        memories: list[Memory],
        query: str | None = None
    ) -> list[Chunk]:
        """
        Chunk a list of Memory objects.
        
        Args:
            memories: List of memories to chunk
            query: Optional query for relevance
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        current_content = []
        current_tokens = 0
        max_chunk_tokens = self.config.max_tokens // 2
        
        for memory in memories:
            memory_tokens = self.token_counter.count_tokens(memory.content)
            
            # If single memory exceeds max, chunk it separately
            if memory_tokens > max_chunk_tokens:
                # Flush current chunk if any
                if current_content:
                    chunk = self._create_chunk_from_memories(
                        current_content,
                        len(chunks),
                        current_tokens
                    )
                    chunks.append(chunk)
                    current_content = []
                    current_tokens = 0
                
                # Chunk the large memory
                content_type = self._detect_content_type(memory)
                memory_chunks = self.semantic_chunker.chunk_content(
                    content=memory.content,
                    content_type=content_type,
                    token_estimator=self.token_counter.count_tokens
                )
                
                # Add memory metadata to chunks
                for mc in memory_chunks:
                    mc.metadata["memory_id"] = memory.id
                    mc.metadata["memory_type"] = memory.memory_type.value
                    mc.metadata["importance"] = memory.importance
                    mc.metadata["timestamp"] = memory.created_at
                    chunks.append(mc)
            
            # Check if adding this memory would exceed limit
            elif current_tokens + memory_tokens > max_chunk_tokens:
                # Create chunk from current content
                if current_content:
                    chunk = self._create_chunk_from_memories(
                        current_content,
                        len(chunks),
                        current_tokens
                    )
                    chunks.append(chunk)
                
                # Start new chunk with this memory
                current_content = [memory]
                current_tokens = memory_tokens
            else:
                # Add to current chunk
                current_content.append(memory)
                current_tokens += memory_tokens
        
        # Create final chunk if any content remains
        if current_content:
            chunk = self._create_chunk_from_memories(
                current_content,
                len(chunks),
                current_tokens
            )
            chunks.append(chunk)
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        # Cache chunks for navigation
        for chunk in chunks:
            self._chunk_cache[chunk.id] = chunk
        
        return chunks

    def _create_chunk_from_memories(
        self,
        memories: list[Memory],
        chunk_index: int,
        token_count: int
    ) -> Chunk:
        """Create a Chunk from a list of Memory objects."""
        # Combine memory contents
        content_parts = []
        for memory in memories:
            if memory.memory_type == MemoryType.CONVERSATION:
                content_parts.append(memory.content)
            elif memory.memory_type == MemoryType.CODE:
                content_parts.append(f"```\n{memory.content}\n```")
            else:
                content_parts.append(memory.content)
        
        content = "\n\n".join(content_parts)
        
        # Create chunk
        chunk = Chunk(
            id=f"chunk_{chunk_index}",
            content=content,
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated later
            token_count=token_count,
            metadata={
                "memory_ids": [m.id for m in memories],
                "memory_types": [m.memory_type.value for m in memories],
                "importance": max(m.importance for m in memories),
                "timestamp": max(m.created_at for m in memories)
            }
        )
        
        return chunk

    def _detect_content_type(self, memory: Memory) -> str:
        """Detect content type from memory."""
        if memory.memory_type == MemoryType.CODE:
            return "code"
        elif memory.memory_type == MemoryType.DOCUMENT:
            # Check if it's markdown
            if any(marker in memory.content for marker in ['# ', '## ', '```']):
                return "markdown"
        return "text"

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
        return self.chunk_formatter.format_chunk(
            chunk=chunk,
            format_type=format_type,
            memory=memory,
            include_navigation=include_navigation
        )

    def select_chunks(
        self,
        chunks: list[Chunk],
        query: str | None = None,
        memories: list[Memory] | None = None,
        max_chunks: int | None = None,
        max_tokens: int | None = None,
        min_score: float = 0.0
    ) -> list[ScoredChunk]:
        """
        Select and rank chunks based on priority scoring.
        
        Args:
            chunks: List of chunks to select from
            query: Optional query text for relevance scoring
            memories: Optional list of associated memories
            max_chunks: Maximum number of chunks to return
            max_tokens: Maximum total tokens (overrides max_chunks if specified)
            min_score: Minimum score threshold for selection
            
        Returns:
            List of ScoredChunk objects, sorted by priority
        """
        # Get scored chunks
        scored_chunks = self.chunk_selector.select_chunks(
            chunks=chunks,
            query=query,
            memories=memories,
            max_chunks=None,  # Don't limit yet
            min_score=min_score
        )
        
        # Apply token budget if specified
        if max_tokens is not None:
            scored_chunks = self._apply_token_budget(scored_chunks, max_tokens)
        
        # Apply max_chunks limit
        if max_chunks is not None:
            scored_chunks = scored_chunks[:max_chunks]
        
        return scored_chunks

    def _apply_token_budget(
        self,
        scored_chunks: list[ScoredChunk],
        max_tokens: int
    ) -> list[ScoredChunk]:
        """
        Select chunks that fit within token budget.
        
        Args:
            scored_chunks: Scored chunks sorted by priority
            max_tokens: Maximum total tokens
            
        Returns:
            Filtered list of chunks that fit within budget
        """
        selected = []
        total_tokens = 0
        
        for scored_chunk in scored_chunks:
            chunk_tokens = scored_chunk.chunk.token_count
            
            if total_tokens + chunk_tokens <= max_tokens:
                selected.append(scored_chunk)
                total_tokens += chunk_tokens
            else:
                # Check if we can fit a truncated version
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # Only if meaningful space remains
                    # Truncate the chunk content
                    truncated_content = self.token_counter.truncate_to_token_limit(
                        scored_chunk.chunk.content,
                        remaining_tokens
                    )
                    
                    # Create truncated chunk
                    truncated_chunk = Chunk(
                        id=scored_chunk.chunk.id + "_truncated",
                        content=truncated_content,
                        chunk_index=scored_chunk.chunk.chunk_index,
                        total_chunks=scored_chunk.chunk.total_chunks,
                        token_count=remaining_tokens,
                        relevance_score=scored_chunk.chunk.relevance_score,
                        metadata={**scored_chunk.chunk.metadata, "truncated": True}
                    )
                    
                    # Create new scored chunk
                    truncated_scored = ScoredChunk(
                        chunk=truncated_chunk,
                        relevance_score=scored_chunk.relevance_score,
                        importance_score=scored_chunk.importance_score,
                        recency_score=scored_chunk.recency_score,
                        final_score=scored_chunk.final_score,
                        metadata={**scored_chunk.metadata, "truncated": True}
                    )
                    
                    selected.append(truncated_scored)
                
                break
        
        return selected

    def get_next_chunk(
        self,
        chunk_id: str,
        direction: Literal["forward", "backward"] = "forward"
    ) -> Chunk | None:
        """
        Navigate through chunks sequentially.
        
        Args:
            chunk_id: ID of current chunk
            direction: Navigation direction ('forward' or 'backward')
            
        Returns:
            Next chunk or None if at boundary
        """
        # Get current chunk from cache
        current_chunk = self._chunk_cache.get(chunk_id)
        
        if current_chunk is None:
            return None
        
        # Calculate next index
        if direction == "forward":
            next_index = current_chunk.chunk_index + 1
            if next_index >= current_chunk.total_chunks:
                return None
        else:  # backward
            next_index = current_chunk.chunk_index - 1
            if next_index < 0:
                return None
        
        # Find chunk with next index
        next_chunk_id = f"chunk_{next_index}"
        return self._chunk_cache.get(next_chunk_id)

    def stream_chunks(
        self,
        content: str | list[Memory],
        content_type: str = "text",
        query: str | None = None,
        format_type: FormatType | str = FormatType.MARKDOWN,
        max_chunks: int | None = None
    ) -> Iterator[FormattedChunk]:
        """
        Stream chunks incrementally for large contexts.
        
        Args:
            content: Content to chunk and stream
            content_type: Type of content
            query: Optional query for relevance ranking
            format_type: Output format type
            max_chunks: Maximum number of chunks to stream
            
        Yields:
            FormattedChunk objects one at a time
        """
        # Chunk the content
        chunks = self.chunk_content(content, content_type, query)
        
        # Select and rank chunks if query provided
        if query:
            memories = content if isinstance(content, list) else None
            scored_chunks = self.select_chunks(
                chunks=chunks,
                query=query,
                memories=memories,
                max_chunks=max_chunks
            )
            chunks_to_stream = [sc.chunk for sc in scored_chunks]
        else:
            chunks_to_stream = chunks[:max_chunks] if max_chunks else chunks
        
        # Stream formatted chunks
        for chunk in chunks_to_stream:
            formatted = self.format_chunk(chunk, format_type)
            yield formatted

    def process_and_format(
        self,
        content: str | list[Memory],
        query: str | None = None,
        content_type: str = "text",
        format_type: FormatType | str = FormatType.MARKDOWN,
        max_chunks: int | None = None,
        max_tokens: int | None = None,
        return_single_string: bool = False
    ) -> list[FormattedChunk] | str:
        """
        Complete pipeline: chunk, select, and format content.
        
        Args:
            content: Content to process
            query: Optional query for relevance ranking
            content_type: Type of content
            format_type: Output format type
            max_chunks: Maximum number of chunks
            max_tokens: Maximum total tokens
            return_single_string: If True, return concatenated string instead of list
            
        Returns:
            List of FormattedChunk objects or single concatenated string
        """
        # Chunk the content
        chunks = self.chunk_content(content, content_type, query)
        
        # Select and rank chunks
        memories = content if isinstance(content, list) else None
        scored_chunks = self.select_chunks(
            chunks=chunks,
            query=query,
            memories=memories,
            max_chunks=max_chunks,
            max_tokens=max_tokens
        )
        
        # Format chunks
        formatted_chunks = []
        for scored_chunk in scored_chunks:
            formatted = self.format_chunk(scored_chunk.chunk, format_type)
            formatted_chunks.append(formatted)
        
        # Return as single string if requested
        if return_single_string:
            separator = "\n\n---\n\n"
            return separator.join(fc.content for fc in formatted_chunks)
        
        return formatted_chunks

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the chunk manager.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "model_name": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "overlap_tokens": self.config.overlap_tokens,
            "cached_chunks": len(self._chunk_cache),
            "weights": {
                "relevance": self.config.relevance_weight,
                "importance": self.config.importance_weight,
                "recency": self.config.recency_weight
            }
        }


def create_chunk_manager(
    model_name: str = "gpt-4",
    max_tokens: int | None = None,
    embedding_fn: Any = None,
    **kwargs
) -> ChunkManager:
    """
    Factory function to create a ChunkManager.
    
    Args:
        model_name: Name of the target model
        max_tokens: Maximum tokens (auto-detected if None)
        embedding_fn: Optional embedding function
        **kwargs: Additional configuration options
        
    Returns:
        ChunkManager instance
    """
    config = ChunkManagerConfig(
        model_name=model_name,
        max_tokens=max_tokens,
        **kwargs
    )
    
    return ChunkManager(config=config, embedding_fn=embedding_fn)
