"""Priority-based chunk selection for intelligent context retrieval."""

import time
from typing import Any
from dataclasses import dataclass

from .models import Chunk, Memory


@dataclass
class ScoredChunk:
    """A chunk with computed relevance score."""
    chunk: Chunk
    relevance_score: float
    importance_score: float
    recency_score: float
    final_score: float
    metadata: dict[str, Any]


class ChunkSelector:
    """
    Priority-based chunk selection system.
    
    Features:
    - Relevance scoring based on semantic similarity
    - Importance-based ranking from memory metadata
    - Recency boosting for time-sensitive information
    - Configurable selection strategy
    """

    def __init__(
        self,
        relevance_weight: float = 0.5,
        importance_weight: float = 0.3,
        recency_weight: float = 0.2,
        recency_decay_hours: float = 168.0,  # 1 week
        embedding_fn: Any = None
    ):
        """
        Initialize chunk selector.
        
        Args:
            relevance_weight: Weight for relevance score (0-1)
            importance_weight: Weight for importance score (0-1)
            recency_weight: Weight for recency score (0-1)
            recency_decay_hours: Hours for recency to decay to 0.5
            embedding_fn: Optional function to compute embeddings
        """
        # Normalize weights
        total = relevance_weight + importance_weight + recency_weight
        if total == 0:
            total = 1.0
        
        self.relevance_weight = relevance_weight / total
        self.importance_weight = importance_weight / total
        self.recency_weight = recency_weight / total
        self.recency_decay_hours = recency_decay_hours
        self.embedding_fn = embedding_fn

    def compute_relevance_score(
        self,
        chunk: Chunk,
        query: str | None = None,
        query_embedding: list[float] | None = None
    ) -> float:
        """
        Compute relevance score for a chunk.
        
        Args:
            chunk: The chunk to score
            query: Optional query text
            query_embedding: Optional pre-computed query embedding
            
        Returns:
            Relevance score between 0 and 1
        """
        # If chunk already has a relevance score, use it
        if chunk.relevance_score > 0:
            return chunk.relevance_score
        
        # If no query provided, return neutral score
        if not query and not query_embedding:
            return 0.5
        
        # Use embedding-based similarity if available
        if self.embedding_fn and query:
            try:
                chunk_embedding = self._get_chunk_embedding(chunk)
                if not query_embedding:
                    query_embedding = self.embedding_fn(query)
                
                similarity = self._cosine_similarity(chunk_embedding, query_embedding)
                # Normalize from [-1, 1] to [0, 1]
                return (similarity + 1.0) / 2.0
            except Exception:
                pass
        
        # Fallback: lexical similarity
        if query:
            return self._lexical_similarity(chunk.content, query)
        
        return 0.5

    def compute_importance_score(self, chunk: Chunk, memory: Memory | None = None) -> float:
        """
        Compute importance score for a chunk.
        
        Args:
            chunk: The chunk to score
            memory: Optional associated memory with importance rating
            
        Returns:
            Importance score between 0 and 1
        """
        # Check if importance is in chunk metadata
        if "importance" in chunk.metadata:
            importance = chunk.metadata["importance"]
            # Normalize from 1-10 scale to 0-1
            return min(max(importance / 10.0, 0.0), 1.0)
        
        # Use memory importance if available
        if memory:
            # Memory importance is typically 1-10
            return min(max(memory.importance / 10.0, 0.0), 1.0)
        
        # Default: medium importance
        return 0.5

    def compute_recency_score(
        self,
        chunk: Chunk,
        memory: Memory | None = None,
        current_time: float | None = None
    ) -> float:
        """
        Compute recency score for a chunk with exponential decay.
        
        Args:
            chunk: The chunk to score
            memory: Optional associated memory with timestamp
            current_time: Optional current timestamp (uses time.time() if None)
            
        Returns:
            Recency score between 0 and 1
        """
        if current_time is None:
            current_time = time.time()
        
        # Get timestamp from chunk metadata or memory
        timestamp = None
        if "timestamp" in chunk.metadata:
            timestamp = chunk.metadata["timestamp"]
        elif "created_at" in chunk.metadata:
            timestamp = chunk.metadata["created_at"]
        elif memory:
            timestamp = memory.created_at
        
        if timestamp is None:
            # No timestamp available, return neutral score
            return 0.5
        
        # Calculate age in hours
        age_seconds = current_time - timestamp
        age_hours = age_seconds / 3600.0
        
        # Exponential decay: score = exp(-age / decay_constant)
        # At decay_hours, score should be ~0.5
        decay_constant = self.recency_decay_hours / 0.693  # ln(2) ≈ 0.693
        
        import math
        recency_score = math.exp(-age_hours / decay_constant)
        
        return min(max(recency_score, 0.0), 1.0)

    def compute_final_score(
        self,
        chunk: Chunk,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        memory: Memory | None = None,
        current_time: float | None = None
    ) -> tuple[float, float, float, float]:
        """
        Compute final weighted score for a chunk.
        
        Args:
            chunk: The chunk to score
            query: Optional query text
            query_embedding: Optional pre-computed query embedding
            memory: Optional associated memory
            current_time: Optional current timestamp
            
        Returns:
            Tuple of (relevance, importance, recency, final_score)
        """
        relevance = self.compute_relevance_score(chunk, query, query_embedding)
        importance = self.compute_importance_score(chunk, memory)
        recency = self.compute_recency_score(chunk, memory, current_time)
        
        final_score = (
            relevance * self.relevance_weight +
            importance * self.importance_weight +
            recency * self.recency_weight
        )
        
        return relevance, importance, recency, final_score

    def select_chunks(
        self,
        chunks: list[Chunk],
        query: str | None = None,
        query_embedding: list[float] | None = None,
        memories: list[Memory] | None = None,
        max_chunks: int | None = None,
        min_score: float = 0.0,
        current_time: float | None = None
    ) -> list[ScoredChunk]:
        """
        Select and rank chunks based on priority scoring.
        
        Args:
            chunks: List of chunks to select from
            query: Optional query text for relevance scoring
            query_embedding: Optional pre-computed query embedding
            memories: Optional list of associated memories (aligned with chunks)
            max_chunks: Maximum number of chunks to return
            min_score: Minimum score threshold for selection
            current_time: Optional current timestamp
            
        Returns:
            List of ScoredChunk objects, sorted by final score (descending)
        """
        scored_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Get associated memory if available
            memory = memories[i] if memories and i < len(memories) else None
            
            # Compute scores
            relevance, importance, recency, final_score = self.compute_final_score(
                chunk, query, query_embedding, memory, current_time
            )
            
            # Apply minimum score threshold
            if final_score < min_score:
                continue
            
            scored_chunk = ScoredChunk(
                chunk=chunk,
                relevance_score=relevance,
                importance_score=importance,
                recency_score=recency,
                final_score=final_score,
                metadata={
                    "weights": {
                        "relevance": self.relevance_weight,
                        "importance": self.importance_weight,
                        "recency": self.recency_weight
                    }
                }
            )
            scored_chunks.append(scored_chunk)
        
        # Sort by final score (descending)
        scored_chunks.sort(key=lambda sc: sc.final_score, reverse=True)
        
        # Apply max_chunks limit
        if max_chunks is not None:
            scored_chunks = scored_chunks[:max_chunks]
        
        return scored_chunks

    def _get_chunk_embedding(self, chunk: Chunk) -> list[float]:
        """Get or compute embedding for a chunk."""
        # Check if embedding is cached in metadata
        if "embedding" in chunk.metadata:
            return chunk.metadata["embedding"]
        
        # Compute new embedding
        if self.embedding_fn:
            embedding = self.embedding_fn(chunk.content)
            # Cache it
            chunk.metadata["embedding"] = embedding
            return embedding
        
        raise ValueError("No embedding function available")

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

    def _lexical_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
