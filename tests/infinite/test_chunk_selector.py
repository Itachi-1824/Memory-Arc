"""Unit tests for priority-based chunk selection."""

import time
import pytest
from core.infinite import (
    ChunkSelector,
    ScoredChunk,
    Chunk,
    Memory,
    MemoryType,
    BoundaryType,
)


class TestChunkSelector:
    """Test suite for ChunkSelector class."""

    def test_initialization_default(self):
        """Test selector initialization with default parameters."""
        selector = ChunkSelector()
        
        # Weights should be normalized to sum to 1.0
        total = selector.relevance_weight + selector.importance_weight + selector.recency_weight
        assert abs(total - 1.0) < 0.001
        
        # Default decay should be 1 week
        assert selector.recency_decay_hours == 168.0

    def test_initialization_custom_weights(self):
        """Test selector initialization with custom weights."""
        selector = ChunkSelector(
            relevance_weight=0.6,
            importance_weight=0.3,
            recency_weight=0.1
        )
        
        # Weights should be normalized
        total = selector.relevance_weight + selector.importance_weight + selector.recency_weight
        assert abs(total - 1.0) < 0.001
        
        # Relative proportions should be maintained
        assert selector.relevance_weight > selector.importance_weight
        assert selector.importance_weight > selector.recency_weight

    def test_initialization_zero_weights(self):
        """Test selector handles zero weights gracefully."""
        selector = ChunkSelector(
            relevance_weight=0.0,
            importance_weight=0.0,
            recency_weight=0.0
        )
        
        # Should not crash, weights should be normalized
        total = selector.relevance_weight + selector.importance_weight + selector.recency_weight
        assert total >= 0.0

    def test_compute_relevance_score_with_existing_score(self):
        """Test relevance scoring uses existing chunk score."""
        selector = ChunkSelector()
        chunk = Chunk(
            id="chunk_1",
            content="test content",
            chunk_index=0,
            total_chunks=1,
            token_count=10,
            relevance_score=0.8
        )
        
        score = selector.compute_relevance_score(chunk)
        
        assert score == 0.8

    def test_compute_relevance_score_no_query(self):
        """Test relevance scoring without query returns neutral score."""
        selector = ChunkSelector()
        chunk = Chunk(
            id="chunk_1",
            content="test content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        score = selector.compute_relevance_score(chunk)
        
        assert score == 0.5

    def test_compute_relevance_score_lexical_similarity(self):
        """Test relevance scoring with lexical similarity."""
        selector = ChunkSelector()
        chunk = Chunk(
            id="chunk_1",
            content="machine learning algorithms and models",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        # Query with overlapping words
        score = selector.compute_relevance_score(chunk, query="machine learning systems")
        
        # Should have some relevance due to word overlap
        assert 0.0 < score < 1.0

    def test_compute_relevance_score_with_embedding(self):
        """Test relevance scoring with embedding function."""
        def mock_embedding(text):
            # Simple mock: return vector based on text length
            return [float(len(text)), 1.0, 0.5]
        
        selector = ChunkSelector(embedding_fn=mock_embedding)
        chunk = Chunk(
            id="chunk_1",
            content="test content here",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        score = selector.compute_relevance_score(chunk, query="test query")
        
        # Should compute embedding-based similarity
        assert 0.0 <= score <= 1.0

    def test_compute_importance_score_from_chunk_metadata(self):
        """Test importance scoring from chunk metadata."""
        selector = ChunkSelector()
        chunk = Chunk(
            id="chunk_1",
            content="important content",
            chunk_index=0,
            total_chunks=1,
            token_count=10,
            metadata={"importance": 8}
        )
        
        score = selector.compute_importance_score(chunk)
        
        # Should normalize 8/10 to 0.8
        assert abs(score - 0.8) < 0.01

    def test_compute_importance_score_from_memory(self):
        """Test importance scoring from associated memory."""
        selector = ChunkSelector()
        chunk = Chunk(
            id="chunk_1",
            content="content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        memory = Memory(
            id="mem_1",
            context_id="ctx_1",
            content="content",
            memory_type=MemoryType.FACT,
            created_at=time.time(),
            importance=7
        )
        
        score = selector.compute_importance_score(chunk, memory)
        
        # Should normalize 7/10 to 0.7
        assert abs(score - 0.7) < 0.01

    def test_compute_importance_score_default(self):
        """Test importance scoring with no metadata or memory."""
        selector = ChunkSelector()
        chunk = Chunk(
            id="chunk_1",
            content="content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        score = selector.compute_importance_score(chunk)
        
        # Should return default medium importance
        assert score == 0.5

    def test_compute_recency_score_recent(self):
        """Test recency scoring for recent content."""
        selector = ChunkSelector(recency_decay_hours=168.0)
        current = time.time()
        
        chunk = Chunk(
            id="chunk_1",
            content="content",
            chunk_index=0,
            total_chunks=1,
            token_count=10,
            metadata={"timestamp": current - 3600}  # 1 hour ago
        )
        
        score = selector.compute_recency_score(chunk, current_time=current)
        
        # Recent content should have high score
        assert score > 0.9

    def test_compute_recency_score_old(self):
        """Test recency scoring for old content."""
        selector = ChunkSelector(recency_decay_hours=168.0)
        current = time.time()
        
        chunk = Chunk(
            id="chunk_1",
            content="content",
            chunk_index=0,
            total_chunks=1,
            token_count=10,
            metadata={"timestamp": current - (365 * 24 * 3600)}  # 1 year ago
        )
        
        score = selector.compute_recency_score(chunk, current_time=current)
        
        # Old content should have low score
        assert score < 0.1

    def test_compute_recency_score_decay_point(self):
        """Test recency scoring at decay point."""
        selector = ChunkSelector(recency_decay_hours=168.0)
        current = time.time()
        
        chunk = Chunk(
            id="chunk_1",
            content="content",
            chunk_index=0,
            total_chunks=1,
            token_count=10,
            metadata={"timestamp": current - (168 * 3600)}  # 1 week ago
        )
        
        score = selector.compute_recency_score(chunk, current_time=current)
        
        # At decay point, score should be around 0.5
        assert 0.4 < score < 0.6

    def test_compute_recency_score_from_memory(self):
        """Test recency scoring from memory timestamp."""
        selector = ChunkSelector()
        current = time.time()
        
        chunk = Chunk(
            id="chunk_1",
            content="content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        memory = Memory(
            id="mem_1",
            context_id="ctx_1",
            content="content",
            memory_type=MemoryType.CONVERSATION,
            created_at=current - 7200,  # 2 hours ago
            importance=5
        )
        
        score = selector.compute_recency_score(chunk, memory, current_time=current)
        
        # Recent memory should have high score
        assert score > 0.9

    def test_compute_recency_score_no_timestamp(self):
        """Test recency scoring without timestamp."""
        selector = ChunkSelector()
        chunk = Chunk(
            id="chunk_1",
            content="content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        score = selector.compute_recency_score(chunk)
        
        # Should return neutral score
        assert score == 0.5

    def test_compute_final_score(self):
        """Test final score computation with all components."""
        selector = ChunkSelector(
            relevance_weight=0.5,
            importance_weight=0.3,
            recency_weight=0.2
        )
        current = time.time()
        
        chunk = Chunk(
            id="chunk_1",
            content="machine learning content",
            chunk_index=0,
            total_chunks=1,
            token_count=10,
            metadata={
                "importance": 8,
                "timestamp": current - 3600
            }
        )
        
        relevance, importance, recency, final = selector.compute_final_score(
            chunk,
            query="machine learning",
            current_time=current
        )
        
        # All scores should be in valid range
        assert 0.0 <= relevance <= 1.0
        assert 0.0 <= importance <= 1.0
        assert 0.0 <= recency <= 1.0
        assert 0.0 <= final <= 1.0
        
        # Final score should be weighted combination
        expected = (
            relevance * selector.relevance_weight +
            importance * selector.importance_weight +
            recency * selector.recency_weight
        )
        assert abs(final - expected) < 0.001

    def test_select_chunks_basic(self):
        """Test basic chunk selection."""
        selector = ChunkSelector()
        current = time.time()
        
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                content=f"content {i}",
                chunk_index=i,
                total_chunks=3,
                token_count=10,
                metadata={"importance": 5 + i, "timestamp": current - (i * 3600)}
            )
            for i in range(3)
        ]
        
        scored = selector.select_chunks(chunks, current_time=current)
        
        # Should return all chunks
        assert len(scored) == 3
        
        # Should be sorted by score (descending)
        for i in range(len(scored) - 1):
            assert scored[i].final_score >= scored[i + 1].final_score

    def test_select_chunks_with_query(self):
        """Test chunk selection with query relevance."""
        selector = ChunkSelector(relevance_weight=1.0, importance_weight=0.0, recency_weight=0.0)
        
        chunks = [
            Chunk(
                id="chunk_1",
                content="machine learning algorithms",
                chunk_index=0,
                total_chunks=3,
                token_count=10
            ),
            Chunk(
                id="chunk_2",
                content="cooking recipes",
                chunk_index=1,
                total_chunks=3,
                token_count=10
            ),
            Chunk(
                id="chunk_3",
                content="machine learning models",
                chunk_index=2,
                total_chunks=3,
                token_count=10
            ),
        ]
        
        scored = selector.select_chunks(chunks, query="machine learning")
        
        # Relevant chunks should score higher
        assert scored[0].chunk.content in ["machine learning algorithms", "machine learning models"]

    def test_select_chunks_max_limit(self):
        """Test chunk selection with max limit."""
        selector = ChunkSelector()
        
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                content=f"content {i}",
                chunk_index=i,
                total_chunks=10,
                token_count=10
            )
            for i in range(10)
        ]
        
        scored = selector.select_chunks(chunks, max_chunks=3)
        
        # Should return only top 3
        assert len(scored) == 3

    def test_select_chunks_min_score_threshold(self):
        """Test chunk selection with minimum score threshold."""
        selector = ChunkSelector()
        current = time.time()
        
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                content=f"content {i}",
                chunk_index=i,
                total_chunks=5,
                token_count=10,
                metadata={"importance": i, "timestamp": current}
            )
            for i in range(1, 6)
        ]
        
        scored = selector.select_chunks(chunks, min_score=0.5, current_time=current)
        
        # Only chunks with score >= 0.5 should be returned
        for sc in scored:
            assert sc.final_score >= 0.5

    def test_select_chunks_with_memories(self):
        """Test chunk selection with associated memories."""
        selector = ChunkSelector()
        current = time.time()
        
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                content=f"content {i}",
                chunk_index=i,
                total_chunks=3,
                token_count=10
            )
            for i in range(3)
        ]
        
        memories = [
            Memory(
                id=f"mem_{i}",
                context_id="ctx_1",
                content=f"content {i}",
                memory_type=MemoryType.FACT,
                created_at=current - (i * 3600),
                importance=5 + i
            )
            for i in range(3)
        ]
        
        scored = selector.select_chunks(chunks, memories=memories, current_time=current)
        
        # Should use memory metadata for scoring
        assert len(scored) == 3
        assert all(sc.importance_score > 0 for sc in scored)

    def test_scored_chunk_structure(self):
        """Test ScoredChunk contains all required fields."""
        selector = ChunkSelector()
        chunk = Chunk(
            id="chunk_1",
            content="test",
            chunk_index=0,
            total_chunks=1,
            token_count=10,
            metadata={"importance": 7}
        )
        
        scored = selector.select_chunks([chunk])
        
        assert len(scored) == 1
        sc = scored[0]
        
        # Check all fields are present
        assert sc.chunk == chunk
        assert 0.0 <= sc.relevance_score <= 1.0
        assert 0.0 <= sc.importance_score <= 1.0
        assert 0.0 <= sc.recency_score <= 1.0
        assert 0.0 <= sc.final_score <= 1.0
        assert "weights" in sc.metadata


class TestRelevanceScoring:
    """Test suite for relevance scoring algorithm."""

    def test_relevance_high_overlap(self):
        """Test relevance with high word overlap."""
        selector = ChunkSelector()
        chunk = Chunk(
            id="chunk_1",
            content="python programming language tutorial",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        score = selector.compute_relevance_score(chunk, query="python programming tutorial")
        
        # High overlap should give high score
        assert score > 0.5

    def test_relevance_no_overlap(self):
        """Test relevance with no word overlap."""
        selector = ChunkSelector()
        chunk = Chunk(
            id="chunk_1",
            content="cooking recipes food",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        score = selector.compute_relevance_score(chunk, query="quantum physics")
        
        # No overlap should give low score
        assert score < 0.3

    def test_relevance_case_insensitive(self):
        """Test relevance is case insensitive."""
        selector = ChunkSelector()
        chunk = Chunk(
            id="chunk_1",
            content="Python Programming",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        score1 = selector.compute_relevance_score(chunk, query="python programming")
        score2 = selector.compute_relevance_score(chunk, query="PYTHON PROGRAMMING")
        
        # Should be the same regardless of case
        assert abs(score1 - score2) < 0.001


class TestImportanceRanking:
    """Test suite for importance-based ranking."""

    def test_importance_ranking_order(self):
        """Test chunks are ranked by importance."""
        selector = ChunkSelector(
            relevance_weight=0.0,
            importance_weight=1.0,
            recency_weight=0.0
        )
        
        chunks = [
            Chunk(
                id="chunk_low",
                content="low importance",
                chunk_index=0,
                total_chunks=3,
                token_count=10,
                metadata={"importance": 3}
            ),
            Chunk(
                id="chunk_high",
                content="high importance",
                chunk_index=1,
                total_chunks=3,
                token_count=10,
                metadata={"importance": 9}
            ),
            Chunk(
                id="chunk_med",
                content="medium importance",
                chunk_index=2,
                total_chunks=3,
                token_count=10,
                metadata={"importance": 6}
            ),
        ]
        
        scored = selector.select_chunks(chunks)
        
        # Should be ordered by importance
        assert scored[0].chunk.id == "chunk_high"
        assert scored[1].chunk.id == "chunk_med"
        assert scored[2].chunk.id == "chunk_low"

    def test_importance_bounds(self):
        """Test importance scores are properly bounded."""
        selector = ChunkSelector()
        
        # Test extreme values
        chunk_low = Chunk(
            id="chunk_1",
            content="test",
            chunk_index=0,
            total_chunks=1,
            token_count=10,
            metadata={"importance": 0}
        )
        chunk_high = Chunk(
            id="chunk_2",
            content="test",
            chunk_index=0,
            total_chunks=1,
            token_count=10,
            metadata={"importance": 15}  # Above max
        )
        
        score_low = selector.compute_importance_score(chunk_low)
        score_high = selector.compute_importance_score(chunk_high)
        
        # Should be bounded to [0, 1]
        assert 0.0 <= score_low <= 1.0
        assert 0.0 <= score_high <= 1.0


class TestRecencyBoosting:
    """Test suite for recency boosting."""

    def test_recency_boost_recent_over_old(self):
        """Test recent content scores higher than old content."""
        selector = ChunkSelector(
            relevance_weight=0.0,
            importance_weight=0.0,
            recency_weight=1.0
        )
        current = time.time()
        
        chunks = [
            Chunk(
                id="chunk_old",
                content="old content",
                chunk_index=0,
                total_chunks=2,
                token_count=10,
                metadata={"timestamp": current - (365 * 24 * 3600)}  # 1 year ago
            ),
            Chunk(
                id="chunk_recent",
                content="recent content",
                chunk_index=1,
                total_chunks=2,
                token_count=10,
                metadata={"timestamp": current - 3600}  # 1 hour ago
            ),
        ]
        
        scored = selector.select_chunks(chunks, current_time=current)
        
        # Recent should score higher
        assert scored[0].chunk.id == "chunk_recent"
        assert scored[1].chunk.id == "chunk_old"

    def test_recency_exponential_decay(self):
        """Test recency follows exponential decay."""
        selector = ChunkSelector(recency_decay_hours=168.0)
        current = time.time()
        
        # Test at different time points
        times = [0, 84, 168, 336]  # 0h, 3.5d, 7d, 14d
        scores = []
        
        for hours_ago in times:
            chunk = Chunk(
                id=f"chunk_{hours_ago}",
                content="content",
                chunk_index=0,
                total_chunks=1,
                token_count=10,
                metadata={"timestamp": current - (hours_ago * 3600)}
            )
            score = selector.compute_recency_score(chunk, current_time=current)
            scores.append(score)
        
        # Scores should decrease over time
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]


class TestChunkSelectionStrategy:
    """Test suite for overall chunk selection strategy."""

    def test_balanced_selection(self):
        """Test selection balances all three factors."""
        selector = ChunkSelector(
            relevance_weight=0.33,
            importance_weight=0.33,
            recency_weight=0.34
        )
        current = time.time()
        
        chunks = [
            # High relevance, low importance, old
            Chunk(
                id="chunk_relevant",
                content="machine learning algorithms",
                chunk_index=0,
                total_chunks=3,
                token_count=10,
                metadata={"importance": 2, "timestamp": current - (365 * 24 * 3600)}
            ),
            # Low relevance, high importance, old
            Chunk(
                id="chunk_important",
                content="cooking recipes",
                chunk_index=1,
                total_chunks=3,
                token_count=10,
                metadata={"importance": 9, "timestamp": current - (365 * 24 * 3600)}
            ),
            # Low relevance, low importance, recent
            Chunk(
                id="chunk_recent",
                content="random content",
                chunk_index=2,
                total_chunks=3,
                token_count=10,
                metadata={"importance": 2, "timestamp": current - 3600}
            ),
        ]
        
        scored = selector.select_chunks(chunks, query="machine learning", current_time=current)
        
        # All chunks should be scored
        assert len(scored) == 3
        
        # Each should have different strengths
        for sc in scored:
            assert 0.0 < sc.final_score < 1.0

    def test_empty_chunks_list(self):
        """Test selection with empty chunks list."""
        selector = ChunkSelector()
        
        scored = selector.select_chunks([])
        
        assert len(scored) == 0

    def test_single_chunk(self):
        """Test selection with single chunk."""
        selector = ChunkSelector()
        chunk = Chunk(
            id="chunk_1",
            content="test",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        scored = selector.select_chunks([chunk])
        
        assert len(scored) == 1
        assert scored[0].chunk == chunk

