"""Tests for result composition system."""

import pytest
import time
import uuid

from core.infinite.result_composer import (
    ResultComposer,
    MemoryGroup,
    ContextBreadcrumb
)
from core.infinite.retrieval_strategies import ScoredMemory
from core.infinite.models import (
    Memory,
    MemoryType,
    QueryAnalysis,
    QueryIntent,
    RetrievalResult
)


@pytest.fixture
def sample_scored_memories():
    """Create sample scored memories for testing."""
    current_time = time.time()
    
    memories = [
        # High-scoring code memory
        ScoredMemory(
            memory=Memory(
                id=str(uuid.uuid4()),
                context_id="test",
                content="def login(user, password): return authenticate(user, password)",
                memory_type=MemoryType.CODE,
                created_at=current_time - 3600,
                importance=9
            ),
            score=0.95,
            strategy="semantic",
            metadata={"similarity": 0.95}
        ),
        # Medium-scoring preference memory
        ScoredMemory(
            memory=Memory(
                id=str(uuid.uuid4()),
                context_id="test",
                content="I prefer dark mode for coding",
                memory_type=MemoryType.PREFERENCE,
                created_at=current_time - 7200,
                importance=7
            ),
            score=0.75,
            strategy="semantic",
            metadata={"similarity": 0.75}
        ),
        # High-scoring conversation memory
        ScoredMemory(
            memory=Memory(
                id=str(uuid.uuid4()),
                context_id="test",
                content="We discussed authentication implementation yesterday",
                memory_type=MemoryType.CONVERSATION,
                created_at=current_time - 86400,
                importance=8
            ),
            score=0.85,
            strategy="temporal",
            metadata={"timestamp": current_time - 86400}
        ),
        # Low-scoring fact memory
        ScoredMemory(
            memory=Memory(
                id=str(uuid.uuid4()),
                context_id="test",
                content="Python was created by Guido van Rossum",
                memory_type=MemoryType.FACT,
                created_at=current_time - 172800,
                importance=5
            ),
            score=0.45,
            strategy="fulltext",
            metadata={"keyword_matches": 2}
        ),
        # Another code memory (fused from multiple strategies)
        ScoredMemory(
            memory=Memory(
                id=str(uuid.uuid4()),
                context_id="test",
                content="class AuthManager: def __init__(self): pass",
                memory_type=MemoryType.CODE,
                created_at=current_time - 1800,
                importance=8
            ),
            score=0.88,
            strategy="fused",
            metadata={
                "strategies": ["semantic", "structural"],
                "strategy_scores": {"semantic": 0.85, "structural": 0.90}
            }
        ),
    ]
    
    return memories


@pytest.fixture
def query_analysis_code():
    """Create a code-related query analysis."""
    return QueryAnalysis(
        intent=QueryIntent.CODE,
        entities=[("function", "login")],
        temporal_expressions=[],
        code_patterns=["login(", "authenticate"],
        keywords=["login", "authentication"],
        confidence=0.9
    )


@pytest.fixture
def query_analysis_mixed():
    """Create a mixed query analysis."""
    return QueryAnalysis(
        intent=QueryIntent.MIXED,
        entities=[],
        temporal_expressions=[("yesterday", time.time() - 86400)],
        code_patterns=["login"],
        keywords=["authentication", "yesterday"],
        confidence=0.7
    )


class TestResultComposer:
    """Test suite for ResultComposer."""
    
    def test_initialization(self):
        """Test composer initialization with various configurations."""
        # Default configuration
        composer = ResultComposer()
        assert composer.interleave_by_type is True
        assert composer.max_per_type is None
        assert composer.include_breadcrumbs is True
        assert composer.min_confidence == 0.0
        
        # Custom configuration
        composer = ResultComposer(
            interleave_by_type=False,
            max_per_type=5,
            include_breadcrumbs=False,
            min_confidence=0.5
        )
        assert composer.interleave_by_type is False
        assert composer.max_per_type == 5
        assert composer.include_breadcrumbs is False
        assert composer.min_confidence == 0.5
    
    def test_compose_basic(self, sample_scored_memories, query_analysis_code):
        """Test basic result composition."""
        composer = ResultComposer()
        
        result = composer.compose(
            scored_memories=sample_scored_memories,
            query_analysis=query_analysis_code,
            retrieval_time_ms=50.0,
            limit=10
        )
        
        # Check result structure
        assert isinstance(result, RetrievalResult)
        assert len(result.memories) <= 10
        assert result.total_found == len(sample_scored_memories)
        assert result.retrieval_time_ms == 50.0
        assert result.query_analysis == query_analysis_code
        
        # Check metadata
        assert "memory_groups" in result.metadata
        assert "breadcrumbs" in result.metadata
        assert "overall_confidence" in result.metadata
        assert "composition_strategy" in result.metadata
    
    def test_filter_by_confidence(self, sample_scored_memories, query_analysis_code):
        """Test filtering by minimum confidence threshold."""
        # Filter with threshold 0.5
        composer = ResultComposer(min_confidence=0.5)
        
        result = composer.compose(
            scored_memories=sample_scored_memories,
            query_analysis=query_analysis_code,
            retrieval_time_ms=50.0
        )
        
        # Should filter out the low-scoring fact memory (0.45)
        assert all(sm.score >= 0.5 for sm in sample_scored_memories 
                  if any(m.id == sm.memory.id for m in result.memories))
        assert result.metadata["filtered_count"] == 1
    
    def test_group_by_type(self, sample_scored_memories):
        """Test memory grouping by type."""
        composer = ResultComposer()
        
        groups = composer._group_by_type(sample_scored_memories)
        
        # Check groups exist
        assert MemoryType.CODE in groups
        assert MemoryType.PREFERENCE in groups
        assert MemoryType.CONVERSATION in groups
        assert MemoryType.FACT in groups
        
        # Check CODE group
        code_group = groups[MemoryType.CODE]
        assert code_group.memory_type == MemoryType.CODE
        assert code_group.total_count == 2
        assert len(code_group.memories) == 2
        
        # Check memories are sorted by score within group
        assert code_group.memories[0].score >= code_group.memories[1].score
        
        # Check average score calculation
        expected_avg = (0.95 + 0.88) / 2
        assert abs(code_group.avg_score - expected_avg) < 0.01
    
    def test_group_by_type_with_limit(self, sample_scored_memories):
        """Test memory grouping with max_per_type limit."""
        composer = ResultComposer(max_per_type=1)
        
        groups = composer._group_by_type(sample_scored_memories)
        
        # Each group should have at most 1 memory
        for group in groups.values():
            assert len(group.memories) <= 1
        
        # CODE group should have the highest-scoring code memory
        code_group = groups[MemoryType.CODE]
        assert code_group.memories[0].score == 0.95
    
    def test_interleave_results(self, sample_scored_memories):
        """Test result interleaving by type."""
        composer = ResultComposer(interleave_by_type=True)
        
        groups = composer._group_by_type(sample_scored_memories)
        interleaved = composer._interleave_results(groups, limit=None)
        
        # Should have all memories
        assert len(interleaved) == len(sample_scored_memories)
        
        # Check interleaving pattern - types should alternate
        types_sequence = [sm.memory.memory_type for sm in interleaved]
        
        # First few should be from different types (round-robin)
        # Groups are sorted by avg score, so CODE should be first
        assert types_sequence[0] == MemoryType.CODE
    
    def test_interleave_with_limit(self, sample_scored_memories):
        """Test result interleaving with limit."""
        composer = ResultComposer(interleave_by_type=True)
        
        groups = composer._group_by_type(sample_scored_memories)
        interleaved = composer._interleave_results(groups, limit=3)
        
        # Should respect limit
        assert len(interleaved) <= 3
    
    def test_no_interleaving(self, sample_scored_memories, query_analysis_code):
        """Test composition without interleaving (pure ranking)."""
        composer = ResultComposer(interleave_by_type=False)
        
        result = composer.compose(
            scored_memories=sample_scored_memories,
            query_analysis=query_analysis_code,
            retrieval_time_ms=50.0,
            limit=5
        )
        
        # Results should be sorted by score
        # Build a map of memory ID to score
        score_map = {sm.memory.id: sm.score for sm in sample_scored_memories}
        # Get scores in the order they appear in result
        result_scores = [score_map[m.id] for m in result.memories if m.id in score_map]
        assert result_scores == sorted(result_scores, reverse=True)
        
        assert result.metadata["composition_strategy"] == "ranked"
    
    def test_generate_breadcrumbs(self, sample_scored_memories, query_analysis_code):
        """Test context breadcrumb generation."""
        composer = ResultComposer(include_breadcrumbs=True)
        
        breadcrumbs = composer._generate_breadcrumbs(
            sample_scored_memories[:3],
            query_analysis_code
        )
        
        # Should have breadcrumb for each memory
        assert len(breadcrumbs) == 3
        
        # Check breadcrumb structure
        for bc in breadcrumbs:
            assert isinstance(bc, ContextBreadcrumb)
            assert bc.memory_id
            assert len(bc.retrieval_path) > 0
            assert len(bc.strategies_used) > 0
            assert 0 <= bc.confidence <= 1
            assert bc.reasoning
            
            # Path should start with query intent
            assert bc.retrieval_path[0].startswith("query_intent:")
            
            # Path should end with selection
            assert bc.retrieval_path[-1] == "selected"
    
    def test_breadcrumb_strategies(self, sample_scored_memories):
        """Test strategy extraction in breadcrumbs."""
        composer = ResultComposer()
        
        # Test single strategy
        single_strategy = sample_scored_memories[0]
        strategies = composer._extract_strategies(single_strategy)
        assert strategies == ["semantic"]
        
        # Test fused strategies
        fused_strategy = sample_scored_memories[4]
        strategies = composer._extract_strategies(fused_strategy)
        assert "semantic" in strategies
        assert "structural" in strategies
    
    def test_calculate_result_confidence(self, sample_scored_memories, query_analysis_code):
        """Test confidence calculation for individual results."""
        composer = ResultComposer()
        
        # High-scoring memory with high importance
        high_score = sample_scored_memories[0]
        confidence = composer._calculate_result_confidence(high_score, query_analysis_code)
        assert confidence > 0.7
        
        # Low-scoring memory
        low_score = sample_scored_memories[3]
        confidence = composer._calculate_result_confidence(low_score, query_analysis_code)
        assert confidence < 0.6
        
        # Fused memory (multiple strategies)
        fused = sample_scored_memories[4]
        confidence = composer._calculate_result_confidence(fused, query_analysis_code)
        # Should get consensus boost
        assert confidence > fused.score * 0.5
    
    def test_calculate_overall_confidence(self, sample_scored_memories, query_analysis_code):
        """Test overall confidence calculation."""
        composer = ResultComposer()
        
        # With all memories
        confidence = composer._calculate_overall_confidence(
            sample_scored_memories,
            query_analysis_code
        )
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably confident
        
        # With only high-scoring memories
        high_scoring = [sm for sm in sample_scored_memories if sm.score >= 0.8]
        confidence_high = composer._calculate_overall_confidence(
            high_scoring,
            query_analysis_code
        )
        assert confidence_high > confidence
        
        # With very few results (should be penalized)
        few_results = sample_scored_memories[:1]
        confidence_few = composer._calculate_overall_confidence(
            few_results,
            query_analysis_code
        )
        # Should be penalized for having few results
        assert confidence_few < 1.0
    
    def test_generate_reasoning(self, sample_scored_memories, query_analysis_code):
        """Test reasoning generation."""
        composer = ResultComposer()
        
        # Semantic strategy
        semantic_memory = sample_scored_memories[0]
        reasoning = composer._generate_reasoning(semantic_memory, query_analysis_code)
        assert "semantically similar" in reasoning
        assert "very high relevance" in reasoning or "high relevance" in reasoning
        assert "important" in reasoning  # importance=9
        
        # Temporal strategy
        temporal_memory = sample_scored_memories[2]
        reasoning = composer._generate_reasoning(temporal_memory, query_analysis_code)
        assert "temporally relevant" in reasoning
        
        # Fused strategy
        fused_memory = sample_scored_memories[4]
        reasoning = composer._generate_reasoning(fused_memory, query_analysis_code)
        assert "multiple strategies" in reasoning
        assert "semantic" in reasoning
        assert "structural" in reasoning
    
    def test_compose_with_limit(self, sample_scored_memories, query_analysis_code):
        """Test composition with result limit."""
        composer = ResultComposer()
        
        result = composer.compose(
            scored_memories=sample_scored_memories,
            query_analysis=query_analysis_code,
            retrieval_time_ms=50.0,
            limit=3
        )
        
        # Should respect limit
        assert len(result.memories) <= 3
        assert result.total_found == len(sample_scored_memories)
    
    def test_compose_empty_results(self, query_analysis_code):
        """Test composition with no results."""
        composer = ResultComposer()
        
        result = composer.compose(
            scored_memories=[],
            query_analysis=query_analysis_code,
            retrieval_time_ms=10.0
        )
        
        assert len(result.memories) == 0
        assert result.total_found == 0
        assert result.metadata["overall_confidence"] == 0.0
    
    def test_memory_type_matching(self, sample_scored_memories, query_analysis_code):
        """Test reasoning includes type matching for query intent."""
        composer = ResultComposer()
        
        # Code memory with code query
        code_memory = sample_scored_memories[0]
        reasoning = composer._generate_reasoning(code_memory, query_analysis_code)
        assert "matches code query intent" in reasoning
        
        # Preference memory with code query (shouldn't match)
        pref_memory = sample_scored_memories[1]
        reasoning = composer._generate_reasoning(pref_memory, query_analysis_code)
        assert "matches code query intent" not in reasoning
    
    def test_group_memories_by_type_utility(self):
        """Test utility method for grouping plain memories."""
        composer = ResultComposer()
        current_time = time.time()
        
        memories = [
            Memory(
                id="1",
                context_id="test",
                content="code",
                memory_type=MemoryType.CODE,
                created_at=current_time
            ),
            Memory(
                id="2",
                context_id="test",
                content="fact",
                memory_type=MemoryType.FACT,
                created_at=current_time
            ),
            Memory(
                id="3",
                context_id="test",
                content="more code",
                memory_type=MemoryType.CODE,
                created_at=current_time
            ),
        ]
        
        groups = composer.group_memories_by_type(memories)
        
        assert len(groups[MemoryType.CODE]) == 2
        assert len(groups[MemoryType.FACT]) == 1
    
    def test_full_integration(self, sample_scored_memories, query_analysis_mixed):
        """Test full integration with all features enabled."""
        composer = ResultComposer(
            interleave_by_type=True,
            max_per_type=2,
            include_breadcrumbs=True,
            min_confidence=0.4
        )
        
        result = composer.compose(
            scored_memories=sample_scored_memories,
            query_analysis=query_analysis_mixed,
            retrieval_time_ms=75.5,
            limit=10
        )
        
        # Verify all components
        assert isinstance(result, RetrievalResult)
        assert len(result.memories) > 0
        assert result.total_found == len(sample_scored_memories)
        assert result.retrieval_time_ms == 75.5
        
        # Check metadata completeness
        metadata = result.metadata
        assert "memory_groups" in metadata
        assert "breadcrumbs" in metadata
        assert "overall_confidence" in metadata
        assert "composition_strategy" in metadata
        assert "filtered_count" in metadata
        
        # Verify memory groups
        assert len(metadata["memory_groups"]) > 0
        for group in metadata["memory_groups"]:
            assert "type" in group
            assert "count" in group
            assert "avg_score" in group
            # Respect max_per_type
            assert group["count"] <= 2
        
        # Verify breadcrumbs
        assert len(metadata["breadcrumbs"]) == len(result.memories)
        for bc in metadata["breadcrumbs"]:
            assert "memory_id" in bc
            assert "path" in bc
            assert "strategies" in bc
            assert "confidence" in bc
            assert "reasoning" in bc
        
        # Verify confidence
        assert 0 <= metadata["overall_confidence"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
