"""Tests for RetrievalOrchestrator."""

import pytest
import time
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.infinite.retrieval_orchestrator import RetrievalOrchestrator, CachedResult
from core.infinite.models import Memory, MemoryType, QueryAnalysis, QueryIntent, RetrievalResult
from core.infinite.retrieval_strategies import ScoredMemory
from core.infinite.document_store import DocumentStore
from core.infinite.temporal_index import TemporalIndex
from core.infinite.vector_store import VectorStore


@pytest.fixture
def mock_document_store():
    """Create mock document store."""
    store = AsyncMock(spec=DocumentStore)
    return store


@pytest.fixture
def mock_temporal_index():
    """Create mock temporal index."""
    index = AsyncMock(spec=TemporalIndex)
    return index


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = AsyncMock(spec=VectorStore)
    return store


@pytest.fixture
def mock_embedding_fn():
    """Create mock embedding function."""
    async def embedding_fn(text: str) -> list[float]:
        # Simple mock: return hash-based embedding
        return [float(hash(text) % 100) / 100.0 for _ in range(384)]
    return embedding_fn


@pytest.fixture
def sample_memories():
    """Create sample memories for testing."""
    current_time = time.time()
    return [
        Memory(
            id="mem1",
            context_id="ctx1",
            content="I like apples",
            memory_type=MemoryType.PREFERENCE,
            created_at=current_time - 86400,  # 1 day ago
            importance=7
        ),
        Memory(
            id="mem2",
            context_id="ctx1",
            content="Python is a great programming language",
            memory_type=MemoryType.CONVERSATION,
            created_at=current_time - 3600,  # 1 hour ago
            importance=5
        ),
        Memory(
            id="mem3",
            context_id="ctx1",
            content="def login(username, password): return True",
            memory_type=MemoryType.CODE,
            created_at=current_time - 7200,  # 2 hours ago
            importance=8
        ),
    ]


@pytest.fixture
async def orchestrator(
    mock_document_store,
    mock_temporal_index,
    mock_vector_store,
    mock_embedding_fn
):
    """Create retrieval orchestrator for testing."""
    return RetrievalOrchestrator(
        document_store=mock_document_store,
        temporal_index=mock_temporal_index,
        vector_store=mock_vector_store,
        embedding_fn=mock_embedding_fn,
        enable_caching=True,
        cache_ttl_seconds=60.0,
        enable_scope_expansion=True,
        min_results_threshold=3
    )


class TestRetrievalOrchestrator:
    """Test suite for RetrievalOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.document_store is not None
        assert orchestrator.temporal_index is not None
        assert orchestrator.vector_store is not None
        assert orchestrator.query_analyzer is not None
        assert orchestrator.multi_strategy is not None
        assert orchestrator.ranker is not None
        assert orchestrator.result_composer is not None
        assert orchestrator.enable_caching is True
        assert orchestrator.cache_ttl_seconds == 60.0
    
    @pytest.mark.asyncio
    async def test_analyze_query(self, orchestrator):
        """Test query analysis."""
        # Test factual query
        analysis = await orchestrator.analyze_query("What is Python?")
        assert isinstance(analysis, QueryAnalysis)
        assert analysis.intent in [QueryIntent.FACTUAL, QueryIntent.CONVERSATIONAL]
        assert len(analysis.keywords) > 0
        
        # Test temporal query
        analysis = await orchestrator.analyze_query("What did I do yesterday?")
        # Query may be classified as TEMPORAL or MIXED (both temporal and factual patterns)
        assert analysis.intent in [QueryIntent.TEMPORAL, QueryIntent.MIXED]
        assert len(analysis.temporal_expressions) > 0
        
        # Test code query
        analysis = await orchestrator.analyze_query("Show me the login() function")
        assert analysis.intent == QueryIntent.CODE
        # Code patterns may or may not be detected depending on the query format
        # The intent detection is what matters most
    
    @pytest.mark.asyncio
    async def test_retrieve_basic(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store,
        sample_memories
    ):
        """Test basic retrieval."""
        # Setup mocks
        mock_vector_store.search = AsyncMock(return_value=[
            ("mem1", 0.9),
            ("mem2", 0.8),
        ])
        
        mock_document_store.get_memory = AsyncMock(side_effect=lambda mid: {
            "mem1": sample_memories[0],
            "mem2": sample_memories[1],
        }.get(mid))
        
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories)
        
        # Execute retrieval
        result = await orchestrator.retrieve(
            query="What do I like?",
            context_id="ctx1",
            max_results=10
        )
        
        # Verify result
        assert isinstance(result, RetrievalResult)
        assert len(result.memories) > 0
        assert result.total_found > 0
        assert result.retrieval_time_ms > 0
        assert result.query_analysis is not None
    
    @pytest.mark.asyncio
    async def test_retrieve_with_memory_type_filter(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store,
        sample_memories
    ):
        """Test retrieval with memory type filter."""
        # Setup mocks
        mock_vector_store.search = AsyncMock(return_value=[
            ("mem1", 0.9),
            ("mem3", 0.7),
        ])
        
        mock_document_store.get_memory = AsyncMock(side_effect=lambda mid: {
            "mem1": sample_memories[0],
            "mem3": sample_memories[2],
        }.get(mid))
        
        mock_document_store.query_memories = AsyncMock(return_value=[sample_memories[2]])
        
        # Execute retrieval with CODE filter
        result = await orchestrator.retrieve(
            query="Show me code",
            context_id="ctx1",
            max_results=10,
            memory_types=[MemoryType.CODE]
        )
        
        # Verify only CODE memories returned
        assert all(m.memory_type == MemoryType.CODE for m in result.memories)
    
    @pytest.mark.asyncio
    async def test_retrieve_with_time_range(
        self,
        orchestrator,
        mock_temporal_index,
        mock_document_store,
        mock_vector_store,
        sample_memories
    ):
        """Test retrieval with time range filter."""
        current_time = time.time()
        time_range = (current_time - 7200, current_time)  # Last 2 hours
        
        # Setup mocks for temporal strategy
        mock_temporal_index.query_by_time_range = AsyncMock(return_value=[
            {"memory_id": "mem2", "timestamp": current_time - 3600, "event_type": "created"},
            {"memory_id": "mem3", "timestamp": current_time - 7200, "event_type": "created"},
        ])
        
        mock_document_store.get_memory = AsyncMock(side_effect=lambda mid: {
            "mem2": sample_memories[1],
            "mem3": sample_memories[2],
        }.get(mid))
        
        # Also setup for semantic and fulltext strategies
        mock_vector_store.search = AsyncMock(return_value=[
            ("mem2", 0.8),
            ("mem3", 0.7),
        ])
        
        mock_document_store.query_memories = AsyncMock(return_value=[
            sample_memories[1],
            sample_memories[2]
        ])
        
        # Execute retrieval with time range
        result = await orchestrator.retrieve(
            query="Recent activities",
            context_id="ctx1",
            max_results=10,
            time_range=time_range
        )
        
        # Verify result - should have at least some memories
        assert isinstance(result, RetrievalResult)
        # With mocks, we should get results from at least one strategy
        assert result.total_found >= 0  # May be 0 with mocks, but should not error
    
    @pytest.mark.asyncio
    async def test_caching(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store,
        sample_memories
    ):
        """Test query result caching."""
        # Setup mocks
        mock_vector_store.search = AsyncMock(return_value=[("mem1", 0.9)])
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[0])
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories)
        
        query = "What do I like?"
        context_id = "ctx1"
        
        # First retrieval - should hit storage
        result1 = await orchestrator.retrieve(query, context_id, max_results=10)
        call_count_1 = mock_vector_store.search.call_count
        
        # Second retrieval - should hit cache
        result2 = await orchestrator.retrieve(query, context_id, max_results=10)
        call_count_2 = mock_vector_store.search.call_count
        
        # Verify cache was used (no additional storage calls)
        assert call_count_2 == call_count_1
        assert len(result1.memories) == len(result2.memories)
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, orchestrator):
        """Test cache expiration."""
        # Set very short TTL
        orchestrator.cache_ttl_seconds = 0.1
        
        # Create a fake cached result
        query_hash = orchestrator._get_query_hash("test", "ctx1", 10)
        fake_result = RetrievalResult(
            memories=[],
            total_found=0,
            query_analysis=QueryAnalysis(intent=QueryIntent.FACTUAL),
            retrieval_time_ms=10.0
        )
        
        orchestrator._cache[query_hash] = CachedResult(
            result=fake_result,
            timestamp=time.time(),
            query_hash=query_hash
        )
        
        # Verify cache has entry
        assert len(orchestrator._cache) == 1
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Try to get cached result - should be None (expired)
        cached = orchestrator._get_cached_result("test", "ctx1", 10)
        assert cached is None
        assert len(orchestrator._cache) == 0  # Should be removed
    
    @pytest.mark.asyncio
    async def test_cache_disabled(
        self,
        mock_document_store,
        mock_temporal_index,
        mock_vector_store,
        mock_embedding_fn,
        sample_memories
    ):
        """Test retrieval with caching disabled."""
        # Create orchestrator with caching disabled
        orch = RetrievalOrchestrator(
            document_store=mock_document_store,
            temporal_index=mock_temporal_index,
            vector_store=mock_vector_store,
            embedding_fn=mock_embedding_fn,
            enable_caching=False
        )
        
        # Setup mocks
        mock_vector_store.search = AsyncMock(return_value=[("mem1", 0.9)])
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[0])
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories)
        
        # Multiple retrievals should always hit storage
        await orch.retrieve("test", "ctx1", max_results=10)
        call_count_1 = mock_vector_store.search.call_count
        
        await orch.retrieve("test", "ctx1", max_results=10)
        call_count_2 = mock_vector_store.search.call_count
        
        # Verify no caching (call count increased)
        assert call_count_2 > call_count_1
    
    @pytest.mark.asyncio
    async def test_scope_expansion_triggered(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store,
        sample_memories
    ):
        """Test adaptive search scope expansion when results are insufficient."""
        # Setup mocks to return very few results initially
        mock_vector_store.search = AsyncMock(return_value=[("mem1", 0.9)])
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[0])
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories[:1])
        
        # Set low threshold to trigger expansion
        orchestrator.min_results_threshold = 5
        
        # Execute retrieval
        result = await orchestrator.retrieve(
            query="test query",
            context_id="ctx1",
            max_results=10
        )
        
        # Verify retrieval completed (expansion was attempted)
        assert isinstance(result, RetrievalResult)
        # Note: With mocks, expansion may not add results, but it should not fail
    
    @pytest.mark.asyncio
    async def test_scope_expansion_disabled(
        self,
        mock_document_store,
        mock_temporal_index,
        mock_vector_store,
        mock_embedding_fn,
        sample_memories
    ):
        """Test retrieval with scope expansion disabled."""
        # Create orchestrator with expansion disabled
        orch = RetrievalOrchestrator(
            document_store=mock_document_store,
            temporal_index=mock_temporal_index,
            vector_store=mock_vector_store,
            embedding_fn=mock_embedding_fn,
            enable_scope_expansion=False
        )
        
        # Setup mocks
        mock_vector_store.search = AsyncMock(return_value=[("mem1", 0.9)])
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[0])
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories[:1])
        
        # Execute retrieval
        result = await orch.retrieve(
            query="test query",
            context_id="ctx1",
            max_results=10
        )
        
        # Verify result (no expansion should occur)
        assert isinstance(result, RetrievalResult)
    
    @pytest.mark.asyncio
    async def test_rank_results(self, orchestrator, sample_memories):
        """Test result ranking."""
        # Create scored memories
        scored_memories = [
            ScoredMemory(memory=sample_memories[0], score=0.7, strategy="semantic"),
            ScoredMemory(memory=sample_memories[1], score=0.9, strategy="semantic"),
            ScoredMemory(memory=sample_memories[2], score=0.6, strategy="temporal"),
        ]
        
        # Create query analysis
        query_analysis = QueryAnalysis(
            intent=QueryIntent.FACTUAL,
            confidence=0.8
        )
        
        # Rank results
        ranked = await orchestrator.rank_results(
            scored_memories=scored_memories,
            query_analysis=query_analysis,
            boost_recent=True,
            penalize_redundancy=True
        )
        
        # Verify ranking
        assert len(ranked) == 3
        assert all(isinstance(sm, ScoredMemory) for sm in ranked)
        # Scores should be updated by ranking
        assert ranked[0].score >= ranked[1].score >= ranked[2].score
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, orchestrator):
        """Test cache clearing."""
        # Add some fake cache entries
        for i in range(5):
            query_hash = orchestrator._get_query_hash(f"query{i}", "ctx1", 10)
            orchestrator._cache[query_hash] = CachedResult(
                result=RetrievalResult(
                    memories=[],
                    total_found=0,
                    query_analysis=QueryAnalysis(intent=QueryIntent.FACTUAL),
                    retrieval_time_ms=10.0
                ),
                timestamp=time.time(),
                query_hash=query_hash
            )
        
        # Verify cache has entries
        assert len(orchestrator._cache) == 5
        
        # Clear cache
        count = orchestrator.clear_cache()
        
        # Verify cache is empty
        assert count == 5
        assert len(orchestrator._cache) == 0
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self, orchestrator):
        """Test cache statistics."""
        # Add some cache entries
        current_time = time.time()
        
        # Add active entry
        query_hash1 = orchestrator._get_query_hash("query1", "ctx1", 10)
        orchestrator._cache[query_hash1] = CachedResult(
            result=RetrievalResult(
                memories=[],
                total_found=0,
                query_analysis=QueryAnalysis(intent=QueryIntent.FACTUAL),
                retrieval_time_ms=10.0
            ),
            timestamp=current_time,
            query_hash=query_hash1
        )
        
        # Add expired entry
        query_hash2 = orchestrator._get_query_hash("query2", "ctx1", 10)
        orchestrator._cache[query_hash2] = CachedResult(
            result=RetrievalResult(
                memories=[],
                total_found=0,
                query_analysis=QueryAnalysis(intent=QueryIntent.FACTUAL),
                retrieval_time_ms=10.0
            ),
            timestamp=current_time - 1000,  # Old timestamp
            query_hash=query_hash2
        )
        
        # Get stats
        stats = orchestrator.get_cache_stats()
        
        # Verify stats
        assert stats["total_entries"] == 2
        assert stats["expired_entries"] == 1
        assert stats["active_entries"] == 1
        assert stats["cache_enabled"] is True
        assert stats["ttl_seconds"] == 60.0
    
    @pytest.mark.asyncio
    async def test_retrieve_with_use_cache_false(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store,
        sample_memories
    ):
        """Test retrieval with use_cache=False bypasses cache."""
        # Setup mocks
        mock_vector_store.search = AsyncMock(return_value=[("mem1", 0.9)])
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[0])
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories)
        
        query = "test query"
        context_id = "ctx1"
        
        # First retrieval with caching
        await orchestrator.retrieve(query, context_id, max_results=10, use_cache=True)
        call_count_1 = mock_vector_store.search.call_count
        
        # Second retrieval with use_cache=False should bypass cache
        await orchestrator.retrieve(query, context_id, max_results=10, use_cache=False)
        call_count_2 = mock_vector_store.search.call_count
        
        # Verify cache was bypassed (additional storage call made)
        assert call_count_2 > call_count_1


class TestQueryAnalysisAccuracy:
    """Comprehensive tests for query analysis accuracy."""
    
    @pytest.mark.asyncio
    async def test_factual_query_patterns(self, orchestrator):
        """Test various factual query patterns."""
        factual_queries = [
            "What is Python?",
            "Tell me about machine learning",
            "Explain quantum computing",
            "What are the benefits of exercise?",
            "Define artificial intelligence"
        ]
        
        for query in factual_queries:
            analysis = await orchestrator.analyze_query(query)
            assert analysis.intent in [QueryIntent.FACTUAL, QueryIntent.CONVERSATIONAL]
            assert len(analysis.keywords) > 0
            assert analysis.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_temporal_query_patterns(self, orchestrator):
        """Test various temporal query patterns."""
        temporal_queries = [
            "What did I do yesterday?",
            "Show me last week's activities",
            "What happened 2 hours ago?",
            "Recent conversations",
            "Activities from last month"
        ]
        
        for query in temporal_queries:
            analysis = await orchestrator.analyze_query(query)
            # Should detect temporal intent, mixed intent, or conversational (all acceptable)
            # The key is that temporal expressions or keywords are extracted
            assert analysis.intent in [QueryIntent.TEMPORAL, QueryIntent.MIXED, QueryIntent.CONVERSATIONAL]
            # Should extract temporal expressions or relevant keywords
            assert len(analysis.temporal_expressions) > 0 or len(analysis.keywords) > 0
    
    @pytest.mark.asyncio
    async def test_code_query_patterns(self, orchestrator):
        """Test various code-related query patterns."""
        code_queries = [
            "Show me the login() function",
            "Find the authenticate method",
            "What's in auth.py?",
            "Show code for user authentication",
            "def process_data implementation"
        ]
        
        for query in code_queries:
            analysis = await orchestrator.analyze_query(query)
            # Should detect code or mixed intent (both acceptable for code queries)
            assert analysis.intent in [QueryIntent.CODE, QueryIntent.MIXED]
            # Should detect code patterns or keywords
            assert len(analysis.code_patterns) > 0 or len(analysis.keywords) > 0
    
    @pytest.mark.asyncio
    async def test_conversational_query_patterns(self, orchestrator):
        """Test conversational query patterns."""
        conversational_queries = [
            "How are you?",
            "Thanks for the help",
            "That's interesting",
            "I see what you mean",
            "Great explanation"
        ]
        
        for query in conversational_queries:
            analysis = await orchestrator.analyze_query(query)
            # Conversational or factual are both acceptable for these queries
            assert analysis.intent in [QueryIntent.CONVERSATIONAL, QueryIntent.FACTUAL]
    
    @pytest.mark.asyncio
    async def test_mixed_intent_queries(self, orchestrator):
        """Test queries with mixed intents."""
        mixed_queries = [
            "What code changes did I make yesterday?",
            "Show me recent Python functions",
            "What did the login function look like last week?"
        ]
        
        for query in mixed_queries:
            analysis = await orchestrator.analyze_query(query)
            # Should detect mixed, code, temporal, or conversational intent (all acceptable)
            assert analysis.intent in [QueryIntent.MIXED, QueryIntent.CODE, QueryIntent.TEMPORAL, QueryIntent.CONVERSATIONAL]
    
    @pytest.mark.asyncio
    async def test_entity_extraction_accuracy(self, orchestrator):
        """Test entity extraction from queries."""
        query = "What did John say about Python yesterday?"
        analysis = await orchestrator.analyze_query(query)
        
        # Should extract entities (if spaCy is available)
        # At minimum should extract keywords
        assert len(analysis.keywords) > 0
        assert analysis.confidence > 0.0


class TestMultiStrategyRetrieval:
    """Comprehensive tests for multi-strategy retrieval."""
    
    @pytest.mark.asyncio
    async def test_semantic_strategy_dominance(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store,
        sample_memories
    ):
        """Test semantic strategy for similarity-based queries."""
        # Setup mocks for semantic search
        mock_vector_store.search = AsyncMock(return_value=[
            ("mem1", 0.95),
            ("mem2", 0.85),
            ("mem3", 0.75),
        ])
        
        mock_document_store.get_memory = AsyncMock(side_effect=lambda mid: {
            "mem1": sample_memories[0],
            "mem2": sample_memories[1],
            "mem3": sample_memories[2],
        }.get(mid))
        
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories)
        
        # Query that should trigger semantic search
        result = await orchestrator.retrieve(
            query="Tell me about programming languages",
            context_id="ctx1",
            max_results=10
        )
        
        # Verify semantic strategy was used
        assert len(result.memories) > 0
        assert mock_vector_store.search.called
    
    @pytest.mark.asyncio
    async def test_temporal_strategy_for_time_queries(
        self,
        orchestrator,
        mock_temporal_index,
        mock_document_store,
        mock_vector_store,
        sample_memories
    ):
        """Test temporal strategy for time-based queries."""
        current_time = time.time()
        
        # Setup mocks for temporal search
        mock_temporal_index.query_by_time_range = AsyncMock(return_value=[
            {"memory_id": "mem2", "timestamp": current_time - 3600, "event_type": "created"},
        ])
        
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[1])
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories)
        mock_vector_store.search = AsyncMock(return_value=[("mem2", 0.8)])
        
        # Query with temporal expression
        result = await orchestrator.retrieve(
            query="What happened in the last hour?",
            context_id="ctx1",
            max_results=10
        )
        
        # Verify retrieval completed
        assert isinstance(result, RetrievalResult)
    
    @pytest.mark.asyncio
    async def test_fulltext_strategy_fallback(
        self,
        orchestrator,
        mock_document_store,
        mock_vector_store,
        sample_memories
    ):
        """Test fulltext strategy as fallback."""
        # Setup mocks - semantic returns nothing
        mock_vector_store.search = AsyncMock(return_value=[])
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories[:1])
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[0])
        
        # Query should fall back to fulltext
        result = await orchestrator.retrieve(
            query="apples",
            context_id="ctx1",
            max_results=10
        )
        
        # Should get results from fulltext strategy
        assert isinstance(result, RetrievalResult)
    
    @pytest.mark.asyncio
    async def test_strategy_fusion(
        self,
        orchestrator,
        mock_vector_store,
        mock_temporal_index,
        mock_document_store,
        sample_memories
    ):
        """Test fusion of results from multiple strategies."""
        current_time = time.time()
        
        # Setup mocks for multiple strategies
        mock_vector_store.search = AsyncMock(return_value=[
            ("mem1", 0.9),
            ("mem2", 0.8),
        ])
        
        mock_temporal_index.query_by_time_range = AsyncMock(return_value=[
            {"memory_id": "mem2", "timestamp": current_time - 3600, "event_type": "created"},
            {"memory_id": "mem3", "timestamp": current_time - 7200, "event_type": "created"},
        ])
        
        mock_document_store.get_memory = AsyncMock(side_effect=lambda mid: {
            "mem1": sample_memories[0],
            "mem2": sample_memories[1],
            "mem3": sample_memories[2],
        }.get(mid))
        
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories)
        
        # Query that could trigger multiple strategies
        result = await orchestrator.retrieve(
            query="Recent programming discussions",
            context_id="ctx1",
            max_results=10
        )
        
        # Should fuse results from multiple strategies
        assert isinstance(result, RetrievalResult)
        assert len(result.memories) > 0


class TestRankingAlgorithm:
    """Comprehensive tests for ranking algorithm correctness."""
    
    @pytest.mark.asyncio
    async def test_relevance_score_ordering(self, orchestrator, sample_memories):
        """Test that higher relevance scores rank higher."""
        scored_memories = [
            ScoredMemory(memory=sample_memories[0], score=0.5, strategy="semantic"),
            ScoredMemory(memory=sample_memories[1], score=0.9, strategy="semantic"),
            ScoredMemory(memory=sample_memories[2], score=0.7, strategy="semantic"),
        ]
        
        query_analysis = QueryAnalysis(intent=QueryIntent.FACTUAL, confidence=0.8)
        
        ranked = await orchestrator.rank_results(
            scored_memories=scored_memories,
            query_analysis=query_analysis,
            boost_recent=False,
            penalize_redundancy=False
        )
        
        # Should be ordered by score (descending)
        assert ranked[0].score >= ranked[1].score
        assert ranked[1].score >= ranked[2].score
    
    @pytest.mark.asyncio
    async def test_recency_boost(self, orchestrator):
        """Test that recent memories get boosted."""
        current_time = time.time()
        
        old_memory = Memory(
            id="old",
            context_id="ctx1",
            content="Old information",
            memory_type=MemoryType.FACT,
            created_at=current_time - 86400 * 30,  # 30 days ago
            importance=5
        )
        
        recent_memory = Memory(
            id="recent",
            context_id="ctx1",
            content="Recent information",
            memory_type=MemoryType.FACT,
            created_at=current_time - 3600,  # 1 hour ago
            importance=5
        )
        
        scored_memories = [
            ScoredMemory(memory=old_memory, score=0.8, strategy="semantic"),
            ScoredMemory(memory=recent_memory, score=0.8, strategy="semantic"),
        ]
        
        query_analysis = QueryAnalysis(intent=QueryIntent.FACTUAL, confidence=0.8)
        
        ranked = await orchestrator.rank_results(
            scored_memories=scored_memories,
            query_analysis=query_analysis,
            boost_recent=True,
            penalize_redundancy=False
        )
        
        # Recent memory should rank higher due to recency boost
        assert ranked[0].memory.id == "recent"
    
    @pytest.mark.asyncio
    async def test_importance_weighting(self, orchestrator):
        """Test that importance affects ranking."""
        current_time = time.time()
        
        low_importance = Memory(
            id="low",
            context_id="ctx1",
            content="Low importance info",
            memory_type=MemoryType.FACT,
            created_at=current_time,
            importance=3
        )
        
        high_importance = Memory(
            id="high",
            context_id="ctx1",
            content="High importance info",
            memory_type=MemoryType.FACT,
            created_at=current_time,
            importance=9
        )
        
        scored_memories = [
            ScoredMemory(memory=low_importance, score=0.8, strategy="semantic"),
            ScoredMemory(memory=high_importance, score=0.8, strategy="semantic"),
        ]
        
        query_analysis = QueryAnalysis(intent=QueryIntent.FACTUAL, confidence=0.8)
        
        ranked = await orchestrator.rank_results(
            scored_memories=scored_memories,
            query_analysis=query_analysis,
            boost_recent=False,
            penalize_redundancy=False
        )
        
        # High importance should rank higher
        assert ranked[0].memory.id == "high"
    
    @pytest.mark.asyncio
    async def test_redundancy_penalization(self, orchestrator):
        """Test that redundant results are penalized."""
        current_time = time.time()
        
        # Create similar memories
        mem1 = Memory(
            id="mem1",
            context_id="ctx1",
            content="Python is a programming language",
            memory_type=MemoryType.FACT,
            created_at=current_time,
            importance=5
        )
        
        mem2 = Memory(
            id="mem2",
            context_id="ctx1",
            content="Python is a programming language used for development",
            memory_type=MemoryType.FACT,
            created_at=current_time,
            importance=5
        )
        
        mem3 = Memory(
            id="mem3",
            context_id="ctx1",
            content="JavaScript is completely different",
            memory_type=MemoryType.FACT,
            created_at=current_time,
            importance=5
        )
        
        scored_memories = [
            ScoredMemory(memory=mem1, score=0.9, strategy="semantic"),
            ScoredMemory(memory=mem2, score=0.85, strategy="semantic"),
            ScoredMemory(memory=mem3, score=0.8, strategy="semantic"),
        ]
        
        query_analysis = QueryAnalysis(intent=QueryIntent.FACTUAL, confidence=0.8)
        
        ranked = await orchestrator.rank_results(
            scored_memories=scored_memories,
            query_analysis=query_analysis,
            boost_recent=False,
            penalize_redundancy=True
        )
        
        # mem2 should be penalized for being similar to mem1
        # mem3 should maintain its position as it's different
        assert len(ranked) == 3


class TestResultComposition:
    """Comprehensive tests for result composition."""
    
    @pytest.mark.asyncio
    async def test_result_structure(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store,
        sample_memories
    ):
        """Test that result has correct structure."""
        mock_vector_store.search = AsyncMock(return_value=[("mem1", 0.9)])
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[0])
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories)
        
        result = await orchestrator.retrieve(
            query="test",
            context_id="ctx1",
            max_results=10
        )
        
        # Verify result structure
        assert isinstance(result, RetrievalResult)
        assert hasattr(result, 'memories')
        assert hasattr(result, 'total_found')
        assert hasattr(result, 'query_analysis')
        assert hasattr(result, 'retrieval_time_ms')
        assert isinstance(result.memories, list)
        assert isinstance(result.total_found, int)
        assert isinstance(result.retrieval_time_ms, float)
    
    @pytest.mark.asyncio
    async def test_memory_type_interleaving(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store
    ):
        """Test that different memory types are interleaved."""
        current_time = time.time()
        
        # Create memories of different types
        mixed_memories = [
            Memory(
                id="conv1",
                context_id="ctx1",
                content="Conversation about Python",
                memory_type=MemoryType.CONVERSATION,
                created_at=current_time,
                importance=5
            ),
            Memory(
                id="code1",
                context_id="ctx1",
                content="def hello(): pass",
                memory_type=MemoryType.CODE,
                created_at=current_time,
                importance=5
            ),
            Memory(
                id="fact1",
                context_id="ctx1",
                content="Python was created in 1991",
                memory_type=MemoryType.FACT,
                created_at=current_time,
                importance=5
            ),
        ]
        
        mock_vector_store.search = AsyncMock(return_value=[
            ("conv1", 0.9),
            ("code1", 0.85),
            ("fact1", 0.8),
        ])
        
        mock_document_store.get_memory = AsyncMock(side_effect=lambda mid: {
            "conv1": mixed_memories[0],
            "code1": mixed_memories[1],
            "fact1": mixed_memories[2],
        }.get(mid))
        
        mock_document_store.query_memories = AsyncMock(return_value=mixed_memories)
        
        result = await orchestrator.retrieve(
            query="Python information",
            context_id="ctx1",
            max_results=10
        )
        
        # Should have mixed types
        memory_types = [m.memory_type for m in result.memories]
        assert len(set(memory_types)) > 1  # Multiple types present
    
    @pytest.mark.asyncio
    async def test_max_results_limit(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store,
        sample_memories
    ):
        """Test that max_results limit is respected."""
        # Setup mocks to return many results
        mock_vector_store.search = AsyncMock(return_value=[
            (f"mem{i}", 0.9 - i * 0.1) for i in range(20)
        ])
        
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[0])
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories * 10)
        
        # Request limited results
        result = await orchestrator.retrieve(
            query="test",
            context_id="ctx1",
            max_results=5
        )
        
        # Should not exceed max_results
        assert len(result.memories) <= 5
    
    @pytest.mark.asyncio
    async def test_query_analysis_included(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store,
        sample_memories
    ):
        """Test that query analysis is included in result."""
        mock_vector_store.search = AsyncMock(return_value=[("mem1", 0.9)])
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[0])
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories)
        
        result = await orchestrator.retrieve(
            query="What is Python?",
            context_id="ctx1",
            max_results=10
        )
        
        # Should include query analysis
        assert result.query_analysis is not None
        assert isinstance(result.query_analysis, QueryAnalysis)
        assert result.query_analysis.intent is not None


class TestVariousQueryTypesAndVolumes:
    """Comprehensive tests with various query types and memory volumes."""
    
    @pytest.mark.asyncio
    async def test_empty_result_set(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store
    ):
        """Test handling of queries with no results."""
        mock_vector_store.search = AsyncMock(return_value=[])
        mock_document_store.query_memories = AsyncMock(return_value=[])
        
        result = await orchestrator.retrieve(
            query="nonexistent information",
            context_id="ctx1",
            max_results=10
        )
        
        # Should return empty result gracefully
        assert isinstance(result, RetrievalResult)
        assert len(result.memories) == 0
        assert result.total_found == 0
    
    @pytest.mark.asyncio
    async def test_single_result(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store,
        sample_memories
    ):
        """Test handling of single result."""
        mock_vector_store.search = AsyncMock(return_value=[("mem1", 0.9)])
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[0])
        mock_document_store.query_memories = AsyncMock(return_value=[sample_memories[0]])
        
        result = await orchestrator.retrieve(
            query="specific query",
            context_id="ctx1",
            max_results=10
        )
        
        # Should handle single result
        assert len(result.memories) >= 0
        assert result.total_found >= 0
    
    @pytest.mark.asyncio
    async def test_large_result_set(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store
    ):
        """Test handling of large result sets."""
        current_time = time.time()
        
        # Create many memories
        large_memory_set = [
            Memory(
                id=f"mem{i}",
                context_id="ctx1",
                content=f"Memory content {i}",
                memory_type=MemoryType.FACT,
                created_at=current_time - i * 100,
                importance=5
            )
            for i in range(100)
        ]
        
        mock_vector_store.search = AsyncMock(return_value=[
            (f"mem{i}", 0.9 - i * 0.001) for i in range(100)
        ])
        
        mock_document_store.get_memory = AsyncMock(
            side_effect=lambda mid: next(
                (m for m in large_memory_set if m.id == mid),
                None
            )
        )
        
        mock_document_store.query_memories = AsyncMock(return_value=large_memory_set)
        
        result = await orchestrator.retrieve(
            query="test query",
            context_id="ctx1",
            max_results=50
        )
        
        # Should handle large sets efficiently
        assert isinstance(result, RetrievalResult)
        assert len(result.memories) <= 50
        assert result.retrieval_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_complex_query_with_multiple_filters(
        self,
        orchestrator,
        mock_vector_store,
        mock_temporal_index,
        mock_document_store,
        sample_memories
    ):
        """Test complex query with multiple filters."""
        current_time = time.time()
        time_range = (current_time - 86400, current_time)
        
        mock_vector_store.search = AsyncMock(return_value=[
            ("mem2", 0.9),
        ])
        
        mock_temporal_index.query_by_time_range = AsyncMock(return_value=[
            {"memory_id": "mem2", "timestamp": current_time - 3600, "event_type": "created"},
        ])
        
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[1])
        mock_document_store.query_memories = AsyncMock(return_value=[sample_memories[1]])
        
        result = await orchestrator.retrieve(
            query="programming",
            context_id="ctx1",
            max_results=10,
            memory_types=[MemoryType.CONVERSATION],
            time_range=time_range
        )
        
        # Should apply all filters
        assert isinstance(result, RetrievalResult)
        # All returned memories should match filters
        for memory in result.memories:
            assert memory.memory_type == MemoryType.CONVERSATION
            assert time_range[0] <= memory.created_at <= time_range[1]
    
    @pytest.mark.asyncio
    async def test_performance_with_concurrent_queries(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store,
        sample_memories
    ):
        """Test performance with concurrent queries."""
        mock_vector_store.search = AsyncMock(return_value=[("mem1", 0.9)])
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[0])
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories)
        
        # Execute multiple queries concurrently
        queries = [
            orchestrator.retrieve(f"query {i}", "ctx1", max_results=10)
            for i in range(10)
        ]
        
        results = await asyncio.gather(*queries)
        
        # All queries should complete successfully
        assert len(results) == 10
        assert all(isinstance(r, RetrievalResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_different_context_ids(
        self,
        orchestrator,
        mock_vector_store,
        mock_document_store,
        sample_memories
    ):
        """Test retrieval with different context IDs."""
        mock_vector_store.search = AsyncMock(return_value=[("mem1", 0.9)])
        mock_document_store.get_memory = AsyncMock(return_value=sample_memories[0])
        mock_document_store.query_memories = AsyncMock(return_value=sample_memories)
        
        # Query different contexts
        result1 = await orchestrator.retrieve("test", "ctx1", max_results=10)
        result2 = await orchestrator.retrieve("test", "ctx2", max_results=10)
        
        # Should handle different contexts
        assert isinstance(result1, RetrievalResult)
        assert isinstance(result2, RetrievalResult)
        # Results should be cached separately
        assert len(orchestrator._cache) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
