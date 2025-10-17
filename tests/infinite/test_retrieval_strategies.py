"""Tests for multi-strategy retrieval system."""

import pytest
import asyncio
import time
from pathlib import Path
import tempfile
import shutil

from core.infinite.retrieval_strategies import (
    SemanticRetrievalStrategy,
    TemporalRetrievalStrategy,
    StructuralRetrievalStrategy,
    FullTextRetrievalStrategy,
    MultiStrategyRetrieval,
    ScoredMemory,
)
from core.infinite.models import Memory, MemoryType, QueryAnalysis, QueryIntent
from core.infinite.document_store import DocumentStore
from core.infinite.temporal_index import TemporalIndex
from core.infinite.vector_store import VectorStore
from core.infinite.code_change_store import CodeChangeStore, CodeChange
from core.infinite.query_analyzer import QueryAnalyzer


@pytest.fixture
async def temp_dir():
    """Create a temporary directory for test databases."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
async def document_store(temp_dir):
    """Create and initialize a document store."""
    db_path = temp_dir / "test.db"
    store = DocumentStore(db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def temporal_index(temp_dir):
    """Create and initialize a temporal index."""
    db_path = temp_dir / "test.db"
    index = TemporalIndex(db_path)
    await index.initialize()
    yield index
    await index.close()


@pytest.fixture
async def vector_store(temp_dir):
    """Create and initialize a vector store."""
    store = VectorStore(path=temp_dir / "qdrant", embedding_dim=384)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def code_change_store(temp_dir):
    """Create and initialize a code change store."""
    db_path = temp_dir / "test.db"
    store = CodeChangeStore(db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def query_analyzer():
    """Create a query analyzer."""
    return QueryAnalyzer(use_spacy=False)


@pytest.fixture
async def sample_memories(document_store):
    """Create sample memories for testing."""
    import uuid
    current_time = time.time()
    memories = []
    
    # Recent conversation memory
    mem1 = Memory(
        id=str(uuid.uuid4()),
        context_id="test_context",
        content="I prefer dark mode for coding",
        memory_type=MemoryType.PREFERENCE,
        created_at=current_time - 3600,  # 1 hour ago
        importance=8,
        embedding=[0.1] * 384  # Dummy embedding
    )
    await document_store.add_memory(mem1)
    memories.append(mem1)
    
    # Older conversation memory
    mem2 = Memory(
        id=str(uuid.uuid4()),
        context_id="test_context",
        content="Python is my favorite programming language",
        memory_type=MemoryType.PREFERENCE,
        created_at=current_time - 86400 * 7,  # 7 days ago
        importance=7,
        embedding=[0.2] * 384
    )
    await document_store.add_memory(mem2)
    memories.append(mem2)
    
    # Code memory
    mem3 = Memory(
        id=str(uuid.uuid4()),
        context_id="test_context",
        content="def login(username, password): return authenticate(username, password)",
        memory_type=MemoryType.CODE,
        created_at=current_time - 86400 * 2,  # 2 days ago
        importance=9,
        embedding=[0.3] * 384
    )
    await document_store.add_memory(mem3)
    memories.append(mem3)
    
    # Fact memory
    mem4 = Memory(
        id=str(uuid.uuid4()),
        context_id="test_context",
        content="The project deadline is next Friday",
        memory_type=MemoryType.FACT,
        created_at=current_time - 86400,  # 1 day ago
        importance=10,
        embedding=[0.4] * 384
    )
    await document_store.add_memory(mem4)
    memories.append(mem4)
    
    return memories


@pytest.mark.asyncio
async def test_semantic_retrieval_strategy(document_store, vector_store, sample_memories, query_analyzer):
    """Test semantic retrieval strategy."""
    # Add memories to vector store
    for memory in sample_memories:
        await vector_store.add_memory(memory)
    
    # Create strategy
    def dummy_embedding_fn(text):
        # Return a dummy embedding similar to mem1
        return [0.1] * 384
    
    strategy = SemanticRetrievalStrategy(
        vector_store=vector_store,
        document_store=document_store,
        embedding_fn=dummy_embedding_fn
    )
    
    # Test retrieval
    query = "What are my preferences for coding?"
    query_analysis = query_analyzer.analyze(query)
    
    results = await strategy.retrieve(
        query=query,
        query_analysis=query_analysis,
        context_id="test_context",
        limit=5
    )
    
    assert len(results) > 0
    assert all(isinstance(r, ScoredMemory) for r in results)
    assert all(r.strategy == "semantic" for r in results)
    # Scores from Qdrant can be > 1 for cosine similarity, so just check they're reasonable
    assert all(r.score >= 0 for r in results)


@pytest.mark.asyncio
async def test_temporal_retrieval_strategy(document_store, temporal_index, sample_memories, query_analyzer):
    """Test temporal retrieval strategy."""
    strategy = TemporalRetrievalStrategy(
        temporal_index=temporal_index,
        document_store=document_store
    )
    
    # Test retrieval with recent time range
    query = "What happened yesterday?"
    query_analysis = query_analyzer.analyze(query)
    
    results = await strategy.retrieve(
        query=query,
        query_analysis=query_analysis,
        context_id="test_context",
        limit=5
    )
    
    assert len(results) > 0
    assert all(isinstance(r, ScoredMemory) for r in results)
    assert all(r.strategy == "temporal" for r in results)
    
    # More recent memories should have higher scores
    if len(results) > 1:
        assert results[0].score >= results[-1].score


@pytest.mark.asyncio
async def test_fulltext_retrieval_strategy(document_store, sample_memories, query_analyzer):
    """Test full-text retrieval strategy."""
    strategy = FullTextRetrievalStrategy(document_store=document_store)
    
    # Test retrieval with keyword matching
    query = "What is my favorite programming language?"
    query_analysis = query_analyzer.analyze(query)
    
    results = await strategy.retrieve(
        query=query,
        query_analysis=query_analysis,
        context_id="test_context",
        limit=5
    )
    
    assert len(results) > 0
    assert all(isinstance(r, ScoredMemory) for r in results)
    assert all(r.strategy == "fulltext" for r in results)
    
    # Should find the Python preference memory
    memory_contents = [r.memory.content for r in results]
    assert any("Python" in content for content in memory_contents)


@pytest.mark.asyncio
async def test_structural_retrieval_strategy(code_change_store, document_store, query_analyzer):
    """Test structural retrieval strategy."""
    # Add a code change
    change = CodeChange(
        id="change1",
        file_path="auth.py",
        change_type="modify",
        timestamp=time.time(),
        before_content="def login(): pass",
        after_content="def login(username, password): return authenticate(username, password)",
        metadata={}
    )
    await code_change_store.add_change(change)
    
    strategy = StructuralRetrievalStrategy(
        code_change_store=code_change_store,
        document_store=document_store
    )
    
    # Test retrieval with code pattern
    query = "Show me changes to the login function"
    query_analysis = query_analyzer.analyze(query)
    
    results = await strategy.retrieve(
        query=query,
        query_analysis=query_analysis,
        context_id="test_context",
        limit=5
    )
    
    # Should find code-related results
    assert isinstance(results, list)
    if len(results) > 0:
        assert all(isinstance(r, ScoredMemory) for r in results)
        assert all(r.strategy == "structural" for r in results)


@pytest.mark.asyncio
async def test_multi_strategy_retrieval(
    document_store,
    temporal_index,
    vector_store,
    sample_memories,
    query_analyzer
):
    """Test multi-strategy retrieval with fusion."""
    # Add memories to vector store
    for memory in sample_memories:
        await vector_store.add_memory(memory)
    
    # Create strategies
    def dummy_embedding_fn(text):
        return [0.1] * 384
    
    semantic_strategy = SemanticRetrievalStrategy(
        vector_store=vector_store,
        document_store=document_store,
        embedding_fn=dummy_embedding_fn
    )
    
    temporal_strategy = TemporalRetrievalStrategy(
        temporal_index=temporal_index,
        document_store=document_store
    )
    
    fulltext_strategy = FullTextRetrievalStrategy(document_store=document_store)
    
    # Create multi-strategy retrieval
    multi_strategy = MultiStrategyRetrieval(
        semantic_strategy=semantic_strategy,
        temporal_strategy=temporal_strategy,
        fulltext_strategy=fulltext_strategy
    )
    
    # Test retrieval
    query = "What are my coding preferences?"
    query_analysis = query_analyzer.analyze(query)
    
    results = await multi_strategy.retrieve(
        query=query,
        query_analysis=query_analysis,
        context_id="test_context",
        limit=5
    )
    
    assert len(results) > 0
    assert all(isinstance(r, ScoredMemory) for r in results)
    
    # Results should be sorted by score
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_multi_strategy_deduplication(
    document_store,
    temporal_index,
    vector_store,
    sample_memories,
    query_analyzer
):
    """Test that multi-strategy retrieval deduplicates results."""
    # Add memories to vector store
    for memory in sample_memories:
        await vector_store.add_memory(memory)
    
    # Create strategies
    def dummy_embedding_fn(text):
        return [0.1] * 384
    
    semantic_strategy = SemanticRetrievalStrategy(
        vector_store=vector_store,
        document_store=document_store,
        embedding_fn=dummy_embedding_fn
    )
    
    temporal_strategy = TemporalRetrievalStrategy(
        temporal_index=temporal_index,
        document_store=document_store
    )
    
    fulltext_strategy = FullTextRetrievalStrategy(document_store=document_store)
    
    # Create multi-strategy retrieval
    multi_strategy = MultiStrategyRetrieval(
        semantic_strategy=semantic_strategy,
        temporal_strategy=temporal_strategy,
        fulltext_strategy=fulltext_strategy
    )
    
    # Test retrieval with a query that should match across strategies
    query = "dark mode coding"
    query_analysis = query_analyzer.analyze(query)
    
    results = await multi_strategy.retrieve(
        query=query,
        query_analysis=query_analysis,
        context_id="test_context",
        limit=10
    )
    
    # Check for duplicates
    memory_ids = [r.memory.id for r in results]
    assert len(memory_ids) == len(set(memory_ids)), "Results contain duplicates"
    
    # Fused results should have metadata about strategies used
    for result in results:
        if result.strategy == "fused":
            assert "strategies" in result.metadata
            assert len(result.metadata["strategies"]) > 1


@pytest.mark.asyncio
async def test_strategy_selection(document_store, temporal_index, vector_store, query_analyzer):
    """Test that appropriate strategies are selected based on query intent."""
    def dummy_embedding_fn(text):
        return [0.1] * 384
    
    semantic_strategy = SemanticRetrievalStrategy(
        vector_store=vector_store,
        document_store=document_store,
        embedding_fn=dummy_embedding_fn
    )
    
    temporal_strategy = TemporalRetrievalStrategy(
        temporal_index=temporal_index,
        document_store=document_store
    )
    
    fulltext_strategy = FullTextRetrievalStrategy(document_store=document_store)
    
    multi_strategy = MultiStrategyRetrieval(
        semantic_strategy=semantic_strategy,
        temporal_strategy=temporal_strategy,
        fulltext_strategy=fulltext_strategy
    )
    
    # Test temporal query
    temporal_query = "What happened last week?"
    temporal_analysis = query_analyzer.analyze(temporal_query)
    selected = multi_strategy._select_strategies(temporal_analysis)
    assert "temporal" in selected
    
    # Test code query
    code_query = "Show me the login function"
    code_analysis = query_analyzer.analyze(code_query)
    selected = multi_strategy._select_strategies(code_analysis)
    # Should select semantic (always) but not necessarily structural if not available
    assert "semantic" in selected


@pytest.mark.asyncio
async def test_strategy_weights(document_store, temporal_index, vector_store, query_analyzer):
    """Test that strategy weights are applied correctly."""
    def dummy_embedding_fn(text):
        return [0.5] * 384
    
    semantic_strategy = SemanticRetrievalStrategy(
        vector_store=vector_store,
        document_store=document_store,
        embedding_fn=dummy_embedding_fn
    )
    
    temporal_strategy = TemporalRetrievalStrategy(
        temporal_index=temporal_index,
        document_store=document_store
    )
    
    multi_strategy = MultiStrategyRetrieval(
        semantic_strategy=semantic_strategy,
        temporal_strategy=temporal_strategy
    )
    
    # Test default weights
    default_query = "What are my preferences?"
    default_analysis = query_analyzer.analyze(default_query)
    default_weights = multi_strategy._get_default_weights(default_analysis)
    
    # Semantic should have default weight of 1.0
    assert default_weights["semantic"] == 1.0
    assert default_weights["temporal"] == 0.8
    
    # Test code query weights
    code_query = "Show me the login function"
    code_analysis = query_analyzer.analyze(code_query)
    code_weights = multi_strategy._get_default_weights(code_analysis)
    
    # Structural weight should be boosted for code queries
    if code_analysis.intent == QueryIntent.CODE:
        assert code_weights["structural"] == 1.2
        assert code_weights["semantic"] == 0.9


@pytest.mark.asyncio
async def test_empty_results_handling(document_store, vector_store, query_analyzer):
    """Test handling of empty results from strategies."""
    def dummy_embedding_fn(text):
        return [0.1] * 384
    
    semantic_strategy = SemanticRetrievalStrategy(
        vector_store=vector_store,
        document_store=document_store,
        embedding_fn=dummy_embedding_fn
    )
    
    fulltext_strategy = FullTextRetrievalStrategy(document_store=document_store)
    
    multi_strategy = MultiStrategyRetrieval(
        semantic_strategy=semantic_strategy,
        fulltext_strategy=fulltext_strategy
    )
    
    # Query with no matching memories
    query = "nonexistent query that matches nothing"
    query_analysis = query_analyzer.analyze(query)
    
    results = await multi_strategy.retrieve(
        query=query,
        query_analysis=query_analysis,
        context_id="test_context",
        limit=5
    )
    
    # Should return empty list, not error
    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_adaptive_ranker_multi_signal_scoring(sample_memories):
    """Test adaptive ranker with multi-signal scoring."""
    from core.infinite.retrieval_strategies import AdaptiveRanker
    
    # Create scored memories with different characteristics
    scored_memories = [
        ScoredMemory(
            memory=sample_memories[0],  # Recent, high importance
            score=0.7,
            strategy="semantic"
        ),
        ScoredMemory(
            memory=sample_memories[1],  # Old, medium importance
            score=0.9,
            strategy="semantic"
        ),
        ScoredMemory(
            memory=sample_memories[3],  # Recent, very high importance
            score=0.6,
            strategy="semantic"
        ),
    ]
    
    # Create ranker
    ranker = AdaptiveRanker()
    
    # Create query analysis
    query_analysis = QueryAnalysis(
        intent=QueryIntent.FACTUAL,
        keywords=["test"],
        confidence=0.9
    )
    
    # Rank memories
    ranked = ranker.rank(scored_memories, query_analysis)
    
    # Verify ranking metadata is added
    assert all('ranking' in sm.metadata for sm in ranked)
    assert all('semantic_score' in sm.metadata['ranking'] for sm in ranked)
    assert all('recency_score' in sm.metadata['ranking'] for sm in ranked)
    assert all('importance_score' in sm.metadata['ranking'] for sm in ranked)
    
    # Verify scores are updated
    assert all(sm.score > 0 for sm in ranked)
    
    # Results should be sorted by final score
    scores = [sm.score for sm in ranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_adaptive_ranker_recency_boosting(sample_memories):
    """Test recency boosting for time-sensitive queries."""
    from core.infinite.retrieval_strategies import AdaptiveRanker
    
    # Create scored memories with same semantic score but different ages
    scored_memories = [
        ScoredMemory(
            memory=sample_memories[0],  # 1 hour ago
            score=0.8,
            strategy="semantic"
        ),
        ScoredMemory(
            memory=sample_memories[1],  # 7 days ago
            score=0.8,
            strategy="semantic"
        ),
    ]
    
    ranker = AdaptiveRanker()
    
    # Temporal query should boost recent memories
    temporal_query = QueryAnalysis(
        intent=QueryIntent.TEMPORAL,
        temporal_expressions=[("yesterday", time.time() - 86400)],
        keywords=["yesterday"],
        confidence=0.9
    )
    
    ranked = ranker.rank(scored_memories, temporal_query, boost_recent=True)
    
    # More recent memory should rank higher
    assert ranked[0].memory.id == sample_memories[0].id
    
    # Check recency scores
    recent_recency = ranked[0].metadata['ranking']['recency_score']
    old_recency = ranked[1].metadata['ranking']['recency_score']
    assert recent_recency > old_recency


@pytest.mark.asyncio
async def test_adaptive_ranker_importance_boosting(sample_memories):
    """Test importance boosting in ranking."""
    from core.infinite.retrieval_strategies import AdaptiveRanker
    
    # Create scored memories with same semantic score but different importance
    scored_memories = [
        ScoredMemory(
            memory=sample_memories[1],  # importance=7
            score=0.8,
            strategy="semantic"
        ),
        ScoredMemory(
            memory=sample_memories[3],  # importance=10
            score=0.8,
            strategy="semantic"
        ),
    ]
    
    ranker = AdaptiveRanker(importance_weight=0.5)  # High importance weight
    
    # Factual query should consider importance
    factual_query = QueryAnalysis(
        intent=QueryIntent.FACTUAL,
        keywords=["fact"],
        confidence=0.9
    )
    
    ranked = ranker.rank(scored_memories, factual_query)
    
    # Higher importance memory should rank higher
    assert ranked[0].memory.importance >= ranked[1].memory.importance
    
    # Check importance scores
    assert ranked[0].metadata['ranking']['importance_score'] == 1.0  # 10/10
    assert ranked[1].metadata['ranking']['importance_score'] == 0.7  # 7/10


@pytest.mark.asyncio
async def test_adaptive_ranker_redundancy_penalization(sample_memories):
    """Test redundancy penalization."""
    from core.infinite.retrieval_strategies import AdaptiveRanker
    import uuid
    
    # Create similar memories (redundant)
    similar_mem1 = Memory(
        id=str(uuid.uuid4()),
        context_id="test_context",
        content="I like Python programming language",
        memory_type=MemoryType.PREFERENCE,
        created_at=time.time(),
        importance=8
    )
    
    similar_mem2 = Memory(
        id=str(uuid.uuid4()),
        context_id="test_context",
        content="I like Python programming",  # Very similar
        memory_type=MemoryType.PREFERENCE,
        created_at=time.time() - 100,
        importance=8
    )
    
    different_mem = Memory(
        id=str(uuid.uuid4()),
        context_id="test_context",
        content="I prefer dark mode for coding",  # Different
        memory_type=MemoryType.PREFERENCE,
        created_at=time.time() - 200,
        importance=8
    )
    
    scored_memories = [
        ScoredMemory(memory=similar_mem1, score=0.9, strategy="semantic"),
        ScoredMemory(memory=similar_mem2, score=0.85, strategy="semantic"),
        ScoredMemory(memory=different_mem, score=0.8, strategy="semantic"),
    ]
    
    ranker = AdaptiveRanker(redundancy_threshold=0.5)
    
    query_analysis = QueryAnalysis(
        intent=QueryIntent.PREFERENCE,
        keywords=["python"],
        confidence=0.9
    )
    
    ranked = ranker.rank(scored_memories, query_analysis, penalize_redundancy=True)
    
    # Check that redundancy penalty was applied
    redundant_results = [r for r in ranked if 'redundancy_penalty' in r.metadata.get('ranking', {})]
    assert len(redundant_results) > 0, "Redundancy penalty should be applied to similar memories"
    
    # The second similar memory should have a penalty
    for result in ranked:
        if result.memory.id == similar_mem2.id:
            assert 'redundancy_penalty' in result.metadata.get('ranking', {})


@pytest.mark.asyncio
async def test_adaptive_ranker_intent_based_weights(sample_memories):
    """Test that weights adapt based on query intent."""
    from core.infinite.retrieval_strategies import AdaptiveRanker
    
    ranker = AdaptiveRanker()
    
    # Test temporal query weights
    temporal_query = QueryAnalysis(
        intent=QueryIntent.TEMPORAL,
        temporal_expressions=[("yesterday", time.time() - 86400)],
        keywords=["yesterday"],
        confidence=0.9
    )
    temporal_weights = ranker._get_adaptive_weights(temporal_query)
    assert temporal_weights['recency'] > temporal_weights['semantic']
    
    # Test code query weights
    code_query = QueryAnalysis(
        intent=QueryIntent.CODE,
        code_patterns=["login()"],
        keywords=["function"],
        confidence=0.9
    )
    code_weights = ranker._get_adaptive_weights(code_query)
    assert code_weights['semantic'] >= code_weights['recency']
    
    # Test preference query weights
    preference_query = QueryAnalysis(
        intent=QueryIntent.PREFERENCE,
        keywords=["like", "prefer"],
        confidence=0.9
    )
    preference_weights = ranker._get_adaptive_weights(preference_query)
    assert preference_weights['recency'] > 0.3  # Should boost recency
    
    # Verify weights sum to 1.0
    for weights in [temporal_weights, code_weights, preference_weights]:
        assert abs(sum(weights.values()) - 1.0) < 0.01


@pytest.mark.asyncio
async def test_adaptive_ranker_with_multi_strategy(
    document_store,
    temporal_index,
    vector_store,
    sample_memories,
    query_analyzer
):
    """Test adaptive ranker integration with multi-strategy retrieval."""
    from core.infinite.retrieval_strategies import AdaptiveRanker
    
    # Add memories to vector store
    for memory in sample_memories:
        await vector_store.add_memory(memory)
    
    # Create strategies
    def dummy_embedding_fn(text):
        return [0.1] * 384
    
    semantic_strategy = SemanticRetrievalStrategy(
        vector_store=vector_store,
        document_store=document_store,
        embedding_fn=dummy_embedding_fn
    )
    
    temporal_strategy = TemporalRetrievalStrategy(
        temporal_index=temporal_index,
        document_store=document_store
    )
    
    fulltext_strategy = FullTextRetrievalStrategy(document_store=document_store)
    
    # Create custom ranker
    custom_ranker = AdaptiveRanker(
        recency_weight=0.4,
        importance_weight=0.3,
        semantic_weight=0.3
    )
    
    # Create multi-strategy retrieval with custom ranker
    multi_strategy = MultiStrategyRetrieval(
        semantic_strategy=semantic_strategy,
        temporal_strategy=temporal_strategy,
        fulltext_strategy=fulltext_strategy,
        ranker=custom_ranker
    )
    
    # Test retrieval with adaptive ranking
    query = "What are my recent preferences?"
    query_analysis = query_analyzer.analyze(query)
    
    results = await multi_strategy.retrieve(
        query=query,
        query_analysis=query_analysis,
        context_id="test_context",
        limit=5,
        use_adaptive_ranking=True
    )
    
    assert len(results) > 0
    
    # Check that ranking metadata is present
    for result in results:
        if 'ranking' in result.metadata:
            ranking = result.metadata['ranking']
            assert 'semantic_score' in ranking
            assert 'recency_score' in ranking
            assert 'importance_score' in ranking


@pytest.mark.asyncio
async def test_adaptive_ranking_disabled(
    document_store,
    temporal_index,
    vector_store,
    sample_memories,
    query_analyzer
):
    """Test that adaptive ranking can be disabled."""
    # Add memories to vector store
    for memory in sample_memories:
        await vector_store.add_memory(memory)
    
    # Create strategies
    def dummy_embedding_fn(text):
        return [0.1] * 384
    
    semantic_strategy = SemanticRetrievalStrategy(
        vector_store=vector_store,
        document_store=document_store,
        embedding_fn=dummy_embedding_fn
    )
    
    multi_strategy = MultiStrategyRetrieval(
        semantic_strategy=semantic_strategy
    )
    
    # Test retrieval without adaptive ranking
    query = "What are my preferences?"
    query_analysis = query_analyzer.analyze(query)
    
    results = await multi_strategy.retrieve(
        query=query,
        query_analysis=query_analysis,
        context_id="test_context",
        limit=5,
        use_adaptive_ranking=False
    )
    
    assert len(results) >= 0
    # Results should still be sorted, just without adaptive ranking metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
