"""Retrieval orchestrator for intelligent memory retrieval.

This module provides the main orchestration layer that integrates query analysis,
multi-strategy retrieval, adaptive ranking, and result composition with caching
and adaptive search scope expansion.
"""

import asyncio
import time
import logging
import hashlib
from typing import Optional, Callable
from dataclasses import dataclass, field

from .models import Memory, MemoryType, QueryAnalysis, RetrievalResult
from .query_analyzer import QueryAnalyzer
from .retrieval_strategies import (
    MultiStrategyRetrieval,
    SemanticRetrievalStrategy,
    TemporalRetrievalStrategy,
    StructuralRetrievalStrategy,
    FullTextRetrievalStrategy,
    AdaptiveRanker,
    ScoredMemory
)
from .result_composer import ResultComposer
from .document_store import DocumentStore
from .temporal_index import TemporalIndex
from .vector_store import VectorStore
from .code_change_store import CodeChangeStore

logger = logging.getLogger(__name__)


@dataclass
class CachedResult:
    """Cached retrieval result with expiration."""
    result: RetrievalResult
    timestamp: float
    query_hash: str
    metadata: dict = field(default_factory=dict)


class RetrievalOrchestrator:
    """
    Main orchestration layer for intelligent memory retrieval.
    
    Features:
    - Query analysis without AI
    - Multi-strategy retrieval with fusion
    - Adaptive ranking with multi-signal scoring
    - Result composition with breadcrumbs
    - Query result caching
    - Adaptive search scope expansion
    
    Integrates:
    - QueryAnalyzer (Phase 5.1)
    - MultiStrategyRetrieval (Phase 5.2)
    - AdaptiveRanker (Phase 5.3)
    - ResultComposer (Phase 5.4)
    """
    
    def __init__(
        self,
        document_store: DocumentStore,
        temporal_index: TemporalIndex,
        vector_store: VectorStore,
        code_change_store: Optional[CodeChangeStore] = None,
        embedding_fn: Optional[Callable] = None,
        use_spacy: bool = False,
        enable_caching: bool = True,
        cache_ttl_seconds: float = 300.0,
        enable_scope_expansion: bool = True,
        min_results_threshold: int = 3
    ):
        """Initialize retrieval orchestrator.
        
        Args:
            document_store: Document store for memory storage
            temporal_index: Temporal index for time-based queries
            vector_store: Vector store for semantic search
            code_change_store: Optional code change store for structural queries
            embedding_fn: Function to generate embeddings from text
            use_spacy: Whether to use spaCy for enhanced entity extraction
            enable_caching: Whether to enable query result caching
            cache_ttl_seconds: Time-to-live for cached results in seconds
            enable_scope_expansion: Whether to enable adaptive search scope expansion
            min_results_threshold: Minimum results before triggering scope expansion
        """
        self.document_store = document_store
        self.temporal_index = temporal_index
        self.vector_store = vector_store
        self.code_change_store = code_change_store
        self.embedding_fn = embedding_fn
        
        # Initialize query analyzer
        self.query_analyzer = QueryAnalyzer(use_spacy=use_spacy)
        
        # Initialize retrieval strategies
        semantic_strategy = SemanticRetrievalStrategy(
            vector_store=vector_store,
            document_store=document_store,
            embedding_fn=embedding_fn
        )
        
        temporal_strategy = TemporalRetrievalStrategy(
            temporal_index=temporal_index,
            document_store=document_store
        )
        
        fulltext_strategy = FullTextRetrievalStrategy(
            document_store=document_store
        )
        
        structural_strategy = None
        if code_change_store:
            structural_strategy = StructuralRetrievalStrategy(
                code_change_store=code_change_store,
                document_store=document_store
            )
        
        # Initialize adaptive ranker
        self.ranker = AdaptiveRanker()
        
        # Initialize multi-strategy retrieval
        self.multi_strategy = MultiStrategyRetrieval(
            semantic_strategy=semantic_strategy,
            temporal_strategy=temporal_strategy,
            structural_strategy=structural_strategy,
            fulltext_strategy=fulltext_strategy,
            ranker=self.ranker
        )
        
        # Initialize result composer
        self.result_composer = ResultComposer(
            interleave_by_type=True,
            include_breadcrumbs=True
        )
        
        # Caching configuration
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[str, CachedResult] = {}
        
        # Scope expansion configuration
        self.enable_scope_expansion = enable_scope_expansion
        self.min_results_threshold = min_results_threshold
    
    async def retrieve(
        self,
        query: str,
        context_id: str,
        max_results: int = 10,
        memory_types: Optional[list[MemoryType]] = None,
        time_range: Optional[tuple[float, float]] = None,
        include_history: bool = False,
        use_cache: bool = True,
        **kwargs
    ) -> RetrievalResult:
        """Main retrieval interface with full orchestration.
        
        Args:
            query: The query string
            context_id: Context identifier
            max_results: Maximum number of results to return
            memory_types: Optional filter for specific memory types
            time_range: Optional time range filter (start_time, end_time)
            include_history: Whether to include version history
            use_cache: Whether to use cached results if available
            **kwargs: Additional parameters passed to strategies
            
        Returns:
            Complete RetrievalResult with composed data
        """
        start_time = time.time()
        
        # Check cache first
        if self.enable_caching and use_cache:
            cached_result = self._get_cached_result(query, context_id, max_results)
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                # Update retrieval time to include cache lookup
                cached_result.retrieval_time_ms = (time.time() - start_time) * 1000
                return cached_result
        
        # Analyze query
        query_analysis = await self.analyze_query(query)
        logger.info(
            f"Query analysis: intent={query_analysis.intent.value}, "
            f"confidence={query_analysis.confidence:.2f}"
        )
        
        # Prepare retrieval parameters
        retrieval_kwargs = kwargs.copy()
        if time_range:
            retrieval_kwargs['time_range'] = time_range
        if memory_types:
            # Convert to single type if only one specified
            if len(memory_types) == 1:
                retrieval_kwargs['memory_type'] = memory_types[0]
        
        # Execute multi-strategy retrieval
        scored_memories = await self.multi_strategy.retrieve(
            query=query,
            query_analysis=query_analysis,
            context_id=context_id,
            limit=max_results * 2,  # Get more for better ranking
            use_adaptive_ranking=True,
            **retrieval_kwargs
        )
        
        # Check if we need scope expansion
        if self.enable_scope_expansion and len(scored_memories) < self.min_results_threshold:
            logger.info(
                f"Expanding search scope: only {len(scored_memories)} results found, "
                f"threshold is {self.min_results_threshold}"
            )
            scored_memories = await self._expand_search_scope(
                query=query,
                query_analysis=query_analysis,
                context_id=context_id,
                initial_results=scored_memories,
                max_results=max_results,
                **retrieval_kwargs
            )
        
        # Filter by memory types if specified
        if memory_types:
            scored_memories = [
                sm for sm in scored_memories
                if sm.memory.memory_type in memory_types
            ]
        
        # Calculate retrieval time
        retrieval_time_ms = (time.time() - start_time) * 1000
        
        # Compose final result
        result = self.result_composer.compose(
            scored_memories=scored_memories,
            query_analysis=query_analysis,
            retrieval_time_ms=retrieval_time_ms,
            limit=max_results
        )
        
        # Cache the result
        if self.enable_caching:
            self._cache_result(query, context_id, max_results, result)
        
        logger.info(
            f"Retrieved {len(result.memories)} memories in {retrieval_time_ms:.2f}ms "
            f"(total found: {result.total_found})"
        )
        
        return result
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query without AI calls.
        
        Args:
            query: The query string to analyze
            
        Returns:
            QueryAnalysis object with extracted information
        """
        # Use the query analyzer (rule-based, no AI)
        return self.query_analyzer.analyze(query)
    
    async def rank_results(
        self,
        scored_memories: list[ScoredMemory],
        query_analysis: QueryAnalysis,
        boost_recent: bool = True,
        penalize_redundancy: bool = True
    ) -> list[ScoredMemory]:
        """Rank results using adaptive multi-signal scoring.
        
        Args:
            scored_memories: List of memories with initial scores
            query_analysis: Analyzed query information
            boost_recent: Whether to boost recent memories
            penalize_redundancy: Whether to penalize redundant results
            
        Returns:
            Re-ranked list of scored memories
        """
        return self.ranker.rank(
            scored_memories=scored_memories,
            query_analysis=query_analysis,
            boost_recent=boost_recent,
            penalize_redundancy=penalize_redundancy
        )
    
    def _get_query_hash(
        self,
        query: str,
        context_id: str,
        max_results: int
    ) -> str:
        """Generate hash for query caching.
        
        Args:
            query: The query string
            context_id: Context identifier
            max_results: Maximum results
            
        Returns:
            Hash string for cache key
        """
        cache_key = f"{query}|{context_id}|{max_results}"
        return hashlib.sha256(cache_key.encode()).hexdigest()
    
    def _get_cached_result(
        self,
        query: str,
        context_id: str,
        max_results: int
    ) -> Optional[RetrievalResult]:
        """Get cached result if available and not expired.
        
        Args:
            query: The query string
            context_id: Context identifier
            max_results: Maximum results
            
        Returns:
            Cached result or None if not found/expired
        """
        query_hash = self._get_query_hash(query, context_id, max_results)
        
        if query_hash not in self._cache:
            return None
        
        cached = self._cache[query_hash]
        current_time = time.time()
        
        # Check if expired
        if current_time - cached.timestamp > self.cache_ttl_seconds:
            # Remove expired entry
            del self._cache[query_hash]
            return None
        
        return cached.result
    
    def _cache_result(
        self,
        query: str,
        context_id: str,
        max_results: int,
        result: RetrievalResult
    ) -> None:
        """Cache a retrieval result.
        
        Args:
            query: The query string
            context_id: Context identifier
            max_results: Maximum results
            result: The result to cache
        """
        query_hash = self._get_query_hash(query, context_id, max_results)
        
        self._cache[query_hash] = CachedResult(
            result=result,
            timestamp=time.time(),
            query_hash=query_hash,
            metadata={
                "query": query[:100],  # Store truncated query for debugging
                "context_id": context_id,
                "max_results": max_results
            }
        )
        
        # Clean up old cache entries if cache is too large
        if len(self._cache) > 1000:
            self._cleanup_cache()
    
    def _cleanup_cache(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, cached in self._cache.items()
            if current_time - cached.timestamp > self.cache_ttl_seconds
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def _expand_search_scope(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        context_id: str,
        initial_results: list[ScoredMemory],
        max_results: int,
        **kwargs
    ) -> list[ScoredMemory]:
        """Expand search scope when initial results are insufficient.
        
        Strategies for expansion:
        1. Relax score thresholds
        2. Expand time range for temporal queries
        3. Add more retrieval strategies
        4. Broaden semantic search
        
        Args:
            query: The query string
            query_analysis: Analyzed query information
            context_id: Context identifier
            initial_results: Initial retrieval results
            max_results: Maximum results desired
            **kwargs: Additional retrieval parameters
            
        Returns:
            Expanded list of scored memories
        """
        logger.info("Expanding search scope...")
        expanded_results = list(initial_results)
        
        # Strategy 1: Relax semantic search threshold
        if 'min_score' in kwargs:
            # Already tried with custom threshold
            pass
        else:
            # Try with lower threshold
            kwargs_relaxed = kwargs.copy()
            kwargs_relaxed['min_score'] = 0.3  # Lower threshold
            
            additional = await self.multi_strategy.retrieve(
                query=query,
                query_analysis=query_analysis,
                context_id=context_id,
                limit=max_results * 2,
                use_adaptive_ranking=False,  # Don't re-rank yet
                **kwargs_relaxed
            )
            
            # Add new results not already in initial results
            existing_ids = {sm.memory.id for sm in expanded_results}
            for sm in additional:
                if sm.memory.id not in existing_ids:
                    expanded_results.append(sm)
                    existing_ids.add(sm.memory.id)
        
        # Strategy 2: Expand time range for temporal queries
        if query_analysis.temporal_expressions and 'time_range' not in kwargs:
            # Get memories from a broader time range
            current_time = time.time()
            expanded_time_range = (current_time - 90 * 86400, current_time)  # Last 90 days
            
            kwargs_temporal = kwargs.copy()
            kwargs_temporal['time_range'] = expanded_time_range
            
            additional = await self.multi_strategy.retrieve(
                query=query,
                query_analysis=query_analysis,
                context_id=context_id,
                limit=max_results,
                use_adaptive_ranking=False,
                **kwargs_temporal
            )
            
            existing_ids = {sm.memory.id for sm in expanded_results}
            for sm in additional:
                if sm.memory.id not in existing_ids:
                    # Reduce score slightly for expanded scope
                    sm.score *= 0.9
                    expanded_results.append(sm)
                    existing_ids.add(sm.memory.id)
        
        # Strategy 3: Try all memory types if not specified
        if 'memory_type' not in kwargs:
            # Already searching all types
            pass
        
        # Re-rank all expanded results
        if len(expanded_results) > len(initial_results):
            expanded_results = self.ranker.rank(
                scored_memories=expanded_results,
                query_analysis=query_analysis,
                boost_recent=True,
                penalize_redundancy=True
            )
            
            logger.info(
                f"Scope expansion added {len(expanded_results) - len(initial_results)} "
                f"new results"
            )
        
        return expanded_results
    
    def clear_cache(self) -> int:
        """Clear all cached results.
        
        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cache entries")
        return count
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        current_time = time.time()
        expired_count = sum(
            1 for cached in self._cache.values()
            if current_time - cached.timestamp > self.cache_ttl_seconds
        )
        
        return {
            "total_entries": len(self._cache),
            "expired_entries": expired_count,
            "active_entries": len(self._cache) - expired_count,
            "cache_enabled": self.enable_caching,
            "ttl_seconds": self.cache_ttl_seconds
        }
