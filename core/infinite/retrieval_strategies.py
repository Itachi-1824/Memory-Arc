"""Multi-strategy retrieval for infinite context system.

This module implements different retrieval strategies (semantic, temporal,
structural, full-text) and provides result fusion and deduplication.
"""

import asyncio
import time
import logging
from typing import Any, Protocol, Callable, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .models import Memory, MemoryType, QueryAnalysis, QueryIntent
from .document_store import DocumentStore
from .temporal_index import TemporalIndex
from .vector_store import VectorStore
from .code_change_store import CodeChangeStore

logger = logging.getLogger(__name__)


@dataclass
class ScoredMemory:
    """Memory with relevance score from retrieval."""
    memory: Memory
    score: float
    strategy: str  # Which strategy retrieved this
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalStrategy(ABC):
    """Base class for retrieval strategies."""
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        context_id: str,
        limit: int = 10,
        **kwargs
    ) -> list[ScoredMemory]:
        """Retrieve memories using this strategy.
        
        Args:
            query: The query string
            query_analysis: Analyzed query information
            context_id: Context identifier
            limit: Maximum number of results
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of scored memories
        """
        pass


class SemanticRetrievalStrategy(RetrievalStrategy):
    """Semantic search using vector embeddings."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        document_store: DocumentStore,
        embedding_fn: Optional[Callable] = None
    ):
        """Initialize semantic retrieval.
        
        Args:
            vector_store: Vector store for semantic search
            document_store: Document store to fetch full memories
            embedding_fn: Function to generate embeddings from text
        """
        self.vector_store = vector_store
        self.document_store = document_store
        self.embedding_fn = embedding_fn
    
    async def retrieve(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        context_id: str,
        limit: int = 10,
        **kwargs
    ) -> list[ScoredMemory]:
        """Retrieve memories using semantic similarity.
        
        Args:
            query: The query string
            query_analysis: Analyzed query information
            context_id: Context identifier
            limit: Maximum number of results
            **kwargs: Can include 'query_vector', 'memory_type', 'min_score'
            
        Returns:
            List of scored memories
        """
        # Get query vector
        query_vector = kwargs.get('query_vector')
        if query_vector is None:
            if self.embedding_fn is None:
                logger.warning("No embedding function provided for semantic search")
                return []
            query_vector = await self._generate_embedding(query)
        
        # Determine memory type filter from query analysis
        memory_type = kwargs.get('memory_type')
        if memory_type is None and query_analysis.intent == QueryIntent.CODE:
            memory_type = MemoryType.CODE
        elif memory_type is None and query_analysis.intent == QueryIntent.PREFERENCE:
            memory_type = MemoryType.PREFERENCE
        
        # Search vector store
        min_score = kwargs.get('min_score', 0.5)
        results = await self.vector_store.search(
            query_vector=query_vector,
            memory_type=memory_type,
            context_id=context_id,
            limit=limit * 2,  # Get more to account for filtering
            min_score=min_score
        )
        
        # Fetch full memories
        scored_memories = []
        for memory_id, score in results[:limit]:
            memory = await self.document_store.get_memory(memory_id)
            if memory:
                scored_memories.append(ScoredMemory(
                    memory=memory,
                    score=score,
                    strategy="semantic",
                    metadata={"similarity": score}
                ))
        
        return scored_memories
    
    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if asyncio.iscoroutinefunction(self.embedding_fn):
            return await self.embedding_fn(text)
        else:
            return await asyncio.to_thread(self.embedding_fn, text)


class TemporalRetrievalStrategy(RetrievalStrategy):
    """Temporal search based on time ranges and recency."""
    
    def __init__(
        self,
        temporal_index: TemporalIndex,
        document_store: DocumentStore
    ):
        """Initialize temporal retrieval.
        
        Args:
            temporal_index: Temporal index for time-based queries
            document_store: Document store to fetch full memories
        """
        self.temporal_index = temporal_index
        self.document_store = document_store
    
    async def retrieve(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        context_id: str,
        limit: int = 10,
        **kwargs
    ) -> list[ScoredMemory]:
        """Retrieve memories based on temporal relevance.
        
        Args:
            query: The query string
            query_analysis: Analyzed query information
            context_id: Context identifier
            limit: Maximum number of results
            **kwargs: Can include 'time_range', 'boost_recent'
            
        Returns:
            List of scored memories
        """
        # Determine time range from query analysis or kwargs
        time_range = kwargs.get('time_range')
        
        if time_range is None and query_analysis.temporal_expressions:
            # Use the first temporal expression
            _, timestamp = query_analysis.temporal_expressions[0]
            # Create a range around this timestamp (e.g., ±1 day)
            time_range = (timestamp - 86400, timestamp + 86400)
        
        if time_range is None:
            # Default to recent memories (last 30 days)
            current_time = time.time()
            time_range = (current_time - 30 * 86400, current_time)
        
        # Query temporal index
        events = await self.temporal_index.query_by_time_range(
            start_time=time_range[0],
            end_time=time_range[1],
            event_type="created",
            limit=limit * 2
        )
        
        # Fetch memories and score by recency
        scored_memories = []
        current_time = time.time()
        boost_recent = kwargs.get('boost_recent', True)
        
        for event in events:
            memory = await self.document_store.get_memory(event['memory_id'])
            if memory and memory.context_id == context_id:
                # Score based on recency (more recent = higher score)
                if boost_recent:
                    age_days = (current_time - memory.created_at) / 86400
                    # Exponential decay: score = e^(-age/30)
                    score = 2.71828 ** (-age_days / 30)
                else:
                    # Uniform score for temporal matches
                    score = 0.8
                
                scored_memories.append(ScoredMemory(
                    memory=memory,
                    score=score,
                    strategy="temporal",
                    metadata={"timestamp": memory.created_at, "age_days": age_days if boost_recent else None}
                ))
        
        # Sort by score and limit
        scored_memories.sort(key=lambda x: x.score, reverse=True)
        return scored_memories[:limit]


class StructuralRetrievalStrategy(RetrievalStrategy):
    """Structural search for code using AST and symbols."""
    
    def __init__(
        self,
        code_change_store: CodeChangeStore,
        document_store: DocumentStore
    ):
        """Initialize structural retrieval.
        
        Args:
            code_change_store: Code change store for structural queries
            document_store: Document store to fetch full memories
        """
        self.code_change_store = code_change_store
        self.document_store = document_store
    
    async def retrieve(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        context_id: str,
        limit: int = 10,
        **kwargs
    ) -> list[ScoredMemory]:
        """Retrieve code memories using structural patterns.
        
        Args:
            query: The query string
            query_analysis: Analyzed query information
            context_id: Context identifier
            limit: Maximum number of results
            **kwargs: Can include 'file_path', 'function_name'
            
        Returns:
            List of scored memories
        """
        # Only applicable for code queries
        if query_analysis.intent != QueryIntent.CODE and not query_analysis.code_patterns:
            return []
        
        # Extract search parameters
        file_path = kwargs.get('file_path')
        function_name = kwargs.get('function_name')
        
        # If not provided, try to extract from code patterns
        if not file_path and not function_name:
            for pattern in query_analysis.code_patterns:
                if '.' in pattern and any(pattern.endswith(ext) for ext in ['.py', '.js', '.ts', '.java']):
                    file_path = pattern
                elif '(' in pattern:
                    # Likely a function call
                    function_name = pattern.split('(')[0]
        
        # Query code changes
        try:
            changes = await self.code_change_store.query_changes(
                file_path=file_path,
                function_name=function_name,
                limit=limit
            )
            
            # Convert to scored memories
            scored_memories = []
            for change in changes:
                # Create a memory-like representation of the code change
                # In a real implementation, we'd fetch associated memories
                # For now, we'll create a synthetic score based on relevance
                score = 0.7  # Base score for structural matches
                
                # Boost if multiple patterns match
                pattern_matches = sum(1 for p in query_analysis.code_patterns if p in change.file_path)
                score += pattern_matches * 0.1
                
                scored_memories.append(ScoredMemory(
                    memory=Memory(
                        id=change.id,
                        context_id=context_id,
                        content=f"Code change in {change.file_path}",
                        memory_type=MemoryType.CODE,
                        created_at=change.timestamp,
                        metadata=change.metadata
                    ),
                    score=min(score, 1.0),
                    strategy="structural",
                    metadata={"file_path": change.file_path, "change_type": change.change_type}
                ))
            
            return scored_memories[:limit]
        
        except Exception as e:
            logger.error(f"Structural retrieval failed: {e}")
            return []


class FullTextRetrievalStrategy(RetrievalStrategy):
    """Full-text search using keyword matching."""
    
    def __init__(self, document_store: DocumentStore):
        """Initialize full-text retrieval.
        
        Args:
            document_store: Document store for keyword search
        """
        self.document_store = document_store
    
    async def retrieve(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        context_id: str,
        limit: int = 10,
        **kwargs
    ) -> list[ScoredMemory]:
        """Retrieve memories using keyword matching.
        
        Args:
            query: The query string
            query_analysis: Analyzed query information
            context_id: Context identifier
            limit: Maximum number of results
            **kwargs: Can include 'memory_type'
            
        Returns:
            List of scored memories
        """
        # Get all memories in context (with reasonable limit)
        memory_type = kwargs.get('memory_type')
        memories = await self.document_store.query_memories(
            context_id=context_id,
            memory_type=memory_type,
            limit=1000  # Search through more memories
        )
        
        # Score based on keyword matches
        keywords = query_analysis.keywords
        if not keywords:
            # Fall back to simple tokenization
            keywords = query.lower().split()
        
        scored_memories = []
        for memory in memories:
            content_lower = memory.content.lower()
            
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            
            if matches > 0:
                # Score based on match ratio and importance
                score = (matches / len(keywords)) * 0.6 + (memory.importance / 10) * 0.4
                
                scored_memories.append(ScoredMemory(
                    memory=memory,
                    score=min(score, 1.0),
                    strategy="fulltext",
                    metadata={"keyword_matches": matches, "total_keywords": len(keywords)}
                ))
        
        # Sort by score and limit
        scored_memories.sort(key=lambda x: x.score, reverse=True)
        return scored_memories[:limit]


class AdaptiveRanker:
    """Adaptive ranking system with multi-signal scoring.
    
    Features:
    - Multi-signal relevance scoring (semantic, temporal, importance)
    - Recency boosting for time-sensitive queries
    - Importance boosting
    - Redundancy penalization
    - Adaptive ranking based on query intent
    """
    
    def __init__(
        self,
        recency_weight: float = 0.3,
        importance_weight: float = 0.2,
        semantic_weight: float = 0.5,
        redundancy_threshold: float = 0.85
    ):
        """Initialize adaptive ranker.
        
        Args:
            recency_weight: Weight for recency signal (0-1)
            importance_weight: Weight for importance signal (0-1)
            semantic_weight: Weight for semantic similarity signal (0-1)
            redundancy_threshold: Similarity threshold for redundancy detection
        """
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.semantic_weight = semantic_weight
        self.redundancy_threshold = redundancy_threshold
    
    def rank(
        self,
        scored_memories: list[ScoredMemory],
        query_analysis: QueryAnalysis,
        boost_recent: bool = True,
        penalize_redundancy: bool = True
    ) -> list[ScoredMemory]:
        """Rank memories using adaptive multi-signal scoring.
        
        Args:
            scored_memories: List of memories with initial scores
            query_analysis: Analyzed query information
            boost_recent: Whether to boost recent memories
            penalize_redundancy: Whether to penalize redundant results
            
        Returns:
            Re-ranked list of scored memories
        """
        if not scored_memories:
            return []
        
        # Adjust weights based on query intent
        weights = self._get_adaptive_weights(query_analysis)
        
        # Compute multi-signal scores
        current_time = time.time()
        for scored_memory in scored_memories:
            memory = scored_memory.memory
            
            # Semantic score (from retrieval strategy)
            semantic_score = scored_memory.score
            
            # Recency score
            recency_score = self._compute_recency_score(
                memory.created_at,
                current_time,
                boost_recent and self._is_time_sensitive(query_analysis)
            )
            
            # Importance score (normalized to 0-1)
            importance_score = memory.importance / 10.0
            
            # Combine signals with adaptive weights
            final_score = (
                weights['semantic'] * semantic_score +
                weights['recency'] * recency_score +
                weights['importance'] * importance_score
            )
            
            # Store component scores in metadata
            scored_memory.metadata['ranking'] = {
                'semantic_score': semantic_score,
                'recency_score': recency_score,
                'importance_score': importance_score,
                'weights': weights,
                'final_score': final_score
            }
            
            # Update the score
            scored_memory.score = final_score
        
        # Sort by score
        scored_memories.sort(key=lambda x: x.score, reverse=True)
        
        # Apply redundancy penalization
        if penalize_redundancy:
            scored_memories = self._penalize_redundancy(scored_memories)
        
        return scored_memories
    
    def _get_adaptive_weights(self, query_analysis: QueryAnalysis) -> dict[str, float]:
        """Get adaptive weights based on query intent.
        
        Args:
            query_analysis: Analyzed query information
            
        Returns:
            Dictionary of signal weights
        """
        # Start with default weights
        weights = {
            'semantic': self.semantic_weight,
            'recency': self.recency_weight,
            'importance': self.importance_weight
        }
        
        # Adjust based on query intent
        if query_analysis.intent == QueryIntent.TEMPORAL:
            # Boost recency for temporal queries
            weights['recency'] = 0.5
            weights['semantic'] = 0.3
            weights['importance'] = 0.2
        
        elif query_analysis.intent == QueryIntent.CODE:
            # Boost semantic and importance for code queries
            weights['semantic'] = 0.5
            weights['importance'] = 0.3
            weights['recency'] = 0.2
        
        elif query_analysis.intent == QueryIntent.PREFERENCE:
            # Boost recency and semantic for preference queries
            weights['recency'] = 0.4
            weights['semantic'] = 0.4
            weights['importance'] = 0.2
        
        elif query_analysis.intent == QueryIntent.FACTUAL:
            # Boost importance and semantic for factual queries
            weights['semantic'] = 0.5
            weights['importance'] = 0.3
            weights['recency'] = 0.2
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _compute_recency_score(
        self,
        created_at: float,
        current_time: float,
        boost: bool = True
    ) -> float:
        """Compute recency score for a memory.
        
        Args:
            created_at: Memory creation timestamp
            current_time: Current timestamp
            boost: Whether to apply exponential boost
            
        Returns:
            Recency score (0-1)
        """
        age_seconds = current_time - created_at
        age_days = age_seconds / 86400
        
        if not boost:
            # Linear decay over 30 days
            return max(0.0, 1.0 - (age_days / 30))
        
        # Exponential decay with configurable half-life
        # Recent memories get significant boost
        half_life_days = 7  # Score halves every 7 days
        decay_factor = 2 ** (-age_days / half_life_days)
        
        return min(1.0, decay_factor)
    
    def _is_time_sensitive(self, query_analysis: QueryAnalysis) -> bool:
        """Determine if query is time-sensitive.
        
        Args:
            query_analysis: Analyzed query information
            
        Returns:
            True if query is time-sensitive
        """
        # Temporal queries are obviously time-sensitive
        if query_analysis.intent == QueryIntent.TEMPORAL:
            return True
        
        # Queries with temporal expressions are time-sensitive
        if query_analysis.temporal_expressions:
            return True
        
        # Preference queries often care about recent preferences
        if query_analysis.intent == QueryIntent.PREFERENCE:
            return True
        
        return False
    
    def _penalize_redundancy(
        self,
        scored_memories: list[ScoredMemory]
    ) -> list[ScoredMemory]:
        """Penalize redundant memories to promote diversity.
        
        Args:
            scored_memories: Sorted list of scored memories
            
        Returns:
            List with redundancy penalties applied
        """
        if len(scored_memories) <= 1:
            return scored_memories
        
        # Track selected memories
        selected = []
        
        for i, scored_memory in enumerate(scored_memories):
            # Check similarity with already selected memories
            redundancy_penalty = 0.0
            
            for selected_memory in selected:
                similarity = self._compute_content_similarity(
                    scored_memory.memory.content,
                    selected_memory.memory.content
                )
                
                if similarity >= self.redundancy_threshold:
                    # Apply penalty proportional to similarity
                    redundancy_penalty += (similarity - self.redundancy_threshold) * 0.5
            
            # Apply penalty (reduce score)
            if redundancy_penalty > 0:
                original_score = scored_memory.score
                scored_memory.score = max(0.0, original_score - redundancy_penalty)
                
                # Track penalty in metadata
                if 'ranking' not in scored_memory.metadata:
                    scored_memory.metadata['ranking'] = {}
                scored_memory.metadata['ranking']['redundancy_penalty'] = redundancy_penalty
                scored_memory.metadata['ranking']['original_score'] = original_score
            
            selected.append(scored_memory)
        
        # Re-sort after applying penalties
        scored_memories.sort(key=lambda x: x.score, reverse=True)
        
        return scored_memories
    
    def _compute_content_similarity(self, content1: str, content2: str) -> float:
        """Compute similarity between two content strings.
        
        Uses simple token-based Jaccard similarity for efficiency.
        
        Args:
            content1: First content string
            content2: Second content string
            
        Returns:
            Similarity score (0-1)
        """
        # Tokenize and normalize
        tokens1 = set(content1.lower().split())
        tokens2 = set(content2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0


class MultiStrategyRetrieval:
    """Orchestrates multiple retrieval strategies with fusion and deduplication."""
    
    def __init__(
        self,
        semantic_strategy: Optional[SemanticRetrievalStrategy] = None,
        temporal_strategy: Optional[TemporalRetrievalStrategy] = None,
        structural_strategy: Optional[StructuralRetrievalStrategy] = None,
        fulltext_strategy: Optional[FullTextRetrievalStrategy] = None,
        ranker: Optional[AdaptiveRanker] = None
    ):
        """Initialize multi-strategy retrieval.
        
        Args:
            semantic_strategy: Semantic retrieval strategy
            temporal_strategy: Temporal retrieval strategy
            structural_strategy: Structural retrieval strategy
            fulltext_strategy: Full-text retrieval strategy
            ranker: Adaptive ranker for result ranking (creates default if None)
        """
        self.strategies = {}
        
        if semantic_strategy:
            self.strategies['semantic'] = semantic_strategy
        if temporal_strategy:
            self.strategies['temporal'] = temporal_strategy
        if structural_strategy:
            self.strategies['structural'] = structural_strategy
        if fulltext_strategy:
            self.strategies['fulltext'] = fulltext_strategy
        
        # Initialize adaptive ranker
        self.ranker = ranker if ranker is not None else AdaptiveRanker()
    
    async def retrieve(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        context_id: str,
        limit: int = 10,
        strategy_weights: Optional[dict[str, float]] = None,
        use_adaptive_ranking: bool = True,
        **kwargs
    ) -> list[ScoredMemory]:
        """Retrieve memories using multiple strategies with fusion.
        
        Args:
            query: The query string
            query_analysis: Analyzed query information
            context_id: Context identifier
            limit: Maximum number of results
            strategy_weights: Optional weights for each strategy
            use_adaptive_ranking: Whether to use adaptive ranking (default: True)
            **kwargs: Additional parameters passed to strategies
            
        Returns:
            List of deduplicated and fused scored memories
        """
        # Select strategies based on query intent
        selected_strategies = self._select_strategies(query_analysis)
        
        if not selected_strategies:
            # Fall back to all available strategies
            selected_strategies = list(self.strategies.keys())
        
        # Default weights if not provided
        if strategy_weights is None:
            strategy_weights = self._get_default_weights(query_analysis)
        
        # Execute strategies in parallel
        tasks = []
        for strategy_name in selected_strategies:
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                tasks.append(self._execute_strategy(
                    strategy_name, strategy, query, query_analysis, context_id, limit, kwargs
                ))
        
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results from all strategies
        all_scored_memories = []
        for strategy_name, result in zip(selected_strategies, results):
            if isinstance(result, Exception):
                logger.error(f"Strategy {strategy_name} failed: {result}")
                continue
            
            # Apply strategy weight
            weight = strategy_weights.get(strategy_name, 1.0)
            for scored_memory in result:
                scored_memory.score *= weight
                all_scored_memories.append(scored_memory)
        
        # Deduplicate and fuse
        fused_memories = self._fuse_results(all_scored_memories)
        
        # Apply adaptive ranking if enabled
        if use_adaptive_ranking:
            fused_memories = self.ranker.rank(
                scored_memories=fused_memories,
                query_analysis=query_analysis,
                boost_recent=True,
                penalize_redundancy=True
            )
        else:
            # Sort by final score
            fused_memories.sort(key=lambda x: x.score, reverse=True)
        
        return fused_memories[:limit]
    
    async def _execute_strategy(
        self,
        strategy_name: str,
        strategy: RetrievalStrategy,
        query: str,
        query_analysis: QueryAnalysis,
        context_id: str,
        limit: int,
        kwargs: dict
    ) -> list[ScoredMemory]:
        """Execute a single strategy."""
        try:
            return await strategy.retrieve(
                query=query,
                query_analysis=query_analysis,
                context_id=context_id,
                limit=limit,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Strategy {strategy_name} execution failed: {e}")
            return []
    
    def _select_strategies(self, query_analysis: QueryAnalysis) -> list[str]:
        """Select appropriate strategies based on query analysis.
        
        Args:
            query_analysis: Analyzed query information
            
        Returns:
            List of strategy names to use
        """
        strategies = []
        
        # Always use semantic if available
        if 'semantic' in self.strategies:
            strategies.append('semantic')
        
        # Use temporal for temporal queries
        if query_analysis.intent == QueryIntent.TEMPORAL or query_analysis.temporal_expressions:
            if 'temporal' in self.strategies:
                strategies.append('temporal')
        
        # Use structural for code queries
        if query_analysis.intent == QueryIntent.CODE or query_analysis.code_patterns:
            if 'structural' in self.strategies:
                strategies.append('structural')
        
        # Use full-text as fallback or for factual queries
        if query_analysis.intent in (QueryIntent.FACTUAL, QueryIntent.CONVERSATIONAL):
            if 'fulltext' in self.strategies:
                strategies.append('fulltext')
        
        # For mixed queries, use all available strategies
        if query_analysis.intent == QueryIntent.MIXED:
            strategies = list(self.strategies.keys())
        
        return strategies
    
    def _get_default_weights(self, query_analysis: QueryAnalysis) -> dict[str, float]:
        """Get default strategy weights based on query analysis.
        
        Args:
            query_analysis: Analyzed query information
            
        Returns:
            Dictionary of strategy weights
        """
        weights = {
            'semantic': 1.0,
            'temporal': 0.8,
            'structural': 0.9,
            'fulltext': 0.6
        }
        
        # Boost weights based on query intent
        if query_analysis.intent == QueryIntent.TEMPORAL:
            weights['temporal'] = 1.2
        elif query_analysis.intent == QueryIntent.CODE:
            weights['structural'] = 1.2
            weights['semantic'] = 0.9
        elif query_analysis.intent == QueryIntent.FACTUAL:
            weights['fulltext'] = 0.9
        
        return weights
    
    def _fuse_results(self, scored_memories: list[ScoredMemory]) -> list[ScoredMemory]:
        """Fuse and deduplicate results from multiple strategies.
        
        Args:
            scored_memories: List of scored memories from all strategies
            
        Returns:
            Deduplicated list with fused scores
        """
        # Group by memory ID
        memory_groups: dict[str, list[ScoredMemory]] = {}
        
        for scored_memory in scored_memories:
            memory_id = scored_memory.memory.id
            if memory_id not in memory_groups:
                memory_groups[memory_id] = []
            memory_groups[memory_id].append(scored_memory)
        
        # Fuse scores for duplicates
        fused_memories = []
        for memory_id, group in memory_groups.items():
            if len(group) == 1:
                # No duplicates, use as-is
                fused_memories.append(group[0])
            else:
                # Multiple strategies retrieved this memory
                # Use weighted average with boost for consensus
                strategies_used = [sm.strategy for sm in group]
                scores = [sm.score for sm in group]
                
                # Average score with consensus boost
                avg_score = sum(scores) / len(scores)
                consensus_boost = 0.1 * (len(group) - 1)  # Boost for each additional strategy
                final_score = min(avg_score + consensus_boost, 1.0)
                
                # Use the first memory but update score and metadata
                fused = group[0]
                fused.score = final_score
                fused.metadata['strategies'] = strategies_used
                fused.metadata['strategy_scores'] = dict(zip(strategies_used, scores))
                fused.strategy = "fused"
                
                fused_memories.append(fused)
        
        return fused_memories
