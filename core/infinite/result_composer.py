"""Result composition for intelligent memory retrieval.

This module provides result interleaving, memory grouping, context breadcrumb
generation, and confidence score calculation for retrieval results.
"""

import time
from typing import Any
from dataclasses import dataclass, field
from collections import defaultdict

from .models import Memory, MemoryType, QueryAnalysis, QueryIntent, RetrievalResult
from .retrieval_strategies import ScoredMemory


@dataclass
class MemoryGroup:
    """A group of memories of the same type."""
    memory_type: MemoryType
    memories: list[ScoredMemory]
    avg_score: float
    total_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextBreadcrumb:
    """Breadcrumb showing how a memory was retrieved."""
    memory_id: str
    retrieval_path: list[str]  # e.g., ["semantic_search", "temporal_filter", "ranked"]
    strategies_used: list[str]
    confidence: float
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ResultComposer:
    """
    Composes retrieval results with intelligent interleaving and grouping.
    
    Features:
    - Result interleaving by relevance and type
    - Memory grouping by type
    - Context breadcrumb generation
    - Confidence score calculation
    """
    
    def __init__(
        self,
        interleave_by_type: bool = True,
        max_per_type: int | None = None,
        include_breadcrumbs: bool = True,
        min_confidence: float = 0.0
    ):
        """Initialize result composer.
        
        Args:
            interleave_by_type: Whether to interleave results by memory type
            max_per_type: Maximum memories per type (None for unlimited)
            include_breadcrumbs: Whether to generate context breadcrumbs
            min_confidence: Minimum confidence threshold for results
        """
        self.interleave_by_type = interleave_by_type
        self.max_per_type = max_per_type
        self.include_breadcrumbs = include_breadcrumbs
        self.min_confidence = min_confidence
    
    def compose(
        self,
        scored_memories: list[ScoredMemory],
        query_analysis: QueryAnalysis,
        retrieval_time_ms: float,
        limit: int | None = None
    ) -> RetrievalResult:
        """Compose final retrieval result with all enhancements.
        
        Args:
            scored_memories: List of scored memories from retrieval
            query_analysis: Analysis of the original query
            retrieval_time_ms: Time taken for retrieval
            limit: Maximum number of results to return
            
        Returns:
            Complete RetrievalResult with composed data
        """
        # Filter by confidence
        filtered_memories = self._filter_by_confidence(scored_memories)
        
        # Group memories by type
        memory_groups = self._group_by_type(filtered_memories)
        
        # Interleave results
        if self.interleave_by_type:
            interleaved = self._interleave_results(memory_groups, limit)
        else:
            # Just sort by score and limit
            # Need to sort the filtered memories first
            sorted_memories = sorted(filtered_memories, key=lambda x: x.score, reverse=True)
            if limit:
                interleaved = sorted_memories[:limit]
            else:
                interleaved = sorted_memories
        
        # Generate breadcrumbs
        breadcrumbs = []
        if self.include_breadcrumbs:
            breadcrumbs = self._generate_breadcrumbs(interleaved, query_analysis)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            interleaved, query_analysis
        )
        
        # Extract plain memories
        memories = [sm.memory for sm in interleaved]
        
        # Build metadata
        metadata = {
            "memory_groups": [
                {
                    "type": group.memory_type.value,
                    "count": group.total_count,
                    "avg_score": group.avg_score
                }
                for group in memory_groups.values()
            ],
            "breadcrumbs": [
                {
                    "memory_id": bc.memory_id,
                    "path": bc.retrieval_path,
                    "strategies": bc.strategies_used,
                    "confidence": bc.confidence,
                    "reasoning": bc.reasoning
                }
                for bc in breadcrumbs
            ],
            "overall_confidence": overall_confidence,
            "composition_strategy": "interleaved" if self.interleave_by_type else "ranked",
            "filtered_count": len(scored_memories) - len(filtered_memories)
        }
        
        return RetrievalResult(
            memories=memories,
            total_found=len(scored_memories),
            query_analysis=query_analysis,
            retrieval_time_ms=retrieval_time_ms,
            metadata=metadata
        )
    
    def _filter_by_confidence(
        self,
        scored_memories: list[ScoredMemory]
    ) -> list[ScoredMemory]:
        """Filter memories by minimum confidence threshold.
        
        Args:
            scored_memories: List of scored memories
            
        Returns:
            Filtered list of memories
        """
        if self.min_confidence <= 0:
            return scored_memories
        
        return [
            sm for sm in scored_memories
            if sm.score >= self.min_confidence
        ]
    
    def _group_by_type(
        self,
        scored_memories: list[ScoredMemory]
    ) -> dict[MemoryType, MemoryGroup]:
        """Group memories by their type.
        
        Args:
            scored_memories: List of scored memories
            
        Returns:
            Dictionary mapping memory type to memory group
        """
        groups: dict[MemoryType, list[ScoredMemory]] = defaultdict(list)
        
        # Group memories
        for sm in scored_memories:
            groups[sm.memory.memory_type].append(sm)
        
        # Create MemoryGroup objects
        memory_groups = {}
        for memory_type, memories in groups.items():
            # Sort by score within group
            memories.sort(key=lambda x: x.score, reverse=True)
            
            # Apply max_per_type limit
            if self.max_per_type:
                memories = memories[:self.max_per_type]
            
            # Calculate average score
            avg_score = sum(sm.score for sm in memories) / len(memories) if memories else 0.0
            
            memory_groups[memory_type] = MemoryGroup(
                memory_type=memory_type,
                memories=memories,
                avg_score=avg_score,
                total_count=len(memories)
            )
        
        return memory_groups
    
    def _interleave_results(
        self,
        memory_groups: dict[MemoryType, MemoryGroup],
        limit: int | None = None
    ) -> list[ScoredMemory]:
        """Interleave memories from different types by relevance.
        
        Uses a round-robin approach weighted by group average scores.
        
        Args:
            memory_groups: Dictionary of memory groups by type
            limit: Maximum number of results
            
        Returns:
            Interleaved list of scored memories
        """
        if not memory_groups:
            return []
        
        # Sort groups by average score
        sorted_groups = sorted(
            memory_groups.values(),
            key=lambda g: g.avg_score,
            reverse=True
        )
        
        # Create iterators for each group
        group_iters = [iter(group.memories) for group in sorted_groups]
        
        # Interleave results
        interleaved = []
        exhausted = set()
        
        while len(group_iters) > len(exhausted):
            if limit and len(interleaved) >= limit:
                break
            
            for i, group_iter in enumerate(group_iters):
                if i in exhausted:
                    continue
                
                try:
                    memory = next(group_iter)
                    interleaved.append(memory)
                    
                    if limit and len(interleaved) >= limit:
                        break
                except StopIteration:
                    exhausted.add(i)
        
        return interleaved
    
    def _generate_breadcrumbs(
        self,
        scored_memories: list[ScoredMemory],
        query_analysis: QueryAnalysis
    ) -> list[ContextBreadcrumb]:
        """Generate context breadcrumbs showing retrieval path.
        
        Args:
            scored_memories: List of scored memories
            query_analysis: Analysis of the query
            
        Returns:
            List of context breadcrumbs
        """
        breadcrumbs = []
        
        for sm in scored_memories:
            # Build retrieval path
            retrieval_path = self._build_retrieval_path(sm, query_analysis)
            
            # Get strategies used
            strategies_used = self._extract_strategies(sm)
            
            # Calculate confidence for this result
            confidence = self._calculate_result_confidence(sm, query_analysis)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(sm, query_analysis)
            
            breadcrumb = ContextBreadcrumb(
                memory_id=sm.memory.id,
                retrieval_path=retrieval_path,
                strategies_used=strategies_used,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    "score": sm.score,
                    "strategy": sm.strategy,
                    "memory_type": sm.memory.memory_type.value
                }
            )
            breadcrumbs.append(breadcrumb)
        
        return breadcrumbs
    
    def _build_retrieval_path(
        self,
        scored_memory: ScoredMemory,
        query_analysis: QueryAnalysis
    ) -> list[str]:
        """Build the retrieval path for a memory.
        
        Args:
            scored_memory: Scored memory
            query_analysis: Query analysis
            
        Returns:
            List of steps in retrieval path
        """
        path = []
        
        # Start with query analysis
        path.append(f"query_intent:{query_analysis.intent.value}")
        
        # Add strategy used
        if scored_memory.strategy:
            path.append(f"strategy:{scored_memory.strategy}")
        
        # Add ranking if present
        if "ranking" in scored_memory.metadata:
            path.append("adaptive_ranking")
        
        # Add fusion if multiple strategies
        if "strategies" in scored_memory.metadata:
            path.append("multi_strategy_fusion")
        
        # Add final selection
        path.append("selected")
        
        return path
    
    def _extract_strategies(self, scored_memory: ScoredMemory) -> list[str]:
        """Extract strategies used to retrieve this memory.
        
        Args:
            scored_memory: Scored memory
            
        Returns:
            List of strategy names
        """
        # Check if multiple strategies were fused
        if "strategies" in scored_memory.metadata:
            return scored_memory.metadata["strategies"]
        
        # Single strategy
        if scored_memory.strategy:
            return [scored_memory.strategy]
        
        return ["unknown"]
    
    def _calculate_result_confidence(
        self,
        scored_memory: ScoredMemory,
        query_analysis: QueryAnalysis
    ) -> float:
        """Calculate confidence score for a single result.
        
        Confidence is based on:
        - Retrieval score
        - Query analysis confidence
        - Number of strategies that found this result
        - Memory importance
        
        Args:
            scored_memory: Scored memory
            query_analysis: Query analysis
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from retrieval score
        confidence = scored_memory.score * 0.5
        
        # Add query analysis confidence
        confidence += query_analysis.confidence * 0.2
        
        # Boost for multiple strategies (consensus)
        if "strategies" in scored_memory.metadata:
            num_strategies = len(scored_memory.metadata["strategies"])
            consensus_boost = min(num_strategies * 0.1, 0.2)
            confidence += consensus_boost
        
        # Boost for high importance
        importance_boost = (scored_memory.memory.importance / 10.0) * 0.1
        confidence += importance_boost
        
        return min(confidence, 1.0)
    
    def _calculate_overall_confidence(
        self,
        scored_memories: list[ScoredMemory],
        query_analysis: QueryAnalysis
    ) -> float:
        """Calculate overall confidence for the entire result set.
        
        Args:
            scored_memories: List of scored memories
            query_analysis: Query analysis
            
        Returns:
            Overall confidence score (0-1)
        """
        if not scored_memories:
            return 0.0
        
        # Average of top results (weighted toward top)
        top_n = min(5, len(scored_memories))
        top_scores = [sm.score for sm in scored_memories[:top_n]]
        
        # Weighted average (top result has more weight)
        weights = [1.0 / (i + 1) for i in range(top_n)]
        weighted_sum = sum(score * weight for score, weight in zip(top_scores, weights))
        weight_sum = sum(weights)
        
        avg_top_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        # Factor in query analysis confidence
        confidence = avg_top_score * 0.7 + query_analysis.confidence * 0.3
        
        # Penalize if very few results
        if len(scored_memories) < 3:
            confidence *= 0.8
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(
        self,
        scored_memory: ScoredMemory,
        query_analysis: QueryAnalysis
    ) -> str:
        """Generate human-readable reasoning for why this result was selected.
        
        Args:
            scored_memory: Scored memory
            query_analysis: Query analysis
            
        Returns:
            Reasoning string
        """
        reasons = []
        
        # Strategy-based reasoning
        strategy = scored_memory.strategy
        if strategy == "semantic":
            reasons.append("semantically similar to query")
        elif strategy == "temporal":
            reasons.append("temporally relevant")
        elif strategy == "structural":
            reasons.append("matches code structure")
        elif strategy == "fulltext":
            reasons.append("contains matching keywords")
        elif strategy == "fused":
            strategies = scored_memory.metadata.get("strategies", [])
            reasons.append(f"found by multiple strategies ({', '.join(strategies)})")
        
        # Score-based reasoning
        if scored_memory.score >= 0.9:
            reasons.append("very high relevance")
        elif scored_memory.score >= 0.7:
            reasons.append("high relevance")
        elif scored_memory.score >= 0.5:
            reasons.append("moderate relevance")
        
        # Importance-based reasoning
        if scored_memory.memory.importance >= 8:
            reasons.append("marked as important")
        
        # Recency-based reasoning
        if "ranking" in scored_memory.metadata:
            ranking = scored_memory.metadata["ranking"]
            if ranking.get("recency_score", 0) >= 0.8:
                reasons.append("recent memory")
        
        # Memory type reasoning
        memory_type = scored_memory.memory.memory_type
        if query_analysis.intent == QueryIntent.CODE and memory_type == MemoryType.CODE:
            reasons.append("matches code query intent")
        elif query_analysis.intent == QueryIntent.PREFERENCE and memory_type == MemoryType.PREFERENCE:
            reasons.append("matches preference query intent")
        
        return "; ".join(reasons) if reasons else "retrieved by system"
    
    def group_memories_by_type(
        self,
        memories: list[Memory]
    ) -> dict[MemoryType, list[Memory]]:
        """Group plain memories by type (utility method).
        
        Args:
            memories: List of memories
            
        Returns:
            Dictionary mapping memory type to list of memories
        """
        groups: dict[MemoryType, list[Memory]] = defaultdict(list)
        for memory in memories:
            groups[memory.memory_type].append(memory)
        return dict(groups)
