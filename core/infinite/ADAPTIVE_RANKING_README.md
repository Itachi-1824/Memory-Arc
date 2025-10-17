# Adaptive Ranking System

## Overview

The Adaptive Ranking System is a sophisticated multi-signal scoring mechanism that intelligently ranks retrieved memories based on multiple factors. It goes beyond simple semantic similarity to provide context-aware, intent-driven ranking that adapts to different query types.

## Key Features

### 1. Multi-Signal Relevance Scoring

The ranker combines three primary signals:

- **Semantic Score**: Similarity between query and memory content (from retrieval strategies)
- **Recency Score**: How recent the memory is (with exponential decay)
- **Importance Score**: User-assigned importance level (0-10 scale)

Each signal is weighted and combined to produce a final relevance score.

### 2. Adaptive Weight Adjustment

Weights automatically adjust based on query intent:

| Query Intent | Semantic | Recency | Importance | Use Case |
|--------------|----------|---------|------------|----------|
| **Temporal** | 0.3 | 0.5 | 0.2 | "What happened yesterday?" |
| **Code** | 0.5 | 0.2 | 0.3 | "Show me the login function" |
| **Preference** | 0.4 | 0.4 | 0.2 | "What do I like?" |
| **Factual** | 0.5 | 0.2 | 0.3 | "What is the deadline?" |
| **Default** | 0.5 | 0.3 | 0.2 | General queries |

### 3. Recency Boosting

For time-sensitive queries, the system applies exponential decay:

```python
# Exponential decay with 7-day half-life
recency_score = 2^(-age_days / 7)
```

This means:
- Memories from today: score ≈ 1.0
- Memories from 7 days ago: score ≈ 0.5
- Memories from 14 days ago: score ≈ 0.25
- Memories from 30 days ago: score ≈ 0.09

### 4. Importance Boosting

Memories with higher importance (0-10 scale) receive proportional boosts:

```python
importance_score = importance / 10.0
```

Critical memories (importance=10) get maximum boost, while routine memories (importance=5) get moderate boost.

### 5. Redundancy Penalization

To promote diversity in results, the system penalizes redundant memories:

1. Computes content similarity using Jaccard similarity
2. If similarity exceeds threshold (default: 0.85), applies penalty
3. Penalty is proportional to similarity: `penalty = (similarity - threshold) * 0.5`
4. Reduces score of later-ranked similar memories

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AdaptiveRanker                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Analyze Query Intent                             │  │
│  │     - Temporal, Code, Preference, Factual, etc.      │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  2. Adjust Signal Weights                            │  │
│  │     - Boost relevant signals for query type          │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  3. Compute Multi-Signal Scores                      │  │
│  │     - Semantic: from retrieval strategy              │  │
│  │     - Recency: exponential decay                     │  │
│  │     - Importance: normalized 0-1                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  4. Combine Signals                                  │  │
│  │     final_score = Σ(weight_i × signal_i)            │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  5. Apply Redundancy Penalties                       │  │
│  │     - Compare with already-ranked memories           │  │
│  │     - Penalize similar content                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  6. Sort by Final Score                              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Usage

```python
from core.infinite.retrieval_strategies import AdaptiveRanker, ScoredMemory
from core.infinite.models import QueryAnalysis, QueryIntent

# Create ranker with default weights
ranker = AdaptiveRanker()

# Rank memories
ranked_memories = ranker.rank(
    scored_memories=scored_memories,
    query_analysis=query_analysis,
    boost_recent=True,
    penalize_redundancy=True
)
```

### Custom Configuration

```python
# Create ranker with custom weights
ranker = AdaptiveRanker(
    recency_weight=0.4,      # Boost recency
    importance_weight=0.3,   # Boost importance
    semantic_weight=0.3,     # Reduce semantic weight
    redundancy_threshold=0.7 # Lower threshold for redundancy
)
```

### Integration with Multi-Strategy Retrieval

```python
from core.infinite.retrieval_strategies import MultiStrategyRetrieval

# Create multi-strategy retrieval with custom ranker
multi_strategy = MultiStrategyRetrieval(
    semantic_strategy=semantic_strategy,
    temporal_strategy=temporal_strategy,
    fulltext_strategy=fulltext_strategy,
    ranker=ranker  # Use custom ranker
)

# Retrieve with adaptive ranking
results = await multi_strategy.retrieve(
    query=query,
    query_analysis=query_analysis,
    context_id=context_id,
    limit=10,
    use_adaptive_ranking=True  # Enable adaptive ranking
)
```

### Disabling Adaptive Ranking

```python
# Retrieve without adaptive ranking (use strategy scores only)
results = await multi_strategy.retrieve(
    query=query,
    query_analysis=query_analysis,
    context_id=context_id,
    limit=10,
    use_adaptive_ranking=False
)
```

## Ranking Metadata

Each ranked memory includes detailed metadata about the ranking process:

```python
for result in results:
    if 'ranking' in result.metadata:
        ranking = result.metadata['ranking']
        
        # Component scores
        print(f"Semantic: {ranking['semantic_score']}")
        print(f"Recency: {ranking['recency_score']}")
        print(f"Importance: {ranking['importance_score']}")
        
        # Applied weights
        print(f"Weights: {ranking['weights']}")
        
        # Final score
        print(f"Final: {ranking['final_score']}")
        
        # Redundancy penalty (if applied)
        if 'redundancy_penalty' in ranking:
            print(f"Penalty: {ranking['redundancy_penalty']}")
            print(f"Original: {ranking['original_score']}")
```

## Examples

### Example 1: Temporal Query

```python
query = "What happened yesterday?"
query_analysis = QueryAnalysis(
    intent=QueryIntent.TEMPORAL,
    temporal_expressions=[("yesterday", timestamp)],
    keywords=["yesterday"]
)

# Recency weight will be boosted to 0.5
# Recent memories will rank higher
results = ranker.rank(scored_memories, query_analysis)
```

### Example 2: Factual Query

```python
query = "What is the project deadline?"
query_analysis = QueryAnalysis(
    intent=QueryIntent.FACTUAL,
    keywords=["project", "deadline"]
)

# Importance weight will be boosted to 0.3
# High-importance memories will rank higher
results = ranker.rank(scored_memories, query_analysis)
```

### Example 3: Preference Query

```python
query = "What are my coding preferences?"
query_analysis = QueryAnalysis(
    intent=QueryIntent.PREFERENCE,
    keywords=["coding", "preferences"]
)

# Both recency and semantic weights boosted
# Recent preferences rank higher than old ones
results = ranker.rank(scored_memories, query_analysis)
```

## Performance Characteristics

- **Time Complexity**: O(n²) for redundancy penalization, O(n log n) for sorting
- **Space Complexity**: O(n) for storing ranking metadata
- **Typical Latency**: < 10ms for 100 memories, < 50ms for 1000 memories

## Configuration Guidelines

### High Recency Sensitivity

For applications where recent information is critical:

```python
ranker = AdaptiveRanker(
    recency_weight=0.5,
    importance_weight=0.2,
    semantic_weight=0.3
)
```

### High Importance Sensitivity

For applications with critical vs. routine information:

```python
ranker = AdaptiveRanker(
    recency_weight=0.2,
    importance_weight=0.4,
    semantic_weight=0.4
)
```

### Balanced Configuration

For general-purpose applications:

```python
ranker = AdaptiveRanker(
    recency_weight=0.3,
    importance_weight=0.2,
    semantic_weight=0.5
)
```

### Strict Redundancy Control

To maximize diversity in results:

```python
ranker = AdaptiveRanker(
    redundancy_threshold=0.6  # Lower threshold = stricter
)
```

### Lenient Redundancy Control

To allow more similar results:

```python
ranker = AdaptiveRanker(
    redundancy_threshold=0.9  # Higher threshold = more lenient
)
```

## Testing

Comprehensive tests are available in `tests/infinite/test_retrieval_strategies.py`:

```bash
# Run all adaptive ranking tests
pytest tests/infinite/test_retrieval_strategies.py -k "adaptive" -v

# Run specific tests
pytest tests/infinite/test_retrieval_strategies.py::test_adaptive_ranker_multi_signal_scoring -v
pytest tests/infinite/test_retrieval_strategies.py::test_adaptive_ranker_recency_boosting -v
pytest tests/infinite/test_retrieval_strategies.py::test_adaptive_ranker_redundancy_penalization -v
```

## Future Enhancements

1. **Machine Learning Integration**: Learn optimal weights from user feedback
2. **Context-Aware Decay**: Adjust decay rates based on memory type
3. **Semantic Redundancy**: Use embeddings for more accurate redundancy detection
4. **Diversity Metrics**: Track and optimize result diversity
5. **A/B Testing**: Compare ranking strategies with metrics
6. **User Preferences**: Allow per-user weight customization

## Related Components

- **Query Analyzer**: Determines query intent for adaptive weights
- **Multi-Strategy Retrieval**: Orchestrates multiple retrieval strategies
- **Retrieval Strategies**: Provide initial semantic scores
- **Temporal Index**: Enables efficient time-based queries

## References

- Design Document: `.kiro/specs/infinite-context-system/design.md`
- Requirements: `.kiro/specs/infinite-context-system/requirements.md`
- Example: `examples/adaptive_ranking_example.py`
- Tests: `tests/infinite/test_retrieval_strategies.py`
