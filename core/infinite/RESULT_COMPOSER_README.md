# Result Composer

The Result Composer is responsible for intelligently composing retrieval results with interleaving, grouping, breadcrumb generation, and confidence scoring.

## Features

### 1. Result Interleaving
- **Round-robin interleaving** by memory type for diverse results
- **Weighted by group average scores** to prioritize high-quality types
- **Configurable** - can be disabled for pure ranking

### 2. Memory Grouping
- Groups memories by type (CODE, CONVERSATION, PREFERENCE, FACT, etc.)
- Calculates average scores per group
- Supports **max_per_type** limits for balanced results
- Provides group statistics in metadata

### 3. Context Breadcrumbs
- Shows **retrieval path** for each result (query → strategy → ranking → selection)
- Lists **strategies used** (semantic, temporal, structural, fulltext, fused)
- Generates **human-readable reasoning** explaining why each result was selected
- Includes **confidence scores** per result

### 4. Confidence Calculation
- **Per-result confidence**: Based on retrieval score, query confidence, consensus, and importance
- **Overall confidence**: Weighted average of top results with penalties for few results
- Factors in multiple signals for robust scoring

### 5. Filtering
- **Minimum confidence threshold** to filter low-quality results
- Tracks filtered count in metadata

## Usage

### Basic Composition

```python
from core.infinite.result_composer import ResultComposer
from core.infinite.models import QueryAnalysis, QueryIntent

# Create composer
composer = ResultComposer(
    interleave_by_type=True,
    include_breadcrumbs=True
)

# Compose results
result = composer.compose(
    scored_memories=scored_memories,  # From retrieval strategies
    query_analysis=query_analysis,
    retrieval_time_ms=45.5,
    limit=10
)

# Access results
print(f"Found {result.total_found} memories")
print(f"Confidence: {result.metadata['overall_confidence']:.2f}")

for memory in result.memories:
    print(f"[{memory.memory_type.value}] {memory.content}")
```

### Filtered Composition

```python
# Filter by confidence threshold
composer = ResultComposer(
    min_confidence=0.7,
    interleave_by_type=True
)

result = composer.compose(
    scored_memories=scored_memories,
    query_analysis=query_analysis,
    retrieval_time_ms=38.2
)

print(f"Filtered out {result.metadata['filtered_count']} low-confidence results")
```

### Grouped Composition

```python
# Limit results per type
composer = ResultComposer(
    max_per_type=2,  # Max 2 memories per type
    interleave_by_type=True
)

result = composer.compose(
    scored_memories=scored_memories,
    query_analysis=query_analysis,
    retrieval_time_ms=52.1
)

# Check group distribution
for group in result.metadata['memory_groups']:
    print(f"{group['type']}: {group['count']} memories")
```

### Pure Ranking (No Interleaving)

```python
# Disable interleaving for pure score-based ranking
composer = ResultComposer(
    interleave_by_type=False
)

result = composer.compose(
    scored_memories=scored_memories,
    query_analysis=query_analysis,
    retrieval_time_ms=41.8,
    limit=5
)

# Results are sorted by score only
```

### Breadcrumb Analysis

```python
composer = ResultComposer(include_breadcrumbs=True)

result = composer.compose(
    scored_memories=scored_memories,
    query_analysis=query_analysis,
    retrieval_time_ms=48.3
)

# Analyze retrieval paths
for breadcrumb in result.metadata['breadcrumbs']:
    print(f"Memory: {breadcrumb['memory_id']}")
    print(f"Path: {' → '.join(breadcrumb['path'])}")
    print(f"Strategies: {', '.join(breadcrumb['strategies'])}")
    print(f"Confidence: {breadcrumb['confidence']:.2f}")
    print(f"Reasoning: {breadcrumb['reasoning']}")
```

## Configuration Options

### Constructor Parameters

- **interleave_by_type** (bool, default=True): Enable round-robin interleaving by memory type
- **max_per_type** (int | None, default=None): Maximum memories per type (None for unlimited)
- **include_breadcrumbs** (bool, default=True): Generate context breadcrumbs
- **min_confidence** (float, default=0.0): Minimum confidence threshold for filtering

### Compose Parameters

- **scored_memories** (list[ScoredMemory]): Memories from retrieval strategies
- **query_analysis** (QueryAnalysis): Analyzed query information
- **retrieval_time_ms** (float): Time taken for retrieval
- **limit** (int | None): Maximum number of results to return

## Result Structure

### RetrievalResult

```python
@dataclass
class RetrievalResult:
    memories: list[Memory]           # Final composed memories
    total_found: int                 # Total before filtering/limiting
    query_analysis: QueryAnalysis    # Original query analysis
    retrieval_time_ms: float         # Retrieval time
    metadata: dict[str, Any]         # Rich metadata
```

### Metadata Contents

```python
{
    "memory_groups": [
        {
            "type": "code",
            "count": 2,
            "avg_score": 0.915
        },
        ...
    ],
    "breadcrumbs": [
        {
            "memory_id": "uuid",
            "path": ["query_intent:code", "strategy:semantic", "selected"],
            "strategies": ["semantic"],
            "confidence": 0.87,
            "reasoning": "semantically similar to query; very high relevance; marked as important"
        },
        ...
    ],
    "overall_confidence": 0.82,
    "composition_strategy": "interleaved",  # or "ranked"
    "filtered_count": 1
}
```

## Interleaving Algorithm

The interleaving algorithm:

1. **Groups memories by type** and sorts groups by average score
2. **Creates iterators** for each group
3. **Round-robin selection** from groups (highest avg score first)
4. **Continues until** limit reached or all groups exhausted

This ensures diverse results while prioritizing high-quality memory types.

## Confidence Calculation

### Per-Result Confidence

```
confidence = (
    retrieval_score * 0.5 +
    query_confidence * 0.2 +
    consensus_boost +      # 0.1 per additional strategy
    importance_boost       # (importance/10) * 0.1
)
```

### Overall Confidence

```
overall = (
    weighted_avg_top_scores * 0.7 +
    query_confidence * 0.3
) * penalty_for_few_results
```

## Reasoning Generation

The reasoning system explains why each result was selected:

- **Strategy-based**: "semantically similar", "temporally relevant", "matches code structure"
- **Score-based**: "very high relevance", "high relevance", "moderate relevance"
- **Importance-based**: "marked as important"
- **Recency-based**: "recent memory"
- **Intent matching**: "matches code query intent"

## Integration with Retrieval System

The Result Composer is designed to work with the Multi-Strategy Retrieval system:

```python
from core.infinite.retrieval_strategies import MultiStrategyRetrieval
from core.infinite.result_composer import ResultComposer

# Retrieve with multiple strategies
retrieval = MultiStrategyRetrieval(...)
scored_memories = await retrieval.retrieve(query, query_analysis, context_id)

# Compose results
composer = ResultComposer()
result = composer.compose(
    scored_memories=scored_memories,
    query_analysis=query_analysis,
    retrieval_time_ms=retrieval_time
)
```

## Testing

Comprehensive tests cover:
- Basic composition
- Confidence filtering
- Memory grouping
- Interleaving logic
- Breadcrumb generation
- Confidence calculation
- Reasoning generation
- Edge cases (empty results, single result, etc.)

Run tests:
```bash
pytest tests/infinite/test_result_composer.py -v
```

## Performance

- **Time Complexity**: O(n log n) for sorting + O(n) for interleaving
- **Space Complexity**: O(n) for grouping and metadata
- **Typical Latency**: < 5ms for 100 results

## See Also

- **Multi-Strategy Retrieval**: Provides scored memories for composition
- **Query Analyzer**: Provides query analysis for adaptive composition
- **Adaptive Ranker**: Ranks memories before composition
