# Priority-Based Chunk Selection

## Overview

The `ChunkSelector` provides intelligent, priority-based selection of content chunks using a weighted scoring system that combines:

1. **Relevance Scoring**: How well the chunk matches the query
2. **Importance Ranking**: Priority based on content importance
3. **Recency Boosting**: Time-based scoring with exponential decay

## Features

- **Multi-Factor Scoring**: Combines relevance, importance, and recency
- **Configurable Weights**: Customize the balance between factors
- **Flexible Decay**: Adjustable recency decay rates
- **Embedding Support**: Optional semantic similarity via embeddings
- **Lexical Fallback**: Works without embeddings using word overlap
- **Threshold Filtering**: Minimum score requirements
- **Result Limiting**: Control maximum number of results

## Usage

### Basic Selection

```python
from core.infinite import ChunkSelector, Chunk

# Create selector with default weights
selector = ChunkSelector()

# Select chunks
scored_chunks = selector.select_chunks(
    chunks=my_chunks,
    query="machine learning",
    max_chunks=10
)

# Access results
for sc in scored_chunks:
    print(f"Score: {sc.final_score:.3f}")
    print(f"Content: {sc.chunk.content}")
```

### Custom Weight Configuration

```python
# Relevance-focused (80% relevance, 10% importance, 10% recency)
selector = ChunkSelector(
    relevance_weight=0.8,
    importance_weight=0.1,
    recency_weight=0.1
)

# Recency-focused (10% relevance, 10% importance, 80% recency)
selector = ChunkSelector(
    relevance_weight=0.1,
    importance_weight=0.1,
    recency_weight=0.8,
    recency_decay_hours=24.0  # Fast decay (1 day)
)

# Importance-focused (10% relevance, 80% importance, 10% recency)
selector = ChunkSelector(
    relevance_weight=0.1,
    importance_weight=0.8,
    recency_weight=0.1
)
```

### With Embedding Function

```python
def my_embedding_fn(text: str) -> list[float]:
    # Your embedding model here
    return model.encode(text)

selector = ChunkSelector(embedding_fn=my_embedding_fn)

scored_chunks = selector.select_chunks(
    chunks=my_chunks,
    query="deep learning",
    current_time=time.time()
)
```

### With Associated Memories

```python
from core.infinite import Memory, MemoryType

# Create memories with importance ratings
memories = [
    Memory(
        id="mem_1",
        context_id="user_123",
        content="Important fact",
        memory_type=MemoryType.FACT,
        created_at=time.time(),
        importance=9  # High importance
    ),
    # ... more memories
]

# Select chunks using memory metadata
scored_chunks = selector.select_chunks(
    chunks=my_chunks,
    memories=memories,
    query="important facts",
    current_time=time.time()
)
```

### Minimum Score Threshold

```python
# Only return chunks with score >= 0.5
scored_chunks = selector.select_chunks(
    chunks=my_chunks,
    query="relevant content",
    min_score=0.5
)
```

## Scoring Algorithm

### Relevance Score (0-1)

- **With Embeddings**: Cosine similarity between chunk and query embeddings
- **Without Embeddings**: Lexical similarity (word overlap / union)
- **No Query**: Returns neutral score (0.5)

### Importance Score (0-1)

- Normalized from chunk metadata or memory importance (1-10 scale)
- Default: 0.5 (medium importance)

### Recency Score (0-1)

- Exponential decay: `score = exp(-age / decay_constant)`
- At `recency_decay_hours`, score ≈ 0.5
- Recent content scores near 1.0
- Old content scores near 0.0

### Final Score

```
final_score = (
    relevance * relevance_weight +
    importance * importance_weight +
    recency * recency_weight
)
```

Weights are automatically normalized to sum to 1.0.

## Configuration Parameters

### ChunkSelector Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `relevance_weight` | float | 0.5 | Weight for relevance scoring |
| `importance_weight` | float | 0.3 | Weight for importance scoring |
| `recency_weight` | float | 0.2 | Weight for recency scoring |
| `recency_decay_hours` | float | 168.0 | Hours for recency to decay to 0.5 (1 week) |
| `embedding_fn` | Callable | None | Optional embedding function |

### select_chunks Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunks` | list[Chunk] | Required | Chunks to select from |
| `query` | str | None | Query text for relevance scoring |
| `query_embedding` | list[float] | None | Pre-computed query embedding |
| `memories` | list[Memory] | None | Associated memories for metadata |
| `max_chunks` | int | None | Maximum chunks to return |
| `min_score` | float | 0.0 | Minimum score threshold |
| `current_time` | float | None | Current timestamp (uses time.time() if None) |

## ScoredChunk Structure

```python
@dataclass
class ScoredChunk:
    chunk: Chunk                    # Original chunk
    relevance_score: float          # Relevance component (0-1)
    importance_score: float         # Importance component (0-1)
    recency_score: float            # Recency component (0-1)
    final_score: float              # Weighted final score (0-1)
    metadata: dict[str, Any]        # Additional metadata
```

## Examples

### Example 1: Balanced Selection

```python
selector = ChunkSelector(
    relevance_weight=0.33,
    importance_weight=0.33,
    recency_weight=0.34
)

scored = selector.select_chunks(
    chunks=chunks,
    query="machine learning",
    current_time=time.time()
)

# Results balanced across all three factors
```

### Example 2: Recent Content Only

```python
selector = ChunkSelector(
    relevance_weight=0.0,
    importance_weight=0.0,
    recency_weight=1.0,
    recency_decay_hours=24.0  # 1 day
)

scored = selector.select_chunks(
    chunks=chunks,
    current_time=time.time(),
    max_chunks=5
)

# Returns 5 most recent chunks
```

### Example 3: High-Quality Content

```python
selector = ChunkSelector(
    relevance_weight=0.3,
    importance_weight=0.7,
    recency_weight=0.0
)

scored = selector.select_chunks(
    chunks=chunks,
    query="important information",
    min_score=0.7  # Only high-scoring chunks
)

# Returns important, relevant chunks regardless of age
```

## Performance Considerations

- **Embedding Computation**: Cache embeddings in chunk metadata to avoid recomputation
- **Large Chunk Lists**: Use `max_chunks` to limit results
- **Score Threshold**: Use `min_score` to filter low-quality results early
- **Recency Decay**: Adjust `recency_decay_hours` based on your use case

## Integration with Other Components

### With SemanticChunker

```python
from core.infinite import SemanticChunker, ChunkSelector

# Create chunks
chunker = SemanticChunker(max_chunk_size=1000)
chunks = chunker.chunk_content(content, "text")

# Select best chunks
selector = ChunkSelector()
scored = selector.select_chunks(chunks, query="relevant topic")
```

### With TokenCounter

```python
from core.infinite import ChunkSelector, TokenCounter

selector = ChunkSelector()
counter = TokenCounter("gpt-4")

# Select chunks that fit in context window
scored = selector.select_chunks(chunks, query="query", max_chunks=10)

# Verify token budget
total_tokens = sum(counter.count_tokens(sc.chunk.content) for sc in scored)
```

## Testing

Run the test suite:

```bash
pytest tests/infinite/test_chunk_selector.py -v
```

Test coverage includes:
- Initialization and configuration
- Relevance scoring (with/without embeddings)
- Importance ranking
- Recency boosting with exponential decay
- Final score computation
- Chunk selection with various filters
- Edge cases and boundary conditions

## Implementation Details

### Relevance Scoring

1. Check if chunk has pre-computed relevance score
2. If embedding function available, compute semantic similarity
3. Fallback to lexical similarity (word overlap)
4. Return neutral score (0.5) if no query provided

### Importance Scoring

1. Check chunk metadata for importance value
2. Check associated memory for importance value
3. Normalize from 1-10 scale to 0-1 range
4. Return default (0.5) if no importance data

### Recency Scoring

1. Extract timestamp from chunk metadata or memory
2. Calculate age in hours
3. Apply exponential decay formula
4. Clamp result to [0, 1] range
5. Return neutral score (0.5) if no timestamp

### Selection Process

1. Iterate through all chunks
2. Compute relevance, importance, and recency scores
3. Calculate weighted final score
4. Filter by minimum score threshold
5. Sort by final score (descending)
6. Apply maximum chunk limit
7. Return list of ScoredChunk objects

## Future Enhancements

- **Diversity Boosting**: Penalize redundant chunks
- **Context Awareness**: Consider chunk position and relationships
- **Learning**: Adapt weights based on user feedback
- **Multi-Query**: Support multiple queries with different weights
- **Clustering**: Group similar chunks before selection
