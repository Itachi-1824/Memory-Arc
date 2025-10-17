# RetrievalOrchestrator

The `RetrievalOrchestrator` is the main orchestration layer for intelligent memory retrieval in the infinite context system. It integrates all retrieval components to provide a unified, high-performance interface for querying memories.

## Overview

The orchestrator combines:
- **Query Analysis** (Phase 5.1): Rule-based query understanding without AI
- **Multi-Strategy Retrieval** (Phase 5.2): Semantic, temporal, structural, and full-text search
- **Adaptive Ranking** (Phase 5.3): Multi-signal relevance scoring
- **Result Composition** (Phase 5.4): Intelligent result interleaving and breadcrumbs
- **Query Caching**: Performance optimization through result caching
- **Adaptive Scope Expansion**: Automatic search broadening when results are insufficient

## Features

### 1. Intelligent Query Analysis
- Intent classification (factual, temporal, code, preference, conversational)
- Entity extraction using patterns or spaCy NER
- Temporal expression parsing (e.g., "yesterday", "2 weeks ago")
- Code pattern detection (functions, files, classes)
- Keyword extraction with stop word filtering

### 2. Multi-Strategy Retrieval
- **Semantic Search**: Vector similarity using embeddings
- **Temporal Search**: Time-based queries with recency boosting
- **Structural Search**: Code-specific queries using AST patterns
- **Full-Text Search**: Keyword matching with importance weighting
- **Strategy Fusion**: Combines results from multiple strategies with consensus boosting

### 3. Adaptive Ranking
- Multi-signal scoring (semantic, temporal, importance)
- Intent-aware weight adjustment
- Recency boosting for time-sensitive queries
- Redundancy penalization for diverse results
- Confidence scoring for each result

### 4. Result Composition
- Type-based interleaving for diverse results
- Context breadcrumbs showing retrieval path
- Confidence scores and reasoning
- Metadata-rich results for transparency

### 5. Performance Optimization
- Query result caching with configurable TTL
- Cache hit/miss tracking
- Automatic cache cleanup
- Sub-second retrieval for most queries

### 6. Adaptive Scope Expansion
- Automatic detection of insufficient results
- Progressive relaxation of search constraints
- Time range expansion for temporal queries
- Score threshold adjustment
- Maintains result quality while improving coverage

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  RetrievalOrchestrator                      │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │Query Analyzer│  │Multi-Strategy│  │Result        │     │
│  │              │→ │Retrieval     │→ │Composer      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                           ↓                                 │
│                    ┌──────────────┐                        │
│                    │Adaptive      │                        │
│                    │Ranker        │                        │
│                    └──────────────┘                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Query Cache (TTL-based)                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Setup

```python
from core.infinite.retrieval_orchestrator import RetrievalOrchestrator
from core.infinite.document_store import DocumentStore
from core.infinite.temporal_index import TemporalIndex
from core.infinite.vector_store import VectorStore

# Initialize storage components
document_store = DocumentStore(db_path="memories.db")
await document_store.initialize()

temporal_index = TemporalIndex(db_path="temporal.db")
await temporal_index.initialize()

vector_store = VectorStore(collection_name="memories")
await vector_store.initialize()

# Create orchestrator
orchestrator = RetrievalOrchestrator(
    document_store=document_store,
    temporal_index=temporal_index,
    vector_store=vector_store,
    embedding_fn=your_embedding_function,
    enable_caching=True,
    cache_ttl_seconds=300.0,
    enable_scope_expansion=True
)
```

### Basic Retrieval

```python
# Simple query
result = await orchestrator.retrieve(
    query="What are my preferences?",
    context_id="user_123",
    max_results=10
)

print(f"Found {len(result.memories)} memories")
print(f"Query intent: {result.query_analysis.intent.value}")
print(f"Retrieval time: {result.retrieval_time_ms:.2f}ms")

for memory in result.memories:
    print(f"- [{memory.memory_type.value}] {memory.content}")
```

### Filtered Retrieval

```python
# Filter by memory type
result = await orchestrator.retrieve(
    query="Show me code",
    context_id="user_123",
    max_results=10,
    memory_types=[MemoryType.CODE]
)

# Filter by time range
import time
current_time = time.time()
last_week = (current_time - 7 * 86400, current_time)

result = await orchestrator.retrieve(
    query="Recent activities",
    context_id="user_123",
    max_results=10,
    time_range=last_week
)
```

### Query Analysis

```python
# Analyze query without retrieval
analysis = await orchestrator.analyze_query("What did I do yesterday?")

print(f"Intent: {analysis.intent.value}")
print(f"Confidence: {analysis.confidence:.2f}")
print(f"Keywords: {analysis.keywords}")
print(f"Temporal expressions: {analysis.temporal_expressions}")
print(f"Code patterns: {analysis.code_patterns}")
```

### Manual Ranking

```python
from core.infinite.retrieval_strategies import ScoredMemory

# Get scored memories from somewhere
scored_memories = [...]

# Rank them
ranked = await orchestrator.rank_results(
    scored_memories=scored_memories,
    query_analysis=analysis,
    boost_recent=True,
    penalize_redundancy=True
)
```

### Cache Management

```python
# Get cache statistics
stats = orchestrator.get_cache_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Active entries: {stats['active_entries']}")
print(f"Expired entries: {stats['expired_entries']}")

# Clear cache
cleared = orchestrator.clear_cache()
print(f"Cleared {cleared} entries")

# Bypass cache for specific query
result = await orchestrator.retrieve(
    query="Fresh query",
    context_id="user_123",
    max_results=10,
    use_cache=False
)
```

## Configuration

### Constructor Parameters

- **document_store**: DocumentStore instance for memory storage
- **temporal_index**: TemporalIndex instance for time-based queries
- **vector_store**: VectorStore instance for semantic search
- **code_change_store**: Optional CodeChangeStore for structural queries
- **embedding_fn**: Function to generate embeddings from text
- **use_spacy**: Whether to use spaCy for enhanced entity extraction (default: False)
- **enable_caching**: Whether to enable query result caching (default: True)
- **cache_ttl_seconds**: Time-to-live for cached results (default: 300.0)
- **enable_scope_expansion**: Whether to enable adaptive search scope expansion (default: True)
- **min_results_threshold**: Minimum results before triggering scope expansion (default: 3)

### Retrieve Parameters

- **query**: The query string
- **context_id**: Context identifier
- **max_results**: Maximum number of results to return (default: 10)
- **memory_types**: Optional filter for specific memory types
- **time_range**: Optional time range filter (start_time, end_time)
- **include_history**: Whether to include version history (default: False)
- **use_cache**: Whether to use cached results if available (default: True)
- **kwargs**: Additional parameters passed to retrieval strategies

## Performance

### Typical Performance Metrics

- **Query analysis**: < 10ms (rule-based, no AI)
- **Cached retrieval**: < 5ms
- **Uncached retrieval**: 50-200ms (depending on memory volume)
- **Scope expansion**: +20-50ms when triggered

### Optimization Tips

1. **Enable caching** for frequently repeated queries
2. **Adjust cache TTL** based on your update frequency
3. **Use memory type filters** to narrow search scope
4. **Set appropriate min_results_threshold** for scope expansion
5. **Provide time ranges** for temporal queries to improve performance

## Query Intent Detection

The orchestrator automatically detects query intent:

| Intent | Example Queries | Strategies Used |
|--------|----------------|-----------------|
| **FACTUAL** | "What is X?", "Tell me about Y" | Semantic, Full-text |
| **TEMPORAL** | "Yesterday", "Last week", "2 days ago" | Temporal, Semantic |
| **CODE** | "Show function X", "auth.py", "class User" | Structural, Semantic |
| **PREFERENCE** | "I like X", "I prefer Y" | Semantic, Temporal |
| **CONVERSATIONAL** | General conversation | Semantic, Full-text |
| **MIXED** | Multiple intents detected | All strategies |

## Result Metadata

Each `RetrievalResult` includes rich metadata:

```python
result = await orchestrator.retrieve(...)

# Query analysis
print(result.query_analysis.intent)
print(result.query_analysis.confidence)

# Retrieval metrics
print(result.total_found)
print(result.retrieval_time_ms)

# Memory groups
for group in result.metadata['memory_groups']:
    print(f"{group['type']}: {group['count']} memories, avg score: {group['avg_score']:.2f}")

# Breadcrumbs (retrieval path for each memory)
for breadcrumb in result.metadata['breadcrumbs']:
    print(f"Memory {breadcrumb['memory_id']}:")
    print(f"  Path: {' → '.join(breadcrumb['path'])}")
    print(f"  Strategies: {breadcrumb['strategies']}")
    print(f"  Confidence: {breadcrumb['confidence']:.2f}")
    print(f"  Reasoning: {breadcrumb['reasoning']}")

# Overall confidence
print(f"Overall confidence: {result.metadata['overall_confidence']:.2f}")
```

## Adaptive Scope Expansion

When initial results are insufficient, the orchestrator automatically:

1. **Relaxes score thresholds** for semantic search
2. **Expands time ranges** for temporal queries
3. **Broadens search scope** across memory types
4. **Re-ranks expanded results** to maintain quality

This ensures users always get useful results, even for challenging queries.

## Integration with Other Components

The orchestrator integrates seamlessly with:

- **ChunkManager**: For breaking large results into model-appropriate chunks
- **DynamicMemoryStore**: For version-aware retrieval
- **CodeChangeTracker**: For code-specific queries
- **MemoryManager**: As the main retrieval interface

## Error Handling

The orchestrator handles errors gracefully:

- **Strategy failures**: Falls back to other strategies
- **Cache errors**: Bypasses cache and retrieves fresh
- **Storage errors**: Returns partial results when possible
- **Timeout handling**: Configurable timeouts for each component

## Testing

Comprehensive test suite covers:

- Query analysis accuracy
- Multi-strategy retrieval
- Caching behavior
- Scope expansion
- Result composition
- Performance benchmarks

Run tests:
```bash
pytest tests/infinite/test_retrieval_orchestrator.py -v
```

## Examples

See example files:
- `examples/retrieval_orchestrator_example.py`: Full example with real storage
- `examples/retrieval_orchestrator_standalone.py`: Standalone example with mocks

## Future Enhancements

Planned improvements:
- Machine learning-based query understanding
- Personalized ranking based on user feedback
- Cross-context retrieval
- Real-time result streaming
- Advanced caching strategies (LRU, LFU)
- Distributed retrieval for multi-node deployments

## Related Components

- **QueryAnalyzer**: Rule-based query understanding
- **MultiStrategyRetrieval**: Strategy orchestration and fusion
- **AdaptiveRanker**: Multi-signal relevance scoring
- **ResultComposer**: Result interleaving and composition
