# Infinite Context API Reference

## Overview

The Infinite Context System provides unlimited memory storage and retrieval for AI applications. This API reference documents all public classes, methods, and data models.

## Table of Contents

- [InfiniteContextEngine](#infinitecontextengine) - Main integration class
- [Configuration](#configuration) - Configuration and presets
- [Data Models](#data-models) - Core data structures
- [Memory Operations](#memory-operations) - Adding and retrieving memories
- [Code Tracking](#code-tracking) - Code change tracking (optional)
- [Monitoring](#monitoring) - Health checks and metrics
- [Error Handling](#error-handling) - Exception handling and recovery

---

## InfiniteContextEngine

The primary interface for the infinite context system.

### Class Definition

```python
from core.infinite import InfiniteContextEngine, InfiniteContextConfig

engine = InfiniteContextEngine(
    config: InfiniteContextConfig | None = None,
    embedding_fn: Callable[[str], list[float]] | None = None,
    # Legacy parameters (for backward compatibility)
    storage_path: str | Path | None = None,
    vector_store_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    model_name: str | None = None,
    max_tokens: int | None = None,
    enable_code_tracking: bool | None = None,
    code_watch_path: str | Path | None = None,
    enable_caching: bool | None = None,
    use_spacy: bool | None = None
)
```

### Parameters

- **config** (`InfiniteContextConfig | None`): Configuration object (recommended approach)
- **embedding_fn** (`Callable[[str], list[float]] | None`): Function to generate embeddings from text
- **storage_path** (`str | Path | None`): Path to SQLite database directory (legacy)
- **vector_store_path** (`str | Path | None`): Path to Qdrant vector store (legacy)
- **cache_path** (`str | Path | None`): Path to LMDB embedding cache (legacy)
- **model_name** (`str | None`): Target model name for chunking (legacy)
- **max_tokens** (`int | None`): Maximum tokens for model (legacy)
- **enable_code_tracking** (`bool | None`): Enable code change tracking (legacy)
- **code_watch_path** (`str | Path | None`): Path to watch for code changes (legacy)
- **enable_caching** (`bool | None`): Enable embedding cache (legacy)
- **use_spacy** (`bool | None`): Use spaCy for enhanced NLP (legacy)

### Basic Usage

```python
import asyncio
from core.infinite import InfiniteContextEngine, InfiniteContextConfig, MemoryType

async def main():
    # Using configuration object (recommended)
    config = InfiniteContextConfig(
        storage_path="./data/my_context",
        model_name="gpt-4",
        enable_caching=True
    )
    
    engine = InfiniteContextEngine(config=config)
    await engine.initialize()
    
    try:
        # Add a memory
        memory_id = await engine.add_memory(
            content="User prefers dark mode",
            memory_type=MemoryType.PREFERENCE,
            context_id="user_123"
        )
        
        # Retrieve memories
        result = await engine.retrieve(
            query="What are the user's preferences?",
            context_id="user_123"
        )
        
        print(f"Found {result.total_found} memories")
        for memory in result.memories:
            print(f"- {memory.content}")
    
    finally:
        await engine.shutdown()

asyncio.run(main())
```

### Using Context Manager

```python
async def main():
    config = InfiniteContextConfig.balanced()
    
    async with InfiniteContextEngine(config=config) as engine:
        # Engine is automatically initialized
        await engine.add_memory(
            content="Important information",
            memory_type=MemoryType.FACT
        )
        # Engine is automatically shut down on exit
```

---

## Configuration

### InfiniteContextConfig

Configuration dataclass for the infinite context system.

```python
from core.infinite import InfiniteContextConfig

config = InfiniteContextConfig(
    # Storage paths
    storage_path: str = "./data/infinite_context",
    vector_store_path: str | None = None,
    cache_path: str | None = None,
    
    # Model configuration
    model_name: str = "gpt-4",
    max_tokens: int | None = None,
    
    # Code tracking
    enable_code_tracking: bool = False,
    code_watch_path: str | None = None,
    code_ignore_patterns: list[str] = [...],
    
    # Caching
    enable_caching: bool = True,
    cache_max_size_gb: float = 10.0,
    
    # NLP features
    use_spacy: bool = False,
    
    # Memory settings
    similarity_threshold: float = 0.7,
    default_importance: int = 5,
    
    # Retrieval settings
    default_max_results: int = 10,
    enable_query_caching: bool = True,
    query_cache_ttl_seconds: int = 300,
    
    # Performance tuning
    batch_size: int = 100,
    max_concurrent_queries: int = 10,
    embedding_batch_size: int = 50,
    
    # Vector store settings
    vector_embedding_dim: int = 1536,
    vector_distance_metric: str = "cosine",
    
    # Chunking settings
    chunk_overlap_tokens: int = 100,
    preserve_structure: bool = True
)
```

### Configuration Presets

Three preset configurations are available:

#### Minimal Configuration

Optimized for low resource usage and fast startup:

```python
config = InfiniteContextConfig.minimal()
```

Features:
- Caching disabled
- spaCy disabled
- Code tracking disabled
- 1GB cache limit
- Smaller batch sizes

#### Balanced Configuration (Default)

Optimized for general usage with good performance:

```python
config = InfiniteContextConfig.balanced()
```

Features:
- Caching enabled
- spaCy disabled
- Code tracking disabled
- 5GB cache limit
- Standard batch sizes

#### Performance Configuration

Optimized for high-throughput production workloads:

```python
config = InfiniteContextConfig.performance()
```

Features:
- Caching enabled
- spaCy enabled
- Code tracking disabled
- 20GB cache limit
- Large batch sizes
- Query caching enabled

### Loading Configuration from File

```python
from core.infinite import load_config_from_file

# Load from JSON
config = load_config_from_file("config.json")

# Load from YAML (requires PyYAML)
config = load_config_from_file("config.yaml")
```

Example `config.json`:

```json
{
  "storage_path": "./data/my_context",
  "model_name": "gpt-4",
  "enable_caching": true,
  "cache_max_size_gb": 10.0,
  "use_spacy": false,
  "similarity_threshold": 0.7,
  "default_max_results": 10
}
```

### Saving Configuration to File

```python
from core.infinite import save_config_to_file

config = InfiniteContextConfig.performance()

# Save as JSON
save_config_to_file(config, "config.json", format="json")

# Save as YAML
save_config_to_file(config, "config.yaml", format="yaml")
```

---

## Data Models

### Memory

Represents a memory entry with versioning support.

```python
@dataclass
class Memory:
    id: str
    context_id: str
    content: str
    memory_type: MemoryType
    created_at: float
    importance: int = 5
    updated_at: float | None = None
    version: int = 1
    parent_id: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
```

### MemoryType

Enum defining types of memories:

```python
class MemoryType(Enum):
    CONVERSATION = "conversation"
    CODE = "code"
    FACT = "fact"
    SUMMARY = "summary"
    PREFERENCE = "preference"
    DOCUMENT = "document"
```

### RetrievalResult

Result of memory retrieval operations.

```python
@dataclass
class RetrievalResult:
    memories: list[Memory]
    total_found: int
    query_analysis: QueryAnalysis
    retrieval_time_ms: float
    chunks: list[Chunk] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### QueryAnalysis

Analysis of a user query.

```python
@dataclass
class QueryAnalysis:
    intent: QueryIntent
    entities: list[tuple[str, str]] = field(default_factory=list)
    temporal_expressions: list[tuple[str, float]] = field(default_factory=list)
    code_patterns: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

### QueryIntent

Enum defining query intent types:

```python
class QueryIntent(Enum):
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    CODE = "code"
    CONVERSATIONAL = "conversational"
    PREFERENCE = "preference"
    MIXED = "mixed"
```

### Chunk

A chunk of content with metadata.

```python
@dataclass
class Chunk:
    id: str
    content: str
    chunk_index: int
    total_chunks: int
    token_count: int
    relevance_score: float = 0.0
    start_pos: int = 0
    end_pos: int = 0
    boundary_type: BoundaryType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### SystemMetrics

System-wide metrics for monitoring.

```python
@dataclass
class SystemMetrics:
    total_memories: int = 0
    total_code_changes: int = 0
    active_contexts: int = 0
    storage_size_bytes: int = 0
    embedding_cache_size_bytes: int = 0
    vector_store_size_bytes: int = 0
    avg_query_latency_ms: float = 0.0
    p95_query_latency_ms: float = 0.0
    p99_query_latency_ms: float = 0.0
    total_queries: int = 0
    cache_hit_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    uptime_seconds: float = 0.0
    last_error: str | None = None
    error_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_updated: float = 0.0
```

---

## Memory Operations

### Initialization

#### `async initialize()`

Initialize all components. Must be called before using the engine.

```python
engine = InfiniteContextEngine(config=config)
await engine.initialize()
```

**Raises:**
- `RuntimeError`: If initialization fails

#### `async shutdown()`

Shutdown all components with proper cleanup.

```python
await engine.shutdown()
```

### Adding Memories

#### `async add_memory()`

Add a new memory to the system.

```python
memory_id = await engine.add_memory(
    content: str,
    memory_type: str | MemoryType = MemoryType.CONVERSATION,
    context_id: str = "default",
    importance: int = 5,
    supersedes: str | None = None,
    thread_id: str | None = None,
    metadata: dict[str, Any] | None = None
) -> str
```

**Parameters:**
- **content** (`str`): Memory content
- **memory_type** (`str | MemoryType`): Type of memory
- **context_id** (`str`): Context identifier for grouping memories
- **importance** (`int`): Importance score (1-10)
- **supersedes** (`str | None`): ID of memory this supersedes (creates new version)
- **thread_id** (`str | None`): Optional thread identifier
- **metadata** (`dict | None`): Optional metadata dictionary

**Returns:** Memory ID (`str`)

**Raises:**
- `RuntimeError`: If engine not initialized

**Examples:**

```python
# Add a simple conversation memory
memory_id = await engine.add_memory(
    content="User asked about pricing",
    memory_type=MemoryType.CONVERSATION,
    context_id="user_123"
)

# Add a preference with high importance
memory_id = await engine.add_memory(
    content="User prefers email notifications",
    memory_type=MemoryType.PREFERENCE,
    context_id="user_123",
    importance=8
)

# Add a fact with metadata
memory_id = await engine.add_memory(
    content="Paris is the capital of France",
    memory_type=MemoryType.FACT,
    context_id="geography",
    metadata={"source": "wikipedia", "verified": True}
)

# Update a preference (create new version)
old_memory_id = "mem_123"
new_memory_id = await engine.add_memory(
    content="User prefers SMS notifications",
    memory_type=MemoryType.PREFERENCE,
    context_id="user_123",
    supersedes=old_memory_id  # Links to previous version
)
```

### Retrieving Memories

#### `async retrieve()`

Retrieve relevant memories for a query.

```python
result = await engine.retrieve(
    query: str,
    context_id: str = "default",
    max_results: int = 10,
    memory_types: list[str | MemoryType] | None = None,
    time_range: tuple[float, float] | None = None,
    include_history: bool = False,
    return_chunks: bool = False
) -> RetrievalResult
```

**Parameters:**
- **query** (`str`): Search query
- **context_id** (`str`): Context identifier to search within
- **max_results** (`int`): Maximum number of results to return
- **memory_types** (`list | None`): Filter by memory types (None = all types)
- **time_range** (`tuple | None`): Optional time range filter (start_time, end_time)
- **include_history** (`bool`): Whether to include version history
- **return_chunks** (`bool`): Whether to chunk results for model consumption

**Returns:** `RetrievalResult`

**Raises:**
- `RuntimeError`: If engine not initialized

**Examples:**

```python
# Basic retrieval
result = await engine.retrieve(
    query="What are the user's preferences?",
    context_id="user_123"
)

print(f"Found {result.total_found} memories in {result.retrieval_time_ms}ms")
for memory in result.memories:
    print(f"- {memory.content} (importance: {memory.importance})")

# Filter by memory type
result = await engine.retrieve(
    query="user settings",
    context_id="user_123",
    memory_types=[MemoryType.PREFERENCE, MemoryType.FACT]
)

# Time-based retrieval (last 7 days)
import time
week_ago = time.time() - (7 * 24 * 60 * 60)
now = time.time()

result = await engine.retrieve(
    query="recent conversations",
    context_id="user_123",
    time_range=(week_ago, now)
)

# Retrieve with chunking for large contexts
result = await engine.retrieve(
    query="explain the entire codebase",
    context_id="project_x",
    max_results=50,
    return_chunks=True
)

if result.chunks:
    print(f"Content split into {len(result.chunks)} chunks")
    for chunk in result.chunks:
        print(f"Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}")
        print(f"Tokens: {chunk.token_count}")
        print(f"Relevance: {chunk.relevance_score}")
```

#### `async query_at_time()`

Query memories as they existed at a specific time.

```python
result = await engine.query_at_time(
    query: str,
    timestamp: float,
    context_id: str = "default",
    max_results: int = 10
) -> RetrievalResult
```

**Parameters:**
- **query** (`str`): Search query
- **timestamp** (`float`): Unix timestamp to query at
- **context_id** (`str`): Context identifier
- **max_results** (`int`): Maximum number of results

**Returns:** `RetrievalResult`

**Examples:**

```python
import time
from datetime import datetime, timedelta

# Query memories from 6 months ago
six_months_ago = (datetime.now() - timedelta(days=180)).timestamp()

result = await engine.query_at_time(
    query="user preferences",
    timestamp=six_months_ago,
    context_id="user_123"
)

print(f"User preferences 6 months ago:")
for memory in result.memories:
    print(f"- {memory.content}")

# Compare with current preferences
current_result = await engine.retrieve(
    query="user preferences",
    context_id="user_123"
)

print(f"\nCurrent user preferences:")
for memory in current_result.memories:
    print(f"- {memory.content}")
```

### Version History

#### `async get_version_history()`

Get version history for a memory.

```python
history = await engine.get_version_history(
    memory_id: str
) -> list[Memory]
```

**Parameters:**
- **memory_id** (`str`): ID of the memory

**Returns:** List of `Memory` objects in chronological order

**Examples:**

```python
# Get version history
history = await engine.get_version_history("mem_123")

print(f"Memory has {len(history)} versions:")
for i, version in enumerate(history, 1):
    print(f"{i}. {version.content} (created: {version.created_at})")

# Show evolution of a preference
memory_id = "pref_456"
history = await engine.get_version_history(memory_id)

print("Preference evolution:")
for version in history:
    timestamp = datetime.fromtimestamp(version.created_at)
    print(f"- {timestamp}: {version.content}")
```

### Contradiction Detection

#### `async detect_contradictions()`

Detect contradictory memories.

```python
contradictions = await engine.detect_contradictions(
    context_id: str,
    entity_type: str | None = None
) -> list[tuple[Memory, Memory, float]]
```

**Parameters:**
- **context_id** (`str`): Context identifier
- **entity_type** (`str | None`): Optional entity type filter

**Returns:** List of `(memory1, memory2, similarity_score)` tuples

**Examples:**

```python
# Detect all contradictions
contradictions = await engine.detect_contradictions(
    context_id="user_123"
)

print(f"Found {len(contradictions)} contradictions:")
for mem1, mem2, similarity in contradictions:
    print(f"\nContradiction (similarity: {similarity:.2f}):")
    print(f"  1. {mem1.content}")
    print(f"  2. {mem2.content}")

# Filter by entity type
contradictions = await engine.detect_contradictions(
    context_id="user_123",
    entity_type="preference"
)
```

---

## Code Tracking

Code tracking is an optional feature that provides 1:1 diff tracking with AST analysis for code files.

### Enabling Code Tracking

```python
config = InfiniteContextConfig(
    storage_path="./data/my_context",
    enable_code_tracking=True,
    code_watch_path="./src",  # Directory to watch
    code_ignore_patterns=["*.pyc", "__pycache__", ".git"]
)

engine = InfiniteContextEngine(config=config)
await engine.initialize()
```

### Starting Code Tracking

#### `async start_code_tracking()`

Start automatic code change tracking.

```python
await engine.start_code_tracking()
```

**Raises:**
- `RuntimeError`: If code tracking not enabled

**Example:**

```python
# Start tracking
await engine.start_code_tracking()
print("Code tracking started")

# Make some code changes...
# Changes are automatically tracked

# Stop tracking
await engine.stop_code_tracking()
```

### Stopping Code Tracking

#### `async stop_code_tracking()`

Stop automatic code change tracking.

```python
await engine.stop_code_tracking()
```

---

## Monitoring

### Health Checks

#### `get_health_status()`

Get health status of all components.

```python
status = engine.get_health_status() -> dict[str, str]
```

**Returns:** Dictionary mapping component names to status:
- `"healthy"`: Component is functioning normally
- `"degraded"`: Component has issues but is operational
- `"down"`: Component is not functioning
- `"not_initialized"`: Engine not initialized

**Example:**

```python
status = engine.get_health_status()

print("System Health:")
for component, health in status.items():
    emoji = "✅" if health == "healthy" else "⚠️" if health == "degraded" else "❌"
    print(f"{emoji} {component}: {health}")

# Check if system is healthy
if status["engine"] == "healthy":
    print("\nAll systems operational")
else:
    print("\nSystem has issues")
```

### Metrics

#### `get_metrics()`

Get current system metrics.

```python
metrics = engine.get_metrics() -> SystemMetrics
```

**Returns:** `SystemMetrics` object

**Example:**

```python
metrics = engine.get_metrics()

print(f"System Metrics:")
print(f"  Total Memories: {metrics.total_memories}")
print(f"  Total Queries: {metrics.total_queries}")
print(f"  Avg Query Latency: {metrics.avg_query_latency_ms:.2f}ms")
print(f"  P95 Query Latency: {metrics.p95_query_latency_ms:.2f}ms")
print(f"  P99 Query Latency: {metrics.p99_query_latency_ms:.2f}ms")
print(f"  Cache Hit Rate: {metrics.cache_hit_rate:.2%}")
print(f"  Storage Size: {metrics.storage_size_bytes / 1024 / 1024:.2f}MB")
print(f"  Uptime: {metrics.uptime_seconds:.0f}s")

if metrics.last_error:
    print(f"  Last Error: {metrics.last_error}")
    print(f"  Error Count: {metrics.error_count}")
```

---

## Error Handling

### Common Exceptions

#### RuntimeError

Raised when operations are attempted on an uninitialized engine:

```python
try:
    await engine.add_memory("test")
except RuntimeError as e:
    print(f"Error: {e}")
    # Initialize the engine first
    await engine.initialize()
```

#### ValueError

Raised for invalid configuration:

```python
try:
    config = InfiniteContextConfig(
        similarity_threshold=1.5  # Invalid: must be 0.0-1.0
    )
    config.validate()
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

### Graceful Degradation

The engine includes automatic fallback mechanisms:

```python
# If vector store fails, falls back to full-text search
result = await engine.retrieve(
    query="test query",
    context_id="user_123"
)

# Check if fallback was used
if result.metadata.get("fallback"):
    print("Warning: Using fallback retrieval (reduced accuracy)")
```

### Error Recovery

The engine tracks errors and attempts recovery:

```python
# Check for errors
metrics = engine.get_metrics()
if metrics.error_count > 0:
    print(f"System has encountered {metrics.error_count} errors")
    print(f"Last error: {metrics.last_error}")
    
    # Engine will attempt automatic recovery on next operation
```

---

## Complete Example

Here's a complete example demonstrating all major features:

```python
import asyncio
from datetime import datetime, timedelta
from core.infinite import (
    InfiniteContextEngine,
    InfiniteContextConfig,
    MemoryType
)

async def main():
    # Create configuration
    config = InfiniteContextConfig.balanced()
    config.storage_path = "./data/demo"
    
    # Initialize engine
    async with InfiniteContextEngine(config=config) as engine:
        context_id = "user_demo"
        
        # Add various types of memories
        print("Adding memories...")
        
        await engine.add_memory(
            content="User prefers dark mode",
            memory_type=MemoryType.PREFERENCE,
            context_id=context_id,
            importance=8
        )
        
        await engine.add_memory(
            content="User is interested in machine learning",
            memory_type=MemoryType.FACT,
            context_id=context_id,
            importance=7
        )
        
        await engine.add_memory(
            content="Discussed API design patterns",
            memory_type=MemoryType.CONVERSATION,
            context_id=context_id,
            importance=6
        )
        
        # Retrieve memories
        print("\nRetrieving memories...")
        result = await engine.retrieve(
            query="What do we know about the user?",
            context_id=context_id,
            max_results=10
        )
        
        print(f"Found {result.total_found} memories in {result.retrieval_time_ms:.2f}ms")
        print(f"Query intent: {result.query_analysis.intent.value}")
        
        for memory in result.memories:
            print(f"\n- {memory.content}")
            print(f"  Type: {memory.memory_type.value}")
            print(f"  Importance: {memory.importance}")
            print(f"  Created: {datetime.fromtimestamp(memory.created_at)}")
        
        # Update a preference (create new version)
        print("\nUpdating preference...")
        old_pref_id = result.memories[0].id
        
        new_pref_id = await engine.add_memory(
            content="User prefers light mode",
            memory_type=MemoryType.PREFERENCE,
            context_id=context_id,
            supersedes=old_pref_id,
            importance=8
        )
        
        # Get version history
        print("\nVersion history:")
        history = await engine.get_version_history(new_pref_id)
        for i, version in enumerate(history, 1):
            print(f"{i}. {version.content}")
        
        # Detect contradictions
        print("\nChecking for contradictions...")
        contradictions = await engine.detect_contradictions(context_id)
        
        if contradictions:
            print(f"Found {len(contradictions)} contradictions")
            for mem1, mem2, similarity in contradictions:
                print(f"\n  Contradiction (similarity: {similarity:.2f}):")
                print(f"    - {mem1.content}")
                print(f"    - {mem2.content}")
        else:
            print("No contradictions found")
        
        # Check health and metrics
        print("\nSystem Health:")
        status = engine.get_health_status()
        for component, health in status.items():
            print(f"  {component}: {health}")
        
        print("\nSystem Metrics:")
        metrics = engine.get_metrics()
        print(f"  Total Memories: {metrics.total_memories}")
        print(f"  Total Queries: {metrics.total_queries}")
        print(f"  Avg Latency: {metrics.avg_query_latency_ms:.2f}ms")
        print(f"  Cache Hit Rate: {metrics.cache_hit_rate:.2%}")
        print(f"  Uptime: {metrics.uptime_seconds:.0f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## See Also

- [Architecture Documentation](../ARCHITECTURE.md)
- [Migration Guide](../MIGRATION_GUIDE.md)
- [Performance Tuning Guide](../PERFORMANCE_TUNING.md)
- [Examples](../../examples/)

