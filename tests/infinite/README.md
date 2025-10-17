# Infinite Context System Tests

This directory contains tests for the Infinite Context System implementation.

## Structure

```
tests/infinite/
├── __init__.py           # Package marker
├── conftest.py           # Pytest fixtures and configuration
├── test_utils.py         # Test utility functions
├── test_setup.py         # Setup verification tests
└── README.md             # This file
```

## Running Tests

### Run all infinite context tests:
```bash
pytest tests/infinite/ -v
```

### Run specific test file:
```bash
pytest tests/infinite/test_setup.py -v
```

### Run with coverage:
```bash
pytest tests/infinite/ --cov=core/infinite --cov-report=html
```

### Run async tests only:
```bash
pytest tests/infinite/ -v -m asyncio
```

### Run benchmarks:
```bash
pytest tests/infinite/ --benchmark-only
```

## Fixtures Available

### Paths
- `temp_dir`: Temporary directory for test files
- `test_db_path`: Path for test SQLite database
- `test_vector_db_path`: Path for test Qdrant vector database
- `test_cache_path`: Path for test LMDB cache

### Data
- `sample_memories`: List of sample memory dictionaries
- `sample_code_changes`: List of sample code change dictionaries

## Test Utilities

### `generate_test_embedding(text, dimensions=384)`
Generate deterministic test embeddings (not real embeddings, just for testing).

### `assert_memory_equal(mem1, mem2, ignore_fields=None)`
Assert two memory dictionaries are equal, optionally ignoring certain fields.

### `wait_for_condition(condition_fn, timeout=5.0, interval=0.1)`
Wait for a condition function to return True.

## Test Markers

Use markers to categorize and filter tests:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (skip with `-m "not slow"`)
- `@pytest.mark.storage` - Storage layer tests
- `@pytest.mark.memory` - Memory management tests
- `@pytest.mark.code` - Code tracking tests
- `@pytest.mark.retrieval` - Retrieval tests
- `@pytest.mark.benchmark` - Performance benchmarks

## Writing Tests

### Example unit test:
```python
import pytest

@pytest.mark.unit
@pytest.mark.storage
def test_document_store(test_db_path):
    from core.infinite.storage import DocumentStore
    
    store = DocumentStore(test_db_path)
    # ... test code ...
```

### Example async test:
```python
import pytest

@pytest.mark.asyncio
@pytest.mark.integration
async def test_memory_retrieval(temp_dir):
    from core.infinite.memory_store import DynamicMemoryStore
    
    store = DynamicMemoryStore(temp_dir)
    await store.add_memory("test content")
    # ... test code ...
```

### Example benchmark:
```python
import pytest

@pytest.mark.benchmark
def test_embedding_cache_performance(benchmark, test_cache_path):
    from core.infinite.cache import EmbeddingCache
    
    cache = EmbeddingCache(test_cache_path)
    result = benchmark(cache.get, "test_key")
    # ... assertions ...
```

## Best Practices

1. **Use fixtures** for setup/teardown
2. **Mark tests appropriately** for easy filtering
3. **Test one thing per test** - keep tests focused
4. **Use descriptive names** - test names should explain what they test
5. **Clean up resources** - fixtures handle this automatically
6. **Mock external dependencies** - use pytest-mock for mocking
7. **Test edge cases** - not just happy paths
8. **Keep tests fast** - mark slow tests with `@pytest.mark.slow`

## Coverage Goals

- **Unit tests**: 90%+ coverage
- **Integration tests**: Cover all major workflows
- **Performance tests**: Verify sub-200ms retrieval for 1M memories
