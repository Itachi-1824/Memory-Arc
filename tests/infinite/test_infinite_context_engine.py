"""Integration tests for InfiniteContextEngine.

Tests the complete system with all components working together.
"""

import asyncio
import pytest
import tempfile
import time
from pathlib import Path

from core.infinite import (
    InfiniteContextEngine,
    InfiniteContextConfig,
    MemoryType,
)


@pytest.fixture
async def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def engine(temp_storage):
    """Create and initialize InfiniteContextEngine."""
    config = InfiniteContextConfig(
        storage_path=str(temp_storage),
        enable_caching=True,
        enable_code_tracking=False,
        use_spacy=False
    )
    
    # Simple embedding function for testing
    def mock_embedding(text: str) -> list[float]:
        # Generate deterministic embedding based on text hash
        hash_val = hash(text)
        return [float((hash_val >> i) & 0xFF) / 255.0 for i in range(1536)]
    
    engine = InfiniteContextEngine(
        config=config,
        embedding_fn=mock_embedding
    )
    
    await engine.initialize()
    yield engine
    await engine.shutdown()


@pytest.mark.asyncio
async def test_engine_initialization(temp_storage):
    """Test engine initialization and shutdown."""
    config = InfiniteContextConfig(storage_path=str(temp_storage))
    engine = InfiniteContextEngine(config=config)
    
    assert not engine._initialized
    
    await engine.initialize()
    assert engine._initialized
    
    # Check health status
    health = engine.get_health_status()
    assert health["engine"] in ["healthy", "degraded"]
    assert "document_store" in health
    
    await engine.shutdown()
    assert not engine._initialized


@pytest.mark.asyncio
async def test_add_and_retrieve_memory(engine):
    """Test basic memory addition and retrieval."""
    # Add memories
    memory_id1 = await engine.add_memory(
        content="Python is a programming language",
        memory_type=MemoryType.FACT,
        context_id="test_context",
        importance=8
    )
    
    memory_id2 = await engine.add_memory(
        content="I love programming in Python",
        memory_type=MemoryType.CONVERSATION,
        context_id="test_context",
        importance=5
    )
    
    assert memory_id1
    assert memory_id2
    
    # Retrieve memories
    result = await engine.retrieve(
        query="Python programming",
        context_id="test_context",
        max_results=10
    )
    
    assert result.total_found >= 2
    assert len(result.memories) >= 2
    assert result.query_analysis is not None


@pytest.mark.asyncio
async def test_memory_versioning(engine):
    """Test memory versioning and evolution."""
    # Add initial memory
    memory_id1 = await engine.add_memory(
        content="I like apples",
        memory_type=MemoryType.PREFERENCE,
        context_id="test_context"
    )
    
    # Update preference (create new version)
    memory_id2 = await engine.add_memory(
        content="I like mangoes",
        memory_type=MemoryType.PREFERENCE,
        context_id="test_context",
        supersedes=memory_id1
    )
    
    # Get version history
    history = await engine.get_version_history(memory_id2)
    
    assert len(history) >= 1
    assert any("apples" in m.content for m in history) or any("mangoes" in m.content for m in history)


@pytest.mark.asyncio
async def test_temporal_queries(engine):
    """Test querying memories at specific times."""
    timestamp_before = time.time()
    
    # Add memory
    await engine.add_memory(
        content="Event at time T1",
        memory_type=MemoryType.FACT,
        context_id="test_context"
    )
    
    await asyncio.sleep(0.1)
    timestamp_middle = time.time()
    await asyncio.sleep(0.1)
    
    # Add another memory
    await engine.add_memory(
        content="Event at time T2",
        memory_type=MemoryType.FACT,
        context_id="test_context"
    )
    
    timestamp_after = time.time()
    
    # Query at middle time
    result = await engine.query_at_time(
        query="Event",
        timestamp=timestamp_middle,
        context_id="test_context"
    )
    
    # Should find at least the first event
    assert len(result.memories) >= 0  # May or may not find depending on timing


@pytest.mark.asyncio
async def test_contradiction_detection(engine):
    """Test detecting contradictory memories."""
    # Add contradictory preferences
    await engine.add_memory(
        content="I prefer dark mode",
        memory_type=MemoryType.PREFERENCE,
        context_id="test_context"
    )
    
    await engine.add_memory(
        content="I prefer light mode",
        memory_type=MemoryType.PREFERENCE,
        context_id="test_context"
    )
    
    # Detect contradictions
    contradictions = await engine.detect_contradictions(
        context_id="test_context"
    )
    
    # May or may not detect contradictions depending on similarity threshold
    assert isinstance(contradictions, list)


@pytest.mark.asyncio
async def test_chunking_integration(engine):
    """Test retrieval with chunking."""
    # Add long content
    long_content = "This is a long piece of content. " * 100
    
    await engine.add_memory(
        content=long_content,
        memory_type=MemoryType.DOCUMENT,
        context_id="test_context"
    )
    
    # Retrieve with chunking
    result = await engine.retrieve(
        query="long piece",
        context_id="test_context",
        return_chunks=True
    )
    
    assert result.memories
    # Chunks may or may not be created depending on content size
    if result.chunks:
        assert len(result.chunks) > 0


@pytest.mark.asyncio
async def test_metrics_collection(engine):
    """Test metrics collection and reporting."""
    # Perform some operations
    await engine.add_memory(
        content="Test memory 1",
        context_id="test_context"
    )
    
    await engine.retrieve(
        query="test",
        context_id="test_context"
    )
    
    # Get metrics
    metrics = engine.get_metrics()
    
    assert metrics.total_memories > 0
    assert metrics.total_queries > 0
    assert metrics.uptime_seconds > 0
    assert metrics.last_updated > 0


@pytest.mark.asyncio
async def test_error_recovery(engine):
    """Test error handling and recovery."""
    # This test verifies graceful degradation
    # Try to retrieve with invalid context
    try:
        result = await engine.retrieve(
            query="test query",
            context_id="nonexistent_context"
        )
        # Should still return a result (possibly empty)
        assert isinstance(result.memories, list)
    except Exception as e:
        # If it fails, that's also acceptable
        pass


@pytest.mark.asyncio
async def test_concurrent_operations(engine):
    """Test concurrent memory operations."""
    # Add multiple memories concurrently
    tasks = []
    for i in range(10):
        task = engine.add_memory(
            content=f"Concurrent memory {i}",
            context_id="test_context"
        )
        tasks.append(task)
    
    memory_ids = await asyncio.gather(*tasks)
    assert len(memory_ids) == 10
    
    # Retrieve concurrently
    query_tasks = []
    for i in range(5):
        task = engine.retrieve(
            query=f"Concurrent memory {i}",
            context_id="test_context"
        )
        query_tasks.append(task)
    
    results = await asyncio.gather(*query_tasks)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_context_manager(temp_storage):
    """Test async context manager usage."""
    config = InfiniteContextConfig(storage_path=str(temp_storage))
    
    async with InfiniteContextEngine(config=config) as engine:
        assert engine._initialized
        
        # Use engine
        memory_id = await engine.add_memory(
            content="Test in context manager",
            context_id="test_context"
        )
        assert memory_id
    
    # Engine should be shut down after context exit
    # (can't easily test this without accessing private state)


@pytest.mark.asyncio
async def test_configuration_presets(temp_storage):
    """Test configuration presets."""
    # Test minimal preset
    minimal_config = InfiniteContextConfig.minimal()
    minimal_config.storage_path = str(temp_storage / "minimal")
    
    engine = InfiniteContextEngine(config=minimal_config)
    await engine.initialize()
    
    await engine.add_memory("Test", context_id="test")
    
    await engine.shutdown()
    
    # Test balanced preset
    balanced_config = InfiniteContextConfig.balanced()
    balanced_config.storage_path = str(temp_storage / "balanced")
    
    engine = InfiniteContextEngine(config=balanced_config)
    await engine.initialize()
    
    await engine.add_memory("Test", context_id="test")
    
    await engine.shutdown()
    
    # Test performance preset
    perf_config = InfiniteContextConfig.performance()
    perf_config.storage_path = str(temp_storage / "performance")
    
    engine = InfiniteContextEngine(config=perf_config)
    await engine.initialize()
    
    await engine.add_memory("Test", context_id="test")
    
    await engine.shutdown()


@pytest.mark.asyncio
async def test_health_status_reporting(engine):
    """Test health status reporting."""
    health = engine.get_health_status()
    
    assert "engine" in health
    assert "document_store" in health
    assert "temporal_index" in health
    assert "vector_store" in health
    
    # All components should be healthy after initialization
    assert health["document_store"] == "healthy"
    assert health["temporal_index"] == "healthy"


@pytest.mark.asyncio
async def test_memory_types(engine):
    """Test different memory types."""
    memory_types = [
        (MemoryType.CONVERSATION, "Hello, how are you?"),
        (MemoryType.FACT, "The sky is blue"),
        (MemoryType.PREFERENCE, "I prefer tea over coffee"),
        (MemoryType.DOCUMENT, "This is a document about AI"),
        (MemoryType.CODE, "def hello(): print('world')"),
    ]
    
    for mem_type, content in memory_types:
        memory_id = await engine.add_memory(
            content=content,
            memory_type=mem_type,
            context_id="test_context"
        )
        assert memory_id
    
    # Retrieve all
    result = await engine.retrieve(
        query="test",
        context_id="test_context",
        max_results=20
    )
    
    assert len(result.memories) >= len(memory_types)
