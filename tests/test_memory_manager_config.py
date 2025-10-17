"""Quick test to verify MemoryManager configuration support."""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory_manager import MemoryManager
from core.models import MemoryEntry
from config import MemoryConfig


async def test_backward_compatibility():
    """Test that old API still works."""
    print("Testing backward compatibility...")
    
    # Old way - should work without config
    memory = MemoryManager(
        context_id="test_old",
        storage_path="./test_data",
        max_stm_length=10,
    )
    
    assert memory.config.mode == "heuristic"
    assert memory.config.stm_max_length == 10
    assert memory.processor is not None
    print("✓ Backward compatibility works")


async def test_heuristic_mode():
    """Test heuristic mode configuration."""
    print("\nTesting heuristic mode...")
    
    config = MemoryConfig(
        mode="heuristic",
        stm_max_length=5,
    )
    
    memory = MemoryManager(
        context_id="test_heuristic",
        config=config,
    )
    
    assert memory.config.mode == "heuristic"
    assert memory.processor is not None
    print("✓ Heuristic mode works")
    
    # Test adding a message
    entry = MemoryEntry(
        role="user",
        content="Hello, this is a test message",
    )
    await memory.add_message(entry)
    print("✓ Message added successfully")


async def test_disabled_mode():
    """Test disabled mode configuration."""
    print("\nTesting disabled mode...")
    
    config = MemoryConfig(
        mode="disabled",
        ltm_enabled=False,
    )
    
    memory = MemoryManager(
        context_id="test_disabled",
        config=config,
    )
    
    assert memory.config.mode == "disabled"
    assert memory.processor is not None
    assert memory.vector_memory is None
    print("✓ Disabled mode works")


async def test_metrics():
    """Test metrics retrieval."""
    print("\nTesting metrics retrieval...")
    
    config = MemoryConfig(
        mode="heuristic",
        enable_metrics=True,
    )
    
    memory = MemoryManager(
        context_id="test_metrics",
        config=config,
    )
    
    metrics = memory.get_metrics()
    assert "mode" in metrics
    assert metrics["mode"] == "heuristic"
    assert "context_id" in metrics
    print("✓ Metrics retrieval works")
    print(f"  Metrics: {metrics}")


async def test_preset_config():
    """Test preset configuration."""
    print("\nTesting preset configuration...")
    
    config = MemoryConfig.from_preset("offline")
    
    memory = MemoryManager(
        context_id="test_preset",
        config=config,
    )
    
    assert memory.config.mode == "heuristic"
    assert memory.processor is not None
    print("✓ Preset configuration works")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing MemoryManager Configuration Support")
    print("=" * 60)
    
    try:
        await test_backward_compatibility()
        await test_heuristic_mode()
        await test_disabled_mode()
        await test_metrics()
        await test_preset_config()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
