"""Test script to verify all package exports are working correctly."""

import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_exports():
    """Test that core module exports work."""
    print("Testing core module exports...")
    from core import (
        MemoryEntry,
        MemoryManager,
        VectorMemoryManager,
        MemoryProcessor,
        ProcessingMetrics,
        AIProcessor,
        HeuristicProcessor,
        HybridProcessor,
        DisabledProcessor,
    )
    print("✓ All core exports successful")
    return True

def test_config_exports():
    """Test that configuration exports work."""
    print("\nTesting configuration exports...")
    from config import (
        MemoryConfig,
        HeuristicConfig,
        HybridConfig,
    )
    print("✓ All configuration exports successful")
    return True

def test_preset_exports():
    """Test that preset exports work."""
    print("\nTesting preset exports...")
    from presets import ConfigPresets
    print("✓ Preset exports successful")
    return True

def test_adapter_exports():
    """Test that adapter exports work."""
    print("\nTesting adapter exports...")
    from adapters import (
        AIAdapter,
        AdapterRegistry,
        AdapterNotFoundError,
    )
    print("✓ All adapter exports successful")
    return True

def test_main_package_exports():
    """Test that main package exports work."""
    print("\nTesting main package exports...")
    
    # Test individual imports
    try:
        from core.models import MemoryEntry
        from core.memory_manager import MemoryManager
        from core.vector_memory import VectorMemoryManager
        from core.processors import (
            MemoryProcessor,
            ProcessingMetrics,
            AIProcessor,
            HeuristicProcessor,
            HybridProcessor,
            DisabledProcessor,
        )
        from config import (
            MemoryConfig,
            HeuristicConfig,
            HybridConfig,
        )
        from presets import ConfigPresets
        from adapters.ai_adapter import AIAdapter
        from adapters.registry import AdapterRegistry, AdapterNotFoundError
        
        print("✓ All main package exports successful")
        
        # Print all exported classes
        print("\nExported classes:")
        print("  Core: MemoryEntry, MemoryManager, VectorMemoryManager")
        print("  Processors: MemoryProcessor, ProcessingMetrics, AIProcessor, HeuristicProcessor, HybridProcessor, DisabledProcessor")
        print("  Configuration: MemoryConfig, HeuristicConfig, HybridConfig, ConfigPresets")
        print("  Adapters: AIAdapter, AdapterRegistry, AdapterNotFoundError")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Package Export Verification")
    print("=" * 60)
    
    results = []
    
    # Test individual module exports
    results.append(test_core_exports())
    results.append(test_config_exports())
    results.append(test_preset_exports())
    results.append(test_adapter_exports())
    results.append(test_main_package_exports())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All export tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ Some export tests failed")
        print("=" * 60)
        sys.exit(1)
