"""Simple test to verify configuration classes work."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from memory_system.config import MemoryConfig, HeuristicConfig, HybridConfig


def test_config_creation():
    """Test basic config creation."""
    print("Testing MemoryConfig creation...")
    
    # Test default config
    config = MemoryConfig()
    assert config.mode == "heuristic"
    assert config.stm_max_length == 150
    assert config.ltm_enabled == True
    print("✓ Default config works")
    
    # Test custom config
    config = MemoryConfig(
        mode="ai",
        stm_max_length=100,
        ai_adapter_name="openai",
    )
    assert config.mode == "ai"
    assert config.stm_max_length == 100
    assert config.ai_adapter_name == "openai"
    print("✓ Custom config works")


def test_preset_config():
    """Test preset configurations."""
    print("\nTesting preset configurations...")
    
    # Test offline preset
    config = MemoryConfig.from_preset("offline")
    assert config.mode == "heuristic"
    print("✓ Offline preset works")
    
    # Test chatbot preset
    config = MemoryConfig.from_preset("chatbot")
    assert config.mode == "hybrid"
    print("✓ Chatbot preset works")
    
    # Test coding-agent preset
    config = MemoryConfig.from_preset("coding-agent")
    assert config.mode == "heuristic"
    print("✓ Coding-agent preset works")
    
    # Test assistant preset
    config = MemoryConfig.from_preset("assistant")
    assert config.mode == "ai"
    print("✓ Assistant preset works")


def test_config_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    # Valid config
    config = MemoryConfig(mode="heuristic")
    errors = config.validate()
    assert len(errors) == 0
    print("✓ Valid config passes validation")
    
    # Invalid mode
    config = MemoryConfig(mode="invalid")  # type: ignore
    errors = config.validate()
    assert len(errors) > 0
    assert any("mode" in err.lower() for err in errors)
    print("✓ Invalid mode detected")
    
    # AI mode without adapter
    config = MemoryConfig(mode="ai")
    errors = config.validate()
    assert len(errors) > 0
    assert any("adapter" in err.lower() for err in errors)
    print("✓ Missing AI adapter detected")


def test_preset_override():
    """Test preset with overrides."""
    print("\nTesting preset overrides...")
    
    config = MemoryConfig.from_preset(
        "offline",
        stm_max_length=200,
        storage_path="./custom_data"
    )
    
    assert config.mode == "heuristic"  # From preset
    assert config.stm_max_length == 200  # Overridden
    assert config.storage_path == "./custom_data"  # Overridden
    print("✓ Preset overrides work")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Configuration System")
    print("=" * 60)
    
    try:
        test_config_creation()
        test_preset_config()
        test_config_validation()
        test_preset_override()
        
        print("\n" + "=" * 60)
        print("All configuration tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
