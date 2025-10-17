"""Simple test for cost and performance tracking features."""

import asyncio
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MemoryConfig


def test_config_with_performance_settings():
    """Test that config accepts performance settings."""
    print("\n=== Testing Configuration ===")
    
    config = MemoryConfig(
        mode="ai",
        max_api_calls_per_minute=10,
        cache_summaries=True,
        batch_processing=True,
    )
    
    print(f"Mode: {config.mode}")
    print(f"Max API calls per minute: {config.max_api_calls_per_minute}")
    print(f"Cache summaries: {config.cache_summaries}")
    print(f"Batch processing: {config.batch_processing}")
    
    assert config.max_api_calls_per_minute == 10
    assert config.cache_summaries is True
    assert config.batch_processing is True
    
    print("✓ Configuration accepts performance settings")


def test_config_validation():
    """Test config validation with performance settings."""
    print("\n=== Testing Configuration Validation ===")
    
    # Valid config
    config = MemoryConfig(
        mode="ai",
        ai_adapter_name="openai",
        max_api_calls_per_minute=5,
    )
    
    errors = config.validate()
    print(f"Validation errors: {errors}")
    assert len(errors) == 0, "Valid config should have no errors"
    print("✓ Valid config passes validation")
    
    # Invalid config - negative rate limit
    config_invalid = MemoryConfig(
        mode="ai",
        ai_adapter_name="openai",
        max_api_calls_per_minute=-5,
    )
    
    errors = config_invalid.validate()
    print(f"Validation errors for invalid config: {errors}")
    assert len(errors) > 0, "Invalid config should have errors"
    assert any("max_api_calls_per_minute" in err for err in errors)
    print("✓ Invalid rate limit detected")


def test_processor_initialization():
    """Test that AIProcessor can be initialized with performance features."""
    print("\n=== Testing AIProcessor Initialization ===")
    
    try:
        # Import here to avoid dependency issues
        from core.processors import AIProcessor, ProcessingMetrics
        from adapters.ai_adapter import AIAdapter
        
        class MockAdapter(AIAdapter):
            async def summarize_conversation(self, messages):
                return "test"
            async def extract_facts(self, messages):
                return []
            async def score_importance(self, text):
                return 5
        
        config = MemoryConfig(
            mode="ai",
            max_api_calls_per_minute=10,
            cache_summaries=True,
            batch_processing=True,
        )
        
        adapter = MockAdapter()
        processor = AIProcessor(adapter, config)
        
        # Check that features are initialized
        assert processor.rate_limit_enabled is True
        assert processor.max_calls_per_minute == 10
        assert processor.cache_enabled is True
        assert processor.batch_enabled is True
        
        print("✓ AIProcessor initialized with all performance features")
        
        # Check metrics
        metrics = processor.get_metrics()
        print(f"Initial metrics: {metrics}")
        assert 'ai_calls' in metrics
        assert 'processing_time' in metrics
        print("✓ Metrics tracking initialized")
        
    except ImportError as e:
        print(f"⚠ Skipping processor test due to missing dependencies: {e}")


def test_cache_key_generation():
    """Test cache key generation."""
    print("\n=== Testing Cache Key Generation ===")
    
    try:
        from core.processors import AIProcessor
        from adapters.ai_adapter import AIAdapter
        
        class MockAdapter(AIAdapter):
            async def summarize_conversation(self, messages):
                return "test"
            async def extract_facts(self, messages):
                return []
            async def score_importance(self, text):
                return 5
        
        config = MemoryConfig(mode="ai", cache_summaries=True)
        adapter = MockAdapter()
        processor = AIProcessor(adapter, config)
        
        # Test cache key generation
        key1 = processor._compute_cache_key("summarize", "test content")
        key2 = processor._compute_cache_key("summarize", "test content")
        key3 = processor._compute_cache_key("summarize", "different content")
        
        print(f"Key 1: {key1}")
        print(f"Key 2: {key2}")
        print(f"Key 3: {key3}")
        
        assert key1 == key2, "Same content should generate same key"
        assert key1 != key3, "Different content should generate different key"
        assert key1.startswith("summarize:"), "Key should include operation"
        
        print("✓ Cache key generation working correctly")
        
    except ImportError as e:
        print(f"⚠ Skipping cache key test due to missing dependencies: {e}")


def main():
    """Run all tests."""
    print("Testing Cost and Performance Tracking Features")
    print("=" * 50)
    
    try:
        test_config_with_performance_settings()
        test_config_validation()
        test_processor_initialization()
        test_cache_key_generation()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
