"""Practical examples of configuration validation."""

from config import MemoryConfig, HybridConfig


def example_1_validate_before_use():
    """Example: Validate configuration before using it."""
    print("Example 1: Validate configuration before use")
    print("-" * 60)
    
    # Create a configuration
    config = MemoryConfig(
        mode="hybrid",
        ai_adapter_name="openai",
        stm_max_length=200
    )
    
    # Validate before using
    errors = config.validate()
    
    if errors:
        print("❌ Configuration is invalid:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Configuration is valid and ready to use!")
        return True


def example_2_catch_common_mistakes():
    """Example: Catch common configuration mistakes."""
    print("\nExample 2: Catch common configuration mistakes")
    print("-" * 60)
    
    # Forgot to set adapter name for AI mode
    config = MemoryConfig(mode="ai")
    
    errors = config.validate()
    
    if errors:
        print("❌ Configuration error detected:")
        for error in errors:
            print(f"  - {error}")
        print("\n💡 Fix: Set ai_adapter_name, e.g., ai_adapter_name='openai'")
        return False
    
    return True


def example_3_validate_hybrid_settings():
    """Example: Validate hybrid mode settings."""
    print("\nExample 3: Validate hybrid mode settings")
    print("-" * 60)
    
    # Invalid probability value
    config = MemoryConfig(
        mode="hybrid",
        ai_adapter_name="ollama",
        hybrid_config=HybridConfig(
            ai_probability=1.5,  # Invalid: must be 0-1
            ai_threshold_importance=15  # Invalid: must be 1-10
        )
    )
    
    errors = config.validate()
    
    if errors:
        print("❌ Configuration errors detected:")
        for error in errors:
            print(f"  - {error}")
        print("\n💡 Fix:")
        print("  - Set ai_probability between 0 and 1 (e.g., 0.1 for 10%)")
        print("  - Set ai_threshold_importance between 1 and 10")
        return False
    
    return True


def example_4_validate_file_paths():
    """Example: Validate file paths."""
    print("\nExample 4: Validate file paths")
    print("-" * 60)
    
    # Empty paths
    config = MemoryConfig(
        storage_path="",
        vector_db_path=""
    )
    
    errors = config.validate()
    
    if errors:
        print("❌ Configuration errors detected:")
        for error in errors:
            print(f"  - {error}")
        print("\n💡 Fix: Provide valid paths for storage_path and vector_db_path")
        return False
    
    return True


def example_5_comprehensive_validation():
    """Example: Comprehensive validation with multiple issues."""
    print("\nExample 5: Comprehensive validation")
    print("-" * 60)
    
    # Multiple issues
    config = MemoryConfig(
        mode="invalid_mode",  # type: ignore
        stm_max_length=-100,
        storage_path="",
        log_level="TRACE"
    )
    
    errors = config.validate()
    
    print(f"Found {len(errors)} validation errors:")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    
    print("\n💡 This demonstrates that validate() catches ALL errors at once,")
    print("   allowing you to fix them all before retrying.")
    
    return len(errors) > 0


if __name__ == "__main__":
    print("=" * 60)
    print("Configuration Validation - Practical Examples")
    print("=" * 60)
    
    example_1_validate_before_use()
    example_2_catch_common_mistakes()
    example_3_validate_hybrid_settings()
    example_4_validate_file_paths()
    example_5_comprehensive_validation()
    
    print("\n" + "=" * 60)
    print("✅ Validation examples completed!")
    print("=" * 60)
