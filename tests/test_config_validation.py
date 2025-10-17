"""Test configuration validation."""

from config import MemoryConfig, HeuristicConfig, HybridConfig


def test_valid_heuristic_config():
    """Test that valid heuristic config passes validation."""
    config = MemoryConfig(mode="heuristic")
    errors = config.validate()
    assert errors == [], f"Expected no errors, got: {errors}"
    print("✓ Valid heuristic config passes validation")


def test_valid_ai_config():
    """Test that valid AI config passes validation."""
    config = MemoryConfig(
        mode="ai",
        ai_adapter_name="openai"
    )
    errors = config.validate()
    assert errors == [], f"Expected no errors, got: {errors}"
    print("✓ Valid AI config passes validation")


def test_valid_hybrid_config():
    """Test that valid hybrid config passes validation."""
    config = MemoryConfig(
        mode="hybrid",
        ai_adapter_name="ollama"
    )
    errors = config.validate()
    assert errors == [], f"Expected no errors, got: {errors}"
    print("✓ Valid hybrid config passes validation")


def test_valid_disabled_config():
    """Test that valid disabled config passes validation."""
    config = MemoryConfig(mode="disabled")
    errors = config.validate()
    assert errors == [], f"Expected no errors, got: {errors}"
    print("✓ Valid disabled config passes validation")


def test_invalid_mode():
    """Test that invalid mode is caught."""
    config = MemoryConfig(mode="invalid")  # type: ignore
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("Invalid mode" in err for err in errors), f"Expected mode error, got: {errors}"
    print(f"✓ Invalid mode detected: {errors[0]}")


def test_ai_mode_missing_adapter():
    """Test that AI mode without adapter name is caught."""
    config = MemoryConfig(mode="ai")
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("ai_adapter_name" in err for err in errors), f"Expected adapter error, got: {errors}"
    print(f"✓ AI mode missing adapter detected: {errors[0]}")


def test_hybrid_mode_missing_adapter():
    """Test that hybrid mode without adapter name is caught."""
    config = MemoryConfig(mode="hybrid")
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("ai_adapter_name" in err for err in errors), f"Expected adapter error, got: {errors}"
    print(f"✓ Hybrid mode missing adapter detected: {errors[0]}")


def test_invalid_stm_max_length():
    """Test that invalid stm_max_length is caught."""
    config = MemoryConfig(stm_max_length=-10)
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("stm_max_length" in err for err in errors), f"Expected stm_max_length error, got: {errors}"
    print(f"✓ Invalid stm_max_length detected: {errors[0]}")


def test_invalid_max_api_calls():
    """Test that invalid max_api_calls_per_minute is caught."""
    config = MemoryConfig(
        mode="ai",
        ai_adapter_name="openai",
        max_api_calls_per_minute=-5
    )
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("max_api_calls_per_minute" in err for err in errors), f"Expected max_api_calls error, got: {errors}"
    print(f"✓ Invalid max_api_calls_per_minute detected: {errors[0]}")


def test_invalid_hybrid_probability():
    """Test that invalid ai_probability is caught."""
    config = MemoryConfig(
        mode="hybrid",
        ai_adapter_name="openai",
        hybrid_config=HybridConfig(ai_probability=1.5)
    )
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("ai_probability" in err for err in errors), f"Expected probability error, got: {errors}"
    print(f"✓ Invalid ai_probability detected: {errors[0]}")


def test_invalid_hybrid_threshold():
    """Test that invalid ai_threshold_importance is caught."""
    config = MemoryConfig(
        mode="hybrid",
        ai_adapter_name="openai",
        hybrid_config=HybridConfig(ai_threshold_importance=15)
    )
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("ai_threshold_importance" in err for err in errors), f"Expected threshold error, got: {errors}"
    print(f"✓ Invalid ai_threshold_importance detected: {errors[0]}")


def test_invalid_heuristic_summary_length():
    """Test that invalid summary_max_length is caught."""
    config = MemoryConfig(
        heuristic_config=HeuristicConfig(summary_max_length=-100)
    )
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("summary_max_length" in err for err in errors), f"Expected summary_max_length error, got: {errors}"
    print(f"✓ Invalid summary_max_length detected: {errors[0]}")


def test_invalid_top_keywords():
    """Test that invalid top_keywords is caught."""
    config = MemoryConfig(
        heuristic_config=HeuristicConfig(top_keywords=0)
    )
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("top_keywords" in err for err in errors), f"Expected top_keywords error, got: {errors}"
    print(f"✓ Invalid top_keywords detected: {errors[0]}")


def test_invalid_min_keyword_length():
    """Test that invalid min_keyword_length is caught."""
    config = MemoryConfig(
        heuristic_config=HeuristicConfig(min_keyword_length=-1)
    )
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("min_keyword_length" in err for err in errors), f"Expected min_keyword_length error, got: {errors}"
    print(f"✓ Invalid min_keyword_length detected: {errors[0]}")


def test_empty_storage_path():
    """Test that empty storage_path is caught."""
    config = MemoryConfig(storage_path="")
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("storage_path" in err for err in errors), f"Expected storage_path error, got: {errors}"
    print(f"✓ Empty storage_path detected: {errors[0]}")


def test_empty_vector_db_path():
    """Test that empty vector_db_path is caught."""
    config = MemoryConfig(vector_db_path="")
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("vector_db_path" in err for err in errors), f"Expected vector_db_path error, got: {errors}"
    print(f"✓ Empty vector_db_path detected: {errors[0]}")


def test_invalid_log_level():
    """Test that invalid log_level is caught."""
    config = MemoryConfig(log_level="INVALID")
    errors = config.validate()
    assert len(errors) > 0, "Expected validation errors"
    assert any("log_level" in err for err in errors), f"Expected log_level error, got: {errors}"
    print(f"✓ Invalid log_level detected: {errors[0]}")


def test_multiple_errors():
    """Test that multiple errors are all caught."""
    config = MemoryConfig(
        mode="ai",  # Missing adapter name
        stm_max_length=-10,  # Invalid
        max_api_calls_per_minute=-5,  # Invalid
        storage_path="",  # Empty
        log_level="INVALID"  # Invalid
    )
    errors = config.validate()
    assert len(errors) >= 5, f"Expected at least 5 errors, got {len(errors)}: {errors}"
    print(f"✓ Multiple errors detected ({len(errors)} errors):")
    for error in errors:
        print(f"  - {error}")


if __name__ == "__main__":
    print("Testing configuration validation...\n")
    
    # Valid configs
    test_valid_heuristic_config()
    test_valid_ai_config()
    test_valid_hybrid_config()
    test_valid_disabled_config()
    
    print("\n" + "="*60)
    print("Testing invalid configurations...\n")
    
    # Invalid configs
    test_invalid_mode()
    test_ai_mode_missing_adapter()
    test_hybrid_mode_missing_adapter()
    test_invalid_stm_max_length()
    test_invalid_max_api_calls()
    test_invalid_hybrid_probability()
    test_invalid_hybrid_threshold()
    test_invalid_heuristic_summary_length()
    test_invalid_top_keywords()
    test_invalid_min_keyword_length()
    test_empty_storage_path()
    test_empty_vector_db_path()
    test_invalid_log_level()
    test_multiple_errors()
    
    print("\n" + "="*60)
    print("✅ All validation tests passed!")
