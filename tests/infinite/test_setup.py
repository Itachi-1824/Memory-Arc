"""Test to verify infinite context test setup is working."""

import pytest
from pathlib import Path


def test_temp_dir_fixture(temp_dir):
    """Test that temp_dir fixture works."""
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    
    # Create a test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("hello")
    assert test_file.exists()


def test_sample_memories_fixture(sample_memories):
    """Test that sample_memories fixture works."""
    assert len(sample_memories) == 4
    assert sample_memories[0]["content"] == "I like apples"
    assert sample_memories[0]["memory_type"] == "preference"


def test_sample_code_changes_fixture(sample_code_changes):
    """Test that sample_code_changes fixture works."""
    assert len(sample_code_changes) == 2
    assert sample_code_changes[0]["file_path"] == "test.py"
    assert sample_code_changes[0]["change_type"] == "modify"


@pytest.mark.asyncio
async def test_async_support():
    """Test that async tests work."""
    import asyncio
    
    async def async_function():
        await asyncio.sleep(0.01)
        return "success"
    
    result = await async_function()
    assert result == "success"


def test_test_utils_import():
    """Test that test utilities can be imported."""
    import sys
    from pathlib import Path
    
    # Add tests directory to path
    tests_dir = Path(__file__).parent.parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    
    from infinite.test_utils import generate_test_embedding, assert_memory_equal
    
    # Test embedding generation
    embedding = generate_test_embedding("test text", dimensions=10)
    assert len(embedding) == 10
    assert all(-1 <= v <= 1 for v in embedding)
    
    # Test same text generates same embedding
    embedding2 = generate_test_embedding("test text", dimensions=10)
    assert embedding == embedding2
