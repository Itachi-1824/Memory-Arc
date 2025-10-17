"""Pytest configuration and fixtures for infinite context tests."""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def test_db_path(temp_dir: Path) -> Path:
    """Get path for test database."""
    return temp_dir / "test.db"


@pytest.fixture
def test_vector_db_path(temp_dir: Path) -> Path:
    """Get path for test vector database."""
    return temp_dir / "vector_db"


@pytest.fixture
def test_cache_path(temp_dir: Path) -> Path:
    """Get path for test embedding cache."""
    return temp_dir / "cache"


@pytest.fixture
def sample_memories() -> list[dict]:
    """Generate sample memory data for testing."""
    return [
        {
            "content": "I like apples",
            "memory_type": "preference",
            "importance": 5,
        },
        {
            "content": "Paris is the capital of France",
            "memory_type": "fact",
            "importance": 7,
        },
        {
            "content": "def hello(): print('world')",
            "memory_type": "code",
            "importance": 6,
        },
        {
            "content": "Meeting notes from yesterday",
            "memory_type": "document",
            "importance": 4,
        },
    ]


@pytest.fixture
def sample_code_changes() -> list[dict]:
    """Generate sample code change data for testing."""
    return [
        {
            "file_path": "test.py",
            "before_content": "def old_function():\n    pass",
            "after_content": "def new_function():\n    return True",
            "change_type": "modify",
        },
        {
            "file_path": "new_file.py",
            "before_content": None,
            "after_content": "# New file\nprint('hello')",
            "change_type": "add",
        },
    ]
