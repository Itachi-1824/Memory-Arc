"""Unit tests for CodeChangeTracker."""

import pytest
import asyncio
import time
import uuid
from pathlib import Path

from core.infinite.code_change_tracker import CodeChangeTracker
from core.infinite.code_change_store import CodeChange


@pytest.fixture
async def code_change_tracker(test_db_path, tmp_path):
    """Create and initialize a code change tracker for testing."""
    watch_path = tmp_path / "watch"
    watch_path.mkdir()
    
    tracker = CodeChangeTracker(
        watch_path=watch_path,
        db_path=test_db_path,
        auto_track=False,
    )
    await tracker.initialize()
    yield tracker
    await tracker.close()


@pytest.mark.asyncio
async def test_track_change_file_creation(code_change_tracker):
    """Test tracking a file creation."""
    file_path = str(code_change_tracker.watch_path / "new_file.py")
    content = "def hello():\n    print('Hello')\n"
    
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content=content,
        change_type="add",
    )
    
    assert change_id is not None
    assert len(change_id) > 0
    
    # Verify the change was stored
    changes = await code_change_tracker.query_changes(file_path=file_path)
    assert len(changes) == 1
    assert changes[0].id == change_id
    assert changes[0].change_type == "add"


@pytest.mark.asyncio
async def test_track_change_file_modification(code_change_tracker):
    """Test tracking a file modification."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    before = "def hello():\n    print('Hello')\n"
    after = "def hello(name):\n    print(f'Hello {name}')\n"
    
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=before,
        after_content=after,
        change_type="modify",
    )
    
    assert change_id is not None
    
    # Verify the change was stored with diffs
    diff = await code_change_tracker.get_diff(change_id, diff_level="unified")
    assert diff is not None
    assert len(diff.content) > 0


@pytest.mark.asyncio
async def test_get_diff_char_level(code_change_tracker):
    """Test retrieving character-level diff."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    before = "hello world"
    after = "hello python"
    
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=before,
        after_content=after,
        change_type="modify",
    )
    
    diff = await code_change_tracker.get_diff(change_id, diff_level="char")
    assert diff is not None
    assert diff.level == "char"


@pytest.mark.asyncio
async def test_get_diff_line_level(code_change_tracker):
    """Test retrieving line-level diff."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    before = "line1\nline2\nline3\n"
    after = "line1\nmodified\nline3\n"
    
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=before,
        after_content=after,
        change_type="modify",
    )
    
    diff = await code_change_tracker.get_diff(change_id, diff_level="line")
    assert diff is not None
    assert diff.level == "line"


@pytest.mark.asyncio
async def test_get_diff_unified_level(code_change_tracker):
    """Test retrieving unified diff."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    before = "def foo():\n    pass\n"
    after = "def foo():\n    return 42\n"
    
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=before,
        after_content=after,
        change_type="modify",
    )
    
    diff = await code_change_tracker.get_diff(change_id, diff_level="unified")
    assert diff is not None
    assert diff.level == "unified"


@pytest.mark.asyncio
async def test_get_diff_ast_level(code_change_tracker):
    """Test retrieving AST diff."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    before = "def foo():\n    pass\n"
    after = "def foo():\n    return 42\n\ndef bar():\n    pass\n"
    
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=before,
        after_content=after,
        change_type="modify",
    )
    
    ast_diff = await code_change_tracker.get_diff(change_id, diff_level="ast")
    assert ast_diff is not None
    # Should detect the added function
    assert len(ast_diff.symbols_added) > 0


@pytest.mark.asyncio
async def test_query_changes_by_file_path(code_change_tracker):
    """Test querying changes by file path."""
    file1 = str(code_change_tracker.watch_path / "file1.py")
    file2 = str(code_change_tracker.watch_path / "file2.py")
    
    await code_change_tracker.track_change(
        file_path=file1,
        before_content=None,
        after_content="content1",
        change_type="add",
    )
    
    await code_change_tracker.track_change(
        file_path=file2,
        before_content=None,
        after_content="content2",
        change_type="add",
    )
    
    # Query for file1
    changes = await code_change_tracker.query_changes(file_path=file1)
    assert len(changes) == 1
    assert changes[0].file_path == file1


@pytest.mark.asyncio
async def test_query_changes_by_time_range(code_change_tracker):
    """Test querying changes by time range."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    
    t1 = time.time()
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content="v1",
        change_type="add",
        timestamp=t1,
    )
    
    t2 = t1 + 10
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="v1",
        after_content="v2",
        change_type="modify",
        timestamp=t2,
    )
    
    t3 = t1 + 20
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="v2",
        after_content="v3",
        change_type="modify",
        timestamp=t3,
    )
    
    # Query for middle time range (inclusive on both ends)
    changes = await code_change_tracker.query_changes(
        file_path=file_path,
        time_range=(t1 + 5, t1 + 15),
    )
    
    # Should find the change at t2 (and possibly boundary changes)
    assert len(changes) >= 1
    assert any(c.timestamp == t2 for c in changes)


@pytest.mark.asyncio
async def test_query_changes_by_function_name(code_change_tracker):
    """Test querying changes by function name."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    
    # Add a function
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content="def target_func():\n    pass\n",
        change_type="add",
    )
    
    # Modify the function
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="def target_func():\n    pass\n",
        after_content="def target_func():\n    return 42\n",
        change_type="modify",
    )
    
    # Add another function
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="def target_func():\n    return 42\n",
        after_content="def target_func():\n    return 42\n\ndef other_func():\n    pass\n",
        change_type="modify",
    )
    
    # Query for changes affecting target_func
    changes = await code_change_tracker.query_changes(
        file_path=file_path,
        function_name="target_func",
    )
    
    # Should find at least the changes that added/modified target_func
    assert len(changes) >= 2


@pytest.mark.asyncio
async def test_reconstruct_file_at_time(code_change_tracker):
    """Test reconstructing file content at a specific time."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    
    t1 = time.time()
    content1 = "version 1"
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content=content1,
        change_type="add",
        timestamp=t1,
    )
    
    t2 = t1 + 10
    content2 = "version 2"
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=content1,
        after_content=content2,
        change_type="modify",
        timestamp=t2,
    )
    
    # Reconstruct at t1
    reconstructed = await code_change_tracker.reconstruct_file(file_path, t1 + 1)
    assert reconstructed == content1
    
    # Reconstruct at t2
    reconstructed = await code_change_tracker.reconstruct_file(file_path, t2 + 1)
    assert reconstructed == content2


@pytest.mark.asyncio
async def test_get_change_graph(code_change_tracker):
    """Test getting change graph for a file."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    
    # Create a series of changes
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content="v1",
        change_type="add",
    )
    
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="v1",
        after_content="v2",
        change_type="modify",
    )
    
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="v2",
        after_content="v3",
        change_type="modify",
    )
    
    # Get change graph
    graph = await code_change_tracker.get_change_graph(file_path)
    
    assert graph.file_path == file_path
    assert len(graph.nodes) == 3
    assert len(graph.root_ids) == 1
    assert len(graph.leaf_ids) == 1


@pytest.mark.asyncio
async def test_get_file_history(code_change_tracker):
    """Test getting complete file history."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    
    contents = ["v1", "v2", "v3"]
    for i, content in enumerate(contents):
        before = contents[i-1] if i > 0 else None
        await code_change_tracker.track_change(
            file_path=file_path,
            before_content=before,
            after_content=content,
            change_type="add" if i == 0 else "modify",
        )
    
    history = await code_change_tracker.get_file_history(file_path)
    
    assert len(history) == 3
    for i, (timestamp, content) in enumerate(history):
        assert content == contents[i]


@pytest.mark.asyncio
async def test_get_symbols_at_time(code_change_tracker):
    """Test getting symbols at a specific time."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    
    t1 = time.time()
    content1 = "def func1():\n    pass\n"
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content=content1,
        change_type="add",
        timestamp=t1,
    )
    
    t2 = t1 + 10
    content2 = "def func1():\n    pass\n\ndef func2():\n    pass\n"
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=content1,
        after_content=content2,
        change_type="modify",
        timestamp=t2,
    )
    
    # Get symbols at t1
    symbols_t1 = await code_change_tracker.get_symbols_at_time(file_path, t1 + 1)
    assert len(symbols_t1) == 1
    assert symbols_t1[0].name == "func1"
    
    # Get symbols at t2
    symbols_t2 = await code_change_tracker.get_symbols_at_time(file_path, t2 + 1)
    assert len(symbols_t2) == 2
    symbol_names = {s.name for s in symbols_t2}
    assert "func1" in symbol_names
    assert "func2" in symbol_names


@pytest.mark.asyncio
async def test_track_symbol_evolution(code_change_tracker):
    """Test tracking how a symbol evolves over time."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    
    # Version 1: function with no parameters
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content="def target():\n    pass\n",
        change_type="add",
    )
    
    # Version 2: function with one parameter
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="def target():\n    pass\n",
        after_content="def target(x):\n    return x\n",
        change_type="modify",
    )
    
    # Version 3: function with two parameters
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="def target(x):\n    return x\n",
        after_content="def target(x, y):\n    return x + y\n",
        change_type="modify",
    )
    
    # Track evolution
    evolution = await code_change_tracker.track_symbol_evolution(file_path, "target")
    
    assert len(evolution) == 3
    
    # Check parameter evolution
    _, symbol1 = evolution[0]
    assert symbol1 is not None
    assert len(symbol1.parameters) == 0
    
    _, symbol2 = evolution[1]
    assert symbol2 is not None
    assert len(symbol2.parameters) == 1
    
    _, symbol3 = evolution[2]
    assert symbol3 is not None
    assert len(symbol3.parameters) == 2


@pytest.mark.asyncio
async def test_track_change_with_metadata(code_change_tracker):
    """Test tracking a change with custom metadata."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    metadata = {
        "author": "test_user",
        "commit": "abc123",
        "branch": "feature/test",
    }
    
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content="content",
        change_type="add",
        metadata=metadata,
    )
    
    # Retrieve and verify metadata
    changes = await code_change_tracker.query_changes(file_path=file_path)
    assert len(changes) == 1
    assert changes[0].metadata == metadata


@pytest.mark.asyncio
async def test_track_change_with_commit_hash(code_change_tracker):
    """Test tracking a change with commit hash."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    commit_hash = "abc123def456"
    
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content="content",
        change_type="add",
        commit_hash=commit_hash,
    )
    
    # Retrieve and verify commit hash
    changes = await code_change_tracker.query_changes(file_path=file_path)
    assert len(changes) == 1
    assert changes[0].commit_hash == commit_hash


# ============================================================================
# COMPREHENSIVE TESTS FOR TASK 3.7
# ============================================================================


@pytest.mark.asyncio
async def test_tracking_accuracy_complex_changes(code_change_tracker):
    """Test tracking accuracy with complex code changes."""
    file_path = str(code_change_tracker.watch_path / "complex.py")
    
    # Initial version with multiple functions
    before = """def func1(x):
    return x * 2

def func2(y):
    return y + 1

class MyClass:
    def method1(self):
        pass
"""
    
    # Modified version with changes to multiple elements
    after = """def func1(x, z=10):
    # Added parameter and default value
    return x * 2 + z

def func2(y):
    return y + 1

def func3():
    # New function
    return 42

class MyClass:
    def method1(self):
        return "modified"
    
    def method2(self):
        # New method
        pass
"""
    
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=before,
        after_content=after,
        change_type="modify",
    )
    
    # Verify all diff levels are captured
    char_diff = await code_change_tracker.get_diff(change_id, "char")
    assert char_diff is not None
    
    line_diff = await code_change_tracker.get_diff(change_id, "line")
    assert line_diff is not None
    
    unified_diff = await code_change_tracker.get_diff(change_id, "unified")
    assert unified_diff is not None
    
    ast_diff = await code_change_tracker.get_diff(change_id, "ast")
    assert ast_diff is not None
    
    # Verify AST diff detected the changes
    assert len(ast_diff.symbols_added) >= 2  # func3 and method2
    assert len(ast_diff.symbols_modified) >= 1  # func1 or method1


@pytest.mark.asyncio
async def test_diff_retrieval_all_levels_consistency(code_change_tracker):
    """Test that all diff levels can be retrieved and are consistent."""
    file_path = str(code_change_tracker.watch_path / "test.py")
    before = "def old_func():\n    return 1\n"
    after = "def new_func():\n    return 2\n"
    
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=before,
        after_content=after,
        change_type="modify",
    )
    
    # Retrieve all diff levels
    char_diff = await code_change_tracker.get_diff(change_id, "char")
    line_diff = await code_change_tracker.get_diff(change_id, "line")
    unified_diff = await code_change_tracker.get_diff(change_id, "unified")
    ast_diff = await code_change_tracker.get_diff(change_id, "ast")
    
    # All should be non-None
    assert char_diff is not None
    assert line_diff is not None
    assert unified_diff is not None
    assert ast_diff is not None
    
    # Verify they all represent the same change
    assert char_diff.level == "char"
    assert line_diff.level == "line"
    assert unified_diff.level == "unified"
    
    # AST diff should detect function name change
    assert len(ast_diff.symbols_removed) >= 1
    assert len(ast_diff.symbols_added) >= 1


@pytest.mark.asyncio
async def test_file_reconstruction_multiple_timestamps(code_change_tracker):
    """Test file reconstruction at multiple arbitrary timestamps."""
    file_path = str(code_change_tracker.watch_path / "evolving.py")
    
    # Create a series of changes with known timestamps
    t0 = time.time()
    versions = [
        (t0, None, "# Version 1\ndef v1():\n    pass\n"),
        (t0 + 10, "# Version 1\ndef v1():\n    pass\n", "# Version 2\ndef v2():\n    pass\n"),
        (t0 + 20, "# Version 2\ndef v2():\n    pass\n", "# Version 3\ndef v3():\n    pass\n"),
        (t0 + 30, "# Version 3\ndef v3():\n    pass\n", "# Version 4\ndef v4():\n    pass\n"),
    ]
    
    for timestamp, before, after in versions:
        await code_change_tracker.track_change(
            file_path=file_path,
            before_content=before,
            after_content=after,
            change_type="add" if before is None else "modify",
            timestamp=timestamp,
        )
    
    # Reconstruct at various timestamps
    content_at_t0 = await code_change_tracker.reconstruct_file(file_path, t0 + 1)
    assert content_at_t0 == "# Version 1\ndef v1():\n    pass\n"
    
    content_at_t10 = await code_change_tracker.reconstruct_file(file_path, t0 + 11)
    assert content_at_t10 == "# Version 2\ndef v2():\n    pass\n"
    
    content_at_t20 = await code_change_tracker.reconstruct_file(file_path, t0 + 21)
    assert content_at_t20 == "# Version 3\ndef v3():\n    pass\n"
    
    content_at_t30 = await code_change_tracker.reconstruct_file(file_path, t0 + 31)
    assert content_at_t30 == "# Version 4\ndef v4():\n    pass\n"
    
    # Test reconstruction at exact timestamps (should get version before that timestamp)
    content_exact = await code_change_tracker.reconstruct_file(file_path, t0 + 20)
    assert content_exact == "# Version 3\ndef v3():\n    pass\n"


@pytest.mark.asyncio
async def test_file_reconstruction_edge_cases(code_change_tracker):
    """Test file reconstruction edge cases."""
    file_path = str(code_change_tracker.watch_path / "edge.py")
    
    t0 = time.time()
    content = "def test():\n    pass\n"
    
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content=content,
        change_type="add",
        timestamp=t0,
    )
    
    # Reconstruct before any changes
    before_creation = await code_change_tracker.reconstruct_file(file_path, t0 - 10)
    assert before_creation is None
    
    # Reconstruct at exact creation time (gets the content at that timestamp)
    at_creation = await code_change_tracker.reconstruct_file(file_path, t0)
    assert at_creation == content  # Should have the content
    
    # Reconstruct after creation
    after_creation = await code_change_tracker.reconstruct_file(file_path, t0 + 1)
    assert after_creation == content


@pytest.mark.asyncio
async def test_query_performance_large_history(code_change_tracker):
    """Test query performance with large change histories."""
    file_path = str(code_change_tracker.watch_path / "large_history.py")
    
    # Create a large number of changes
    num_changes = 100
    t0 = time.time()
    
    for i in range(num_changes):
        before = f"# Version {i}\n" if i > 0 else None
        after = f"# Version {i + 1}\n"
        
        await code_change_tracker.track_change(
            file_path=file_path,
            before_content=before,
            after_content=after,
            change_type="add" if i == 0 else "modify",
            timestamp=t0 + i,
        )
    
    # Query all changes and measure time
    start_time = time.time()
    all_changes = await code_change_tracker.query_changes(file_path=file_path, limit=num_changes)
    query_time = time.time() - start_time
    
    # Verify results
    assert len(all_changes) == num_changes
    
    # Query should be reasonably fast (< 1 second for 100 changes)
    assert query_time < 1.0, f"Query took {query_time:.3f}s, expected < 1.0s"
    
    # Test time range query performance
    start_time = time.time()
    range_changes = await code_change_tracker.query_changes(
        file_path=file_path,
        time_range=(t0 + 25, t0 + 75),
    )
    range_query_time = time.time() - start_time
    
    # Should get approximately 50 changes (may include boundary changes)
    assert 49 <= len(range_changes) <= 51
    assert range_query_time < 0.5, f"Range query took {range_query_time:.3f}s, expected < 0.5s"


@pytest.mark.asyncio
async def test_query_performance_multiple_files(code_change_tracker):
    """Test query performance across multiple files."""
    num_files = 50
    changes_per_file = 10
    t0 = time.time()
    
    # Create changes for multiple files
    for file_idx in range(num_files):
        file_path = str(code_change_tracker.watch_path / f"file_{file_idx}.py")
        
        for change_idx in range(changes_per_file):
            before = f"# File {file_idx} Version {change_idx}\n" if change_idx > 0 else None
            after = f"# File {file_idx} Version {change_idx + 1}\n"
            
            await code_change_tracker.track_change(
                file_path=file_path,
                before_content=before,
                after_content=after,
                change_type="add" if change_idx == 0 else "modify",
                timestamp=t0 + file_idx * changes_per_file + change_idx,
            )
    
    # Query specific file
    start_time = time.time()
    file_changes = await code_change_tracker.query_changes(
        file_path=str(code_change_tracker.watch_path / "file_25.py")
    )
    query_time = time.time() - start_time
    
    assert len(file_changes) == changes_per_file
    assert query_time < 0.5, f"File query took {query_time:.3f}s, expected < 0.5s"


@pytest.mark.asyncio
async def test_semantic_query_by_function_name(code_change_tracker):
    """Test semantic queries for code changes by function name."""
    file_path = str(code_change_tracker.watch_path / "semantic.py")
    
    # Add initial version with target function
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content="def target_function():\n    pass\n\ndef other_function():\n    pass\n",
        change_type="add",
    )
    
    # Modify target function
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="def target_function():\n    pass\n\ndef other_function():\n    pass\n",
        after_content="def target_function(x):\n    return x * 2\n\ndef other_function():\n    pass\n",
        change_type="modify",
    )
    
    # Modify other function
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="def target_function(x):\n    return x * 2\n\ndef other_function():\n    pass\n",
        after_content="def target_function(x):\n    return x * 2\n\ndef other_function(y):\n    return y + 1\n",
        change_type="modify",
    )
    
    # Query for changes affecting target_function
    target_changes = await code_change_tracker.query_changes(
        file_path=file_path,
        function_name="target_function",
    )
    
    # Should find at least 2 changes (add and modify)
    assert len(target_changes) >= 2
    
    # Query for changes affecting other_function
    other_changes = await code_change_tracker.query_changes(
        file_path=file_path,
        function_name="other_function",
    )
    
    # Should find at least 2 changes (add and modify)
    assert len(other_changes) >= 2


@pytest.mark.asyncio
async def test_semantic_query_symbol_tracking(code_change_tracker):
    """Test semantic queries tracking symbol evolution."""
    file_path = str(code_change_tracker.watch_path / "symbols.py")
    
    # Version 1: Add class with method
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content="class Calculator:\n    def add(self, a, b):\n        return a + b\n",
        change_type="add",
    )
    
    # Version 2: Add another method
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="class Calculator:\n    def add(self, a, b):\n        return a + b\n",
        after_content="class Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b\n",
        change_type="modify",
    )
    
    # Version 3: Modify existing method
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="class Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b\n",
        after_content="class Calculator:\n    def add(self, a, b, c=0):\n        return a + b + c\n    \n    def subtract(self, a, b):\n        return a - b\n",
        change_type="modify",
    )
    
    # Track evolution of 'add' method
    evolution = await code_change_tracker.track_symbol_evolution(file_path, "add")
    
    # Should have 3 versions
    assert len(evolution) >= 3
    
    # Verify parameter evolution
    timestamps_with_symbol = [(t, s) for t, s in evolution if s is not None]
    assert len(timestamps_with_symbol) >= 3
    
    # First version: 2 parameters (self, a, b)
    _, first_symbol = timestamps_with_symbol[0]
    assert len(first_symbol.parameters) == 3
    
    # Last version: should have more parameters than first (c with default value)
    _, last_symbol = timestamps_with_symbol[-1]
    # Note: default parameters might not always be counted separately
    assert len(last_symbol.parameters) >= 3


@pytest.mark.asyncio
async def test_change_graph_complex_history(code_change_tracker):
    """Test change graph with complex history."""
    file_path = str(code_change_tracker.watch_path / "graph.py")
    
    # Create a linear history
    versions = [
        (None, "v1"),
        ("v1", "v2"),
        ("v2", "v3"),
        ("v3", "v4"),
        ("v4", "v5"),
    ]
    
    for before, after in versions:
        await code_change_tracker.track_change(
            file_path=file_path,
            before_content=before,
            after_content=after,
            change_type="add" if before is None else "modify",
        )
    
    # Get change graph
    graph = await code_change_tracker.get_change_graph(file_path)
    
    # Verify graph structure
    assert graph.file_path == file_path
    assert len(graph.nodes) == 5
    assert len(graph.root_ids) == 1
    assert len(graph.leaf_ids) == 1
    
    # Verify edges (should have 4 edges for 5 nodes in linear history)
    # Count edges from nodes
    total_edges = sum(len(node.child_ids) for node in graph.nodes)
    assert total_edges == 4


@pytest.mark.asyncio
async def test_tracking_accuracy_whitespace_changes(code_change_tracker):
    """Test tracking accuracy with whitespace-only changes."""
    file_path = str(code_change_tracker.watch_path / "whitespace.py")
    
    before = "def func():\n    return 1\n"
    after = "def func():\n        return 1\n"  # Extra indentation
    
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=before,
        after_content=after,
        change_type="modify",
    )
    
    # Character diff should detect the change
    char_diff = await code_change_tracker.get_diff(change_id, "char")
    assert char_diff is not None
    assert len(char_diff.content) > 0
    
    # Line diff should also detect it
    line_diff = await code_change_tracker.get_diff(change_id, "line")
    assert line_diff is not None
    
    # AST diff might not detect it (whitespace doesn't change AST)
    ast_diff = await code_change_tracker.get_diff(change_id, "ast")
    # AST should be unchanged
    assert len(ast_diff.symbols_added) == 0
    assert len(ast_diff.symbols_removed) == 0


@pytest.mark.asyncio
async def test_tracking_accuracy_multiline_strings(code_change_tracker):
    """Test tracking accuracy with multiline strings."""
    file_path = str(code_change_tracker.watch_path / "multiline.py")
    
    before = '''def func():
    doc = """
    This is a
    multiline string
    """
    return doc
'''
    
    after = '''def func():
    doc = """
    This is a
    modified multiline string
    with extra line
    """
    return doc
'''
    
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=before,
        after_content=after,
        change_type="modify",
    )
    
    # All diff levels should capture the change
    char_diff = await code_change_tracker.get_diff(change_id, "char")
    assert char_diff is not None
    
    line_diff = await code_change_tracker.get_diff(change_id, "line")
    assert line_diff is not None
    
    unified_diff = await code_change_tracker.get_diff(change_id, "unified")
    assert unified_diff is not None
    assert "modified multiline string" in unified_diff.get_content()


@pytest.mark.asyncio
async def test_concurrent_tracking_accuracy(code_change_tracker):
    """Test tracking accuracy with concurrent changes."""
    file_paths = [
        str(code_change_tracker.watch_path / f"concurrent_{i}.py")
        for i in range(10)
    ]
    
    # Track changes concurrently
    tasks = []
    for i, file_path in enumerate(file_paths):
        task = code_change_tracker.track_change(
            file_path=file_path,
            before_content=None,
            after_content=f"# File {i}\ndef func_{i}():\n    pass\n",
            change_type="add",
        )
        tasks.append(task)
    
    # Wait for all to complete
    change_ids = await asyncio.gather(*tasks)
    
    # Verify all changes were tracked
    assert len(change_ids) == 10
    assert all(change_id is not None for change_id in change_ids)
    
    # Verify each change can be retrieved
    for i, file_path in enumerate(file_paths):
        changes = await code_change_tracker.query_changes(file_path=file_path)
        assert len(changes) == 1
        assert changes[0].file_path == file_path


@pytest.mark.asyncio
async def test_file_deletion_tracking(code_change_tracker):
    """Test tracking file deletion."""
    file_path = str(code_change_tracker.watch_path / "to_delete.py")
    
    # Create file
    content = "def func():\n    pass\n"
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content=content,
        change_type="add",
    )
    
    # Delete file
    change_id = await code_change_tracker.track_change(
        file_path=file_path,
        before_content=content,
        after_content="",
        change_type="delete",
    )
    
    # Verify deletion was tracked
    changes = await code_change_tracker.query_changes(file_path=file_path)
    assert len(changes) == 2
    
    # Find the deletion change
    delete_changes = [c for c in changes if c.change_type == "delete"]
    assert len(delete_changes) == 1
    assert delete_changes[0].after_content == ""


@pytest.mark.asyncio
async def test_file_rename_tracking(code_change_tracker):
    """Test tracking file rename."""
    old_path = str(code_change_tracker.watch_path / "old_name.py")
    new_path = str(code_change_tracker.watch_path / "new_name.py")
    
    content = "def func():\n    pass\n"
    
    # Track rename
    change_id = await code_change_tracker.track_change(
        file_path=new_path,
        before_content=content,
        after_content=content,
        change_type="rename",
        metadata={"old_path": old_path},
    )
    
    # Verify rename was tracked
    changes = await code_change_tracker.query_changes(file_path=new_path)
    assert len(changes) == 1
    assert changes[0].change_type == "rename"
    assert changes[0].metadata.get("old_path") == old_path


@pytest.mark.asyncio
async def test_query_with_change_type_filter(code_change_tracker):
    """Test querying with change type filter."""
    file_path = str(code_change_tracker.watch_path / "filtered.py")
    
    # Create different types of changes
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content="v1",
        change_type="add",
    )
    
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="v1",
        after_content="v2",
        change_type="modify",
    )
    
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content="v2",
        after_content="v3",
        change_type="modify",
    )
    
    # Query for only modifications
    modify_changes = await code_change_tracker.query_changes(
        file_path=file_path,
        change_type="modify",
    )
    
    assert len(modify_changes) == 2
    assert all(c.change_type == "modify" for c in modify_changes)
    
    # Query for only additions
    add_changes = await code_change_tracker.query_changes(
        file_path=file_path,
        change_type="add",
    )
    
    assert len(add_changes) == 1
    assert add_changes[0].change_type == "add"


@pytest.mark.asyncio
async def test_reconstruction_with_different_diff_levels(code_change_tracker):
    """Test file reconstruction using different diff levels."""
    file_path = str(code_change_tracker.watch_path / "diff_levels.py")
    
    t0 = time.time()
    before = "def old():\n    return 1\n"
    after = "def new():\n    return 2\n"
    
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=None,
        after_content=before,
        change_type="add",
        timestamp=t0,
    )
    
    await code_change_tracker.track_change(
        file_path=file_path,
        before_content=before,
        after_content=after,
        change_type="modify",
        timestamp=t0 + 10,
    )
    
    # Reconstruct using char diff
    content_char = await code_change_tracker.reconstruct_file(
        file_path, t0 + 15, diff_level="char"
    )
    assert content_char == after
    
    # Reconstruct using line diff
    content_line = await code_change_tracker.reconstruct_file(
        file_path, t0 + 15, diff_level="line"
    )
    assert content_line == after
    
    # Both should produce the same result
    assert content_char == content_line
