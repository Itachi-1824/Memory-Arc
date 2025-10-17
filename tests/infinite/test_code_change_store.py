"""Unit tests for CodeChangeStore."""

import pytest
import asyncio
import time
import uuid

from core.infinite.code_change_store import (
    CodeChangeStore,
    CodeChange,
    ChangeGraph,
    ChangeGraphNode,
)
from core.infinite.diff_generator import DiffGenerator
from core.infinite.ast_diff import ASTDiffEngine


@pytest.fixture
async def code_change_store(test_db_path):
    """Create and initialize a code change store for testing."""
    store = CodeChangeStore(test_db_path)
    await store.initialize()
    yield store
    await store.close()


def create_test_change(
    file_path: str = "test.py",
    change_type: str = "modify",
    before_content: str | None = "old content",
    after_content: str = "new content",
    **kwargs
) -> CodeChange:
    """Helper to create test code change objects."""
    return CodeChange(
        id=kwargs.get("id", str(uuid.uuid4())),
        file_path=file_path,
        change_type=change_type,
        timestamp=kwargs.get("timestamp", time.time()),
        before_content=before_content,
        after_content=after_content,
        commit_hash=kwargs.get("commit_hash"),
        metadata=kwargs.get("metadata", {}),
    )


# Test CodeChange Storage
@pytest.mark.asyncio
async def test_add_change_success(code_change_store):
    """Test successful code change insertion."""
    change = create_test_change(
        file_path="test.py",
        before_content="def old():\n    pass",
        after_content="def new():\n    return True"
    )
    
    result = await code_change_store.add_change(change, compute_diffs=True)
    assert result is True
    
    # Verify change was stored
    retrieved = await code_change_store.get_change(change.id)
    assert retrieved is not None
    assert retrieved.id == change.id
    assert retrieved.file_path == change.file_path
    assert retrieved.before_content == change.before_content
    assert retrieved.after_content == change.after_content


@pytest.mark.asyncio
async def test_add_change_with_diffs(code_change_store):
    """Test that diffs are computed and stored."""
    change = create_test_change(
        before_content="line 1\nline 2\nline 3",
        after_content="line 1\nmodified line 2\nline 3"
    )
    
    result = await code_change_store.add_change(change, compute_diffs=True)
    assert result is True
    
    retrieved = await code_change_store.get_change(change.id)
    assert retrieved.char_diff is not None
    assert retrieved.line_diff is not None
    assert retrieved.unified_diff is not None


@pytest.mark.asyncio
async def test_add_change_without_diffs(code_change_store):
    """Test adding change without computing diffs."""
    change = create_test_change()
    
    result = await code_change_store.add_change(change, compute_diffs=False)
    assert result is True
    
    retrieved = await code_change_store.get_change(change.id)
    assert retrieved.char_diff is None
    assert retrieved.line_diff is None
    assert retrieved.unified_diff is None


@pytest.mark.asyncio
async def test_add_change_with_metadata(code_change_store):
    """Test adding change with metadata."""
    metadata = {"author": "test_user", "branch": "main", "tags": ["feature"]}
    change = create_test_change(metadata=metadata)
    
    await code_change_store.add_change(change)
    retrieved = await code_change_store.get_change(change.id)
    
    assert retrieved.metadata == metadata


@pytest.mark.asyncio
async def test_add_change_file_creation(code_change_store):
    """Test adding a file creation change."""
    change = create_test_change(
        change_type="add",
        before_content=None,
        after_content="# New file\nprint('hello')"
    )
    
    result = await code_change_store.add_change(change, compute_diffs=False)
    assert result is True
    
    retrieved = await code_change_store.get_change(change.id)
    assert retrieved.change_type == "add"
    assert retrieved.before_content is None


@pytest.mark.asyncio
async def test_add_change_file_deletion(code_change_store):
    """Test adding a file deletion change."""
    change = create_test_change(
        change_type="delete",
        before_content="old content",
        after_content=""
    )
    
    result = await code_change_store.add_change(change)
    assert result is True
    
    retrieved = await code_change_store.get_change(change.id)
    assert retrieved.change_type == "delete"


# Test Change Retrieval
@pytest.mark.asyncio
async def test_get_change_not_found(code_change_store):
    """Test retrieving non-existent change returns None."""
    result = await code_change_store.get_change("nonexistent_id")
    assert result is None


# Test Query Changes
@pytest.mark.asyncio
async def test_query_changes_by_file_path(code_change_store):
    """Test querying changes by file path."""
    change1 = create_test_change(file_path="file1.py")
    change2 = create_test_change(file_path="file1.py")
    change3 = create_test_change(file_path="file2.py")
    
    await code_change_store.add_change(change1)
    await code_change_store.add_change(change2)
    await code_change_store.add_change(change3)
    
    results = await code_change_store.query_changes(file_path="file1.py")
    assert len(results) == 2
    assert all(c.file_path == "file1.py" for c in results)


@pytest.mark.asyncio
async def test_query_changes_by_type(code_change_store):
    """Test querying changes by change type."""
    change1 = create_test_change(change_type="add")
    change2 = create_test_change(change_type="modify")
    change3 = create_test_change(change_type="add")
    
    await code_change_store.add_change(change1)
    await code_change_store.add_change(change2)
    await code_change_store.add_change(change3)
    
    results = await code_change_store.query_changes(change_type="add")
    assert len(results) == 2
    assert all(c.change_type == "add" for c in results)


@pytest.mark.asyncio
async def test_query_changes_by_time_range(code_change_store):
    """Test querying changes by time range."""
    base_time = time.time()
    
    change1 = create_test_change(timestamp=base_time)
    change2 = create_test_change(timestamp=base_time + 10)
    change3 = create_test_change(timestamp=base_time + 20)
    
    await code_change_store.add_change(change1)
    await code_change_store.add_change(change2)
    await code_change_store.add_change(change3)
    
    results = await code_change_store.query_changes(
        time_range=(base_time + 5, base_time + 15)
    )
    assert len(results) == 1
    assert results[0].id == change2.id


@pytest.mark.asyncio
async def test_query_changes_with_limit_offset(code_change_store):
    """Test querying changes with pagination."""
    for i in range(10):
        change = create_test_change(file_path=f"file{i}.py")
        await code_change_store.add_change(change)
    
    # Get first page
    page1 = await code_change_store.query_changes(limit=3, offset=0)
    assert len(page1) == 3
    
    # Get second page
    page2 = await code_change_store.query_changes(limit=3, offset=3)
    assert len(page2) == 3
    
    # Verify no overlap
    page1_ids = {c.id for c in page1}
    page2_ids = {c.id for c in page2}
    assert len(page1_ids & page2_ids) == 0


@pytest.mark.asyncio
async def test_query_changes_ordered_by_time(code_change_store):
    """Test that query results are ordered by timestamp (newest first)."""
    base_time = time.time()
    
    change1 = create_test_change(timestamp=base_time, metadata={"order": 1})
    change2 = create_test_change(timestamp=base_time + 1, metadata={"order": 2})
    change3 = create_test_change(timestamp=base_time + 2, metadata={"order": 3})
    
    await code_change_store.add_change(change1)
    await code_change_store.add_change(change2)
    await code_change_store.add_change(change3)
    
    results = await code_change_store.query_changes()
    assert len(results) == 3
    assert results[0].metadata["order"] == 3
    assert results[1].metadata["order"] == 2
    assert results[2].metadata["order"] == 1


# Test Change Graph Construction
@pytest.mark.asyncio
async def test_get_change_graph_empty(code_change_store):
    """Test getting change graph for file with no changes."""
    graph = await code_change_store.get_change_graph("nonexistent.py")
    
    assert graph.file_path == "nonexistent.py"
    assert len(graph.nodes) == 0
    assert len(graph.root_ids) == 0
    assert len(graph.leaf_ids) == 0


@pytest.mark.asyncio
async def test_get_change_graph_single_change(code_change_store):
    """Test getting change graph for file with single change."""
    change = create_test_change(file_path="single.py")
    await code_change_store.add_change(change)
    
    graph = await code_change_store.get_change_graph("single.py")
    
    assert graph.file_path == "single.py"
    assert len(graph.nodes) == 1
    assert len(graph.root_ids) == 1
    assert len(graph.leaf_ids) == 1
    assert graph.root_ids[0] == change.id
    assert graph.leaf_ids[0] == change.id


@pytest.mark.asyncio
async def test_get_change_graph_multiple_changes(code_change_store):
    """Test getting change graph for file with multiple changes."""
    base_time = time.time()
    
    change1 = create_test_change(file_path="multi.py", timestamp=base_time)
    change2 = create_test_change(file_path="multi.py", timestamp=base_time + 1)
    change3 = create_test_change(file_path="multi.py", timestamp=base_time + 2)
    
    await code_change_store.add_change(change1)
    await code_change_store.add_change(change2)
    await code_change_store.add_change(change3)
    
    graph = await code_change_store.get_change_graph("multi.py")
    
    assert graph.file_path == "multi.py"
    assert len(graph.nodes) == 3
    assert len(graph.root_ids) == 1
    assert len(graph.leaf_ids) == 1
    assert graph.root_ids[0] == change1.id
    assert graph.leaf_ids[0] == change3.id
    
    # Verify parent-child relationships
    assert graph.nodes[0].parent_ids == []
    assert graph.nodes[0].child_ids == [change2.id]
    assert graph.nodes[1].parent_ids == [change1.id]
    assert graph.nodes[1].child_ids == [change3.id]
    assert graph.nodes[2].parent_ids == [change2.id]
    assert graph.nodes[2].child_ids == []


# Test File Reconstruction
@pytest.mark.asyncio
async def test_reconstruct_file_no_changes(code_change_store):
    """Test reconstructing file with no changes returns None."""
    result = await code_change_store.reconstruct_file("nonexistent.py", time.time())
    assert result is None


@pytest.mark.asyncio
async def test_reconstruct_file_at_time(code_change_store):
    """Test reconstructing file at specific time."""
    base_time = time.time()
    
    change1 = create_test_change(
        file_path="recon.py",
        timestamp=base_time,
        after_content="version 1"
    )
    change2 = create_test_change(
        file_path="recon.py",
        timestamp=base_time + 10,
        after_content="version 2"
    )
    change3 = create_test_change(
        file_path="recon.py",
        timestamp=base_time + 20,
        after_content="version 3"
    )
    
    await code_change_store.add_change(change1)
    await code_change_store.add_change(change2)
    await code_change_store.add_change(change3)
    
    # Reconstruct at different times
    content_at_5 = await code_change_store.reconstruct_file("recon.py", base_time + 5)
    assert content_at_5 == "version 1"
    
    content_at_15 = await code_change_store.reconstruct_file("recon.py", base_time + 15)
    assert content_at_15 == "version 2"
    
    content_at_25 = await code_change_store.reconstruct_file("recon.py", base_time + 25)
    assert content_at_25 == "version 3"


@pytest.mark.asyncio
async def test_reconstruct_file_before_first_change(code_change_store):
    """Test reconstructing file before first change returns None."""
    base_time = time.time()
    
    change = create_test_change(
        file_path="test.py",
        timestamp=base_time + 10,
        after_content="content"
    )
    await code_change_store.add_change(change)
    
    result = await code_change_store.reconstruct_file("test.py", base_time)
    assert result is None


# Test File History
@pytest.mark.asyncio
async def test_get_file_history_empty(code_change_store):
    """Test getting history for file with no changes."""
    history = await code_change_store.get_file_history("nonexistent.py")
    assert len(history) == 0


@pytest.mark.asyncio
async def test_get_file_history_multiple_changes(code_change_store):
    """Test getting complete file history."""
    base_time = time.time()
    
    changes = []
    for i in range(5):
        change = create_test_change(
            file_path="history.py",
            timestamp=base_time + i,
            after_content=f"version {i}"
        )
        changes.append(change)
        await code_change_store.add_change(change)
    
    history = await code_change_store.get_file_history("history.py")
    
    assert len(history) == 5
    # History should be in chronological order
    for i, (timestamp, content) in enumerate(history):
        assert content == f"version {i}"
        assert timestamp == base_time + i


@pytest.mark.asyncio
async def test_get_file_history_with_limit(code_change_store):
    """Test getting file history with limit."""
    base_time = time.time()
    
    for i in range(10):
        change = create_test_change(
            file_path="limited.py",
            timestamp=base_time + i,
            after_content=f"version {i}"
        )
        await code_change_store.add_change(change)
    
    history = await code_change_store.get_file_history("limited.py", limit=3)
    assert len(history) == 3


# Test Concurrent Operations
@pytest.mark.asyncio
async def test_concurrent_change_additions(code_change_store):
    """Test concurrent change additions."""
    changes = [
        create_test_change(file_path=f"file{i}.py")
        for i in range(20)
    ]
    
    tasks = [code_change_store.add_change(change) for change in changes]
    results = await asyncio.gather(*tasks)
    
    assert all(r is True for r in results)
    
    # Verify all changes were stored
    for change in changes:
        retrieved = await code_change_store.get_change(change.id)
        assert retrieved is not None


@pytest.mark.asyncio
async def test_concurrent_queries(code_change_store):
    """Test concurrent query operations."""
    # Add test changes
    changes = []
    for i in range(10):
        change = create_test_change(file_path=f"file{i}.py")
        await code_change_store.add_change(change)
        changes.append(change)
    
    # Perform concurrent queries
    tasks = [
        code_change_store.get_change(change.id)
        for change in changes
    ]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 10
    assert all(r is not None for r in results)
