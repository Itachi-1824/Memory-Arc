"""Tests for file system monitoring."""

import asyncio
import time
from pathlib import Path

import pytest

from core.infinite.file_watcher import FileSystemWatcher, FileChange, BatchedChanges


class TestFileChange:
    """Test FileChange dataclass."""
    
    def test_file_change_creation(self):
        """Test creating a FileChange."""
        change = FileChange(
            file_path="/test/file.py",
            change_type="modified",
            timestamp=time.time()
        )
        
        assert change.file_path == "/test/file.py"
        assert change.change_type == "modified"
        assert change.old_path is None
    
    def test_file_change_with_move(self):
        """Test FileChange for move operation."""
        change = FileChange(
            file_path="/test/new.py",
            change_type="moved",
            timestamp=time.time(),
            old_path="/test/old.py"
        )
        
        assert change.file_path == "/test/new.py"
        assert change.old_path == "/test/old.py"
        assert change.change_type == "moved"
    
    def test_file_change_hashable(self):
        """Test that FileChange is hashable for deduplication."""
        change1 = FileChange(
            file_path="/test/file.py",
            change_type="modified",
            timestamp=time.time()
        )
        change2 = FileChange(
            file_path="/test/file.py",
            change_type="modified",
            timestamp=time.time()
        )
        
        # Should be able to use in set
        changes = {change1, change2}
        assert len(changes) == 1  # Deduplicated


class TestBatchedChanges:
    """Test BatchedChanges dataclass."""
    
    def test_batched_changes_creation(self):
        """Test creating a BatchedChanges."""
        batch = BatchedChanges()
        
        assert len(batch.changes) == 0
        assert batch.end_time is None
        assert batch.start_time > 0
    
    def test_add_change(self):
        """Test adding changes to batch."""
        batch = BatchedChanges()
        change = FileChange(
            file_path="/test/file.py",
            change_type="modified",
            timestamp=time.time()
        )
        
        batch.add_change(change)
        assert len(batch.changes) == 1
        assert batch.changes[0] == change
    
    def test_finalize(self):
        """Test finalizing a batch."""
        batch = BatchedChanges()
        batch.finalize()
        
        assert batch.end_time is not None
        assert batch.end_time >= batch.start_time
    
    def test_is_multi_file(self):
        """Test detecting multi-file batches."""
        batch = BatchedChanges()
        
        # Single file
        batch.add_change(FileChange("/test/file1.py", "modified", time.time()))
        assert not batch.is_multi_file()
        
        # Multiple files
        batch.add_change(FileChange("/test/file2.py", "modified", time.time()))
        assert batch.is_multi_file()


class TestFileSystemWatcher:
    """Test FileSystemWatcher class."""
    
    @pytest.fixture
    def watch_dir(self, temp_dir: Path) -> Path:
        """Create a directory to watch."""
        watch_path = temp_dir / "watch"
        watch_path.mkdir()
        return watch_path
    
    @pytest.fixture
    def changes_received(self):
        """Track changes received by callback."""
        return []
    
    @pytest.fixture
    async def callback(self, changes_received):
        """Create a callback function."""
        async def _callback(changes: list[FileChange]):
            changes_received.extend(changes)
        return _callback
    
    @pytest.mark.asyncio
    async def test_watcher_initialization(self, watch_dir: Path, callback):
        """Test creating a FileSystemWatcher."""
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback,
            debounce_seconds=0.1
        )
        
        assert watcher.watch_path == watch_dir
        assert not watcher.is_running()
        assert len(watcher.ignore_patterns) > 0  # Has default patterns
    
    @pytest.mark.asyncio
    async def test_start_stop(self, watch_dir: Path, callback):
        """Test starting and stopping the watcher."""
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback,
            debounce_seconds=0.1
        )
        
        watcher.start()
        assert watcher.is_running()
        
        watcher.stop()
        assert not watcher.is_running()
    
    @pytest.mark.asyncio
    async def test_file_creation_detection(self, watch_dir: Path, callback, changes_received):
        """Test detecting file creation."""
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback,
            debounce_seconds=0.2
        )
        
        watcher.start()
        
        try:
            # Create a file
            test_file = watch_dir / "test.py"
            test_file.write_text("print('hello')")
            
            # Wait for debounce + processing
            await asyncio.sleep(0.5)
            
            # Check that change was detected
            assert len(changes_received) > 0
            assert any(c.change_type == "created" for c in changes_received)
            assert any("test.py" in c.file_path for c in changes_received)
        finally:
            watcher.stop()
    
    @pytest.mark.asyncio
    async def test_file_modification_detection(self, watch_dir: Path, callback, changes_received):
        """Test detecting file modification."""
        # Create file first
        test_file = watch_dir / "test.py"
        test_file.write_text("print('hello')")
        
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback,
            debounce_seconds=0.2
        )
        
        watcher.start()
        
        try:
            # Wait a bit to ensure watcher is ready
            await asyncio.sleep(0.1)
            
            # Modify the file
            test_file.write_text("print('world')")
            
            # Wait for debounce + processing
            await asyncio.sleep(0.5)
            
            # Check that change was detected
            assert len(changes_received) > 0
            assert any(c.change_type == "modified" for c in changes_received)
        finally:
            watcher.stop()
    
    @pytest.mark.asyncio
    async def test_file_deletion_detection(self, watch_dir: Path, callback, changes_received):
        """Test detecting file deletion."""
        # Create file first
        test_file = watch_dir / "test.py"
        test_file.write_text("print('hello')")
        
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback,
            debounce_seconds=0.2
        )
        
        watcher.start()
        
        try:
            # Wait a bit to ensure watcher is ready
            await asyncio.sleep(0.1)
            
            # Delete the file
            test_file.unlink()
            
            # Wait for debounce + processing
            await asyncio.sleep(0.5)
            
            # Check that change was detected
            assert len(changes_received) > 0
            assert any(c.change_type == "deleted" for c in changes_received)
        finally:
            watcher.stop()
    
    @pytest.mark.asyncio
    async def test_debouncing_logic(self, watch_dir: Path, callback, changes_received):
        """Test that debouncing prevents duplicate rapid changes."""
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback,
            debounce_seconds=0.3
        )
        
        watcher.start()
        
        try:
            test_file = watch_dir / "test.py"
            
            # Make rapid changes
            for i in range(5):
                test_file.write_text(f"print({i})")
                await asyncio.sleep(0.05)  # Faster than debounce
            
            # Wait for debounce
            await asyncio.sleep(0.5)
            
            # Should have received changes, but debounced
            # The exact count depends on timing, but should be less than 5
            assert len(changes_received) > 0
            # Debouncing should reduce the number of callbacks
            assert len(changes_received) < 5
        finally:
            watcher.stop()
    
    @pytest.mark.asyncio
    async def test_ignore_patterns(self, watch_dir: Path, callback, changes_received):
        """Test that ignore patterns work."""
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback,
            debounce_seconds=0.2,
            ignore_patterns=["*.pyc", "*.log"]
        )
        
        watcher.start()
        
        try:
            # Create ignored file
            ignored_file = watch_dir / "test.pyc"
            ignored_file.write_text("bytecode")
            
            # Create non-ignored file
            normal_file = watch_dir / "test.py"
            normal_file.write_text("print('hello')")
            
            # Wait for debounce
            await asyncio.sleep(0.5)
            
            # Should only detect the .py file
            assert len(changes_received) > 0
            assert all(".pyc" not in c.file_path for c in changes_received)
            assert any(".py" in c.file_path for c in changes_received)
        finally:
            watcher.stop()
    
    @pytest.mark.asyncio
    async def test_default_ignore_patterns(self, watch_dir: Path, callback, changes_received):
        """Test that default ignore patterns are applied."""
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback,
            debounce_seconds=0.2
        )
        
        # Check default patterns exist
        assert "*.pyc" in watcher.ignore_patterns
        assert "__pycache__/*" in watcher.ignore_patterns
        assert ".git/*" in watcher.ignore_patterns
    
    @pytest.mark.asyncio
    async def test_add_ignore_pattern(self, watch_dir: Path, callback):
        """Test adding custom ignore patterns."""
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback
        )
        
        watcher.add_ignore_pattern("*.tmp")
        assert "*.tmp" in watcher.ignore_patterns
        
        # Should not add duplicates
        watcher.add_ignore_pattern("*.tmp")
        assert watcher.ignore_patterns.count("*.tmp") == 1
    
    @pytest.mark.asyncio
    async def test_load_gitignore(self, watch_dir: Path, callback):
        """Test loading patterns from .gitignore."""
        # Create a .gitignore file
        gitignore = watch_dir / ".gitignore"
        gitignore.write_text("""
# Comment
*.log
temp/
node_modules/

# Another comment
.env
""")
        
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback
        )
        
        watcher.load_gitignore()
        
        # Check that patterns were loaded
        assert any("*.log" in p for p in watcher.ignore_patterns)
        assert any("temp" in p for p in watcher.ignore_patterns)
        assert any("node_modules" in p for p in watcher.ignore_patterns)
        assert any(".env" in p for p in watcher.ignore_patterns)
    
    @pytest.mark.asyncio
    async def test_batch_change_detection(self, watch_dir: Path, callback, changes_received):
        """Test detecting batch changes for multi-file operations."""
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback,
            debounce_seconds=0.2,
            batch_window_seconds=1.0
        )
        
        watcher.start()
        
        try:
            # Create multiple files quickly (simulating multi-file operation)
            for i in range(3):
                test_file = watch_dir / f"test{i}.py"
                test_file.write_text(f"print({i})")
            
            # Wait for debounce
            await asyncio.sleep(0.5)
            
            # Should have detected multiple files
            assert len(changes_received) >= 3
            unique_files = {c.file_path for c in changes_received}
            assert len(unique_files) >= 3
        finally:
            watcher.stop()
    
    @pytest.mark.asyncio
    async def test_nested_directory_monitoring(self, watch_dir: Path, callback, changes_received):
        """Test monitoring nested directories."""
        # Create nested structure
        nested_dir = watch_dir / "subdir" / "nested"
        nested_dir.mkdir(parents=True)
        
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback,
            debounce_seconds=0.2
        )
        
        watcher.start()
        
        try:
            # Create file in nested directory
            test_file = nested_dir / "test.py"
            test_file.write_text("print('nested')")
            
            # Wait for debounce
            await asyncio.sleep(0.5)
            
            # Should detect changes in nested directories
            assert len(changes_received) > 0
            assert any("nested" in c.file_path for c in changes_received)
        finally:
            watcher.stop()
    
    @pytest.mark.asyncio
    async def test_should_ignore_method(self, watch_dir: Path, callback):
        """Test the _should_ignore method."""
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback,
            ignore_patterns=["*.pyc", "test_*"]
        )
        
        # Should ignore
        assert watcher._should_ignore(watch_dir / "file.pyc")
        assert watcher._should_ignore(watch_dir / "test_something.py")
        
        # Should not ignore
        assert not watcher._should_ignore(watch_dir / "file.py")
        assert not watcher._should_ignore(watch_dir / "main.py")
    
    @pytest.mark.asyncio
    async def test_multiple_rapid_changes_same_file(self, watch_dir: Path, callback, changes_received):
        """Test handling multiple rapid changes to the same file."""
        watcher = FileSystemWatcher(
            watch_path=watch_dir,
            callback=callback,
            debounce_seconds=0.3
        )
        
        watcher.start()
        
        try:
            test_file = watch_dir / "test.py"
            
            # Make multiple rapid changes to same file
            for i in range(10):
                test_file.write_text(f"version {i}")
                await asyncio.sleep(0.02)
            
            # Wait for debounce
            await asyncio.sleep(0.5)
            
            # Should have deduplicated changes to same file
            assert len(changes_received) > 0
            # Count changes to test.py
            test_py_changes = [c for c in changes_received if "test.py" in c.file_path]
            # Should be significantly less than 10 due to deduplication
            assert len(test_py_changes) < 10
        finally:
            watcher.stop()
