"""File system monitoring for code change tracking."""

import asyncio
import fnmatch
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable
from collections import defaultdict

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent


@dataclass
class FileChange:
    """Represents a file system change."""
    file_path: str
    change_type: str  # 'created', 'modified', 'deleted', 'moved'
    timestamp: float
    old_path: str | None = None  # For move operations
    
    def __hash__(self):
        """Make FileChange hashable for deduplication."""
        return hash((self.file_path, self.change_type, self.old_path))
    
    def __eq__(self, other):
        """Compare FileChanges for equality (excluding timestamp for deduplication)."""
        if not isinstance(other, FileChange):
            return False
        return (
            self.file_path == other.file_path
            and self.change_type == other.change_type
            and self.old_path == other.old_path
        )


@dataclass
class BatchedChanges:
    """Collection of changes detected in a batch."""
    changes: list[FileChange] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    
    def add_change(self, change: FileChange):
        """Add a change to the batch."""
        self.changes.append(change)
    
    def finalize(self):
        """Mark the batch as complete."""
        self.end_time = time.time()
    
    def is_multi_file(self) -> bool:
        """Check if this batch contains multiple file changes."""
        unique_files = {c.file_path for c in self.changes}
        return len(unique_files) > 1


class FileSystemWatcher:
    """
    Monitors file system changes with debouncing and ignore patterns.
    
    Features:
    - Real-time file change detection
    - Debouncing to handle rapid changes
    - .gitignore-compatible ignore patterns
    - Batch change detection for multi-file operations
    """
    
    def __init__(
        self,
        watch_path: str | Path,
        callback: Callable[[list[FileChange]], Awaitable[None]],
        debounce_seconds: float = 0.5,
        ignore_patterns: list[str] | None = None,
        batch_window_seconds: float = 2.0,
    ):
        """
        Initialize file system watcher.
        
        Args:
            watch_path: Directory to monitor
            callback: Async function to call with detected changes
            debounce_seconds: Time to wait before processing changes
            ignore_patterns: List of glob patterns to ignore (e.g., ['*.pyc', '__pycache__/*'])
            batch_window_seconds: Time window to group related changes
        """
        self.watch_path = Path(watch_path).resolve()
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.batch_window_seconds = batch_window_seconds
        
        # Default ignore patterns
        self.ignore_patterns = ignore_patterns or []
        self._add_default_ignore_patterns()
        
        # Change tracking
        self._pending_changes: dict[str, FileChange] = {}
        self._change_lock = asyncio.Lock()
        self._debounce_task: asyncio.Task | None = None
        self._observer: Observer | None = None
        self._running = False
        
        # Batch tracking
        self._current_batch: BatchedChanges | None = None
        self._batch_lock = asyncio.Lock()
    
    def _add_default_ignore_patterns(self):
        """Add common ignore patterns."""
        default_patterns = [
            '*.pyc',
            '*.pyo',
            '*.pyd',
            '__pycache__/*',
            '.git/*',
            '.svn/*',
            '.hg/*',
            '*.swp',
            '*.swo',
            '*~',
            '.DS_Store',
            'node_modules/*',
            'venv/*',
            'env/*',
            '.venv/*',
            '*.egg-info/*',
            'dist/*',
            'build/*',
            '.pytest_cache/*',
            '.coverage',
            'htmlcov/*',
        ]
        
        # Add defaults that aren't already in ignore_patterns
        for pattern in default_patterns:
            if pattern not in self.ignore_patterns:
                self.ignore_patterns.append(pattern)
    
    def add_ignore_pattern(self, pattern: str):
        """Add an ignore pattern (glob format)."""
        if pattern not in self.ignore_patterns:
            self.ignore_patterns.append(pattern)
    
    def load_gitignore(self, gitignore_path: str | Path | None = None):
        """
        Load ignore patterns from .gitignore file.
        
        Args:
            gitignore_path: Path to .gitignore file. If None, looks in watch_path.
        """
        if gitignore_path is None:
            gitignore_path = self.watch_path / '.gitignore'
        else:
            gitignore_path = Path(gitignore_path)
        
        if not gitignore_path.exists():
            return
        
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Convert gitignore pattern to glob pattern
                    pattern = line.rstrip('/')
                    if not pattern.startswith('*'):
                        pattern = f'*/{pattern}'
                    self.add_ignore_pattern(pattern)
    
    def _should_ignore(self, file_path: str | Path) -> bool:
        """Check if a file should be ignored based on patterns."""
        file_path = Path(file_path)
        
        # Get relative path from watch directory
        try:
            rel_path = file_path.relative_to(self.watch_path)
        except ValueError:
            # File is outside watch path
            return True
        
        rel_path_str = str(rel_path)
        
        # Check against all ignore patterns
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(rel_path_str, pattern):
                return True
            # Also check individual path components
            if fnmatch.fnmatch(file_path.name, pattern):
                return True
        
        return False
    
    async def _handle_change(self, change: FileChange):
        """Handle a file change with debouncing."""
        if self._should_ignore(change.file_path):
            return
        
        async with self._change_lock:
            # Update or add the change (deduplicates rapid changes to same file)
            key = f"{change.file_path}:{change.change_type}"
            self._pending_changes[key] = change
            
            # Add to current batch
            async with self._batch_lock:
                if self._current_batch is None:
                    self._current_batch = BatchedChanges()
                self._current_batch.add_change(change)
            
            # Cancel existing debounce task
            if self._debounce_task and not self._debounce_task.done():
                self._debounce_task.cancel()
            
            # Start new debounce task
            self._debounce_task = asyncio.create_task(self._debounced_process())
    
    async def _debounced_process(self):
        """Process changes after debounce period."""
        try:
            await asyncio.sleep(self.debounce_seconds)
            
            async with self._change_lock:
                if not self._pending_changes:
                    return
                
                # Get all pending changes
                changes = list(self._pending_changes.values())
                self._pending_changes.clear()
            
            # Finalize batch
            async with self._batch_lock:
                if self._current_batch:
                    self._current_batch.finalize()
                    self._current_batch = None
            
            # Call the callback with the changes
            await self.callback(changes)
            
        except asyncio.CancelledError:
            # Debounce was cancelled, another change came in
            pass
    
    def start(self):
        """Start monitoring the file system."""
        if self._running:
            return
        
        self._running = True
        
        # Create event handler
        event_handler = _WatchdogHandler(self)
        
        # Create and start observer
        self._observer = Observer()
        self._observer.schedule(
            event_handler,
            str(self.watch_path),
            recursive=True
        )
        self._observer.start()
    
    def stop(self):
        """Stop monitoring the file system."""
        if not self._running:
            return
        
        self._running = False
        
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        
        # Cancel any pending debounce task
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
    
    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        return self._running
    
    async def wait_for_changes(self, timeout: float | None = None) -> list[FileChange]:
        """
        Wait for the next batch of changes.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            List of file changes detected
        """
        start_time = time.time()
        
        while True:
            async with self._change_lock:
                if self._pending_changes:
                    # Wait for debounce to complete
                    if self._debounce_task:
                        try:
                            await asyncio.wait_for(
                                self._debounce_task,
                                timeout=self.debounce_seconds + 0.1
                            )
                        except asyncio.TimeoutError:
                            pass
                    
                    changes = list(self._pending_changes.values())
                    self._pending_changes.clear()
                    return changes
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                return []
            
            await asyncio.sleep(0.1)


class _WatchdogHandler(FileSystemEventHandler):
    """Internal handler for watchdog events."""
    
    def __init__(self, watcher: FileSystemWatcher):
        self.watcher = watcher
        self._loop = asyncio.get_event_loop()
    
    def _create_change(self, event: FileSystemEvent, change_type: str) -> FileChange:
        """Create a FileChange from a watchdog event."""
        return FileChange(
            file_path=event.src_path,
            change_type=change_type,
            timestamp=time.time(),
            old_path=getattr(event, 'dest_path', None) if hasattr(event, 'dest_path') else None
        )
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if not event.is_directory:
            change = self._create_change(event, 'created')
            asyncio.run_coroutine_threadsafe(
                self.watcher._handle_change(change),
                self._loop
            )
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if not event.is_directory:
            change = self._create_change(event, 'modified')
            asyncio.run_coroutine_threadsafe(
                self.watcher._handle_change(change),
                self._loop
            )
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if not event.is_directory:
            change = self._create_change(event, 'deleted')
            asyncio.run_coroutine_threadsafe(
                self.watcher._handle_change(change),
                self._loop
            )
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file move/rename."""
        if not event.is_directory:
            change = FileChange(
                file_path=event.dest_path,
                change_type='moved',
                timestamp=time.time(),
                old_path=event.src_path
            )
            asyncio.run_coroutine_threadsafe(
                self.watcher._handle_change(change),
                self._loop
            )
