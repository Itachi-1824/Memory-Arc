"""Code change tracker integrating file monitoring, diff generation, and storage."""

import asyncio
import uuid
import time
import logging
from pathlib import Path
from typing import Literal, Callable, Awaitable

from .file_watcher import FileSystemWatcher, FileChange
from .diff_generator import DiffGenerator, DiffLevel, Diff
from .ast_diff import ASTDiffEngine, ASTDiff, LanguageType, Symbol
from .code_change_store import CodeChangeStore, CodeChange, ChangeGraph

logger = logging.getLogger(__name__)


class CodeChangeTracker:
    """
    High-level interface for tracking code changes across a codebase.
    
    Integrates:
    - File system monitoring (FileSystemWatcher)
    - Multi-level diff generation (DiffGenerator)
    - AST analysis (ASTDiffEngine)
    - Change storage and retrieval (CodeChangeStore)
    
    Features:
    - Automatic change detection and tracking
    - Multi-level diff storage (char, line, unified, AST)
    - File reconstruction at any point in time
    - Semantic queries for code changes
    - Change graph visualization
    """
    
    def __init__(
        self,
        watch_path: str | Path,
        db_path: str | Path,
        auto_track: bool = False,
        debounce_seconds: float = 0.5,
        ignore_patterns: list[str] | None = None,
    ):
        """
        Initialize code change tracker.
        
        Args:
            watch_path: Directory to monitor for changes
            db_path: Path to SQLite database for storage
            auto_track: Whether to automatically start tracking changes
            debounce_seconds: Time to wait before processing file changes
            ignore_patterns: List of glob patterns to ignore
        """
        self.watch_path = Path(watch_path).resolve()
        self.db_path = Path(db_path)
        self.auto_track = auto_track
        
        # Initialize components
        self.diff_generator = DiffGenerator()
        self.ast_engine = ASTDiffEngine()
        self.store = CodeChangeStore(
            db_path=db_path,
            diff_generator=self.diff_generator,
            ast_engine=self.ast_engine,
        )
        
        # File watcher (created on demand)
        self._watcher: FileSystemWatcher | None = None
        self._debounce_seconds = debounce_seconds
        self._ignore_patterns = ignore_patterns
        
        # File content cache for tracking changes
        self._file_cache: dict[str, str] = {}
        self._cache_lock = asyncio.Lock()
        
        # Initialization state
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the code change tracker."""
        if self._initialized:
            return
        
        # Initialize storage
        await self.store.initialize()
        
        # Load current file states into cache
        await self._load_initial_state()
        
        # Start auto-tracking if enabled
        if self.auto_track:
            await self.start_tracking()
        
        self._initialized = True
        logger.info(f"CodeChangeTracker initialized for {self.watch_path}")
    
    async def _load_initial_state(self) -> None:
        """Load current state of all files in watch path."""
        async with self._cache_lock:
            for file_path in self.watch_path.rglob('*'):
                if file_path.is_file() and self._should_track_file(file_path):
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        self._file_cache[str(file_path)] = content
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
    
    def _should_track_file(self, file_path: Path) -> bool:
        """Check if a file should be tracked (basic filter for code files)."""
        # Track common code file extensions
        code_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx',
            '.java', '.c', '.cpp', '.h', '.hpp',
            '.cs', '.go', '.rs', '.rb', '.php',
            '.swift', '.kt', '.scala', '.r',
        }
        return file_path.suffix.lower() in code_extensions
    
    async def start_tracking(self) -> None:
        """Start automatic file change tracking."""
        if not self._initialized:
            await self.initialize()
        
        if self._watcher and self._watcher.is_running():
            logger.warning("File tracking is already running")
            return
        
        # Create file watcher
        self._watcher = FileSystemWatcher(
            watch_path=self.watch_path,
            callback=self._handle_file_changes,
            debounce_seconds=self._debounce_seconds,
            ignore_patterns=self._ignore_patterns,
        )
        
        # Load .gitignore if it exists
        self._watcher.load_gitignore()
        
        # Start watching
        self._watcher.start()
        logger.info(f"Started tracking changes in {self.watch_path}")
    
    async def stop_tracking(self) -> None:
        """Stop automatic file change tracking."""
        if self._watcher:
            self._watcher.stop()
            self._watcher = None
            logger.info("Stopped tracking changes")
    
    async def _handle_file_changes(self, changes: list[FileChange]) -> None:
        """Handle detected file changes."""
        for change in changes:
            file_path = Path(change.file_path)
            
            # Skip if not a code file
            if not self._should_track_file(file_path):
                continue
            
            try:
                if change.change_type == 'created':
                    await self._handle_file_created(file_path, change.timestamp)
                elif change.change_type == 'modified':
                    await self._handle_file_modified(file_path, change.timestamp)
                elif change.change_type == 'deleted':
                    await self._handle_file_deleted(file_path, change.timestamp)
                elif change.change_type == 'moved':
                    await self._handle_file_moved(
                        Path(change.old_path) if change.old_path else file_path,
                        file_path,
                        change.timestamp
                    )
            except Exception as e:
                logger.error(f"Failed to handle change for {file_path}: {e}")
    
    async def _handle_file_created(self, file_path: Path, timestamp: float) -> None:
        """Handle file creation."""
        if not file_path.exists():
            return
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return
        
        # Track the change
        await self.track_change(
            file_path=str(file_path),
            before_content=None,
            after_content=content,
            change_type='add',
            timestamp=timestamp,
        )
        
        # Update cache
        async with self._cache_lock:
            self._file_cache[str(file_path)] = content
    
    async def _handle_file_modified(self, file_path: Path, timestamp: float) -> None:
        """Handle file modification."""
        if not file_path.exists():
            return
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return
        
        # Get previous content from cache
        async with self._cache_lock:
            before_content = self._file_cache.get(str(file_path))
        
        # Track the change
        await self.track_change(
            file_path=str(file_path),
            before_content=before_content,
            after_content=content,
            change_type='modify',
            timestamp=timestamp,
        )
        
        # Update cache
        async with self._cache_lock:
            self._file_cache[str(file_path)] = content
    
    async def _handle_file_deleted(self, file_path: Path, timestamp: float) -> None:
        """Handle file deletion."""
        # Get previous content from cache
        async with self._cache_lock:
            before_content = self._file_cache.get(str(file_path))
        
        if not before_content:
            return
        
        # Track the change
        await self.track_change(
            file_path=str(file_path),
            before_content=before_content,
            after_content="",
            change_type='delete',
            timestamp=timestamp,
        )
        
        # Remove from cache
        async with self._cache_lock:
            self._file_cache.pop(str(file_path), None)
    
    async def _handle_file_moved(
        self,
        old_path: Path,
        new_path: Path,
        timestamp: float
    ) -> None:
        """Handle file move/rename."""
        # Get content from cache or file
        async with self._cache_lock:
            content = self._file_cache.get(str(old_path))
        
        if not content and new_path.exists():
            try:
                content = new_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.warning(f"Failed to read {new_path}: {e}")
                return
        
        if not content:
            return
        
        # Track as rename
        await self.track_change(
            file_path=str(new_path),
            before_content=content,
            after_content=content,
            change_type='rename',
            timestamp=timestamp,
            metadata={'old_path': str(old_path)},
        )
        
        # Update cache
        async with self._cache_lock:
            self._file_cache.pop(str(old_path), None)
            self._file_cache[str(new_path)] = content
    
    async def track_change(
        self,
        file_path: str,
        before_content: str | None,
        after_content: str,
        change_type: Literal["add", "modify", "delete", "rename"] = "modify",
        timestamp: float | None = None,
        commit_hash: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Track a code change manually.
        
        Args:
            file_path: Path to the file
            before_content: Content before change (None for new files)
            after_content: Content after change
            change_type: Type of change
            timestamp: When the change occurred (defaults to now)
            commit_hash: Optional git commit hash
            metadata: Optional additional metadata
            
        Returns:
            Change ID
        """
        if not self._initialized:
            await self.initialize()
        
        # Generate change ID
        change_id = str(uuid.uuid4())
        
        # Use current time if not provided
        if timestamp is None:
            timestamp = time.time()
        
        # Create code change object
        code_change = CodeChange(
            id=change_id,
            file_path=file_path,
            change_type=change_type,
            timestamp=timestamp,
            before_content=before_content,
            after_content=after_content,
            commit_hash=commit_hash,
            metadata=metadata or {},
        )
        
        # Store the change (diffs and AST will be computed automatically)
        success = await self.store.add_change(
            code_change,
            compute_diffs=True,
            compute_ast=True,
        )
        
        if not success:
            logger.error(f"Failed to store change {change_id}")
        
        return change_id
    
    async def get_diff(
        self,
        change_id: str,
        diff_level: DiffLevel = "unified",
    ) -> Diff | ASTDiff | None:
        """
        Get diff for a specific change at the requested level.
        
        Args:
            change_id: ID of the change
            diff_level: Level of diff to retrieve ('char', 'line', 'unified', or 'ast')
            
        Returns:
            Diff object at the requested level, or None if not found
        """
        if not self._initialized:
            await self.initialize()
        
        # Retrieve the change
        change = await self.store.get_change(change_id)
        if not change:
            return None
        
        # Return the requested diff level
        if diff_level == "char":
            return change.char_diff
        elif diff_level == "line":
            return change.line_diff
        elif diff_level == "unified":
            return change.unified_diff
        elif diff_level == "ast":
            # AST diff needs to be reconstructed from stored data
            # For now, return the stored ast_diff
            return change.ast_diff
        
        return None
    
    async def query_changes(
        self,
        file_path: str | None = None,
        function_name: str | None = None,
        time_range: tuple[float, float] | None = None,
        change_type: str | None = None,
        semantic_query: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CodeChange]:
        """
        Query code changes with multiple filter options.
        
        Args:
            file_path: Filter by file path
            function_name: Filter by function name (searches AST symbols)
            time_range: Filter by timestamp range (start, end)
            change_type: Filter by change type
            semantic_query: Semantic search query (not implemented yet)
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching code changes
        """
        if not self._initialized:
            await self.initialize()
        
        # Query from store
        changes = await self.store.query_changes(
            file_path=file_path,
            change_type=change_type,
            time_range=time_range,
            limit=limit,
            offset=offset,
        )
        
        # Filter by function name if specified
        if function_name:
            filtered_changes = []
            for change in changes:
                # Check if the change affects the specified function
                if await self._change_affects_function(change, function_name):
                    filtered_changes.append(change)
            changes = filtered_changes
        
        # TODO: Implement semantic query filtering
        # This would require embedding the changes and doing similarity search
        
        return changes
    
    async def _change_affects_function(
        self,
        change: CodeChange,
        function_name: str
    ) -> bool:
        """Check if a change affects a specific function."""
        # Check AST diff for function changes
        if change.ast_diff:
            # Check added symbols
            for symbol in change.ast_diff.symbols_added:
                if symbol.name == function_name:
                    return True
            
            # Check removed symbols
            for symbol in change.ast_diff.symbols_removed:
                if symbol.name == function_name:
                    return True
            
            # Check modified symbols
            for before_symbol, after_symbol in change.ast_diff.symbols_modified:
                if before_symbol.name == function_name or after_symbol.name == function_name:
                    return True
        
        # Fallback: check if function name appears in the diff
        if change.unified_diff:
            diff_content = change.unified_diff.get_content()
            if function_name in diff_content:
                return True
        
        return False
    
    async def reconstruct_file(
        self,
        file_path: str,
        at_time: float,
        diff_level: DiffLevel = "char",
    ) -> str | None:
        """
        Reconstruct file content as it existed at a specific time.
        
        Args:
            file_path: Path to the file
            at_time: Timestamp to reconstruct at
            diff_level: Which diff level to use for reconstruction
            
        Returns:
            Reconstructed file content, or None if not found
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.store.reconstruct_file(
            file_path=file_path,
            at_time=at_time,
            diff_level=diff_level,
        )
    
    async def get_change_graph(self, file_path: str) -> ChangeGraph:
        """
        Get the complete change history graph for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            ChangeGraph representing the file's evolution
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.store.get_change_graph(file_path)
    
    async def get_file_history(
        self,
        file_path: str,
        limit: int = 100,
    ) -> list[tuple[float, str]]:
        """
        Get the complete history of a file as (timestamp, content) pairs.
        
        Args:
            file_path: Path to the file
            limit: Maximum number of history entries
            
        Returns:
            List of (timestamp, content) tuples
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.store.get_file_history(file_path, limit)
    
    async def get_symbols_at_time(
        self,
        file_path: str,
        at_time: float,
    ) -> list[Symbol]:
        """
        Get all symbols (functions, classes, etc.) in a file at a specific time.
        
        Args:
            file_path: Path to the file
            at_time: Timestamp to query
            
        Returns:
            List of symbols present at that time
        """
        if not self._initialized:
            await self.initialize()
        
        # Reconstruct file content at the specified time
        content = await self.reconstruct_file(file_path, at_time)
        if not content:
            return []
        
        # Detect language and extract symbols
        language = self.ast_engine.detect_language(file_path, content)
        if language == LanguageType.UNKNOWN:
            return []
        
        return self.ast_engine.extract_symbols(content, language)
    
    async def track_symbol_evolution(
        self,
        file_path: str,
        symbol_name: str,
    ) -> list[tuple[float, Symbol | None]]:
        """
        Track how a symbol evolved over time.
        
        Args:
            file_path: Path to the file
            symbol_name: Name of the symbol to track
            
        Returns:
            List of (timestamp, symbol) tuples showing evolution
        """
        if not self._initialized:
            await self.initialize()
        
        # Get file history
        history = await self.get_file_history(file_path)
        
        evolution = []
        for timestamp, content in history:
            # Extract symbols at this point in time
            language = self.ast_engine.detect_language(file_path, content)
            if language == LanguageType.UNKNOWN:
                continue
            
            symbols = self.ast_engine.extract_symbols(content, language)
            
            # Find the target symbol
            target_symbol = None
            for symbol in symbols:
                if symbol.name == symbol_name:
                    target_symbol = symbol
                    break
            
            evolution.append((timestamp, target_symbol))
        
        return evolution
    
    async def close(self) -> None:
        """Close the code change tracker and release resources."""
        await self.stop_tracking()
        await self.store.close()
        self._initialized = False
        logger.info("CodeChangeTracker closed")
