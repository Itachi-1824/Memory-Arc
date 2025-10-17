"""Code change storage and retrieval for infinite context system."""

import sqlite3
import json
import asyncio
import time
from typing import Any, Literal
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import logging

from .diff_generator import Diff, DiffGenerator, DiffLevel
from .ast_diff import ASTDiff, ASTDiffEngine, LanguageType

logger = logging.getLogger(__name__)


@dataclass
class CodeChange:
    """Represents a code change with diffs at multiple levels."""
    id: str
    file_path: str
    change_type: Literal["add", "modify", "delete", "rename"]
    timestamp: float
    before_content: str | None
    after_content: str
    char_diff: Diff | None = None
    line_diff: Diff | None = None
    unified_diff: Diff | None = None
    ast_diff: ASTDiff | None = None
    commit_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert code change to dictionary for storage."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "change_type": self.change_type,
            "timestamp": self.timestamp,
            "before_content": self.before_content,
            "after_content": self.after_content,
            "commit_hash": self.commit_hash,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeChange":
        """Create code change from dictionary."""
        return cls(
            id=data["id"],
            file_path=data["file_path"],
            change_type=data["change_type"],
            timestamp=data["timestamp"],
            before_content=data.get("before_content"),
            after_content=data["after_content"],
            commit_hash=data.get("commit_hash"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ChangeGraphNode:
    """Represents a node in the change graph."""
    change_id: str
    file_path: str
    timestamp: float
    change_type: str
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change_id": self.change_id,
            "file_path": self.file_path,
            "timestamp": self.timestamp,
            "change_type": self.change_type,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
        }


@dataclass
class ChangeGraph:
    """Represents the complete change history graph for a file."""
    file_path: str
    nodes: list[ChangeGraphNode]
    root_ids: list[str]  # Changes with no parents (initial state)
    leaf_ids: list[str]  # Changes with no children (current state)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "nodes": [node.to_dict() for node in self.nodes],
            "root_ids": self.root_ids,
            "leaf_ids": self.leaf_ids,
        }


class CodeChangeStore:
    """
    Storage and retrieval for code changes with multi-level diffs.
    
    Features:
    - Store code changes with before/after states
    - Multi-level diff storage (char, line, unified, AST)
    - Change graph construction
    - File reconstruction at any point in time
    - Efficient querying by file, time, or semantic content
    """
    
    def __init__(
        self,
        db_path: str | Path,
        diff_generator: DiffGenerator | None = None,
        ast_engine: ASTDiffEngine | None = None,
    ):
        """
        Initialize code change store.
        
        Args:
            db_path: Path to SQLite database file
            diff_generator: Optional diff generator (creates default if None)
            ast_engine: Optional AST diff engine (creates default if None)
        """
        self.db_path = Path(db_path)
        self.diff_generator = diff_generator or DiffGenerator()
        self.ast_engine = ast_engine
        self._conn: sqlite3.Connection | None = None
        self._conn_lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the code change store."""
        if self._initialized:
            return
        
        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create connection
        self._conn = await self._create_connection()
        
        # Ensure schema exists (should already be created by DocumentStore)
        # But we'll verify the code_changes table exists
        await self._verify_schema()
        
        self._initialized = True
        logger.info(f"CodeChangeStore initialized at {self.db_path}")
    
    async def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        def _connect():
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        
        return await asyncio.to_thread(_connect)
    
    async def _verify_schema(self) -> None:
        """Verify that the code_changes table exists."""
        def _execute():
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='code_changes'
            """)
            result = cursor.fetchone()
            
            if not result:
                # Create the table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE code_changes (
                        id TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        change_type TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        before_content TEXT,
                        after_content TEXT,
                        diff_text TEXT,
                        ast_diff TEXT,
                        commit_hash TEXT,
                        metadata TEXT
                    )
                """)
                
                # Create indices
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_code_changes_file 
                    ON code_changes(file_path, timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_code_changes_timestamp 
                    ON code_changes(timestamp)
                """)
                
                self._conn.commit()
        
        await asyncio.to_thread(_execute)
    
    async def add_change(
        self,
        change: CodeChange,
        compute_diffs: bool = True,
        compute_ast: bool = True,
    ) -> bool:
        """
        Add a code change to the store.
        
        Args:
            change: CodeChange object to store
            compute_diffs: Whether to compute and store diffs
            compute_ast: Whether to compute and store AST diff
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        # Compute diffs if requested
        if compute_diffs and change.before_content and change.after_content:
            diffs = self.diff_generator.generate_all_diffs(
                change.before_content,
                change.after_content,
                before_name=f"{change.file_path}@before",
                after_name=f"{change.file_path}@after",
            )
            change.char_diff = diffs["char"]
            change.line_diff = diffs["line"]
            change.unified_diff = diffs["unified"]
        
        # Compute AST diff if requested
        if compute_ast and self.ast_engine and change.before_content and change.after_content:
            try:
                language = self.ast_engine.detect_language(change.file_path)
                if language != LanguageType.UNKNOWN:
                    change.ast_diff = self.ast_engine.compute_ast_diff(
                        change.before_content,
                        change.after_content,
                        language,
                    )
            except Exception as e:
                logger.warning(f"Failed to compute AST diff for {change.file_path}: {e}")
        
        # Store in database
        async with self._conn_lock:
            def _execute():
                cursor = self._conn.cursor()
                
                # Serialize diffs
                diff_data = {}
                if change.char_diff:
                    diff_data["char"] = {
                        "content": change.char_diff.content,
                        "compressed": change.char_diff.compressed,
                        "compression_ratio": change.char_diff.compression_ratio,
                    }
                if change.line_diff:
                    diff_data["line"] = {
                        "content": change.line_diff.content,
                        "compressed": change.line_diff.compressed,
                        "compression_ratio": change.line_diff.compression_ratio,
                    }
                if change.unified_diff:
                    diff_data["unified"] = {
                        "content": change.unified_diff.content,
                        "compressed": change.unified_diff.compressed,
                        "compression_ratio": change.unified_diff.compression_ratio,
                    }
                
                diff_text = json.dumps(diff_data) if diff_data else None
                
                # Serialize AST diff
                ast_diff_text = None
                if change.ast_diff:
                    ast_diff_text = json.dumps(change.ast_diff.to_dict())
                
                cursor.execute("""
                    INSERT INTO code_changes (
                        id, file_path, change_type, timestamp,
                        before_content, after_content, diff_text, ast_diff,
                        commit_hash, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    change.id,
                    change.file_path,
                    change.change_type,
                    change.timestamp,
                    change.before_content,
                    change.after_content,
                    diff_text,
                    ast_diff_text,
                    change.commit_hash,
                    json.dumps(change.metadata),
                ))
                
                self._conn.commit()
                return True
            
            try:
                return await asyncio.to_thread(_execute)
            except sqlite3.IntegrityError as e:
                logger.error(f"Failed to add code change {change.id}: {e}")
                return False
    
    async def get_change(self, change_id: str) -> CodeChange | None:
        """
        Retrieve a code change by ID.
        
        Args:
            change_id: ID of the change to retrieve
            
        Returns:
            CodeChange object if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._conn_lock:
            def _execute():
                cursor = self._conn.cursor()
                cursor.execute("""
                    SELECT * FROM code_changes WHERE id = ?
                """, (change_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
            
            row_dict = await asyncio.to_thread(_execute)
            if not row_dict:
                return None
            
            return self._row_to_change(row_dict)
    
    def _row_to_change(self, row_dict: dict[str, Any]) -> CodeChange:
        """Convert database row to CodeChange object."""
        # Parse metadata
        metadata = {}
        if row_dict.get("metadata"):
            metadata = json.loads(row_dict["metadata"])
        
        # Create base change object
        change = CodeChange(
            id=row_dict["id"],
            file_path=row_dict["file_path"],
            change_type=row_dict["change_type"],
            timestamp=row_dict["timestamp"],
            before_content=row_dict.get("before_content"),
            after_content=row_dict["after_content"],
            commit_hash=row_dict.get("commit_hash"),
            metadata=metadata,
        )
        
        # Parse diffs
        if row_dict.get("diff_text"):
            diff_data = json.loads(row_dict["diff_text"])
            
            if "char" in diff_data:
                d = diff_data["char"]
                change.char_diff = Diff(
                    level="char",
                    content=d["content"],
                    compressed=d["compressed"],
                    compression_ratio=d["compression_ratio"],
                )
            
            if "line" in diff_data:
                d = diff_data["line"]
                change.line_diff = Diff(
                    level="line",
                    content=d["content"],
                    compressed=d["compressed"],
                    compression_ratio=d["compression_ratio"],
                )
            
            if "unified" in diff_data:
                d = diff_data["unified"]
                change.unified_diff = Diff(
                    level="unified",
                    content=d["content"],
                    compressed=d["compressed"],
                    compression_ratio=d["compression_ratio"],
                )
        
        # Parse AST diff
        if row_dict.get("ast_diff"):
            try:
                ast_diff_data = json.loads(row_dict["ast_diff"])
                # Reconstruct ASTDiff from stored data
                from .ast_diff import ASTDiff, ASTNodeChange, Symbol, LanguageType, ChangeType
                
                # Reconstruct symbols
                symbols_added = [Symbol(**s) for s in ast_diff_data.get("symbols_added", [])]
                symbols_removed = [Symbol(**s) for s in ast_diff_data.get("symbols_removed", [])]
                symbols_modified = [
                    (Symbol(**before), Symbol(**after))
                    for before, after in ast_diff_data.get("symbols_modified", [])
                ]
                
                # Reconstruct changes
                changes = []
                for change_data in ast_diff_data.get("changes", []):
                    changes.append(ASTNodeChange(
                        change_type=ChangeType(change_data["change_type"]),
                        node_type=change_data["node_type"],
                        path=change_data["path"],
                        before_text=change_data.get("before_text"),
                        after_text=change_data.get("after_text"),
                        start_line=change_data.get("start_line"),
                        end_line=change_data.get("end_line"),
                        metadata=change_data.get("metadata", {}),
                    ))
                
                change.ast_diff = ASTDiff(
                    language=LanguageType(ast_diff_data["language"]),
                    changes=changes,
                    symbols_added=symbols_added,
                    symbols_removed=symbols_removed,
                    symbols_modified=symbols_modified,
                    metadata=ast_diff_data.get("metadata", {}),
                )
            except Exception as e:
                logger.warning(f"Failed to parse AST diff: {e}")
        
        return change
    
    async def query_changes(
        self,
        file_path: str | None = None,
        change_type: str | None = None,
        time_range: tuple[float, float] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CodeChange]:
        """
        Query code changes with filters.
        
        Args:
            file_path: Filter by file path
            change_type: Filter by change type
            time_range: Filter by timestamp range (start, end)
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching code changes
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._conn_lock:
            def _execute():
                cursor = self._conn.cursor()
                
                query = "SELECT * FROM code_changes WHERE 1=1"
                params = []
                
                if file_path:
                    query += " AND file_path = ?"
                    params.append(file_path)
                
                if change_type:
                    query += " AND change_type = ?"
                    params.append(change_type)
                
                if time_range:
                    query += " AND timestamp >= ? AND timestamp <= ?"
                    params.extend(time_range)
                
                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            
            rows = await asyncio.to_thread(_execute)
            return [self._row_to_change(row) for row in rows]
    
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
        
        # Get all changes for this file
        changes = await self.query_changes(file_path=file_path, limit=10000)
        
        # Sort by timestamp
        changes.sort(key=lambda c: c.timestamp)
        
        # Build graph nodes
        nodes = []
        for i, change in enumerate(changes):
            parent_ids = []
            if i > 0:
                # Link to previous change
                parent_ids.append(changes[i - 1].id)
            
            child_ids = []
            if i < len(changes) - 1:
                # Link to next change
                child_ids.append(changes[i + 1].id)
            
            node = ChangeGraphNode(
                change_id=change.id,
                file_path=change.file_path,
                timestamp=change.timestamp,
                change_type=change.change_type,
                parent_ids=parent_ids,
                child_ids=child_ids,
            )
            nodes.append(node)
        
        # Identify roots and leaves
        root_ids = [nodes[0].change_id] if nodes else []
        leaf_ids = [nodes[-1].change_id] if nodes else []
        
        return ChangeGraph(
            file_path=file_path,
            nodes=nodes,
            root_ids=root_ids,
            leaf_ids=leaf_ids,
        )
    
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
        
        # Get all changes up to the specified time
        changes = await self.query_changes(
            file_path=file_path,
            time_range=(0, at_time),
            limit=10000,
        )
        
        if not changes:
            return None
        
        # Sort by timestamp (oldest first)
        changes.sort(key=lambda c: c.timestamp)
        
        # Start with the first change's after_content
        if not changes:
            return None
        
        # Find the last change before or at the target time
        target_change = None
        for change in changes:
            if change.timestamp <= at_time:
                target_change = change
            else:
                break
        
        if not target_change:
            return None
        
        # Return the after_content of the target change
        return target_change.after_content
    
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
        
        changes = await self.query_changes(file_path=file_path, limit=limit)
        changes.sort(key=lambda c: c.timestamp)
        
        history = []
        for change in changes:
            if change.after_content:
                history.append((change.timestamp, change.after_content))
        
        return history
    
    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._initialized = False
        logger.info("CodeChangeStore closed")

