"""SQLite-based document store for infinite context system."""

import sqlite3
import json
import asyncio
from typing import Any
from pathlib import Path
from contextlib import asynccontextmanager
import logging

from .models import Memory, MemoryType

logger = logging.getLogger(__name__)


class DocumentStore:
    """SQLite document store with async interface and connection pooling."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path, pool_size: int = 5):
        """Initialize document store.
        
        Args:
            db_path: Path to SQLite database file
            pool_size: Number of connections in the pool
        """
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self._pool: list[sqlite3.Connection] = []
        self._pool_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database schema and connection pool."""
        if self._initialized:
            return

        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create initial connection to set up schema
        conn = await self._create_connection()
        try:
            await self._create_schema(conn)
            await self._create_indices(conn)
            await self._initialize_migrations(conn)
        finally:
            conn.close()

        # Initialize connection pool
        async with self._pool_lock:
            for _ in range(self.pool_size):
                conn = await self._create_connection()
                self._pool.append(conn)

        self._initialized = True
        logger.info(f"DocumentStore initialized at {self.db_path}")

    async def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        def _connect():
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        
        return await asyncio.to_thread(_connect)

    @asynccontextmanager
    async def _get_connection(self):
        """Get a connection from the pool."""
        async with self._pool_lock:
            if not self._pool:
                # Pool exhausted, create temporary connection
                conn = await self._create_connection()
                temp_conn = True
            else:
                conn = self._pool.pop()
                temp_conn = False

        try:
            yield conn
        finally:
            async with self._pool_lock:
                if temp_conn:
                    conn.close()
                else:
                    self._pool.append(conn)

    async def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create database schema."""
        def _execute():
            cursor = conn.cursor()
            
            # Memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    thread_id TEXT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL,
                    importance INTEGER DEFAULT 5,
                    version INTEGER DEFAULT 1,
                    parent_id TEXT,
                    metadata TEXT,
                    FOREIGN KEY (parent_id) REFERENCES memories(id)
                )
            """)

            # Temporal index table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temporal_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                )
            """)

            # Code changes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS code_changes (
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

            # Entities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                )
            """)

            # Schema version table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                )
            """)

            conn.commit()

        await asyncio.to_thread(_execute)

    async def _create_indices(self, conn: sqlite3.Connection) -> None:
        """Create database indices for optimized queries."""
        def _execute():
            cursor = conn.cursor()
            
            # Memories indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_context 
                ON memories(context_id, created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type 
                ON memories(memory_type, importance)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_parent 
                ON memories(parent_id)
            """)

            # Temporal index indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_temporal_timestamp 
                ON temporal_index(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_temporal_memory 
                ON temporal_index(memory_id, timestamp)
            """)

            # Code changes indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_code_changes_file 
                ON code_changes(file_path, timestamp)
            """)

            # Entities indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_type_value 
                ON entities(entity_type, entity_value)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_memory 
                ON entities(memory_id)
            """)

            conn.commit()

        await asyncio.to_thread(_execute)

    async def _initialize_migrations(self, conn: sqlite3.Connection) -> None:
        """Initialize migration system."""
        import time
        def _execute():
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO schema_version (version, applied_at)
                VALUES (?, ?)
            """, (self.SCHEMA_VERSION, time.time()))
            conn.commit()

        await asyncio.to_thread(_execute)

    async def add_memory(self, memory: Memory) -> bool:
        """Add a new memory to the store.
        
        Args:
            memory: Memory object to store
            
        Returns:
            True if successful, False otherwise
        """
        async with self._get_connection() as conn:
            def _execute():
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO memories (
                        id, context_id, thread_id, content, memory_type,
                        created_at, updated_at, importance, version, parent_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    memory.context_id,
                    memory.thread_id,
                    memory.content,
                    memory.memory_type.value,
                    memory.created_at,
                    memory.updated_at,
                    memory.importance,
                    memory.version,
                    memory.parent_id,
                    json.dumps(memory.metadata),
                ))
                
                # Add temporal index entry
                cursor.execute("""
                    INSERT INTO temporal_index (memory_id, timestamp, event_type)
                    VALUES (?, ?, ?)
                """, (memory.id, memory.created_at, "created"))
                
                conn.commit()
                return True

            try:
                return await asyncio.to_thread(_execute)
            except sqlite3.IntegrityError as e:
                logger.error(f"Failed to add memory {memory.id}: {e}")
                return False

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory object if found, None otherwise
        """
        async with self._get_connection() as conn:
            def _execute():
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM memories WHERE id = ?
                """, (memory_id,))
                row = cursor.fetchone()
                return dict(row) if row else None

            row_dict = await asyncio.to_thread(_execute)
            if not row_dict:
                return None

            # Parse metadata
            if row_dict.get("metadata"):
                row_dict["metadata"] = json.loads(row_dict["metadata"])
            else:
                row_dict["metadata"] = {}

            return Memory.from_dict(row_dict)

    async def update_memory(self, memory: Memory) -> bool:
        """Update an existing memory.
        
        Args:
            memory: Memory object with updated data
            
        Returns:
            True if successful, False otherwise
        """
        async with self._get_connection() as conn:
            def _execute():
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE memories SET
                        content = ?,
                        memory_type = ?,
                        updated_at = ?,
                        importance = ?,
                        version = ?,
                        parent_id = ?,
                        metadata = ?
                    WHERE id = ?
                """, (
                    memory.content,
                    memory.memory_type.value,
                    memory.updated_at,
                    memory.importance,
                    memory.version,
                    memory.parent_id,
                    json.dumps(memory.metadata),
                    memory.id,
                ))
                
                # Add temporal index entry
                if memory.updated_at:
                    cursor.execute("""
                        INSERT INTO temporal_index (memory_id, timestamp, event_type)
                        VALUES (?, ?, ?)
                    """, (memory.id, memory.updated_at, "updated"))
                
                conn.commit()
                return cursor.rowcount > 0

            try:
                return await asyncio.to_thread(_execute)
            except sqlite3.Error as e:
                logger.error(f"Failed to update memory {memory.id}: {e}")
                return False

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if successful, False otherwise
        """
        async with self._get_connection() as conn:
            def _execute():
                cursor = conn.cursor()
                
                # Delete from temporal index
                cursor.execute("""
                    DELETE FROM temporal_index WHERE memory_id = ?
                """, (memory_id,))
                
                # Delete from entities
                cursor.execute("""
                    DELETE FROM entities WHERE memory_id = ?
                """, (memory_id,))
                
                # Delete memory
                cursor.execute("""
                    DELETE FROM memories WHERE id = ?
                """, (memory_id,))
                
                conn.commit()
                return cursor.rowcount > 0

            try:
                return await asyncio.to_thread(_execute)
            except sqlite3.Error as e:
                logger.error(f"Failed to delete memory {memory_id}: {e}")
                return False

    async def query_memories(
        self,
        context_id: str | None = None,
        memory_type: MemoryType | None = None,
        min_importance: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Query memories with filters.
        
        Args:
            context_id: Filter by context ID
            memory_type: Filter by memory type
            min_importance: Minimum importance score
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching memories
        """
        async with self._get_connection() as conn:
            def _execute():
                cursor = conn.cursor()
                
                query = "SELECT * FROM memories WHERE 1=1"
                params = []
                
                if context_id:
                    query += " AND context_id = ?"
                    params.append(context_id)
                
                if memory_type:
                    query += " AND memory_type = ?"
                    params.append(memory_type.value)
                
                if min_importance is not None:
                    query += " AND importance >= ?"
                    params.append(min_importance)
                
                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

            rows = await asyncio.to_thread(_execute)
            
            memories = []
            for row_dict in rows:
                if row_dict.get("metadata"):
                    row_dict["metadata"] = json.loads(row_dict["metadata"])
                else:
                    row_dict["metadata"] = {}
                memories.append(Memory.from_dict(row_dict))
            
            return memories

    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions with rollback support."""
        async with self._get_connection() as conn:
            def _begin():
                conn.execute("BEGIN")
            
            def _commit():
                conn.commit()
            
            def _rollback():
                conn.rollback()

            await asyncio.to_thread(_begin)
            try:
                yield conn
                await asyncio.to_thread(_commit)
            except Exception:
                await asyncio.to_thread(_rollback)
                raise

    async def close(self) -> None:
        """Close all connections in the pool."""
        async with self._pool_lock:
            for conn in self._pool:
                conn.close()
            self._pool.clear()
        self._initialized = False
        logger.info("DocumentStore closed")
