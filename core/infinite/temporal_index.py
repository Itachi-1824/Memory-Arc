"""Temporal indexing system for time-based memory queries."""

import asyncio
import sqlite3
from typing import Any
from pathlib import Path
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


class TemporalIndex:
    """Time-aware indexing system for tracking memory evolution."""

    def __init__(self, db_path: str | Path):
        """Initialize temporal index.
        
        Args:
            db_path: Path to SQLite database file (shared with DocumentStore)
        """
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize temporal index connection."""
        if self._initialized:
            return

        def _connect():
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            return conn

        self._conn = await asyncio.to_thread(_connect)
        self._initialized = True
        logger.info("TemporalIndex initialized")

    async def add_event(
        self,
        memory_id: str,
        timestamp: float,
        event_type: str
    ) -> bool:
        """Add a temporal event for a memory.
        
        Args:
            memory_id: ID of the memory
            timestamp: Event timestamp
            event_type: Type of event ('created', 'updated', 'superseded')
            
        Returns:
            True if successful
        """
        if not self._conn:
            raise RuntimeError("TemporalIndex not initialized")

        def _execute():
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT INTO temporal_index (memory_id, timestamp, event_type)
                VALUES (?, ?, ?)
            """, (memory_id, timestamp, event_type))
            self._conn.commit()
            return True

        try:
            return await asyncio.to_thread(_execute)
        except sqlite3.Error as e:
            logger.error(f"Failed to add temporal event: {e}")
            return False

    async def get_events(
        self,
        memory_id: str,
        event_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Get all temporal events for a memory.
        
        Args:
            memory_id: ID of the memory
            event_type: Optional filter by event type
            
        Returns:
            List of event dictionaries
        """
        if not self._conn:
            raise RuntimeError("TemporalIndex not initialized")

        def _execute():
            cursor = self._conn.cursor()
            
            if event_type:
                cursor.execute("""
                    SELECT * FROM temporal_index
                    WHERE memory_id = ? AND event_type = ?
                    ORDER BY timestamp ASC
                """, (memory_id, event_type))
            else:
                cursor.execute("""
                    SELECT * FROM temporal_index
                    WHERE memory_id = ?
                    ORDER BY timestamp ASC
                """, (memory_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        return await asyncio.to_thread(_execute)

    async def query_by_time_range(
        self,
        start_time: float,
        end_time: float,
        event_type: str | None = None,
        limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Query events within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            event_type: Optional filter by event type
            limit: Maximum number of results
            
        Returns:
            List of event dictionaries
        """
        if not self._conn:
            raise RuntimeError("TemporalIndex not initialized")

        def _execute():
            cursor = self._conn.cursor()
            
            query = """
                SELECT * FROM temporal_index
                WHERE timestamp >= ? AND timestamp <= ?
            """
            params = [start_time, end_time]
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            query += " ORDER BY timestamp ASC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        return await asyncio.to_thread(_execute)

    async def get_memories_at_time(
        self,
        timestamp: float,
        context_id: str | None = None
    ) -> list[str]:
        """Get all memory IDs that existed at a specific time.
        
        Args:
            timestamp: The point in time to query
            context_id: Optional filter by context
            
        Returns:
            List of memory IDs that existed at that time
        """
        if not self._conn:
            raise RuntimeError("TemporalIndex not initialized")

        def _execute():
            cursor = self._conn.cursor()
            
            # Get all memories created before or at the timestamp
            # and not superseded before that timestamp
            query = """
                SELECT DISTINCT ti.memory_id
                FROM temporal_index ti
                INNER JOIN memories m ON ti.memory_id = m.id
                WHERE ti.event_type = 'created' AND ti.timestamp <= ?
            """
            params = [timestamp]
            
            if context_id:
                query += " AND m.context_id = ?"
                params.append(context_id)
            
            # Exclude memories that were superseded before the timestamp
            query += """
                AND ti.memory_id NOT IN (
                    SELECT memory_id FROM temporal_index
                    WHERE event_type = 'superseded' AND timestamp <= ?
                )
            """
            params.append(timestamp)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [row["memory_id"] for row in rows]

        return await asyncio.to_thread(_execute)

    async def get_latest_event_time(
        self,
        memory_id: str,
        event_type: str | None = None
    ) -> float | None:
        """Get the timestamp of the latest event for a memory.
        
        Args:
            memory_id: ID of the memory
            event_type: Optional filter by event type
            
        Returns:
            Timestamp of latest event, or None if no events found
        """
        if not self._conn:
            raise RuntimeError("TemporalIndex not initialized")

        def _execute():
            cursor = self._conn.cursor()
            
            if event_type:
                cursor.execute("""
                    SELECT MAX(timestamp) as latest FROM temporal_index
                    WHERE memory_id = ? AND event_type = ?
                """, (memory_id, event_type))
            else:
                cursor.execute("""
                    SELECT MAX(timestamp) as latest FROM temporal_index
                    WHERE memory_id = ?
                """, (memory_id,))
            
            row = cursor.fetchone()
            return row["latest"] if row and row["latest"] else None

        return await asyncio.to_thread(_execute)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception as e:
                logger.warning(f"Error closing TemporalIndex connection: {e}")
            finally:
                self._conn = None
        self._initialized = False
        logger.info("TemporalIndex closed")
