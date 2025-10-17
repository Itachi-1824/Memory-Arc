"""Backward compatibility layer for InfiniteContextEngine.

This module provides compatibility wrappers to integrate InfiniteContextEngine
with the existing MemoryManager API, allowing gradual migration.
"""

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Any, Callable

from .infinite_context_engine import InfiniteContextEngine
from .models import InfiniteContextConfig, Memory, MemoryType

logger = logging.getLogger(__name__)


class InfiniteMemoryManagerAdapter:
    """
    Adapter that wraps InfiniteContextEngine to work with existing MemoryManager API.
    
    This allows existing code using MemoryManager to benefit from infinite context
    features without requiring immediate refactoring.
    
    Usage:
        # Old code
        memory = MemoryManager(context_id="user_123")
        
        # New code with infinite context
        memory = InfiniteMemoryManagerAdapter(context_id="user_123")
        
        # API remains the same
        await memory.add_memory("Hello world")
        results = await memory.retrieve("hello")
    """
    
    def __init__(
        self,
        context_id: str,
        config: InfiniteContextConfig | None = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
        enable_infinite_features: bool = True,
        **kwargs
    ):
        """
        Initialize adapter.
        
        Args:
            context_id: Context identifier for memory isolation
            config: InfiniteContextConfig (optional)
            embedding_fn: Embedding function
            enable_infinite_features: Whether to enable infinite context features
            **kwargs: Additional arguments passed to InfiniteContextEngine
        """
        self.context_id = context_id
        self.enable_infinite_features = enable_infinite_features
        
        if not enable_infinite_features:
            warnings.warn(
                "InfiniteMemoryManagerAdapter with enable_infinite_features=False "
                "provides limited functionality. Consider enabling infinite features.",
                DeprecationWarning,
                stacklevel=2
            )
        
        # Create engine
        if config is None:
            config = InfiniteContextConfig.balanced()
        
        self.engine = InfiniteContextEngine(
            config=config,
            embedding_fn=embedding_fn,
            **kwargs
        )
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the engine."""
        if not self._initialized:
            await self.engine.initialize()
            self._initialized = True
    
    async def add_memory(
        self,
        content: str,
        memory_type: str = "conversation",
        importance: int = 5,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Add a memory (compatible with MemoryManager API).
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score (1-10)
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.engine.add_memory(
            content=content,
            memory_type=memory_type,
            context_id=self.context_id,
            importance=importance,
            metadata=metadata
        )
    
    async def retrieve(
        self,
        query: str,
        max_results: int = 10,
        memory_types: list[str] | None = None
    ) -> list[Memory]:
        """
        Retrieve memories (compatible with MemoryManager API).
        
        Args:
            query: Search query
            max_results: Maximum results to return
            memory_types: Filter by memory types
            
        Returns:
            List of Memory objects
        """
        if not self._initialized:
            await self.initialize()
        
        result = await self.engine.retrieve(
            query=query,
            context_id=self.context_id,
            max_results=max_results,
            memory_types=memory_types
        )
        
        return result.memories
    
    async def get_version_history(self, memory_id: str) -> list[Memory]:
        """Get version history for a memory (infinite context feature)."""
        if not self._initialized:
            await self.initialize()
        
        if not self.enable_infinite_features:
            warnings.warn(
                "get_version_history requires enable_infinite_features=True",
                DeprecationWarning,
                stacklevel=2
            )
            return []
        
        return await self.engine.get_version_history(memory_id)
    
    async def query_at_time(
        self,
        query: str,
        timestamp: float,
        max_results: int = 10
    ) -> list[Memory]:
        """Query memories at a specific time (infinite context feature)."""
        if not self._initialized:
            await self.initialize()
        
        if not self.enable_infinite_features:
            warnings.warn(
                "query_at_time requires enable_infinite_features=True",
                DeprecationWarning,
                stacklevel=2
            )
            return []
        
        result = await self.engine.query_at_time(
            query=query,
            timestamp=timestamp,
            context_id=self.context_id,
            max_results=max_results
        )
        
        return result.memories
    
    async def close(self) -> None:
        """Close the engine."""
        if self._initialized:
            await self.engine.shutdown()
            self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


def migrate_memory_data(
    old_db_path: str | Path,
    new_storage_path: str | Path,
    context_id: str = "default"
) -> None:
    """
    Migrate data from old MemoryManager database to InfiniteContextEngine.
    
    This is a utility function to help migrate existing memory data to the
    new infinite context system.
    
    Args:
        old_db_path: Path to old SQLite database
        new_storage_path: Path to new storage directory
        context_id: Context ID to assign to migrated memories
    """
    import sqlite3
    
    old_db_path = Path(old_db_path)
    new_storage_path = Path(new_storage_path)
    
    if not old_db_path.exists():
        raise FileNotFoundError(f"Old database not found: {old_db_path}")
    
    logger.info(f"Migrating data from {old_db_path} to {new_storage_path}")
    
    # Read old data
    conn = sqlite3.connect(old_db_path)
    cursor = conn.cursor()
    
    try:
        # Check if old memories table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        if not cursor.fetchone():
            logger.warning("No 'memories' table found in old database")
            return
        
        # Read all memories
        cursor.execute("SELECT * FROM memories")
        rows = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        logger.info(f"Found {len(rows)} memories to migrate")
        
        # Create new engine
        config = InfiniteContextConfig(storage_path=str(new_storage_path))
        engine = InfiniteContextEngine(config=config)
        
        async def do_migration():
            await engine.initialize()
            
            migrated = 0
            for row in rows:
                try:
                    # Convert row to dict
                    memory_dict = dict(zip(column_names, row))
                    
                    # Extract fields
                    content = memory_dict.get("content", "")
                    memory_type = memory_dict.get("memory_type", "conversation")
                    importance = memory_dict.get("importance", 5)
                    
                    # Add to new system
                    await engine.add_memory(
                        content=content,
                        memory_type=memory_type,
                        context_id=context_id,
                        importance=importance
                    )
                    
                    migrated += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate memory: {e}")
            
            await engine.shutdown()
            logger.info(f"Successfully migrated {migrated}/{len(rows)} memories")
        
        # Run migration
        asyncio.run(do_migration())
        
    finally:
        conn.close()


def create_feature_flags() -> dict[str, bool]:
    """
    Create feature flags for gradual rollout of infinite context features.
    
    Returns:
        Dictionary of feature flags
    """
    return {
        "enable_versioning": True,
        "enable_temporal_queries": True,
        "enable_code_tracking": False,  # Opt-in
        "enable_contradiction_detection": True,
        "enable_advanced_chunking": True,
        "enable_multi_strategy_retrieval": True,
        "enable_query_caching": True,
        "enable_embedding_cache": True,
    }


def check_deprecation(feature_name: str) -> None:
    """
    Check if a feature is deprecated and issue warning.
    
    Args:
        feature_name: Name of the feature to check
    """
    deprecated_features = {
        "simple_memory_store": "Use DynamicMemoryStore instead",
        "basic_retrieval": "Use RetrievalOrchestrator instead",
        "manual_chunking": "Use ChunkManager instead",
    }
    
    if feature_name in deprecated_features:
        warnings.warn(
            f"{feature_name} is deprecated. {deprecated_features[feature_name]}",
            DeprecationWarning,
            stacklevel=3
        )
