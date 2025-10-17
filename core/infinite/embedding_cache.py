"""LMDB-based embedding cache for infinite context system."""

import lmdb
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Any
from collections import OrderedDict
import asyncio

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LMDB-based embedding cache with LRU eviction."""

    def __init__(
        self,
        cache_dir: str | Path,
        model_name: str = "default",
        max_size_bytes: int = 10 * 1024 * 1024 * 1024,  # 10GB default
    ):
        """Initialize embedding cache.
        
        Args:
            cache_dir: Directory for cache storage
            model_name: Name of the embedding model (for separate caches)
            max_size_bytes: Maximum cache size in bytes
        """
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.max_size_bytes = max_size_bytes
        
        # Create model-specific cache directory
        self.db_path = self.cache_dir / f"embeddings_{model_name}"
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # LMDB environment
        self._env: lmdb.Environment | None = None
        self._lock = asyncio.Lock()
        
        # LRU tracking (in-memory)
        self._lru_order: OrderedDict[bytes, None] = OrderedDict()
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    async def initialize(self) -> None:
        """Initialize LMDB environment."""
        if self._env is not None:
            return

        def _init():
            self._env = lmdb.open(
                str(self.db_path),
                map_size=self.max_size_bytes,
                max_dbs=1,
                sync=False,  # Async writes for performance
                writemap=True,  # Memory-mapped writes
            )
            
            # Load existing keys for LRU tracking
            with self._env.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    self._lru_order[key] = None

        await asyncio.to_thread(_init)
        logger.info(f"EmbeddingCache initialized for model '{self.model_name}' at {self.db_path}")

    def _compute_key(self, text: str) -> bytes:
        """Compute cache key from text using SHA-256 hash.
        
        Args:
            text: Text to hash
            
        Returns:
            Hash bytes for use as cache key
        """
        return hashlib.sha256(text.encode("utf-8")).digest()

    async def get(self, text: str) -> list[float] | None:
        """Retrieve embedding from cache.
        
        Args:
            text: Text to look up
            
        Returns:
            Embedding vector if found, None otherwise
        """
        if self._env is None:
            await self.initialize()

        key = self._compute_key(text)

        async with self._lock:
            def _get():
                with self._env.begin() as txn:
                    value = txn.get(key)
                    if value is not None:
                        # Update LRU order
                        if key in self._lru_order:
                            self._lru_order.move_to_end(key)
                        return pickle.loads(value)
                    return None

            embedding = await asyncio.to_thread(_get)
            
            if embedding is not None:
                self._hits += 1
            else:
                self._misses += 1
            
            return embedding

    async def put(self, text: str, embedding: list[float]) -> bool:
        """Store embedding in cache.
        
        Args:
            text: Text associated with embedding
            embedding: Embedding vector to store
            
        Returns:
            True if successful, False otherwise
        """
        if self._env is None:
            await self.initialize()

        key = self._compute_key(text)
        value = pickle.dumps(embedding)

        async with self._lock:
            def _put():
                with self._env.begin(write=True) as txn:
                    txn.put(key, value)
                
                # Update LRU order
                if key in self._lru_order:
                    self._lru_order.move_to_end(key)
                else:
                    self._lru_order[key] = None
                
                return True

            try:
                result = await asyncio.to_thread(_put)
                
                # Check if we need to evict
                await self._check_and_evict()
                
                return result
            except Exception as e:
                logger.error(f"Failed to store embedding: {e}")
                return False

    async def _check_and_evict(self) -> None:
        """Check cache size and evict oldest entries if needed."""
        def _get_size():
            stat = self._env.stat()
            return stat["psize"] * (stat["leaf_pages"] + stat["branch_pages"] + stat["overflow_pages"])

        current_size = await asyncio.to_thread(_get_size)
        
        # Evict if we're over 90% of max size
        if current_size > self.max_size_bytes * 0.9:
            await self._evict_lru()

    async def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        # Evict oldest 10% of entries
        num_to_evict = max(1, len(self._lru_order) // 10)
        
        def _evict():
            with self._env.begin(write=True) as txn:
                for _ in range(num_to_evict):
                    if not self._lru_order:
                        break
                    key, _ = self._lru_order.popitem(last=False)
                    txn.delete(key)
                    self._evictions += 1

        await asyncio.to_thread(_evict)
        logger.debug(f"Evicted {num_to_evict} entries from cache")

    async def delete(self, text: str) -> bool:
        """Delete an embedding from cache.
        
        Args:
            text: Text whose embedding should be deleted
            
        Returns:
            True if successful, False otherwise
        """
        if self._env is None:
            await self.initialize()

        key = self._compute_key(text)

        async with self._lock:
            def _delete():
                with self._env.begin(write=True) as txn:
                    result = txn.delete(key)
                
                if key in self._lru_order:
                    del self._lru_order[key]
                
                return result

            try:
                return await asyncio.to_thread(_delete)
            except Exception as e:
                logger.error(f"Failed to delete embedding: {e}")
                return False

    async def clear(self) -> None:
        """Clear all entries from cache."""
        if self._env is None:
            await self.initialize()

        async with self._lock:
            def _clear():
                with self._env.begin(write=True) as txn:
                    cursor = txn.cursor()
                    for key, _ in cursor:
                        txn.delete(key)
                
                self._lru_order.clear()

            await asyncio.to_thread(_clear)
            logger.info("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache metrics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "entries": len(self._lru_order),
            "model_name": self.model_name,
        }

    async def get_size_bytes(self) -> int:
        """Get current cache size in bytes.
        
        Returns:
            Cache size in bytes
        """
        if self._env is None:
            return 0

        def _get_size():
            stat = self._env.stat()
            return stat["psize"] * (stat["leaf_pages"] + stat["branch_pages"] + stat["overflow_pages"])

        return await asyncio.to_thread(_get_size)

    async def close(self) -> None:
        """Close LMDB environment."""
        if self._env is not None:
            async with self._lock:
                def _close():
                    self._env.close()
                    self._env = None

                await asyncio.to_thread(_close)
            logger.info("EmbeddingCache closed")
