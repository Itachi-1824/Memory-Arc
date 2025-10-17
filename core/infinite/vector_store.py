"""Enhanced Qdrant vector store integration for infinite context system."""

import logging
from typing import Any
from pathlib import Path
import asyncio

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from .models import Memory, MemoryType

logger = logging.getLogger(__name__)


class VectorStore:
    """Enhanced Qdrant vector store with multi-collection strategy."""

    COLLECTIONS = {
        "conversations": "infinite_conversations",
        "code": "infinite_code",
        "facts": "infinite_facts",
        "temporal": "infinite_temporal",
    }

    def __init__(
        self,
        path: str | Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        embedding_dim: int = 384,
    ):
        """Initialize vector store.
        
        Args:
            path: Path for local Qdrant storage (if not using remote)
            url: URL for remote Qdrant instance
            api_key: API key for remote Qdrant
            embedding_dim: Dimension of embedding vectors
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required for VectorStore. Install with: pip install qdrant-client")

        self.path = Path(path) if path else None
        self.url = url
        self.api_key = api_key
        self.embedding_dim = embedding_dim
        self._client: QdrantClient | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Qdrant client and create collections."""
        if self._initialized:
            return

        def _init():
            if self.url:
                self._client = QdrantClient(url=self.url, api_key=self.api_key)
            else:
                if self.path:
                    self.path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(self.path) if self.path else ":memory:")

        await asyncio.to_thread(_init)
        
        # Create collections
        await self._create_collections()
        
        self._initialized = True
        logger.info("VectorStore initialized with multi-collection strategy")

    async def _create_collections(self) -> None:
        """Create all required collections."""
        def _create():
            for collection_name in self.COLLECTIONS.values():
                try:
                    self._client.get_collection(collection_name)
                    logger.debug(f"Collection {collection_name} already exists")
                except Exception:
                    # Collection doesn't exist, create it
                    self._client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=self.embedding_dim,
                            distance=Distance.COSINE,
                        ),
                    )
                    logger.info(f"Created collection {collection_name}")

        await asyncio.to_thread(_create)

    def _get_collection_name(self, memory_type: MemoryType) -> str:
        """Get collection name for a memory type.
        
        Args:
            memory_type: Type of memory
            
        Returns:
            Collection name
        """
        if memory_type == MemoryType.CONVERSATION:
            return self.COLLECTIONS["conversations"]
        elif memory_type == MemoryType.CODE:
            return self.COLLECTIONS["code"]
        elif memory_type in (MemoryType.FACT, MemoryType.PREFERENCE):
            return self.COLLECTIONS["facts"]
        else:
            return self.COLLECTIONS["temporal"]

    async def add_memory(self, memory: Memory) -> bool:
        """Add a memory with its embedding to the vector store.
        
        Args:
            memory: Memory object with embedding
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        if memory.embedding is None:
            logger.warning(f"Memory {memory.id} has no embedding, skipping vector store")
            return False

        collection_name = self._get_collection_name(memory.memory_type)

        def _add():
            point = PointStruct(
                id=memory.id,
                vector=memory.embedding,
                payload={
                    "context_id": memory.context_id,
                    "memory_type": memory.memory_type.value,
                    "importance": memory.importance,
                    "created_at": memory.created_at,
                    "version": memory.version,
                },
            )
            self._client.upsert(
                collection_name=collection_name,
                points=[point],
            )
            return True

        try:
            return await asyncio.to_thread(_add)
        except Exception as e:
            logger.error(f"Failed to add memory to vector store: {e}")
            return False

    async def add_memories_batch(self, memories: list[Memory]) -> int:
        """Add multiple memories in batch for better performance.
        
        Args:
            memories: List of memories with embeddings
            
        Returns:
            Number of successfully added memories
        """
        if not self._initialized:
            await self.initialize()

        # Group by collection
        by_collection: dict[str, list[PointStruct]] = {}
        
        for memory in memories:
            if memory.embedding is None:
                continue
            
            collection_name = self._get_collection_name(memory.memory_type)
            
            if collection_name not in by_collection:
                by_collection[collection_name] = []
            
            point = PointStruct(
                id=memory.id,
                vector=memory.embedding,
                payload={
                    "context_id": memory.context_id,
                    "memory_type": memory.memory_type.value,
                    "importance": memory.importance,
                    "created_at": memory.created_at,
                    "version": memory.version,
                },
            )
            by_collection[collection_name].append(point)

        def _add_batch():
            count = 0
            for collection_name, points in by_collection.items():
                try:
                    self._client.upsert(
                        collection_name=collection_name,
                        points=points,
                    )
                    count += len(points)
                except Exception as e:
                    logger.error(f"Failed to add batch to {collection_name}: {e}")
            return count

        try:
            return await asyncio.to_thread(_add_batch)
        except Exception as e:
            logger.error(f"Failed to add memories batch: {e}")
            return 0

    async def search(
        self,
        query_vector: list[float],
        memory_type: MemoryType | None = None,
        context_id: str | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Search for similar memories.
        
        Args:
            query_vector: Query embedding vector
            memory_type: Filter by memory type
            context_id: Filter by context ID
            limit: Maximum number of results
            min_score: Minimum similarity score
            
        Returns:
            List of (memory_id, score) tuples
        """
        if not self._initialized:
            await self.initialize()

        # Determine which collections to search
        if memory_type:
            collections = [self._get_collection_name(memory_type)]
        else:
            collections = list(self.COLLECTIONS.values())

        def _search():
            all_results = []
            
            for collection_name in collections:
                try:
                    # Build filter
                    filter_conditions = []
                    if context_id:
                        filter_conditions.append(
                            FieldCondition(
                                key="context_id",
                                match=MatchValue(value=context_id),
                            )
                        )
                    
                    query_filter = Filter(must=filter_conditions) if filter_conditions else None
                    
                    # Search
                    results = self._client.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        query_filter=query_filter,
                        limit=limit,
                        score_threshold=min_score,
                    )
                    
                    for result in results:
                        all_results.append((result.id, result.score))
                
                except Exception as e:
                    logger.error(f"Failed to search collection {collection_name}: {e}")
            
            # Sort by score and limit
            all_results.sort(key=lambda x: x[1], reverse=True)
            return all_results[:limit]

        try:
            return await asyncio.to_thread(_search)
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            return []

    async def delete_memory(self, memory_id: str, memory_type: MemoryType) -> bool:
        """Delete a memory from the vector store.
        
        Args:
            memory_id: ID of memory to delete
            memory_type: Type of memory (to determine collection)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        collection_name = self._get_collection_name(memory_type)

        def _delete():
            self._client.delete(
                collection_name=collection_name,
                points_selector=[memory_id],
            )
            return True

        try:
            return await asyncio.to_thread(_delete)
        except Exception as e:
            logger.error(f"Failed to delete memory from vector store: {e}")
            return False

    async def create_snapshot(self, snapshot_dir: str | Path) -> str | None:
        """Create a snapshot of all collections.
        
        Args:
            snapshot_dir: Directory to store snapshots
            
        Returns:
            Snapshot name if successful, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        def _snapshot():
            snapshots = []
            for collection_name in self.COLLECTIONS.values():
                try:
                    snapshot = self._client.create_snapshot(collection_name=collection_name)
                    snapshots.append(snapshot.name)
                except Exception as e:
                    logger.error(f"Failed to create snapshot for {collection_name}: {e}")
            
            return ",".join(snapshots) if snapshots else None

        try:
            return await asyncio.to_thread(_snapshot)
        except Exception as e:
            logger.error(f"Failed to create snapshots: {e}")
            return None

    async def get_collection_info(self) -> dict[str, Any]:
        """Get information about all collections.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self._initialized:
            await self.initialize()

        def _get_info():
            info = {}
            for key, collection_name in self.COLLECTIONS.items():
                try:
                    collection = self._client.get_collection(collection_name)
                    info[key] = {
                        "name": collection_name,
                        "vectors_count": collection.vectors_count,
                        "points_count": collection.points_count,
                    }
                except Exception as e:
                    logger.error(f"Failed to get info for {collection_name}: {e}")
                    info[key] = {"error": str(e)}
            return info

        try:
            return await asyncio.to_thread(_get_info)
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    async def close(self) -> None:
        """Close the Qdrant client."""
        if self._client is not None:
            # Qdrant client doesn't need explicit closing
            self._client = None
            self._initialized = False
            logger.info("VectorStore closed")
