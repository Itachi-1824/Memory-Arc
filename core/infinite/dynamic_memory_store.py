"""Dynamic memory store with versioning and temporal tracking."""

import asyncio
import time
import uuid
from typing import Any
from pathlib import Path
import logging

from .models import Memory, MemoryType
from .document_store import DocumentStore
from .temporal_index import TemporalIndex
from .version_graph import VersionGraph, VersionNode
from .semantic_versioning import SemanticVersioning

logger = logging.getLogger(__name__)


class DynamicMemoryStore:
    """Manage memory evolution and versioning for all content types."""

    def __init__(
        self,
        db_path: str | Path,
        similarity_threshold: float = 0.7
    ):
        """Initialize dynamic memory store.
        
        Args:
            db_path: Path to SQLite database
            similarity_threshold: Threshold for semantic similarity (0-1)
        """
        self.db_path = Path(db_path)
        self.document_store = DocumentStore(db_path)
        self.temporal_index = TemporalIndex(db_path)
        self.version_graph = VersionGraph()
        self.semantic_versioning = SemanticVersioning(similarity_threshold)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        await self.document_store.initialize()
        await self.temporal_index.initialize()
        
        # Load existing version graph from database
        await self._load_version_graph()
        
        self._initialized = True
        logger.info("DynamicMemoryStore initialized")

    async def _load_version_graph(self) -> None:
        """Load version graph from database."""
        # Query all memories to rebuild the graph
        memories = await self.document_store.query_memories(limit=100000)
        
        for memory in memories:
            self.version_graph.add_node(
                memory_id=memory.id,
                version=memory.version,
                parent_id=memory.parent_id,
                created_at=memory.created_at
            )
        
        logger.info(f"Loaded {self.version_graph.size()} nodes into version graph")

    async def add_memory(
        self,
        content: str,
        memory_type: str | MemoryType,
        context_id: str,
        importance: int = 5,
        supersedes: str | None = None,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None
    ) -> str:
        """Add new memory, optionally superseding an older version.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            context_id: Context identifier
            importance: Importance score (1-10)
            supersedes: ID of memory this supersedes (creates new version)
            thread_id: Optional thread identifier
            metadata: Optional metadata dictionary
            embedding: Optional embedding vector
            
        Returns:
            ID of the created memory
        """
        if not self._initialized:
            raise RuntimeError("DynamicMemoryStore not initialized")

        # Convert memory_type to enum if string
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        # Generate unique ID
        memory_id = str(uuid.uuid4())
        current_time = time.time()

        # Determine version number
        version = 1
        parent_id = None
        semantic_similarity = None

        if supersedes:
            # Get parent memory to determine version
            parent = await self.document_store.get_memory(supersedes)
            if parent:
                version = parent.version + 1
                parent_id = supersedes
                
                # Compute semantic similarity if embeddings available
                if embedding and parent.embedding:
                    semantic_similarity = self.semantic_versioning.compute_similarity(
                        parent.embedding, embedding
                    )
                
                # Mark parent as superseded in temporal index
                await self.temporal_index.add_event(
                    memory_id=supersedes,
                    timestamp=current_time,
                    event_type="superseded"
                )

        # Create memory object
        memory = Memory(
            id=memory_id,
            context_id=context_id,
            content=content,
            memory_type=memory_type,
            created_at=current_time,
            importance=importance,
            version=version,
            parent_id=parent_id,
            thread_id=thread_id,
            metadata=metadata or {},
            embedding=embedding
        )

        # Store in document store
        success = await self.document_store.add_memory(memory)
        if not success:
            raise RuntimeError(f"Failed to add memory {memory_id}")

        # Add to version graph
        self.version_graph.add_node(
            memory_id=memory_id,
            version=version,
            parent_id=parent_id,
            created_at=current_time,
            semantic_similarity=semantic_similarity
        )

        # Temporal index entry is added by document store
        
        logger.info(f"Added memory {memory_id} (version {version})")
        return memory_id

    async def get_current_version(self, memory_id: str) -> Memory | None:
        """Get the most recent version of a memory.
        
        Args:
            memory_id: ID of any version in the chain
            
        Returns:
            Latest Memory version or None
        """
        if not self._initialized:
            raise RuntimeError("DynamicMemoryStore not initialized")

        # Get the latest node in the version chain
        latest_node = self.version_graph.get_latest_in_chain(memory_id)
        
        if not latest_node:
            # This might be the latest version itself
            return await self.document_store.get_memory(memory_id)
        
        return await self.document_store.get_memory(latest_node.memory_id)

    async def get_version_history(self, memory_id: str) -> list[Memory]:
        """Get all versions of a memory in chronological order.
        
        Args:
            memory_id: ID of any version in the chain
            
        Returns:
            List of Memory objects from oldest to newest
        """
        if not self._initialized:
            raise RuntimeError("DynamicMemoryStore not initialized")

        # Start from the given memory and walk backwards to find root
        current_memory = await self.document_store.get_memory(memory_id)
        if not current_memory:
            return []
        
        # Build chain by following parent_id links
        chain = [current_memory]
        current_id = current_memory.parent_id
        
        while current_id:
            parent = await self.document_store.get_memory(current_id)
            if not parent:
                break
            chain.insert(0, parent)  # Insert at beginning to maintain order
            current_id = parent.parent_id
        
        # Now walk forward from the last memory to get all descendants
        last_id = chain[-1].id
        descendants = await self._get_all_descendants(last_id)
        
        # Add descendants that aren't already in chain
        for desc in descendants:
            if desc.id not in [m.id for m in chain]:
                chain.append(desc)
        
        return chain
    
    async def _get_all_descendants(self, memory_id: str) -> list[Memory]:
        """Get all descendant memories recursively."""
        # Query for direct children
        all_memories = await self.document_store.query_memories(limit=10000)
        children = [m for m in all_memories if m.parent_id == memory_id]
        
        descendants = []
        for child in children:
            descendants.append(child)
            # Recursively get descendants of this child
            child_descendants = await self._get_all_descendants(child.id)
            descendants.extend(child_descendants)
        
        return descendants

    async def query_at_time(
        self,
        query: str,
        timestamp: float,
        context_id: str,
        top_k: int = 10,
        memory_type: MemoryType | None = None
    ) -> list[Memory]:
        """Query memories as they existed at a specific time.
        
        Args:
            query: Search query (currently unused, placeholder for semantic search)
            timestamp: Point in time to query
            context_id: Context identifier
            top_k: Maximum number of results
            memory_type: Optional filter by memory type
            
        Returns:
            List of memories that existed at that time
        """
        if not self._initialized:
            raise RuntimeError("DynamicMemoryStore not initialized")

        # Get memory IDs that existed at the timestamp
        memory_ids = await self.temporal_index.get_memories_at_time(
            timestamp=timestamp,
            context_id=context_id
        )

        # Fetch full memory objects
        memories = []
        for memory_id in memory_ids[:top_k]:
            memory = await self.document_store.get_memory(memory_id)
            if memory:
                # Filter by type if specified
                if memory_type is None or memory.memory_type == memory_type:
                    memories.append(memory)

        # Sort by importance and creation time
        memories.sort(key=lambda m: (m.importance, m.created_at), reverse=True)
        
        return memories[:top_k]

    async def detect_contradictions(
        self,
        context_id: str,
        entity_type: str | None = None,
        min_confidence: float = 0.5
    ) -> list[tuple[Memory, Memory, float]]:
        """Find contradictory memories with similarity score.
        
        Args:
            context_id: Context identifier
            entity_type: Optional filter by entity type
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of (memory1, memory2, confidence) tuples
        """
        if not self._initialized:
            raise RuntimeError("DynamicMemoryStore not initialized")

        # Get all current memories in context
        memories = await self.document_store.query_memories(
            context_id=context_id,
            limit=1000
        )

        # Filter to only current versions (not superseded)
        current_memories = []
        for memory in memories:
            # Check if this memory has been superseded
            events = await self.temporal_index.get_events(
                memory_id=memory.id,
                event_type="superseded"
            )
            if not events and memory.embedding:
                current_memories.append(memory)

        # Find contradictions
        contradictions = []
        for i in range(len(current_memories)):
            for j in range(i + 1, len(current_memories)):
                mem1 = current_memories[i]
                mem2 = current_memories[j]

                if mem1.embedding and mem2.embedding:
                    is_contradiction, confidence = self.semantic_versioning.detect_contradiction(
                        mem1.content,
                        mem2.content,
                        mem1.embedding,
                        mem2.embedding
                    )

                    if is_contradiction and confidence >= min_confidence:
                        contradictions.append((mem1, mem2, confidence))

        # Sort by confidence (highest first)
        contradictions.sort(key=lambda x: x[2], reverse=True)
        
        return contradictions

    async def get_memory_evolution(
        self,
        memory_id: str
    ) -> dict[str, Any]:
        """Get complete evolution information for a memory.
        
        Args:
            memory_id: ID of any version in the chain
            
        Returns:
            Dictionary with evolution details
        """
        if not self._initialized:
            raise RuntimeError("DynamicMemoryStore not initialized")

        # Get version history
        history = await self.get_version_history(memory_id)
        
        # Get version graph info
        chain = self.version_graph.get_version_chain(memory_id)
        root = self.version_graph.get_root(memory_id)
        
        # Check for branching
        branches = []
        for node in chain:
            if self.version_graph.has_branching(node.memory_id):
                children = self.version_graph.get_children(node.memory_id)
                branches.append({
                    "branch_point": node.memory_id,
                    "branches": [c.memory_id for c in children]
                })

        return {
            "root_id": root.memory_id if root else None,
            "version_count": len(history),
            "versions": [m.to_dict() for m in history],
            "branches": branches,
            "current_version": history[-1].to_dict() if history else None
        }

    async def close(self) -> None:
        """Close all connections."""
        await self.document_store.close()
        await self.temporal_index.close()
        self.version_graph.clear()
        self._initialized = False
        logger.info("DynamicMemoryStore closed")
