"""Version graph for tracking memory evolution."""

from dataclasses import dataclass, field
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class VersionNode:
    """Node in the version graph representing a memory version."""
    memory_id: str
    version: int
    parent_id: str | None
    children_ids: list[str] = field(default_factory=list)
    created_at: float = 0.0
    semantic_similarity: float | None = None  # Similarity to parent


class VersionGraph:
    """Directed acyclic graph (DAG) for tracking memory versions."""

    def __init__(self):
        """Initialize empty version graph."""
        self._nodes: dict[str, VersionNode] = {}
        self._roots: set[str] = set()  # Memory IDs with no parent

    def add_node(
        self,
        memory_id: str,
        version: int,
        parent_id: str | None,
        created_at: float,
        semantic_similarity: float | None = None
    ) -> bool:
        """Add a node to the version graph.
        
        Args:
            memory_id: ID of the memory
            version: Version number
            parent_id: ID of parent memory (None for root)
            created_at: Creation timestamp
            semantic_similarity: Similarity score to parent (0-1)
            
        Returns:
            True if successful
        """
        if memory_id in self._nodes:
            logger.warning(f"Node {memory_id} already exists in graph")
            return False

        node = VersionNode(
            memory_id=memory_id,
            version=version,
            parent_id=parent_id,
            created_at=created_at,
            semantic_similarity=semantic_similarity
        )

        self._nodes[memory_id] = node

        # Update parent's children list
        if parent_id:
            if parent_id in self._nodes:
                self._nodes[parent_id].children_ids.append(memory_id)
            else:
                logger.warning(f"Parent {parent_id} not found for {memory_id}")
        else:
            # This is a root node
            self._roots.add(memory_id)

        return True

    def get_node(self, memory_id: str) -> VersionNode | None:
        """Get a node from the graph.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            VersionNode if found, None otherwise
        """
        return self._nodes.get(memory_id)

    def get_parent(self, memory_id: str) -> VersionNode | None:
        """Get the parent node of a memory.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Parent VersionNode if exists, None otherwise
        """
        node = self._nodes.get(memory_id)
        if not node or not node.parent_id:
            return None
        return self._nodes.get(node.parent_id)

    def get_children(self, memory_id: str) -> list[VersionNode]:
        """Get all children nodes of a memory.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            List of child VersionNodes
        """
        node = self._nodes.get(memory_id)
        if not node:
            return []
        
        children = []
        for child_id in node.children_ids:
            child = self._nodes.get(child_id)
            if child:
                children.append(child)
        
        return children

    def get_ancestors(self, memory_id: str) -> list[VersionNode]:
        """Get all ancestor nodes in chronological order (oldest first).
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            List of ancestor VersionNodes from root to parent
        """
        ancestors = []
        current = self._nodes.get(memory_id)
        
        while current and current.parent_id:
            parent = self._nodes.get(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        
        # Reverse to get oldest first
        return list(reversed(ancestors))

    def get_descendants(self, memory_id: str) -> list[VersionNode]:
        """Get all descendant nodes (breadth-first traversal).
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            List of descendant VersionNodes
        """
        descendants = []
        queue = [memory_id]
        visited = set()

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            
            visited.add(current_id)
            node = self._nodes.get(current_id)
            
            if node and current_id != memory_id:
                descendants.append(node)
            
            if node:
                queue.extend(node.children_ids)

        return descendants

    def get_version_chain(self, memory_id: str) -> list[VersionNode]:
        """Get the complete version chain from root to this memory.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            List of VersionNodes from root to current
        """
        ancestors = self.get_ancestors(memory_id)
        current = self._nodes.get(memory_id)
        
        if current:
            return ancestors + [current]
        return ancestors

    def get_root(self, memory_id: str) -> VersionNode | None:
        """Get the root node of a version chain.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Root VersionNode
        """
        ancestors = self.get_ancestors(memory_id)
        if ancestors:
            return ancestors[0]
        
        # Check if this node itself is a root
        node = self._nodes.get(memory_id)
        if node and not node.parent_id:
            return node
        
        return None

    def get_all_roots(self) -> list[VersionNode]:
        """Get all root nodes in the graph.
        
        Returns:
            List of root VersionNodes
        """
        return [self._nodes[root_id] for root_id in self._roots if root_id in self._nodes]

    def has_branching(self, memory_id: str) -> bool:
        """Check if a memory has multiple children (branching).
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            True if memory has multiple children
        """
        node = self._nodes.get(memory_id)
        return node is not None and len(node.children_ids) > 1

    def get_latest_in_chain(self, memory_id: str) -> VersionNode | None:
        """Get the latest (leaf) node in a version chain.
        
        Args:
            memory_id: ID of any memory in the chain
            
        Returns:
            Latest VersionNode in the chain
        """
        descendants = self.get_descendants(memory_id)
        
        # If no descendants, this is the latest
        if not descendants:
            return self._nodes.get(memory_id)
        
        # Find leaf nodes (nodes with no children)
        leaves = [d for d in descendants if not d.children_ids]
        
        # Return the most recent leaf
        if leaves:
            return max(leaves, key=lambda n: n.created_at)
        
        return None

    def detect_conflicts(self, memory_id: str) -> list[tuple[VersionNode, VersionNode]]:
        """Detect conflicting branches from a common ancestor.
        
        Args:
            memory_id: ID of the memory to check
            
        Returns:
            List of conflicting node pairs
        """
        conflicts = []
        node = self._nodes.get(memory_id)
        
        if not node:
            return conflicts

        # Check all ancestors for branching
        ancestors = self.get_ancestors(memory_id)
        for ancestor in ancestors:
            if self.has_branching(ancestor.memory_id):
                # Get all branches from this ancestor
                children = self.get_children(ancestor.memory_id)
                
                # Find which branch our memory is in
                our_branch = None
                for child in children:
                    descendants = self.get_descendants(child.memory_id)
                    if any(d.memory_id == memory_id for d in descendants) or child.memory_id == memory_id:
                        our_branch = child
                        break
                
                # Other branches are potential conflicts
                if our_branch:
                    for child in children:
                        if child.memory_id != our_branch.memory_id:
                            conflicts.append((our_branch, child))

        return conflicts

    def size(self) -> int:
        """Get the number of nodes in the graph.
        
        Returns:
            Number of nodes
        """
        return len(self._nodes)

    def clear(self) -> None:
        """Clear all nodes from the graph."""
        self._nodes.clear()
        self._roots.clear()
