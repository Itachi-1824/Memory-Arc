"""Example usage of CodeChangeStore for tracking code changes."""

import asyncio
import time
import uuid
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct import to avoid core.__init__ issues
from core.infinite import CodeChangeStore, CodeChange


async def main():
    """Demonstrate CodeChangeStore functionality."""
    
    # Initialize the store
    db_path = Path("data/code_changes.db")
    store = CodeChangeStore(db_path)
    await store.initialize()
    
    print("=== CodeChangeStore Example ===\n")
    
    # Example 1: Track a file modification
    print("1. Tracking a file modification...")
    change1 = CodeChange(
        id=str(uuid.uuid4()),
        file_path="auth.py",
        change_type="modify",
        timestamp=time.time(),
        before_content="def login(username, password):\n    return authenticate(username, password)",
        after_content="def login(username: str, password: str) -> bool:\n    return authenticate(username, password)",
        metadata={"author": "developer1", "commit": "abc123"}
    )
    
    await store.add_change(change1, compute_diffs=True)
    print(f"✓ Tracked change {change1.id[:8]}... for {change1.file_path}")
    
    # Retrieve and display the change
    retrieved = await store.get_change(change1.id)
    print(f"  - Change type: {retrieved.change_type}")
    print(f"  - Has diffs: char={retrieved.char_diff is not None}, "
          f"line={retrieved.line_diff is not None}, "
          f"unified={retrieved.unified_diff is not None}")
    print()
    
    # Example 2: Track multiple changes to the same file
    print("2. Tracking multiple changes to the same file...")
    base_time = time.time()
    
    changes = [
        CodeChange(
            id=str(uuid.uuid4()),
            file_path="utils.py",
            change_type="add",
            timestamp=base_time,
            before_content=None,
            after_content="def helper():\n    pass",
        ),
        CodeChange(
            id=str(uuid.uuid4()),
            file_path="utils.py",
            change_type="modify",
            timestamp=base_time + 10,
            before_content="def helper():\n    pass",
            after_content="def helper(x):\n    return x * 2",
        ),
        CodeChange(
            id=str(uuid.uuid4()),
            file_path="utils.py",
            change_type="modify",
            timestamp=base_time + 20,
            before_content="def helper(x):\n    return x * 2",
            after_content="def helper(x: int) -> int:\n    return x * 2",
        ),
    ]
    
    for change in changes:
        await store.add_change(change, compute_diffs=False)
    
    print(f"✓ Tracked {len(changes)} changes to utils.py")
    print()
    
    # Example 3: Query changes by file
    print("3. Querying changes by file...")
    utils_changes = await store.query_changes(file_path="utils.py")
    print(f"✓ Found {len(utils_changes)} changes to utils.py:")
    for change in utils_changes:
        print(f"  - {change.change_type} at {change.timestamp:.0f}")
    print()
    
    # Example 4: Build change graph
    print("4. Building change graph...")
    graph = await store.get_change_graph("utils.py")
    print(f"✓ Change graph for utils.py:")
    print(f"  - Total nodes: {len(graph.nodes)}")
    print(f"  - Root changes: {len(graph.root_ids)}")
    print(f"  - Leaf changes: {len(graph.leaf_ids)}")
    print(f"  - Evolution path:")
    for i, node in enumerate(graph.nodes):
        print(f"    {i+1}. {node.change_type} at {node.timestamp:.0f}")
    print()
    
    # Example 5: Reconstruct file at different times
    print("5. Reconstructing file at different times...")
    
    # At time of first change
    content_v1 = await store.reconstruct_file("utils.py", base_time + 5)
    print(f"✓ Content at t+5s:\n{content_v1}")
    print()
    
    # At time of second change
    content_v2 = await store.reconstruct_file("utils.py", base_time + 15)
    print(f"✓ Content at t+15s:\n{content_v2}")
    print()
    
    # At time of third change
    content_v3 = await store.reconstruct_file("utils.py", base_time + 25)
    print(f"✓ Content at t+25s:\n{content_v3}")
    print()
    
    # Example 6: Get file history
    print("6. Getting complete file history...")
    history = await store.get_file_history("utils.py")
    print(f"✓ File history ({len(history)} versions):")
    for i, (timestamp, content) in enumerate(history):
        print(f"  Version {i+1} at {timestamp:.0f}:")
        print(f"    {content[:50]}...")
    print()
    
    # Example 7: Query by time range
    print("7. Querying changes by time range...")
    time_range_changes = await store.query_changes(
        time_range=(base_time + 5, base_time + 15)
    )
    print(f"✓ Found {len(time_range_changes)} changes in time range")
    print()
    
    # Cleanup
    await store.close()
    print("=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())

