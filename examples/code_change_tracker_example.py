"""Example demonstrating CodeChangeTracker usage."""

import asyncio
import tempfile
import time
import sys
import os
import importlib.util
from pathlib import Path

# Add parent directory to path to import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.infinite.code_change_tracker import CodeChangeTracker


async def main():
    """Demonstrate CodeChangeTracker functionality."""
    
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        watch_path = Path(temp_dir) / "project"
        watch_path.mkdir()
        
        db_path = Path(temp_dir) / "changes.db"
        
        print("=" * 60)
        print("CodeChangeTracker Example")
        print("=" * 60)
        
        # Initialize tracker
        tracker = CodeChangeTracker(
            watch_path=watch_path,
            db_path=db_path,
            auto_track=False,  # Manual tracking for this example
        )
        
        await tracker.initialize()
        print(f"\n✓ Initialized tracker for {watch_path}")
        
        # Example 1: Track a new file creation
        print("\n" + "=" * 60)
        print("Example 1: Track File Creation")
        print("=" * 60)
        
        test_file = watch_path / "example.py"
        initial_content = '''def hello():
    """Say hello."""
    print("Hello, World!")
'''
        
        change_id_1 = await tracker.track_change(
            file_path=str(test_file),
            before_content=None,
            after_content=initial_content,
            change_type="add",
        )
        
        print(f"\n✓ Tracked file creation: {change_id_1}")
        
        # Example 2: Track a modification
        print("\n" + "=" * 60)
        print("Example 2: Track File Modification")
        print("=" * 60)
        
        time.sleep(0.1)  # Small delay to ensure different timestamps
        
        modified_content = '''def hello(name="World"):
    """Say hello to someone."""
    print(f"Hello, {name}!")

def goodbye():
    """Say goodbye."""
    print("Goodbye!")
'''
        
        change_id_2 = await tracker.track_change(
            file_path=str(test_file),
            before_content=initial_content,
            after_content=modified_content,
            change_type="modify",
        )
        
        print(f"\n✓ Tracked file modification: {change_id_2}")
        
        # Example 3: Get diffs at different levels
        print("\n" + "=" * 60)
        print("Example 3: Retrieve Diffs at Different Levels")
        print("=" * 60)
        
        # Get unified diff
        unified_diff = await tracker.get_diff(change_id_2, diff_level="unified")
        if unified_diff:
            print("\nUnified Diff:")
            print("-" * 40)
            print(unified_diff.get_content()[:500])  # First 500 chars
            print("-" * 40)
        
        # Get AST diff
        ast_diff = await tracker.get_diff(change_id_2, diff_level="ast")
        if ast_diff:
            print(f"\nAST Diff Summary:")
            print(f"  - Symbols added: {len(ast_diff.symbols_added)}")
            print(f"  - Symbols removed: {len(ast_diff.symbols_removed)}")
            print(f"  - Symbols modified: {len(ast_diff.symbols_modified)}")
            
            if ast_diff.symbols_added:
                print(f"\n  Added symbols:")
                for symbol in ast_diff.symbols_added:
                    print(f"    - {symbol.symbol_type}: {symbol.name}")
            
            if ast_diff.symbols_modified:
                print(f"\n  Modified symbols:")
                for before, after in ast_diff.symbols_modified:
                    print(f"    - {before.symbol_type}: {before.name}")
        
        # Example 4: Query changes
        print("\n" + "=" * 60)
        print("Example 4: Query Changes")
        print("=" * 60)
        
        changes = await tracker.query_changes(
            file_path=str(test_file),
            limit=10,
        )
        
        print(f"\n✓ Found {len(changes)} changes for {test_file.name}")
        for i, change in enumerate(changes, 1):
            print(f"\n  Change {i}:")
            print(f"    - ID: {change.id}")
            print(f"    - Type: {change.change_type}")
            print(f"    - Timestamp: {change.timestamp}")
        
        # Example 5: Query by function name
        print("\n" + "=" * 60)
        print("Example 5: Query Changes by Function Name")
        print("=" * 60)
        
        goodbye_changes = await tracker.query_changes(
            file_path=str(test_file),
            function_name="goodbye",
        )
        
        print(f"\n✓ Found {len(goodbye_changes)} changes affecting 'goodbye' function")
        
        # Example 6: Reconstruct file at different times
        print("\n" + "=" * 60)
        print("Example 6: File Reconstruction")
        print("=" * 60)
        
        # Get all changes to determine timestamps
        all_changes = await tracker.query_changes(file_path=str(test_file))
        
        if len(all_changes) >= 2:
            # Reconstruct at first change time
            first_time = all_changes[-1].timestamp
            first_content = await tracker.reconstruct_file(
                str(test_file),
                at_time=first_time,
            )
            
            print(f"\nContent at first change (t={first_time:.2f}):")
            print("-" * 40)
            print(first_content)
            print("-" * 40)
            
            # Reconstruct at second change time
            second_time = all_changes[-2].timestamp
            second_content = await tracker.reconstruct_file(
                str(test_file),
                at_time=second_time,
            )
            
            print(f"\nContent at second change (t={second_time:.2f}):")
            print("-" * 40)
            print(second_content)
            print("-" * 40)
        
        # Example 7: Get change graph
        print("\n" + "=" * 60)
        print("Example 7: Change Graph")
        print("=" * 60)
        
        graph = await tracker.get_change_graph(str(test_file))
        
        print(f"\n✓ Change graph for {test_file.name}:")
        print(f"  - Total nodes: {len(graph.nodes)}")
        print(f"  - Root changes: {len(graph.root_ids)}")
        print(f"  - Leaf changes: {len(graph.leaf_ids)}")
        
        print("\n  Evolution path:")
        for i, node in enumerate(graph.nodes, 1):
            print(f"    {i}. {node.change_type} at t={node.timestamp:.2f}")
        
        # Example 8: Track symbol evolution
        print("\n" + "=" * 60)
        print("Example 8: Symbol Evolution")
        print("=" * 60)
        
        hello_evolution = await tracker.track_symbol_evolution(
            str(test_file),
            "hello",
        )
        
        print(f"\n✓ Evolution of 'hello' function:")
        for timestamp, symbol in hello_evolution:
            if symbol:
                params = ", ".join(symbol.parameters) if symbol.parameters else "none"
                print(f"  - t={timestamp:.2f}: {symbol.symbol_type} with params: {params}")
            else:
                print(f"  - t={timestamp:.2f}: not found")
        
        # Example 9: Get symbols at specific time
        print("\n" + "=" * 60)
        print("Example 9: Symbols at Specific Time")
        print("=" * 60)
        
        if all_changes:
            latest_time = all_changes[0].timestamp
            symbols = await tracker.get_symbols_at_time(
                str(test_file),
                at_time=latest_time,
            )
            
            print(f"\n✓ Symbols at t={latest_time:.2f}:")
            for symbol in symbols:
                print(f"  - {symbol.symbol_type}: {symbol.name}")
                if symbol.parameters:
                    print(f"    Parameters: {', '.join(symbol.parameters)}")
                if symbol.docstring:
                    print(f"    Docstring: {symbol.docstring[:50]}...")
        
        # Example 10: File history
        print("\n" + "=" * 60)
        print("Example 10: File History")
        print("=" * 60)
        
        history = await tracker.get_file_history(str(test_file))
        
        print(f"\n✓ File history ({len(history)} versions):")
        for i, (timestamp, content) in enumerate(history, 1):
            lines = content.count('\n') + 1
            print(f"  {i}. t={timestamp:.2f}: {lines} lines")
        
        # Clean up
        await tracker.close()
        print("\n" + "=" * 60)
        print("✓ Example completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
