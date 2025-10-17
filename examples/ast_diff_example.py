"""
Example: AST Diff Engine Usage

This example demonstrates how to use the AST diff engine to:
1. Parse code into Abstract Syntax Trees (AST)
2. Extract symbols (functions, classes, variables)
3. Compute structural diffs between code versions
4. Track symbol references and dependencies

Note: Run this from the project root directory.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from the module to avoid circular imports
from core.infinite.ast_diff import ASTDiffEngine, LanguageType

def example_python_parsing():
    """Example: Parse Python code and extract symbols."""
    print("=" * 60)
    print("Example 1: Python Symbol Extraction")
    print("=" * 60)
    
    code = """
def calculate_total(items):
    '''Calculate the total price of items.'''
    total = 0
    for item in items:
        total += item.price
    return total

class ShoppingCart:
    '''A shopping cart for e-commerce.'''
    
    def __init__(self):
        self.items = []
    
    def add_item(self, item):
        self.items.append(item)
    
    def get_total(self):
        return calculate_total(self.items)
"""
    
    engine = ASTDiffEngine()
    symbols = engine.extract_symbols(code, LanguageType.PYTHON)
    
    print(f"\nFound {len(symbols)} symbols:\n")
    for symbol in symbols:
        parent_info = f" (in {symbol.parent})" if symbol.parent else ""
        params_info = f"({', '.join(symbol.parameters)})" if symbol.parameters else ""
        print(f"  {symbol.symbol_type}: {symbol.name}{params_info}{parent_info}")
        print(f"    Lines {symbol.start_line}-{symbol.end_line}")
        if symbol.docstring:
            print(f"    Doc: {symbol.docstring[:50]}...")
        print()


def example_javascript_parsing():
    """Example: Parse JavaScript code and extract symbols."""
    print("=" * 60)
    print("Example 2: JavaScript Symbol Extraction")
    print("=" * 60)
    
    code = """
function fetchData(url) {
    return fetch(url).then(response => response.json());
}

class DataManager {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.cache = new Map();
    }
    
    async getData(id) {
        if (this.cache.has(id)) {
            return this.cache.get(id);
        }
        const data = await fetchData(`${this.apiUrl}/${id}`);
        this.cache.set(id, data);
        return data;
    }
}
"""
    
    engine = ASTDiffEngine()
    symbols = engine.extract_symbols(code, LanguageType.JAVASCRIPT)
    
    print(f"\nFound {len(symbols)} symbols:\n")
    for symbol in symbols:
        parent_info = f" (in {symbol.parent})" if symbol.parent else ""
        params_info = f"({', '.join(symbol.parameters)})" if symbol.parameters else ""
        print(f"  {symbol.symbol_type}: {symbol.name}{params_info}{parent_info}")
        print(f"    Lines {symbol.start_line}-{symbol.end_line}")
        print()


def example_ast_diff():
    """Example: Compute structural diff between code versions."""
    print("=" * 60)
    print("Example 3: Computing AST Diff")
    print("=" * 60)
    
    before = """
def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result

class Processor:
    def run(self):
        pass
"""
    
    after = """
def process_data(data, multiplier=2):
    result = []
    for item in data:
        result.append(item * multiplier)
    return result

def validate_data(data):
    return all(isinstance(x, (int, float)) for x in data)

class Processor:
    def __init__(self, config):
        self.config = config
    
    def run(self, data):
        if validate_data(data):
            return process_data(data)
        return None
"""
    
    engine = ASTDiffEngine()
    diff = engine.compute_ast_diff(before, after, LanguageType.PYTHON)
    
    print(f"\nStructural Changes:")
    print(f"  Total changes: {diff.metadata['total_changes']}")
    print(f"  Before symbols: {diff.metadata['before_symbol_count']}")
    print(f"  After symbols: {diff.metadata['after_symbol_count']}")
    
    if diff.symbols_added:
        print(f"\n  Added symbols ({len(diff.symbols_added)}):")
        for symbol in diff.symbols_added:
            print(f"    + {symbol.symbol_type}: {symbol.name}")
    
    if diff.symbols_removed:
        print(f"\n  Removed symbols ({len(diff.symbols_removed)}):")
        for symbol in diff.symbols_removed:
            print(f"    - {symbol.symbol_type}: {symbol.name}")
    
    if diff.symbols_modified:
        print(f"\n  Modified symbols ({len(diff.symbols_modified)}):")
        for before_sym, after_sym in diff.symbols_modified:
            print(f"    ~ {before_sym.symbol_type}: {before_sym.name}")
            print(f"      Before: lines {before_sym.start_line}-{before_sym.end_line}")
            print(f"      After:  lines {after_sym.start_line}-{after_sym.end_line}")


def example_symbol_references():
    """Example: Track symbol references and dependencies."""
    print("=" * 60)
    print("Example 4: Symbol References and Dependencies")
    print("=" * 60)
    
    code = """
def calculate_tax(amount, rate):
    return amount * rate

def calculate_total(subtotal, tax_rate):
    tax = calculate_tax(subtotal, tax_rate)
    total = subtotal + tax
    return total

def process_order(order):
    subtotal = sum(item.price for item in order.items)
    total = calculate_total(subtotal, 0.08)
    return total
"""
    
    engine = ASTDiffEngine()
    
    # Track references to 'calculate_tax'
    refs = engine.track_symbol_references(code, LanguageType.PYTHON, "calculate_tax")
    print(f"\nReferences to 'calculate_tax': {len(refs)}")
    for line, col in refs:
        print(f"  Line {line}, Column {col}")
    
    # Extract dependencies
    deps = engine.extract_dependencies(code, LanguageType.PYTHON)
    print(f"\nSymbol Dependencies:")
    for symbol, referenced in deps.items():
        if referenced:
            print(f"  {symbol} uses: {', '.join(referenced)}")


def example_language_detection():
    """Example: Automatic language detection."""
    print("=" * 60)
    print("Example 5: Language Detection")
    print("=" * 60)
    
    engine = ASTDiffEngine()
    
    files = [
        "app.py",
        "utils.js",
        "component.tsx",
        "script.jsx",
        "types.ts",
    ]
    
    print("\nDetected languages:")
    for file in files:
        lang = engine.detect_language(file)
        print(f"  {file} -> {lang.value}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AST Diff Engine Examples")
    print("=" * 60 + "\n")
    
    try:
        example_python_parsing()
        print("\n")
        
        example_javascript_parsing()
        print("\n")
        
        example_ast_diff()
        print("\n")
        
        example_symbol_references()
        print("\n")
        
        example_language_detection()
        
        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
