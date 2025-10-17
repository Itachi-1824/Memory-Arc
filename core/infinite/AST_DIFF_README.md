# AST Diff Engine

## Overview

The AST Diff Engine provides structural code analysis and diff generation using Abstract Syntax Trees (AST). It supports Python, JavaScript, and TypeScript.

## Features

### 1. AST Parsing
- **Python**: Full support for functions, classes, methods, variables
- **JavaScript**: Functions, classes, methods, variables
- **TypeScript**: Uses JavaScript parser (compatible with most TS syntax)

### 2. Symbol Extraction
Extracts code symbols with detailed information:
- Functions and methods (with parameters)
- Classes
- Variables
- Docstrings (Python)
- Line numbers and byte positions
- Parent-child relationships (e.g., methods within classes)

### 3. Structural Diff Generation
Computes differences between code versions at multiple levels:
- **Symbol-level**: Added, removed, and modified symbols
- **Node-level**: AST node changes with paths
- **Content-level**: Actual code changes

### 4. Symbol Reference Tracking
- Track all references to a specific symbol
- Returns line and column positions

### 5. Dependency Extraction
- Analyze which symbols reference which other symbols
- Build dependency graphs

### 6. Language Detection
- Automatic language detection from file extensions
- Supports .py, .js, .jsx, .ts, .tsx

## Usage

```python
from core.infinite.ast_diff import ASTDiffEngine, LanguageType

# Initialize engine
engine = ASTDiffEngine()

# Extract symbols
code = """
def calculate(x, y):
    return x + y

class Calculator:
    def add(self, a, b):
        return calculate(a, b)
"""

symbols = engine.extract_symbols(code, LanguageType.PYTHON)
for symbol in symbols:
    print(f"{symbol.symbol_type}: {symbol.name}")
    # Output:
    # function: calculate
    # class: Calculator
    # method: add

# Compute diff
before = "def foo(): pass"
after = "def foo(): return 1"

diff = engine.compute_ast_diff(before, after, LanguageType.PYTHON)
print(f"Changes: {len(diff.changes)}")
print(f"Added symbols: {len(diff.symbols_added)}")
print(f"Modified symbols: {len(diff.symbols_modified)}")

# Track symbol references
refs = engine.track_symbol_references(code, LanguageType.PYTHON, "calculate")
for line, col in refs:
    print(f"Reference at line {line}, column {col}")

# Extract dependencies
deps = engine.extract_dependencies(code, LanguageType.PYTHON)
for symbol, referenced in deps.items():
    if referenced:
        print(f"{symbol} uses: {', '.join(referenced)}")
```

## Data Models

### Symbol
Represents a code symbol (function, class, variable, etc.):
- `name`: Symbol name
- `symbol_type`: Type (function, class, method, variable)
- `start_line`, `end_line`: Line range
- `start_byte`, `end_byte`: Byte range
- `parent`: Parent symbol (e.g., class for a method)
- `parameters`: Function/method parameters
- `docstring`: Documentation string (Python)

### ASTDiff
Represents structural differences:
- `language`: Programming language
- `changes`: List of AST node changes
- `symbols_added`: Newly added symbols
- `symbols_removed`: Removed symbols
- `symbols_modified`: Modified symbols (before/after pairs)
- `metadata`: Additional information (symbol counts, etc.)

### ASTNodeChange
Represents a change to an AST node:
- `change_type`: ADDED, REMOVED, MODIFIED, UNCHANGED
- `node_type`: Type of AST node
- `path`: Path in the AST tree
- `before_text`, `after_text`: Content before/after
- `start_line`, `end_line`: Line range

## Implementation Details

### Tree-sitter Integration
Uses tree-sitter for robust AST parsing:
- `tree-sitter-python`: Python grammar
- `tree-sitter-javascript`: JavaScript/TypeScript grammar

### Recursive Symbol Extraction
Traverses AST recursively to extract all symbols, maintaining parent-child relationships.

### Diff Algorithm
1. Parse both code versions into ASTs
2. Extract symbols from each version
3. Compare symbol maps to find additions, removals, modifications
4. Build node-level change list
5. Return comprehensive diff object

### Performance
- Efficient parsing using tree-sitter's C-based parsers
- Lazy evaluation where possible
- Configurable depth limits for node traversal

## Requirements

Satisfied requirements from the spec:
- **3.3**: Multi-level diffs (symbol-level and node-level)
- **7.2**: Structural diffs with AST analysis

## Testing

See `examples/ast_diff_example.py` for comprehensive usage examples.

Unit tests are in task 3.4 (separate task).

## Future Enhancements

Potential improvements:
- Support for more languages (Go, Rust, C++, etc.)
- Semantic diff descriptions (auto-generate change summaries)
- Performance optimizations for very large files
- Incremental parsing for real-time analysis
