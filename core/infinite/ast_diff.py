"""AST-based diff generation and symbol extraction for code changes."""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from enum import Enum

try:
    from tree_sitter import Language, Parser, Node, Tree
    import tree_sitter_python as ts_python
    import tree_sitter_javascript as ts_javascript
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Language = None
    Parser = None
    Node = None
    Tree = None


class LanguageType(Enum):
    """Supported programming languages for AST parsing."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"


class ChangeType(Enum):
    """Types of AST node changes."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class Symbol:
    """Represents a code symbol (function, class, variable, etc.)."""
    name: str
    symbol_type: str  # 'function', 'class', 'variable', 'method', etc.
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    parent: str | None = None  # Parent symbol name (e.g., class for a method)
    parameters: list[str] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert symbol to dictionary."""
        return {
            "name": self.name,
            "symbol_type": self.symbol_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "parent": self.parent,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "docstring": self.docstring,
            "metadata": self.metadata,
        }


@dataclass
class ASTNodeChange:
    """Represents a change to an AST node."""
    change_type: ChangeType
    node_type: str
    path: str  # Path in the AST tree (e.g., "module.class.function")
    before_text: str | None = None
    after_text: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert node change to dictionary."""
        return {
            "change_type": self.change_type.value,
            "node_type": self.node_type,
            "path": self.path,
            "before_text": self.before_text,
            "after_text": self.after_text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metadata": self.metadata,
        }


@dataclass
class ASTDiff:
    """Represents structural differences between two ASTs."""
    language: LanguageType
    changes: list[ASTNodeChange]
    symbols_added: list[Symbol]
    symbols_removed: list[Symbol]
    symbols_modified: list[tuple[Symbol, Symbol]]  # (before, after)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert AST diff to dictionary."""
        return {
            "language": self.language.value,
            "changes": [c.to_dict() for c in self.changes],
            "symbols_added": [s.to_dict() for s in self.symbols_added],
            "symbols_removed": [s.to_dict() for s in self.symbols_removed],
            "symbols_modified": [
                (before.to_dict(), after.to_dict())
                for before, after in self.symbols_modified
            ],
            "metadata": self.metadata,
        }


class ASTDiffEngine:
    """
    Engine for parsing code into ASTs and computing structural diffs.
    
    Supports:
    - Python, JavaScript, TypeScript parsing
    - Symbol extraction (functions, classes, variables)
    - Structural diff generation
    - Symbol reference tracking
    """
    
    def __init__(self):
        """Initialize AST diff engine with language parsers."""
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter not available. Install with: "
                "pip install tree-sitter tree-sitter-python tree-sitter-javascript"
            )
        
        # Initialize parsers for each language
        self.parsers: dict[LanguageType, Parser] = {}
        self.languages: dict[LanguageType, Language] = {}
        
        # Python
        self.languages[LanguageType.PYTHON] = Language(ts_python.language())
        python_parser = Parser(self.languages[LanguageType.PYTHON])
        self.parsers[LanguageType.PYTHON] = python_parser
        
        # JavaScript (also used for TypeScript with minor differences)
        self.languages[LanguageType.JAVASCRIPT] = Language(ts_javascript.language())
        js_parser = Parser(self.languages[LanguageType.JAVASCRIPT])
        self.parsers[LanguageType.JAVASCRIPT] = js_parser
        self.parsers[LanguageType.TYPESCRIPT] = js_parser  # Reuse JS parser
    
    def detect_language(self, file_path: str, content: str | None = None) -> LanguageType:
        """
        Detect programming language from file path or content.
        
        Args:
            file_path: Path to the file
            content: Optional file content for detection
            
        Returns:
            Detected language type
        """
        # Simple extension-based detection
        if file_path.endswith('.py'):
            return LanguageType.PYTHON
        elif file_path.endswith('.js') or file_path.endswith('.jsx'):
            return LanguageType.JAVASCRIPT
        elif file_path.endswith('.ts') or file_path.endswith('.tsx'):
            return LanguageType.TYPESCRIPT
        
        return LanguageType.UNKNOWN
    
    def parse(self, code: str, language: LanguageType) -> Optional[Any]:
        """
        Parse code into an AST.
        
        Args:
            code: Source code to parse
            language: Programming language
            
        Returns:
            Parsed AST tree or None if parsing fails
        """
        if language not in self.parsers:
            return None
        
        parser = self.parsers[language]
        tree = parser.parse(bytes(code, "utf-8"))
        return tree
    
    def extract_symbols(self, code: str, language: LanguageType) -> list[Symbol]:
        """
        Extract symbols (functions, classes, variables) from code.
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            List of extracted symbols
        """
        tree = self.parse(code, language)
        if not tree:
            return []
        
        symbols = []
        code_bytes = bytes(code, "utf-8")
        
        if language == LanguageType.PYTHON:
            symbols = self._extract_python_symbols(tree.root_node, code_bytes)
        elif language in (LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT):
            symbols = self._extract_javascript_symbols(tree.root_node, code_bytes)
        
        return symbols
    
    def _extract_python_symbols(
        self,
        node: Node,
        code_bytes: bytes,
        parent: str | None = None
    ) -> list[Symbol]:
        """Extract symbols from Python AST."""
        symbols = []
        
        if node.type == 'function_definition':
            symbol = self._extract_python_function(node, code_bytes, parent)
            if symbol:
                symbols.append(symbol)
                # Recursively extract nested symbols within this function
                for child in node.children:
                    symbols.extend(
                        self._extract_python_symbols(child, code_bytes, symbol.name)
                    )
        
        elif node.type == 'class_definition':
            symbol = self._extract_python_class(node, code_bytes, parent)
            if symbol:
                symbols.append(symbol)
                # Extract methods from class
                for child in node.children:
                    symbols.extend(
                        self._extract_python_symbols(child, code_bytes, symbol.name)
                    )
        
        elif node.type == 'assignment':
            # Extract variable assignments
            symbol = self._extract_python_variable(node, code_bytes, parent)
            if symbol:
                symbols.append(symbol)
        
        else:
            # Recursively process child nodes for other node types
            for child in node.children:
                symbols.extend(
                    self._extract_python_symbols(child, code_bytes, parent)
                )
        
        return symbols
    
    def _extract_python_function(
        self,
        node: Node,
        code_bytes: bytes,
        parent: str | None
    ) -> Symbol | None:
        """Extract function symbol from Python AST node."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Extract parameters
        parameters = []
        params_node = node.child_by_field_name('parameters')
        if params_node:
            for param in params_node.children:
                if param.type == 'identifier':
                    param_name = code_bytes[param.start_byte:param.end_byte].decode('utf-8')
                    parameters.append(param_name)
        
        # Extract docstring
        docstring = None
        body = node.child_by_field_name('body')
        if body and len(body.children) > 0:
            first_stmt = body.children[0]
            if first_stmt.type == 'expression_statement':
                expr = first_stmt.children[0] if first_stmt.children else None
                if expr and expr.type == 'string':
                    docstring = code_bytes[expr.start_byte:expr.end_byte].decode('utf-8')
        
        return Symbol(
            name=name,
            symbol_type='method' if parent else 'function',
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            parent=parent,
            parameters=parameters,
            docstring=docstring,
        )
    
    def _extract_python_class(
        self,
        node: Node,
        code_bytes: bytes,
        parent: str | None
    ) -> Symbol | None:
        """Extract class symbol from Python AST node."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Extract docstring
        docstring = None
        body = node.child_by_field_name('body')
        if body and len(body.children) > 0:
            first_stmt = body.children[0]
            if first_stmt.type == 'expression_statement':
                expr = first_stmt.children[0] if first_stmt.children else None
                if expr and expr.type == 'string':
                    docstring = code_bytes[expr.start_byte:expr.end_byte].decode('utf-8')
        
        return Symbol(
            name=name,
            symbol_type='class',
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            parent=parent,
            docstring=docstring,
        )
    
    def _extract_python_variable(
        self,
        node: Node,
        code_bytes: bytes,
        parent: str | None
    ) -> Symbol | None:
        """Extract variable symbol from Python AST node."""
        # Get the left side of assignment
        left = node.child_by_field_name('left')
        if not left or left.type != 'identifier':
            return None
        
        name = code_bytes[left.start_byte:left.end_byte].decode('utf-8')
        
        return Symbol(
            name=name,
            symbol_type='variable',
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            parent=parent,
        )
    
    def _extract_javascript_symbols(
        self,
        node: Node,
        code_bytes: bytes,
        parent: str | None = None
    ) -> list[Symbol]:
        """Extract symbols from JavaScript/TypeScript AST."""
        symbols = []
        
        if node.type in ('function_declaration', 'function'):
            symbol = self._extract_js_function(node, code_bytes, parent)
            if symbol:
                symbols.append(symbol)
        
        elif node.type == 'class_declaration':
            symbol = self._extract_js_class(node, code_bytes, parent)
            if symbol:
                symbols.append(symbol)
                # Extract methods from class
                for child in node.children:
                    symbols.extend(
                        self._extract_javascript_symbols(child, code_bytes, symbol.name)
                    )
        
        elif node.type == 'method_definition':
            symbol = self._extract_js_method(node, code_bytes, parent)
            if symbol:
                symbols.append(symbol)
        
        elif node.type in ('variable_declaration', 'lexical_declaration'):
            # Extract variable declarations
            for child in node.children:
                if child.type == 'variable_declarator':
                    symbol = self._extract_js_variable(child, code_bytes, parent)
                    if symbol:
                        symbols.append(symbol)
        
        else:
            # Recursively process child nodes for other node types
            for child in node.children:
                symbols.extend(
                    self._extract_javascript_symbols(child, code_bytes, parent)
                )
        
        return symbols
    
    def _extract_js_function(
        self,
        node: Node,
        code_bytes: bytes,
        parent: str | None
    ) -> Symbol | None:
        """Extract function symbol from JavaScript AST node."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Extract parameters
        parameters = []
        params_node = node.child_by_field_name('parameters')
        if params_node:
            for param in params_node.children:
                if param.type in ('identifier', 'required_parameter'):
                    param_name = code_bytes[param.start_byte:param.end_byte].decode('utf-8')
                    parameters.append(param_name)
        
        return Symbol(
            name=name,
            symbol_type='function',
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            parent=parent,
            parameters=parameters,
        )
    
    def _extract_js_class(
        self,
        node: Node,
        code_bytes: bytes,
        parent: str | None
    ) -> Symbol | None:
        """Extract class symbol from JavaScript AST node."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        return Symbol(
            name=name,
            symbol_type='class',
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            parent=parent,
        )
    
    def _extract_js_method(
        self,
        node: Node,
        code_bytes: bytes,
        parent: str | None
    ) -> Symbol | None:
        """Extract method symbol from JavaScript AST node."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Extract parameters
        parameters = []
        params_node = node.child_by_field_name('parameters')
        if params_node:
            for param in params_node.children:
                if param.type in ('identifier', 'required_parameter'):
                    param_name = code_bytes[param.start_byte:param.end_byte].decode('utf-8')
                    parameters.append(param_name)
        
        return Symbol(
            name=name,
            symbol_type='method',
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            parent=parent,
            parameters=parameters,
        )
    
    def _extract_js_variable(
        self,
        node: Node,
        code_bytes: bytes,
        parent: str | None
    ) -> Symbol | None:
        """Extract variable symbol from JavaScript AST node."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        return Symbol(
            name=name,
            symbol_type='variable',
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            parent=parent,
        )
    
    def compute_ast_diff(
        self,
        before: str,
        after: str,
        language: LanguageType
    ) -> ASTDiff:
        """
        Compute structural diff between two code versions.
        
        Args:
            before: Original code
            after: Modified code
            language: Programming language
            
        Returns:
            AST diff with structural changes and symbol differences
        """
        # Extract symbols from both versions
        before_symbols = self.extract_symbols(before, language)
        after_symbols = self.extract_symbols(after, language)
        
        # Create symbol maps for comparison
        before_map = {self._symbol_key(s): s for s in before_symbols}
        after_map = {self._symbol_key(s): s for s in after_symbols}
        
        # Find added, removed, and modified symbols
        symbols_added = []
        symbols_removed = []
        symbols_modified = []
        
        # Find removed symbols
        for key, symbol in before_map.items():
            if key not in after_map:
                symbols_removed.append(symbol)
        
        # Find added and modified symbols
        for key, symbol in after_map.items():
            if key not in before_map:
                symbols_added.append(symbol)
            else:
                # Check if symbol was modified
                before_symbol = before_map[key]
                if self._symbol_changed(before_symbol, symbol, before, after):
                    symbols_modified.append((before_symbol, symbol))
        
        # Compute node-level changes
        changes = self._compute_node_changes(before, after, language)
        
        return ASTDiff(
            language=language,
            changes=changes,
            symbols_added=symbols_added,
            symbols_removed=symbols_removed,
            symbols_modified=symbols_modified,
            metadata={
                "before_symbol_count": len(before_symbols),
                "after_symbol_count": len(after_symbols),
                "total_changes": len(changes),
            }
        )
    
    def _symbol_key(self, symbol: Symbol) -> str:
        """Generate unique key for symbol comparison."""
        parent_part = f"{symbol.parent}." if symbol.parent else ""
        return f"{parent_part}{symbol.name}:{symbol.symbol_type}"
    
    def _symbol_changed(
        self,
        before: Symbol,
        after: Symbol,
        before_code: str,
        after_code: str
    ) -> bool:
        """Check if symbol content has changed."""
        # Extract symbol content from code
        before_content = before_code[before.start_byte:before.end_byte]
        after_content = after_code[after.start_byte:after.end_byte]
        
        # Compare content hashes for efficiency
        before_hash = hashlib.md5(before_content.encode()).hexdigest()
        after_hash = hashlib.md5(after_content.encode()).hexdigest()
        
        return before_hash != after_hash
    
    def _compute_node_changes(
        self,
        before: str,
        after: str,
        language: LanguageType
    ) -> list[ASTNodeChange]:
        """Compute node-level changes between two ASTs."""
        before_tree = self.parse(before, language)
        after_tree = self.parse(after, language)
        
        if not before_tree or not after_tree:
            return []
        
        changes = []
        
        # Build node maps for comparison
        before_nodes = self._build_node_map(before_tree.root_node, bytes(before, "utf-8"))
        after_nodes = self._build_node_map(after_tree.root_node, bytes(after, "utf-8"))
        
        # Find removed nodes
        for path, (node, text) in before_nodes.items():
            if path not in after_nodes:
                changes.append(ASTNodeChange(
                    change_type=ChangeType.REMOVED,
                    node_type=node.type,
                    path=path,
                    before_text=text,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                ))
        
        # Find added and modified nodes
        for path, (node, text) in after_nodes.items():
            if path not in before_nodes:
                changes.append(ASTNodeChange(
                    change_type=ChangeType.ADDED,
                    node_type=node.type,
                    path=path,
                    after_text=text,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                ))
            else:
                before_node, before_text = before_nodes[path]
                if text != before_text:
                    changes.append(ASTNodeChange(
                        change_type=ChangeType.MODIFIED,
                        node_type=node.type,
                        path=path,
                        before_text=before_text,
                        after_text=text,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    ))
        
        return changes
    
    def _build_node_map(
        self,
        node: Node,
        code_bytes: bytes,
        path: str = "root",
        depth: int = 0,
        max_depth: int = 10
    ) -> dict[str, tuple[Node, str]]:
        """Build a map of AST nodes with their paths."""
        if depth > max_depth:
            return {}
        
        node_map = {}
        
        # Only track significant node types
        significant_types = {
            'function_definition', 'class_definition', 'method_definition',
            'function_declaration', 'class_declaration', 'assignment',
            'variable_declaration', 'lexical_declaration', 'if_statement',
            'for_statement', 'while_statement', 'return_statement',
        }
        
        if node.type in significant_types:
            text = code_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            node_map[path] = (node, text)
        
        # Recursively process children
        for i, child in enumerate(node.children):
            child_path = f"{path}.{child.type}[{i}]"
            node_map.update(
                self._build_node_map(child, code_bytes, child_path, depth + 1, max_depth)
            )
        
        return node_map
    
    def track_symbol_references(
        self,
        code: str,
        language: LanguageType,
        symbol_name: str
    ) -> list[tuple[int, int]]:
        """
        Track all references to a specific symbol in the code.
        
        Args:
            code: Source code
            language: Programming language
            symbol_name: Name of symbol to track
            
        Returns:
            List of (line, column) positions where symbol is referenced
        """
        tree = self.parse(code, language)
        if not tree:
            return []
        
        references = []
        code_bytes = bytes(code, "utf-8")
        
        def find_references(node: Node):
            """Recursively find symbol references."""
            if node.type == 'identifier':
                text = code_bytes[node.start_byte:node.end_byte].decode('utf-8')
                if text == symbol_name:
                    references.append((
                        node.start_point[0] + 1,
                        node.start_point[1]
                    ))
            
            for child in node.children:
                find_references(child)
        
        find_references(tree.root_node)
        return references
    
    def extract_dependencies(
        self,
        code: str,
        language: LanguageType
    ) -> dict[str, list[str]]:
        """
        Extract symbol dependencies (which symbols use which other symbols).
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            Dictionary mapping symbol names to list of symbols they reference
        """
        symbols = self.extract_symbols(code, language)
        tree = self.parse(code, language)
        
        if not tree:
            return {}
        
        dependencies = {}
        code_bytes = bytes(code, "utf-8")
        
        # For each symbol, find what it references
        for symbol in symbols:
            refs = set()
            
            # Get the node for this symbol
            symbol_text = code[symbol.start_byte:symbol.end_byte]
            symbol_tree = self.parse(symbol_text, language)
            
            if symbol_tree:
                # Find all identifiers in this symbol's scope
                def collect_identifiers(node: Node):
                    if node.type == 'identifier':
                        text = code_bytes[node.start_byte:node.end_byte].decode('utf-8')
                        # Don't include the symbol itself
                        if text != symbol.name:
                            refs.add(text)
                    
                    for child in node.children:
                        collect_identifiers(child)
                
                collect_identifiers(symbol_tree.root_node)
            
            dependencies[symbol.name] = list(refs)
        
        return dependencies
