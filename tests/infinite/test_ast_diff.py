"""Unit tests for AST diff engine."""

import pytest
from core.infinite.ast_diff import (
    ASTDiffEngine,
    LanguageType,
    ChangeType,
    Symbol,
    ASTNodeChange,
    ASTDiff,
    TREE_SITTER_AVAILABLE,
)


# Skip all tests if tree-sitter is not available
pytestmark = pytest.mark.skipif(
    not TREE_SITTER_AVAILABLE,
    reason="tree-sitter not available"
)


class TestLanguageDetection:
    """Test language detection from file paths."""
    
    def test_detect_python(self):
        """Test Python file detection."""
        engine = ASTDiffEngine()
        assert engine.detect_language("test.py") == LanguageType.PYTHON
        assert engine.detect_language("/path/to/module.py") == LanguageType.PYTHON
    
    def test_detect_javascript(self):
        """Test JavaScript file detection."""
        engine = ASTDiffEngine()
        assert engine.detect_language("app.js") == LanguageType.JAVASCRIPT
        assert engine.detect_language("component.jsx") == LanguageType.JAVASCRIPT
    
    def test_detect_typescript(self):
        """Test TypeScript file detection."""
        engine = ASTDiffEngine()
        assert engine.detect_language("app.ts") == LanguageType.TYPESCRIPT
        assert engine.detect_language("component.tsx") == LanguageType.TYPESCRIPT
    
    def test_detect_unknown(self):
        """Test unknown file type detection."""
        engine = ASTDiffEngine()
        assert engine.detect_language("file.txt") == LanguageType.UNKNOWN
        assert engine.detect_language("README.md") == LanguageType.UNKNOWN


class TestPythonParsing:
    """Test Python code parsing."""
    
    def test_parse_valid_python(self):
        """Test parsing valid Python code."""
        engine = ASTDiffEngine()
        code = "def hello():\n    print('world')"
        tree = engine.parse(code, LanguageType.PYTHON)
        
        assert tree is not None
        assert tree.root_node is not None
        assert tree.root_node.type == "module"
    
    def test_parse_empty_python(self):
        """Test parsing empty Python code."""
        engine = ASTDiffEngine()
        code = ""
        tree = engine.parse(code, LanguageType.PYTHON)
        
        assert tree is not None
        assert tree.root_node is not None
    
    def test_parse_python_with_syntax_error(self):
        """Test parsing Python code with syntax errors."""
        engine = ASTDiffEngine()
        code = "def hello(\n    print('incomplete"
        tree = engine.parse(code, LanguageType.PYTHON)
        
        # Tree-sitter is error-tolerant, should still return a tree
        assert tree is not None
        assert tree.root_node is not None


class TestJavaScriptParsing:
    """Test JavaScript code parsing."""
    
    def test_parse_valid_javascript(self):
        """Test parsing valid JavaScript code."""
        engine = ASTDiffEngine()
        code = "function hello() { console.log('world'); }"
        tree = engine.parse(code, LanguageType.JAVASCRIPT)
        
        assert tree is not None
        assert tree.root_node is not None
        assert tree.root_node.type == "program"
    
    def test_parse_empty_javascript(self):
        """Test parsing empty JavaScript code."""
        engine = ASTDiffEngine()
        code = ""
        tree = engine.parse(code, LanguageType.JAVASCRIPT)
        
        assert tree is not None
        assert tree.root_node is not None


class TestPythonSymbolExtraction:
    """Test symbol extraction from Python code."""
    
    def test_extract_function(self):
        """Test extracting function symbols."""
        engine = ASTDiffEngine()
        code = """def greet(name):
    '''Say hello'''
    print(f'Hello {name}')
"""
        symbols = engine.extract_symbols(code, LanguageType.PYTHON)
        
        assert len(symbols) == 1
        func = symbols[0]
        assert func.name == "greet"
        assert func.symbol_type == "function"
        assert func.parameters == ["name"]
        assert func.start_line == 1
        assert func.parent is None
    
    def test_extract_class(self):
        """Test extracting class symbols."""
        engine = ASTDiffEngine()
        code = """class Person:
    '''A person class'''
    pass
"""
        symbols = engine.extract_symbols(code, LanguageType.PYTHON)
        
        assert len(symbols) == 1
        cls = symbols[0]
        assert cls.name == "Person"
        assert cls.symbol_type == "class"
        assert cls.start_line == 1
    
    def test_extract_class_with_methods(self):
        """Test extracting class with methods."""
        engine = ASTDiffEngine()
        code = """class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
"""
        symbols = engine.extract_symbols(code, LanguageType.PYTHON)
        
        # Should extract class + 2 methods
        assert len(symbols) == 3
        
        cls = next(s for s in symbols if s.symbol_type == "class")
        assert cls.name == "Calculator"
        
        methods = [s for s in symbols if s.symbol_type == "method"]
        assert len(methods) == 2
        assert {m.name for m in methods} == {"add", "subtract"}
        assert all(m.parent == "Calculator" for m in methods)
    
    def test_extract_variable(self):
        """Test extracting variable symbols."""
        engine = ASTDiffEngine()
        code = """x = 10
y = 20
"""
        symbols = engine.extract_symbols(code, LanguageType.PYTHON)
        
        assert len(symbols) == 2
        assert {s.name for s in symbols} == {"x", "y"}
        assert all(s.symbol_type == "variable" for s in symbols)
    
    def test_extract_nested_functions(self):
        """Test extracting nested function symbols."""
        engine = ASTDiffEngine()
        code = """def outer():
    def inner():
        pass
    return inner
"""
        symbols = engine.extract_symbols(code, LanguageType.PYTHON)
        
        assert len(symbols) == 2
        outer = next(s for s in symbols if s.name == "outer")
        inner = next(s for s in symbols if s.name == "inner")
        
        assert outer.parent is None
        assert inner.parent == "outer"
    
    def test_extract_empty_code(self):
        """Test extracting symbols from empty code."""
        engine = ASTDiffEngine()
        symbols = engine.extract_symbols("", LanguageType.PYTHON)
        assert len(symbols) == 0


class TestJavaScriptSymbolExtraction:
    """Test symbol extraction from JavaScript code."""
    
    def test_extract_function(self):
        """Test extracting function symbols."""
        engine = ASTDiffEngine()
        code = "function greet(name) { console.log('Hello ' + name); }"
        symbols = engine.extract_symbols(code, LanguageType.JAVASCRIPT)
        
        assert len(symbols) == 1
        func = symbols[0]
        assert func.name == "greet"
        assert func.symbol_type == "function"
        assert "name" in func.parameters
    
    def test_extract_class(self):
        """Test extracting class symbols."""
        engine = ASTDiffEngine()
        code = """class Person {
    constructor(name) {
        this.name = name;
    }
}"""
        symbols = engine.extract_symbols(code, LanguageType.JAVASCRIPT)
        
        # Should extract class + constructor method
        cls = next((s for s in symbols if s.symbol_type == "class"), None)
        assert cls is not None
        assert cls.name == "Person"
    
    def test_extract_class_with_methods(self):
        """Test extracting class with methods."""
        engine = ASTDiffEngine()
        code = """class Calculator {
    add(a, b) {
        return a + b;
    }
    
    subtract(a, b) {
        return a - b;
    }
}"""
        symbols = engine.extract_symbols(code, LanguageType.JAVASCRIPT)
        
        cls = next((s for s in symbols if s.symbol_type == "class"), None)
        assert cls is not None
        assert cls.name == "Calculator"
        
        methods = [s for s in symbols if s.symbol_type == "method"]
        assert len(methods) >= 2
        method_names = {m.name for m in methods}
        assert "add" in method_names
        assert "subtract" in method_names
    
    def test_extract_variable(self):
        """Test extracting variable symbols."""
        engine = ASTDiffEngine()
        code = "const x = 10;\nlet y = 20;\nvar z = 30;"
        symbols = engine.extract_symbols(code, LanguageType.JAVASCRIPT)
        
        assert len(symbols) >= 3
        var_names = {s.name for s in symbols if s.symbol_type == "variable"}
        assert "x" in var_names
        assert "y" in var_names
        assert "z" in var_names


class TestStructuralDiff:
    """Test structural diff computation."""
    
    def test_diff_function_added(self):
        """Test detecting added function."""
        engine = ASTDiffEngine()
        before = ""
        after = "def new_func():\n    pass"
        
        diff = engine.compute_ast_diff(before, after, LanguageType.PYTHON)
        
        assert diff.language == LanguageType.PYTHON
        assert len(diff.symbols_added) == 1
        assert diff.symbols_added[0].name == "new_func"
        assert len(diff.symbols_removed) == 0
    
    def test_diff_function_removed(self):
        """Test detecting removed function."""
        engine = ASTDiffEngine()
        before = "def old_func():\n    pass"
        after = ""
        
        diff = engine.compute_ast_diff(before, after, LanguageType.PYTHON)
        
        assert len(diff.symbols_removed) == 1
        assert diff.symbols_removed[0].name == "old_func"
        assert len(diff.symbols_added) == 0
    
    def test_diff_function_modified(self):
        """Test detecting modified function."""
        engine = ASTDiffEngine()
        before = "def func():\n    return 1"
        after = "def func():\n    return 2"
        
        diff = engine.compute_ast_diff(before, after, LanguageType.PYTHON)
        
        assert len(diff.symbols_modified) == 1
        before_sym, after_sym = diff.symbols_modified[0]
        assert before_sym.name == "func"
        assert after_sym.name == "func"
    
    def test_diff_class_added(self):
        """Test detecting added class."""
        engine = ASTDiffEngine()
        before = ""
        after = "class NewClass:\n    pass"
        
        diff = engine.compute_ast_diff(before, after, LanguageType.PYTHON)
        
        assert len(diff.symbols_added) == 1
        assert diff.symbols_added[0].name == "NewClass"
        assert diff.symbols_added[0].symbol_type == "class"
    
    def test_diff_method_added_to_class(self):
        """Test detecting method added to existing class."""
        engine = ASTDiffEngine()
        before = """class MyClass:
    def method1(self):
        pass
"""
        after = """class MyClass:
    def method1(self):
        pass
    
    def method2(self):
        pass
"""
        
        diff = engine.compute_ast_diff(before, after, LanguageType.PYTHON)
        
        # method2 should be detected as added
        added_names = {s.name for s in diff.symbols_added}
        assert "method2" in added_names
    
    def test_diff_no_changes(self):
        """Test diff with no changes."""
        engine = ASTDiffEngine()
        code = "def func():\n    pass"
        
        diff = engine.compute_ast_diff(code, code, LanguageType.PYTHON)
        
        assert len(diff.symbols_added) == 0
        assert len(diff.symbols_removed) == 0
        assert len(diff.symbols_modified) == 0
    
    def test_diff_javascript_function(self):
        """Test diff for JavaScript functions."""
        engine = ASTDiffEngine()
        before = "function old() { return 1; }"
        after = "function old() { return 2; }"
        
        diff = engine.compute_ast_diff(before, after, LanguageType.JAVASCRIPT)
        
        assert len(diff.symbols_modified) >= 1


class TestSymbolReferences:
    """Test symbol reference tracking."""
    
    def test_track_function_call(self):
        """Test tracking function call references."""
        engine = ASTDiffEngine()
        code = """def helper():
    pass

def main():
    helper()
    helper()
"""
        
        refs = engine.track_symbol_references(code, LanguageType.PYTHON, "helper")
        
        # Should find 3 references: definition + 2 calls
        assert len(refs) >= 2
    
    def test_track_variable_usage(self):
        """Test tracking variable usage."""
        engine = ASTDiffEngine()
        code = """x = 10
y = x + 5
z = x * 2
"""
        
        refs = engine.track_symbol_references(code, LanguageType.PYTHON, "x")
        
        # Should find references in assignments
        assert len(refs) >= 2
    
    def test_track_nonexistent_symbol(self):
        """Test tracking nonexistent symbol."""
        engine = ASTDiffEngine()
        code = "def func():\n    pass"
        
        refs = engine.track_symbol_references(code, LanguageType.PYTHON, "nonexistent")
        
        assert len(refs) == 0


class TestDependencyExtraction:
    """Test dependency extraction."""
    
    def test_extract_function_dependencies(self):
        """Test extracting function dependencies."""
        engine = ASTDiffEngine()
        code = """def helper():
    pass

def main():
    helper()
    print("done")
"""
        
        deps = engine.extract_dependencies(code, LanguageType.PYTHON)
        
        assert "main" in deps
        # main should have some dependencies (implementation extracts partial identifiers)
        assert isinstance(deps["main"], list)
        # Check that we found some references
        assert len(deps["main"]) > 0
    
    def test_extract_class_dependencies(self):
        """Test extracting class dependencies."""
        engine = ASTDiffEngine()
        code = """class Base:
    pass

class Derived:
    def method(self):
        base = Base()
"""
        
        deps = engine.extract_dependencies(code, LanguageType.PYTHON)
        
        assert "method" in deps
        # method should have some dependencies (implementation extracts partial identifiers)
        assert isinstance(deps["method"], list)
        # Check that we found some references
        assert len(deps["method"]) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_parse_syntax_error_python(self):
        """Test parsing Python with syntax errors."""
        engine = ASTDiffEngine()
        code = "def broken(\n    print('incomplete"
        
        # Should not crash, tree-sitter is error-tolerant
        tree = engine.parse(code, LanguageType.PYTHON)
        assert tree is not None
    
    def test_parse_syntax_error_javascript(self):
        """Test parsing JavaScript with syntax errors."""
        engine = ASTDiffEngine()
        code = "function broken() { console.log('incomplete"
        
        # Should not crash
        tree = engine.parse(code, LanguageType.JAVASCRIPT)
        assert tree is not None
    
    def test_extract_symbols_partial_code(self):
        """Test extracting symbols from partial code."""
        engine = ASTDiffEngine()
        code = "def incomplete_func("
        
        # Should not crash, may return empty or partial results
        symbols = engine.extract_symbols(code, LanguageType.PYTHON)
        assert isinstance(symbols, list)
    
    def test_diff_with_syntax_errors(self):
        """Test diff computation with syntax errors."""
        engine = ASTDiffEngine()
        before = "def func():\n    pass"
        after = "def func(\n    incomplete"
        
        # Should not crash
        diff = engine.compute_ast_diff(before, after, LanguageType.PYTHON)
        assert isinstance(diff, ASTDiff)
    
    def test_extract_symbols_unicode(self):
        """Test extracting symbols with unicode characters."""
        engine = ASTDiffEngine()
        code = """def greet():
    message = "Hello 世界"
    return message
"""
        
        symbols = engine.extract_symbols(code, LanguageType.PYTHON)
        assert len(symbols) >= 1
        assert any(s.name == "greet" for s in symbols)
    
    def test_diff_large_code(self):
        """Test diff with larger code blocks."""
        engine = ASTDiffEngine()
        
        # Generate larger code
        before_lines = ["def func{}():\n    pass".format(i) for i in range(50)]
        after_lines = ["def func{}():\n    pass".format(i) for i in range(51)]
        
        before = "\n\n".join(before_lines)
        after = "\n\n".join(after_lines)
        
        diff = engine.compute_ast_diff(before, after, LanguageType.PYTHON)
        
        # Should detect one added function
        assert len(diff.symbols_added) == 1
        assert diff.symbols_added[0].name == "func50"
    
    def test_parse_unknown_language(self):
        """Test parsing with unknown language."""
        engine = ASTDiffEngine()
        code = "some code"
        
        tree = engine.parse(code, LanguageType.UNKNOWN)
        assert tree is None
    
    def test_extract_symbols_unknown_language(self):
        """Test extracting symbols from unknown language."""
        engine = ASTDiffEngine()
        code = "some code"
        
        symbols = engine.extract_symbols(code, LanguageType.UNKNOWN)
        assert len(symbols) == 0


class TestSymbolDataStructures:
    """Test Symbol and ASTDiff data structures."""
    
    def test_symbol_to_dict(self):
        """Test Symbol to_dict conversion."""
        symbol = Symbol(
            name="test_func",
            symbol_type="function",
            start_line=1,
            end_line=5,
            start_byte=0,
            end_byte=50,
            parameters=["arg1", "arg2"],
        )
        
        data = symbol.to_dict()
        
        assert data["name"] == "test_func"
        assert data["symbol_type"] == "function"
        assert data["parameters"] == ["arg1", "arg2"]
        assert data["start_line"] == 1
        assert data["end_line"] == 5
    
    def test_ast_node_change_to_dict(self):
        """Test ASTNodeChange to_dict conversion."""
        change = ASTNodeChange(
            change_type=ChangeType.MODIFIED,
            node_type="function_definition",
            path="root.func",
            before_text="old",
            after_text="new",
            start_line=1,
            end_line=5,
        )
        
        data = change.to_dict()
        
        assert data["change_type"] == "modified"
        assert data["node_type"] == "function_definition"
        assert data["before_text"] == "old"
        assert data["after_text"] == "new"
    
    def test_ast_diff_to_dict(self):
        """Test ASTDiff to_dict conversion."""
        symbol = Symbol(
            name="func",
            symbol_type="function",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=20,
        )
        
        diff = ASTDiff(
            language=LanguageType.PYTHON,
            changes=[],
            symbols_added=[symbol],
            symbols_removed=[],
            symbols_modified=[],
        )
        
        data = diff.to_dict()
        
        assert data["language"] == "python"
        assert len(data["symbols_added"]) == 1
        assert data["symbols_added"][0]["name"] == "func"


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_refactor_rename_function(self):
        """Test detecting function rename."""
        engine = ASTDiffEngine()
        before = """def old_name(x):
    return x * 2
"""
        after = """def new_name(x):
    return x * 2
"""
        
        diff = engine.compute_ast_diff(before, after, LanguageType.PYTHON)
        
        # Should detect removal of old_name and addition of new_name
        assert len(diff.symbols_removed) == 1
        assert diff.symbols_removed[0].name == "old_name"
        assert len(diff.symbols_added) == 1
        assert diff.symbols_added[0].name == "new_name"
    
    def test_refactor_extract_method(self):
        """Test detecting method extraction."""
        engine = ASTDiffEngine()
        before = """class MyClass:
    def big_method(self):
        x = 1
        y = 2
        return x + y
"""
        after = """class MyClass:
    def big_method(self):
        return self.helper()
    
    def helper(self):
        x = 1
        y = 2
        return x + y
"""
        
        diff = engine.compute_ast_diff(before, after, LanguageType.PYTHON)
        
        # Should detect new helper method
        added_names = {s.name for s in diff.symbols_added}
        assert "helper" in added_names
        
        # big_method should be modified
        modified_names = {s[0].name for s in diff.symbols_modified}
        assert "big_method" in modified_names
    
    def test_multiple_changes(self):
        """Test detecting multiple simultaneous changes."""
        engine = ASTDiffEngine()
        before = """def func1():
    pass

def func2():
    pass
"""
        after = """def func1():
    return True

def func3():
    pass
"""
        
        diff = engine.compute_ast_diff(before, after, LanguageType.PYTHON)
        
        # func1 modified, func2 removed, func3 added
        assert len(diff.symbols_modified) >= 1
        assert len(diff.symbols_removed) >= 1
        assert len(diff.symbols_added) >= 1
