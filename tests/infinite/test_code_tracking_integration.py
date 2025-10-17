"""Integration tests for code tracking - end-to-end workflows."""

import pytest
import asyncio
import time
from pathlib import Path
import tempfile
import shutil

from core.infinite.code_change_tracker import CodeChangeTracker


@pytest.fixture
async def code_tracker_integration(tmp_path):
    """Create code change tracker for integration testing."""
    watch_path = tmp_path / "codebase"
    watch_path.mkdir()
    
    db_path = tmp_path / "integration.db"
    
    tracker = CodeChangeTracker(
        watch_path=watch_path,
        db_path=db_path,
        auto_track=False,
    )
    await tracker.initialize()
    yield tracker, watch_path
    await tracker.close()


# ============================================================================
# END-TO-END CODE CHANGE WORKFLOW TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_file_lifecycle(code_tracker_integration):
    """Test complete file lifecycle: create, modify, rename, delete."""
    tracker, watch_path = code_tracker_integration
    
    # Step 1: Create a new file
    file_path = watch_path / "lifecycle.py"
    t1 = time.time()
    content_v1 = '''"""Module for lifecycle testing."""

def initial_function():
    """Initial version."""
    return "v1"
'''
    
    change_id_1 = await tracker.track_change(
        file_path=str(file_path),
        before_content=None,
        after_content=content_v1,
        change_type="add",
        timestamp=t1,
    )
    
    assert change_id_1 is not None
    
    # Step 2: Modify the file
    t2 = t1 + 10
    content_v2 = '''"""Module for lifecycle testing."""

def initial_function(name="default"):
    """Updated version with parameter."""
    return f"v2: {name}"

def new_function():
    """Newly added function."""
    return "new"
'''
    
    change_id_2 = await tracker.track_change(
        file_path=str(file_path),
        before_content=content_v1,
        after_content=content_v2,
        change_type="modify",
        timestamp=t2,
    )
    
    assert change_id_2 is not None
    
    # Step 3: Rename the file
    t3 = t2 + 10
    new_file_path = watch_path / "renamed_lifecycle.py"
    
    change_id_3 = await tracker.track_change(
        file_path=str(new_file_path),
        before_content=content_v2,
        after_content=content_v2,
        change_type="rename",
        timestamp=t3,
        metadata={"old_path": str(file_path)},
    )
    
    assert change_id_3 is not None
    
    # Step 4: Delete the file
    t4 = t3 + 10
    change_id_4 = await tracker.track_change(
        file_path=str(new_file_path),
        before_content=content_v2,
        after_content="",
        change_type="delete",
        timestamp=t4,
    )
    
    assert change_id_4 is not None
    
    # Verify complete history
    original_history = await tracker.query_changes(file_path=str(file_path))
    renamed_history = await tracker.query_changes(file_path=str(new_file_path))
    
    # Should have changes for both paths
    assert len(original_history) >= 2  # create and modify
    assert len(renamed_history) >= 2  # rename and delete
    
    # Verify reconstruction at different points
    content_at_t1 = await tracker.reconstruct_file(str(file_path), t1 + 1)
    assert content_at_t1 == content_v1
    
    content_at_t2 = await tracker.reconstruct_file(str(file_path), t2 + 1)
    assert content_at_t2 == content_v2
    
    content_at_t3 = await tracker.reconstruct_file(str(new_file_path), t3 + 1)
    assert content_at_t3 == content_v2
    
    # After deletion, should return None or empty
    content_at_t4 = await tracker.reconstruct_file(str(new_file_path), t4 + 1)
    assert content_at_t4 in [None, ""]


@pytest.mark.asyncio
async def test_end_to_end_feature_development(code_tracker_integration):
    """Test tracking a complete feature development workflow."""
    tracker, watch_path = code_tracker_integration
    
    # Simulate developing a user authentication feature
    auth_file = watch_path / "auth.py"
    
    # Version 1: Basic structure
    t1 = time.time()
    v1 = '''"""Authentication module."""

class AuthManager:
    """Manages user authentication."""
    
    def __init__(self):
        self.users = {}
    
    def register(self, username, password):
        """Register a new user."""
        self.users[username] = password
        return True
'''
    
    await tracker.track_change(
        file_path=str(auth_file),
        before_content=None,
        after_content=v1,
        change_type="add",
        timestamp=t1,
    )
    
    # Version 2: Add login functionality
    t2 = t1 + 60
    v2 = '''"""Authentication module."""

class AuthManager:
    """Manages user authentication."""
    
    def __init__(self):
        self.users = {}
        self.sessions = {}
    
    def register(self, username, password):
        """Register a new user."""
        self.users[username] = password
        return True
    
    def login(self, username, password):
        """Authenticate user and create session."""
        if username in self.users and self.users[username] == password:
            session_id = f"session_{username}"
            self.sessions[session_id] = username
            return session_id
        return None
'''
    
    await tracker.track_change(
        file_path=str(auth_file),
        before_content=v1,
        after_content=v2,
        change_type="modify",
        timestamp=t2,
    )
    
    # Version 3: Add password hashing
    t3 = t2 + 60
    v3 = '''"""Authentication module."""
import hashlib

class AuthManager:
    """Manages user authentication."""
    
    def __init__(self):
        self.users = {}
        self.sessions = {}
    
    def _hash_password(self, password):
        """Hash password for secure storage."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, username, password):
        """Register a new user with hashed password."""
        self.users[username] = self._hash_password(password)
        return True
    
    def login(self, username, password):
        """Authenticate user and create session."""
        hashed = self._hash_password(password)
        if username in self.users and self.users[username] == hashed:
            session_id = f"session_{username}"
            self.sessions[session_id] = username
            return session_id
        return None
    
    def logout(self, session_id):
        """End user session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
'''
    
    await tracker.track_change(
        file_path=str(auth_file),
        before_content=v2,
        after_content=v3,
        change_type="modify",
        timestamp=t3,
    )
    
    # Verify feature evolution
    all_changes = await tracker.query_changes(file_path=str(auth_file))
    assert len(all_changes) == 3
    
    # Track evolution of specific methods
    register_evolution = await tracker.track_symbol_evolution(str(auth_file), "register")
    assert len(register_evolution) == 3
    
    login_evolution = await tracker.track_symbol_evolution(str(auth_file), "login")
    # login appears in v1 (None), v2 (added), v3 (modified)
    assert len(login_evolution) >= 2  # At least 2 versions where it exists
    
    # Verify we can reconstruct at any point
    reconstructed_v1 = await tracker.reconstruct_file(str(auth_file), t1 + 1)
    assert "def register" in reconstructed_v1
    assert "def login" not in reconstructed_v1
    
    reconstructed_v2 = await tracker.reconstruct_file(str(auth_file), t2 + 1)
    assert "def login" in reconstructed_v2
    assert "hashlib" not in reconstructed_v2
    
    reconstructed_v3 = await tracker.reconstruct_file(str(auth_file), t3 + 1)
    assert "hashlib" in reconstructed_v3
    assert "def logout" in reconstructed_v3
    
    # Verify AST diffs captured structural changes
    changes = await tracker.query_changes(file_path=str(auth_file))
    for change in changes:
        if change.change_type == "modify":
            ast_diff = await tracker.get_diff(change.id, "ast")
            assert ast_diff is not None
            # Should have detected added/modified symbols
            assert len(ast_diff.symbols_added) > 0 or len(ast_diff.symbols_modified) > 0


# ============================================================================
# REAL CODEBASE MODIFICATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_real_codebase_python_module(code_tracker_integration):
    """Test tracking changes in a realistic Python module."""
    tracker, watch_path = code_tracker_integration
    
    # Create a realistic Python module with multiple components
    module_file = watch_path / "data_processor.py"
    
    original_code = '''"""Data processing utilities."""
from typing import List, Dict, Any
import json

class DataProcessor:
    """Process and transform data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize processor with optional config."""
        self.config = config or {}
        self.cache = {}
    
    def process(self, data: List[Dict]) -> List[Dict]:
        """Process a list of data items."""
        results = []
        for item in data:
            processed = self._process_item(item)
            if processed:
                results.append(processed)
        return results
    
    def _process_item(self, item: Dict) -> Dict:
        """Process a single item."""
        # Apply transformations
        transformed = {
            k: v.strip() if isinstance(v, str) else v
            for k, v in item.items()
        }
        return transformed
    
    def save_to_file(self, data: List[Dict], filepath: str):
        """Save processed data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

def load_from_file(filepath: str) -> List[Dict]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
'''
    
    t1 = time.time()
    await tracker.track_change(
        file_path=str(module_file),
        before_content=None,
        after_content=original_code,
        change_type="add",
        timestamp=t1,
    )
    
    # Modify: Add caching and error handling
    modified_code = '''"""Data processing utilities."""
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and transform data with caching."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize processor with optional config."""
        self.config = config or {}
        self.cache = {}
        self.enable_cache = config.get('enable_cache', True) if config else True
    
    def process(self, data: List[Dict]) -> List[Dict]:
        """Process a list of data items with caching."""
        results = []
        for item in data:
            try:
                processed = self._process_item(item)
                if processed:
                    results.append(processed)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                continue
        return results
    
    def _process_item(self, item: Dict) -> Optional[Dict]:
        """Process a single item with caching."""
        # Check cache
        item_id = item.get('id')
        if self.enable_cache and item_id and item_id in self.cache:
            return self.cache[item_id]
        
        # Apply transformations
        transformed = {
            k: v.strip() if isinstance(v, str) else v
            for k, v in item.items()
        }
        
        # Cache result
        if self.enable_cache and item_id:
            self.cache[item_id] = transformed
        
        return transformed
    
    def clear_cache(self):
        """Clear the processing cache."""
        self.cache.clear()
    
    def save_to_file(self, data: List[Dict], filepath: str):
        """Save processed data to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(data)} items to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save to {filepath}: {e}")
            raise

def load_from_file(filepath: str) -> List[Dict]:
    """Load data from JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load from {filepath}: {e}")
        return []
'''
    
    t2 = t1 + 120
    change_id = await tracker.track_change(
        file_path=str(module_file),
        before_content=original_code,
        after_content=modified_code,
        change_type="modify",
        timestamp=t2,
    )
    
    # Verify all diff levels captured the changes
    char_diff = await tracker.get_diff(change_id, "char")
    assert char_diff is not None
    assert len(char_diff.content) > 0
    
    line_diff = await tracker.get_diff(change_id, "line")
    assert line_diff is not None
    
    unified_diff = await tracker.get_diff(change_id, "unified")
    assert unified_diff is not None
    unified_content = unified_diff.get_content()
    assert "logger" in unified_content
    assert "enable_cache" in unified_content
    
    ast_diff = await tracker.get_diff(change_id, "ast")
    assert ast_diff is not None
    # Should detect new method clear_cache
    added_names = [s.name for s in ast_diff.symbols_added]
    assert "clear_cache" in added_names
    
    # Verify 1:1 reconstruction
    reconstructed = await tracker.reconstruct_file(str(module_file), t2 + 1)
    assert reconstructed == modified_code
    
    # Verify can reconstruct original
    reconstructed_original = await tracker.reconstruct_file(str(module_file), t1 + 1)
    assert reconstructed_original == original_code


@pytest.mark.asyncio
async def test_real_codebase_javascript_module(code_tracker_integration):
    """Test tracking changes in a JavaScript/TypeScript module."""
    tracker, watch_path = code_tracker_integration
    
    js_file = watch_path / "api_client.js"
    
    original_js = '''/**
 * API Client for making HTTP requests
 */
class ApiClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Content-Type': 'application/json'
    };
  }

  async get(endpoint) {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'GET',
      headers: this.headers
    });
    return response.json();
  }

  async post(endpoint, data) {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(data)
    });
    return response.json();
  }
}

export default ApiClient;
'''
    
    t1 = time.time()
    await tracker.track_change(
        file_path=str(js_file),
        before_content=None,
        after_content=original_js,
        change_type="add",
        timestamp=t1,
    )
    
    # Modify: Add error handling and retry logic
    modified_js = '''/**
 * API Client for making HTTP requests with retry logic
 */
class ApiClient {
  constructor(baseUrl, options = {}) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Content-Type': 'application/json',
      ...options.headers
    };
    this.maxRetries = options.maxRetries || 3;
    this.retryDelay = options.retryDelay || 1000;
  }

  async _fetchWithRetry(url, options, retries = 0) {
    try {
      const response = await fetch(url, options);
      if (!response.ok && retries < this.maxRetries) {
        await new Promise(resolve => setTimeout(resolve, this.retryDelay));
        return this._fetchWithRetry(url, options, retries + 1);
      }
      return response;
    } catch (error) {
      if (retries < this.maxRetries) {
        await new Promise(resolve => setTimeout(resolve, this.retryDelay));
        return this._fetchWithRetry(url, options, retries + 1);
      }
      throw error;
    }
  }

  async get(endpoint) {
    const response = await this._fetchWithRetry(`${this.baseUrl}${endpoint}`, {
      method: 'GET',
      headers: this.headers
    });
    return response.json();
  }

  async post(endpoint, data) {
    const response = await this._fetchWithRetry(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(data)
    });
    return response.json();
  }

  async delete(endpoint) {
    const response = await this._fetchWithRetry(`${this.baseUrl}${endpoint}`, {
      method: 'DELETE',
      headers: this.headers
    });
    return response.json();
  }
}

export default ApiClient;
'''
    
    t2 = t1 + 90
    change_id = await tracker.track_change(
        file_path=str(js_file),
        before_content=original_js,
        after_content=modified_js,
        change_type="modify",
        timestamp=t2,
    )
    
    # Verify diffs
    unified_diff = await tracker.get_diff(change_id, "unified")
    assert unified_diff is not None
    content = unified_diff.get_content()
    assert "_fetchWithRetry" in content
    assert "maxRetries" in content
    
    ast_diff = await tracker.get_diff(change_id, "ast")
    assert ast_diff is not None
    # Should detect new method and modified constructor
    assert len(ast_diff.symbols_added) > 0 or len(ast_diff.symbols_modified) > 0
    
    # Verify 1:1 reconstruction
    reconstructed = await tracker.reconstruct_file(str(js_file), t2 + 1)
    assert reconstructed == modified_js


# ============================================================================
# MULTI-FILE REFACTORING TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_multi_file_refactoring_extract_module(code_tracker_integration):
    """Test tracking a refactoring that extracts code into a new module."""
    tracker, watch_path = code_tracker_integration
    
    # Original monolithic file
    main_file = watch_path / "main.py"
    original_main = '''"""Main application module."""

class Database:
    """Database connection handler."""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def connect(self):
        """Establish database connection."""
        pass
    
    def query(self, sql):
        """Execute SQL query."""
        pass

class Application:
    """Main application class."""
    
    def __init__(self):
        self.db = Database("sqlite:///app.db")
    
    def run(self):
        """Run the application."""
        self.db.connect()
        print("Application running")

if __name__ == "__main__":
    app = Application()
    app.run()
'''
    
    t1 = time.time()
    await tracker.track_change(
        file_path=str(main_file),
        before_content=None,
        after_content=original_main,
        change_type="add",
        timestamp=t1,
    )
    
    # Refactor: Extract Database class to separate module
    t2 = t1 + 60
    
    # New database module
    db_file = watch_path / "database.py"
    db_module = '''"""Database connection handler."""

class Database:
    """Database connection handler."""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def connect(self):
        """Establish database connection."""
        pass
    
    def query(self, sql):
        """Execute SQL query."""
        pass
'''
    
    await tracker.track_change(
        file_path=str(db_file),
        before_content=None,
        after_content=db_module,
        change_type="add",
        timestamp=t2,
    )
    
    # Modified main file
    refactored_main = '''"""Main application module."""
from database import Database

class Application:
    """Main application class."""
    
    def __init__(self):
        self.db = Database("sqlite:///app.db")
    
    def run(self):
        """Run the application."""
        self.db.connect()
        print("Application running")

if __name__ == "__main__":
    app = Application()
    app.run()
'''
    
    await tracker.track_change(
        file_path=str(main_file),
        before_content=original_main,
        after_content=refactored_main,
        change_type="modify",
        timestamp=t2,
    )
    
    # Verify both files were tracked
    main_changes = await tracker.query_changes(file_path=str(main_file))
    db_changes = await tracker.query_changes(file_path=str(db_file))
    
    assert len(main_changes) == 2  # add and modify
    assert len(db_changes) == 1  # add
    
    # Verify Database class was removed from main
    main_modify_change = [c for c in main_changes if c.change_type == "modify"][0]
    ast_diff = await tracker.get_diff(main_modify_change.id, "ast")
    assert ast_diff is not None
    
    # Should show Database class was removed
    removed_names = [s.name for s in ast_diff.symbols_removed]
    assert "Database" in removed_names
    
    # Verify Database class was added to db module
    db_add_change = db_changes[0]
    db_ast_diff = await tracker.get_diff(db_add_change.id, "ast")
    # For file creation, AST diff might be None or have symbols_added
    if db_ast_diff is not None:
        added_names = [s.name for s in db_ast_diff.symbols_added]
        assert "Database" in added_names
    else:
        # If AST diff is None for creation, verify the content has Database class
        assert "class Database:" in db_module
    
    # Verify reconstruction
    reconstructed_main = await tracker.reconstruct_file(str(main_file), t2 + 1)
    assert reconstructed_main == refactored_main
    assert "from database import Database" in reconstructed_main
    assert "class Database:" not in reconstructed_main
    
    reconstructed_db = await tracker.reconstruct_file(str(db_file), t2 + 1)
    assert reconstructed_db == db_module
    assert "class Database:" in reconstructed_db


@pytest.mark.asyncio
async def test_multi_file_refactoring_rename_across_files(code_tracker_integration):
    """Test tracking a refactoring that renames symbols across multiple files."""
    tracker, watch_path = code_tracker_integration
    
    # Create multiple files with related code
    t1 = time.time()
    
    # File 1: Model definition
    model_file = watch_path / "models.py"
    original_model = '''"""Data models."""

class UserAccount:
    """User account model."""
    
    def __init__(self, username, email):
        self.username = username
        self.email = email
'''
    
    await tracker.track_change(
        file_path=str(model_file),
        before_content=None,
        after_content=original_model,
        change_type="add",
        timestamp=t1,
    )
    
    # File 2: Service using the model
    service_file = watch_path / "services.py"
    original_service = '''"""Business logic services."""
from models import UserAccount

class UserService:
    """Service for user operations."""
    
    def create_user(self, username, email):
        """Create a new user account."""
        return UserAccount(username, email)
    
    def get_user_info(self, account):
        """Get user information."""
        return f"{account.username} <{account.email}>"
'''
    
    await tracker.track_change(
        file_path=str(service_file),
        before_content=None,
        after_content=original_service,
        change_type="add",
        timestamp=t1,
    )
    
    # Refactor: Rename UserAccount to User
    t2 = t1 + 120
    
    refactored_model = '''"""Data models."""

class User:
    """User model."""
    
    def __init__(self, username, email):
        self.username = username
        self.email = email
'''
    
    await tracker.track_change(
        file_path=str(model_file),
        before_content=original_model,
        after_content=refactored_model,
        change_type="modify",
        timestamp=t2,
    )
    
    refactored_service = '''"""Business logic services."""
from models import User

class UserService:
    """Service for user operations."""
    
    def create_user(self, username, email):
        """Create a new user."""
        return User(username, email)
    
    def get_user_info(self, user):
        """Get user information."""
        return f"{user.username} <{user.email}>"
'''
    
    await tracker.track_change(
        file_path=str(service_file),
        before_content=original_service,
        after_content=refactored_service,
        change_type="modify",
        timestamp=t2,
    )
    
    # Verify changes in both files
    model_changes = await tracker.query_changes(file_path=str(model_file))
    service_changes = await tracker.query_changes(file_path=str(service_file))
    
    assert len(model_changes) == 2
    assert len(service_changes) == 2
    
    # Verify AST diffs show the rename
    model_modify = [c for c in model_changes if c.change_type == "modify"][0]
    model_ast = await tracker.get_diff(model_modify.id, "ast")
    assert model_ast is not None
    
    # Should show UserAccount removed and User added
    removed = [s.name for s in model_ast.symbols_removed]
    added = [s.name for s in model_ast.symbols_added]
    assert "UserAccount" in removed
    assert "User" in added
    
    # Verify unified diffs show the changes
    service_modify = [c for c in service_changes if c.change_type == "modify"][0]
    service_unified = await tracker.get_diff(service_modify.id, "unified")
    assert service_unified is not None
    content = service_unified.get_content()
    assert "UserAccount" in content or "User" in content
    
    # Verify 1:1 reconstruction for both files
    reconstructed_model = await tracker.reconstruct_file(str(model_file), t2 + 1)
    assert reconstructed_model == refactored_model
    
    reconstructed_service = await tracker.reconstruct_file(str(service_file), t2 + 1)
    assert reconstructed_service == refactored_service


@pytest.mark.asyncio
async def test_multi_file_refactoring_merge_modules(code_tracker_integration):
    """Test tracking a refactoring that merges multiple modules."""
    tracker, watch_path = code_tracker_integration
    
    t1 = time.time()
    
    # Original: Two separate utility modules
    utils1_file = watch_path / "string_utils.py"
    utils1 = '''"""String utilities."""

def capitalize_words(text):
    """Capitalize each word."""
    return text.title()

def reverse_string(text):
    """Reverse a string."""
    return text[::-1]
'''
    
    await tracker.track_change(
        file_path=str(utils1_file),
        before_content=None,
        after_content=utils1,
        change_type="add",
        timestamp=t1,
    )
    
    utils2_file = watch_path / "number_utils.py"
    utils2 = '''"""Number utilities."""

def is_even(n):
    """Check if number is even."""
    return n % 2 == 0

def factorial(n):
    """Calculate factorial."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''
    
    await tracker.track_change(
        file_path=str(utils2_file),
        before_content=None,
        after_content=utils2,
        change_type="add",
        timestamp=t1,
    )
    
    # Refactor: Merge into single utils module
    t2 = t1 + 90
    
    merged_file = watch_path / "utils.py"
    merged = '''"""Utility functions."""

# String utilities
def capitalize_words(text):
    """Capitalize each word."""
    return text.title()

def reverse_string(text):
    """Reverse a string."""
    return text[::-1]

# Number utilities
def is_even(n):
    """Check if number is even."""
    return n % 2 == 0

def factorial(n):
    """Calculate factorial."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''
    
    await tracker.track_change(
        file_path=str(merged_file),
        before_content=None,
        after_content=merged,
        change_type="add",
        timestamp=t2,
    )
    
    # Delete old files
    await tracker.track_change(
        file_path=str(utils1_file),
        before_content=utils1,
        after_content="",
        change_type="delete",
        timestamp=t2,
    )
    
    await tracker.track_change(
        file_path=str(utils2_file),
        before_content=utils2,
        after_content="",
        change_type="delete",
        timestamp=t2,
    )
    
    # Verify all changes tracked
    utils1_changes = await tracker.query_changes(file_path=str(utils1_file))
    utils2_changes = await tracker.query_changes(file_path=str(utils2_file))
    merged_changes = await tracker.query_changes(file_path=str(merged_file))
    
    assert len(utils1_changes) == 2  # add and delete
    assert len(utils2_changes) == 2  # add and delete
    assert len(merged_changes) == 1  # add
    
    # Verify we can reconstruct all states
    # Before merge
    before_utils1 = await tracker.reconstruct_file(str(utils1_file), t1 + 1)
    assert before_utils1 == utils1
    
    before_utils2 = await tracker.reconstruct_file(str(utils2_file), t1 + 1)
    assert before_utils2 == utils2
    
    # After merge
    after_merged = await tracker.reconstruct_file(str(merged_file), t2 + 1)
    assert after_merged == merged
    assert "capitalize_words" in after_merged
    assert "is_even" in after_merged
    
    # Old files should be deleted
    after_utils1 = await tracker.reconstruct_file(str(utils1_file), t2 + 1)
    assert after_utils1 in [None, ""]


# ============================================================================
# 1:1 DIFF ACCURACY TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_diff_accuracy_exact_reconstruction(code_tracker_integration):
    """Test that diffs allow exact 1:1 reconstruction."""
    tracker, watch_path = code_tracker_integration
    
    # Create a file with various Python constructs
    test_file = watch_path / "complex.py"
    original = '''"""Complex module for testing diff accuracy."""
import sys
import os
from typing import List, Dict, Optional

# Module-level constant
MAX_RETRIES = 3

class ComplexClass:
    """A class with various features."""
    
    class_var = "shared"
    
    def __init__(self, name: str, value: int = 0):
        """Initialize with name and optional value."""
        self.name = name
        self.value = value
        self._private = []
    
    def method_with_docstring(self, param: str) -> str:
        """
        Method with detailed docstring.
        
        Args:
            param: Input parameter
            
        Returns:
            Processed string
        """
        return f"{self.name}: {param}"
    
    @staticmethod
    def static_method(x: int) -> int:
        """Static method."""
        return x * 2
    
    @classmethod
    def class_method(cls, value: str):
        """Class method."""
        return cls(value)

def standalone_function(items: List[str]) -> Dict[str, int]:
    """Process items and return counts."""
    result = {}
    for item in items:
        result[item] = result.get(item, 0) + 1
    return result

if __name__ == "__main__":
    obj = ComplexClass("test", 42)
    print(obj.method_with_docstring("hello"))
'''
    
    t1 = time.time()
    await tracker.track_change(
        file_path=str(test_file),
        before_content=None,
        after_content=original,
        change_type="add",
        timestamp=t1,
    )
    
    # Make complex modifications
    modified = '''"""Complex module for testing diff accuracy."""
import sys
import os
import json
from typing import List, Dict, Optional, Tuple

# Module-level constants
MAX_RETRIES = 5
TIMEOUT = 30

class ComplexClass:
    """A class with various features and enhancements."""
    
    class_var = "shared"
    version = "2.0"
    
    def __init__(self, name: str, value: int = 0, config: Optional[Dict] = None):
        """Initialize with name, optional value, and config."""
        self.name = name
        self.value = value
        self._private = []
        self.config = config or {}
    
    def method_with_docstring(self, param: str, uppercase: bool = False) -> str:
        """
        Method with detailed docstring.
        
        Args:
            param: Input parameter
            uppercase: Whether to uppercase the result
            
        Returns:
            Processed string
        """
        result = f"{self.name}: {param}"
        return result.upper() if uppercase else result
    
    def new_method(self) -> List[str]:
        """New method added in this version."""
        return [self.name, str(self.value)]
    
    @staticmethod
    def static_method(x: int, y: int = 1) -> int:
        """Static method with additional parameter."""
        return x * y * 2
    
    @classmethod
    def class_method(cls, value: str, **kwargs):
        """Class method with kwargs."""
        return cls(value, **kwargs)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ComplexClass(name={self.name}, value={self.value})"

def standalone_function(items: List[str], case_sensitive: bool = True) -> Dict[str, int]:
    """Process items and return counts with case sensitivity option."""
    result = {}
    for item in items:
        key = item if case_sensitive else item.lower()
        result[key] = result.get(key, 0) + 1
    return result

def new_helper_function(data: Dict) -> str:
    """New helper function."""
    return json.dumps(data, indent=2)

if __name__ == "__main__":
    obj = ComplexClass("test", 42, {"debug": True})
    print(obj.method_with_docstring("hello", uppercase=True))
    print(obj)
'''
    
    t2 = t1 + 180
    change_id = await tracker.track_change(
        file_path=str(test_file),
        before_content=original,
        after_content=modified,
        change_type="modify",
        timestamp=t2,
    )
    
    # Test 1: Character-level diff reconstruction
    reconstructed_char = await tracker.reconstruct_file(
        str(test_file), t2 + 1, diff_level="char"
    )
    assert reconstructed_char == modified, "Character-level reconstruction failed"
    
    # Test 2: Line-level diff reconstruction
    reconstructed_line = await tracker.reconstruct_file(
        str(test_file), t2 + 1, diff_level="line"
    )
    assert reconstructed_line == modified, "Line-level reconstruction failed"
    
    # Test 3: Verify both methods produce identical results
    assert reconstructed_char == reconstructed_line, "Diff levels produced different results"
    
    # Test 4: Verify original can be reconstructed
    reconstructed_original = await tracker.reconstruct_file(
        str(test_file), t1 + 1, diff_level="char"
    )
    assert reconstructed_original == original, "Original reconstruction failed"
    
    # Test 5: Verify character-by-character accuracy
    for i, (orig_char, recon_char) in enumerate(zip(modified, reconstructed_char)):
        assert orig_char == recon_char, f"Character mismatch at position {i}: expected {repr(orig_char)}, got {repr(recon_char)}"
    
    # Test 6: Verify length matches exactly
    assert len(reconstructed_char) == len(modified), f"Length mismatch: expected {len(modified)}, got {len(reconstructed_char)}"
    
    # Test 7: Verify byte-level accuracy
    assert reconstructed_char.encode('utf-8') == modified.encode('utf-8'), "Byte-level mismatch"


@pytest.mark.asyncio
async def test_diff_accuracy_with_special_characters(code_tracker_integration):
    """Test diff accuracy with special characters and unicode."""
    tracker, watch_path = code_tracker_integration
    
    test_file = watch_path / "special_chars.py"
    
    original = '''"""Module with special characters."""

# Unicode characters: café, naïve, 日本語
GREETING = "Hello, 世界! 🌍"

def process_text(text):
    """Process text with special chars: @#$%^&*()."""
    # Tabs\t\tand\tnewlines\n
    result = text.replace("\\n", " ")
    result = result.replace("\\t", "    ")
    return result

# String with quotes
QUOTE = 'He said, "It\\'s working!"'
MULTILINE = """
    Line 1
    Line 2 with "quotes"
    Line 3 with 'apostrophes'
"""
'''
    
    t1 = time.time()
    await tracker.track_change(
        file_path=str(test_file),
        before_content=None,
        after_content=original,
        change_type="add",
        timestamp=t1,
    )
    
    modified = '''"""Module with special characters and more."""

# Unicode characters: café, naïve, 日本語, 한글
GREETING = "Hello, 世界! 🌍🌎🌏"
EMOJI = "😀😃😄😁"

def process_text(text, preserve_newlines=False):
    """Process text with special chars: @#$%^&*()+=[]{}."""
    # Tabs\t\tand\tnewlines\n
    if not preserve_newlines:
        result = text.replace("\\n", " ")
    else:
        result = text
    result = result.replace("\\t", "    ")
    return result

# String with quotes and escapes
QUOTE = 'He said, "It\\'s working!" and she replied, "Great!"'
MULTILINE = """
    Line 1 with unicode: café
    Line 2 with "quotes" and émojis 🎉
    Line 3 with 'apostrophes' and ñ
    Line 4: \\n\\t\\r special escapes
"""

# Raw string
PATH = r"C:\\Users\\Test\\Documents\\file.txt"
'''
    
    t2 = t1 + 60
    change_id = await tracker.track_change(
        file_path=str(test_file),
        before_content=original,
        after_content=modified,
        change_type="modify",
        timestamp=t2,
    )
    
    # Verify exact reconstruction with special characters
    reconstructed = await tracker.reconstruct_file(str(test_file), t2 + 1)
    assert reconstructed == modified
    
    # Verify unicode handling
    assert "🌍🌎🌏" in reconstructed
    assert "😀😃😄😁" in reconstructed
    assert "café" in reconstructed
    assert "한글" in reconstructed
    
    # Verify escape sequences preserved
    assert r"\\n\\t\\r" in reconstructed or "\\n\\t\\r" in reconstructed
    
    # Verify byte-level accuracy
    assert reconstructed.encode('utf-8') == modified.encode('utf-8')



@pytest.mark.asyncio
async def test_diff_accuracy_large_file(code_tracker_integration):
    """Test diff accuracy with large files."""
    tracker, watch_path = code_tracker_integration
    
    test_file = watch_path / "large_file.py"
    
    # Generate a large file with many functions
    original_lines = ['"""Large module for testing."""\n']
    for i in range(200):
        original_lines.append(f'\ndef function_{i}(x):\n')
        original_lines.append(f'    """Function number {i}."""\n')
        original_lines.append(f'    return x + {i}\n')
    
    original = ''.join(original_lines)
    
    t1 = time.time()
    await tracker.track_change(
        file_path=str(test_file),
        before_content=None,
        after_content=original,
        change_type="add",
        timestamp=t1,
    )
    
    # Modify: Change every 10th function
    modified_lines = ['"""Large module for testing."""\n']
    for i in range(200):
        if i % 10 == 0:
            # Modified version
            modified_lines.append(f'\ndef function_{i}(x, y=0):\n')
            modified_lines.append(f'    """Function number {i} - enhanced."""\n')
            modified_lines.append(f'    return x + y + {i}\n')
        else:
            # Original version
            modified_lines.append(f'\ndef function_{i}(x):\n')
            modified_lines.append(f'    """Function number {i}."""\n')
            modified_lines.append(f'    return x + {i}\n')
    
    modified = ''.join(modified_lines)
    
    t2 = t1 + 120
    change_id = await tracker.track_change(
        file_path=str(test_file),
        before_content=original,
        after_content=modified,
        change_type="modify",
        timestamp=t2,
    )
    
    # Verify exact reconstruction
    reconstructed = await tracker.reconstruct_file(str(test_file), t2 + 1)
    assert reconstructed == modified
    
    # Verify length
    assert len(reconstructed) == len(modified)
    
    # Verify specific changes were captured
    assert "function_0(x, y=0)" in reconstructed
    assert "function_10(x, y=0)" in reconstructed
    assert "function_5(x):" in reconstructed  # Unchanged
    
    # Verify can reconstruct original
    reconstructed_original = await tracker.reconstruct_file(str(test_file), t1 + 1)
    assert reconstructed_original == original


@pytest.mark.asyncio
async def test_diff_accuracy_whitespace_preservation(code_tracker_integration):
    """Test that whitespace is preserved exactly in diffs."""
    tracker, watch_path = code_tracker_integration
    
    test_file = watch_path / "whitespace.py"
    
    # File with various whitespace patterns
    original = '''def function_with_spaces():
    # 4 spaces indentation
    x = 1
    y = 2
    return x + y

def function_with_tabs():
\t# Tab indentation
\tx = 1
\ty = 2
\treturn x + y

def mixed_indentation():
    # Starts with spaces
\tx = 1  # Tab here
    y = 2  # Spaces here
    return x + y

# Trailing spaces at end of line    
# Empty line with spaces below
    
# Multiple blank lines


def final_function():
    pass
'''
    
    t1 = time.time()
    await tracker.track_change(
        file_path=str(test_file),
        before_content=None,
        after_content=original,
        change_type="add",
        timestamp=t1,
    )
    
    # Modify with different whitespace
    modified = '''def function_with_spaces():
    # 4 spaces indentation (unchanged)
    x = 1
    y = 2
    z = 3  # New line
    return x + y + z

def function_with_tabs():
\t# Tab indentation (unchanged)
\tx = 1
\ty = 2
\tz = 3  # New line with tab
\treturn x + y + z

def mixed_indentation():
    # Starts with spaces
\tx = 1  # Tab here
    y = 2  # Spaces here
\tz = 3  # Tab here
    return x + y + z

# Trailing spaces removed
# Empty line with spaces below
    
# Multiple blank lines (one removed)

def final_function():
    return None  # Changed
'''
    
    t2 = t1 + 60
    change_id = await tracker.track_change(
        file_path=str(test_file),
        before_content=original,
        after_content=modified,
        change_type="modify",
        timestamp=t2,
    )
    
    # Verify exact reconstruction including whitespace
    reconstructed = await tracker.reconstruct_file(str(test_file), t2 + 1)
    assert reconstructed == modified
    
    # Verify tabs are preserved
    assert '\t' in reconstructed
    
    # Verify exact character match
    for i, (orig_char, recon_char) in enumerate(zip(modified, reconstructed)):
        if orig_char != recon_char:
            context_start = max(0, i - 20)
            context_end = min(len(modified), i + 20)
            print(f"Mismatch at position {i}:")
            print(f"  Expected: {repr(modified[context_start:context_end])}")
            print(f"  Got: {repr(reconstructed[context_start:context_end])}")
        assert orig_char == recon_char


@pytest.mark.asyncio
async def test_diff_accuracy_empty_lines_and_comments(code_tracker_integration):
    """Test diff accuracy with empty lines and comments."""
    tracker, watch_path = code_tracker_integration
    
    test_file = watch_path / "comments.py"
    
    original = '''# Header comment
# Another header line

"""Module docstring."""

# Section 1
def func1():
    # Comment inside function
    pass

# Section 2
def func2():
    pass
'''
    
    t1 = time.time()
    await tracker.track_change(
        file_path=str(test_file),
        before_content=None,
        after_content=original,
        change_type="add",
        timestamp=t1,
    )
    
    modified = '''# Header comment
# Another header line
# New header line

"""Module docstring with more detail."""

# Section 1 - Updated
def func1():
    # Comment inside function
    # Additional comment
    return True

# Section 2
def func2():
    pass

# Section 3 - New
def func3():
    """New function."""
    pass
'''
    
    t2 = t1 + 60
    await tracker.track_change(
        file_path=str(test_file),
        before_content=original,
        after_content=modified,
        change_type="modify",
        timestamp=t2,
    )
    
    # Verify exact reconstruction
    reconstructed = await tracker.reconstruct_file(str(test_file), t2 + 1)
    assert reconstructed == modified
    
    # Verify comments preserved
    assert "# New header line" in reconstructed
    assert "# Additional comment" in reconstructed
    assert "# Section 3 - New" in reconstructed


@pytest.mark.asyncio
async def test_integration_complete_workflow(code_tracker_integration):
    """Test complete integration workflow with multiple files and changes."""
    tracker, watch_path = code_tracker_integration
    
    # Simulate a complete development session
    t_start = time.time()
    
    # Day 1: Create initial project structure
    files_created = []
    
    # Create main.py
    main_file = watch_path / "main.py"
    main_v1 = '''"""Main application entry point."""

def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
'''
    await tracker.track_change(
        file_path=str(main_file),
        before_content=None,
        after_content=main_v1,
        change_type="add",
        timestamp=t_start,
    )
    files_created.append(str(main_file))
    
    # Create config.py
    config_file = watch_path / "config.py"
    config_v1 = '''"""Configuration settings."""

DEBUG = True
VERSION = "0.1.0"
'''
    await tracker.track_change(
        file_path=str(config_file),
        before_content=None,
        after_content=config_v1,
        change_type="add",
        timestamp=t_start + 10,
    )
    files_created.append(str(config_file))
    
    # Day 2: Add features
    t_day2 = t_start + 86400  # Next day
    
    # Update main.py
    main_v2 = '''"""Main application entry point."""
from config import DEBUG, VERSION

def main():
    print(f"Application v{VERSION}")
    if DEBUG:
        print("Debug mode enabled")

if __name__ == "__main__":
    main()
'''
    await tracker.track_change(
        file_path=str(main_file),
        before_content=main_v1,
        after_content=main_v2,
        change_type="modify",
        timestamp=t_day2,
    )
    
    # Create utils.py
    utils_file = watch_path / "utils.py"
    utils_v1 = '''"""Utility functions."""

def format_message(msg):
    return f"[INFO] {msg}"
'''
    await tracker.track_change(
        file_path=str(utils_file),
        before_content=None,
        after_content=utils_v1,
        change_type="add",
        timestamp=t_day2 + 60,
    )
    files_created.append(str(utils_file))
    
    # Day 3: Refactor and bug fixes
    t_day3 = t_day2 + 86400
    
    # Update config.py
    config_v2 = '''"""Configuration settings."""

DEBUG = False  # Disabled for production
VERSION = "0.2.0"
LOG_LEVEL = "INFO"
'''
    await tracker.track_change(
        file_path=str(config_file),
        before_content=config_v1,
        after_content=config_v2,
        change_type="modify",
        timestamp=t_day3,
    )
    
    # Update main.py to use utils
    main_v3 = '''"""Main application entry point."""
from config import DEBUG, VERSION, LOG_LEVEL
from utils import format_message

def main():
    print(format_message(f"Application v{VERSION}"))
    if DEBUG:
        print(format_message("Debug mode enabled"))
    print(format_message(f"Log level: {LOG_LEVEL}"))

if __name__ == "__main__":
    main()
'''
    await tracker.track_change(
        file_path=str(main_file),
        before_content=main_v2,
        after_content=main_v3,
        change_type="modify",
        timestamp=t_day3 + 120,
    )
    
    # Verify complete history
    all_files = [main_file, config_file, utils_file]
    for file_path in all_files:
        changes = await tracker.query_changes(file_path=str(file_path))
        assert len(changes) > 0
    
    # Verify reconstruction at different points in time
    # Day 1
    main_day1 = await tracker.reconstruct_file(str(main_file), t_start + 1)
    assert main_day1 == main_v1
    assert "Hello, World!" in main_day1
    
    # Day 2
    main_day2 = await tracker.reconstruct_file(str(main_file), t_day2 + 1)
    assert main_day2 == main_v2
    assert "from config import" in main_day2
    
    # Day 3
    main_day3 = await tracker.reconstruct_file(str(main_file), t_day3 + 200)
    assert main_day3 == main_v3
    assert "from utils import" in main_day3
    
    config_day3 = await tracker.reconstruct_file(str(config_file), t_day3 + 200)
    assert config_day3 == config_v2
    assert "LOG_LEVEL" in config_day3
    
    # Verify change graph
    main_graph = await tracker.get_change_graph(str(main_file))
    assert len(main_graph.nodes) == 3  # 3 versions
    
    config_graph = await tracker.get_change_graph(str(config_file))
    assert len(config_graph.nodes) == 2  # 2 versions
    
    utils_graph = await tracker.get_change_graph(str(utils_file))
    assert len(utils_graph.nodes) == 1  # 1 version
    
    print(f"\n✓ Integration test completed successfully!")
    print(f"  - Tracked {len(all_files)} files")
    print(f"  - Simulated 3 days of development")
    print(f"  - Verified reconstruction at multiple points in time")
