"""Utility functions for infinite context tests."""

import time
from typing import Any


def generate_test_embedding(text: str, dimensions: int = 384) -> list[float]:
    """
    Generate a deterministic test embedding for a given text.
    
    This is NOT a real embedding - just for testing purposes.
    Uses hash of text to generate consistent values.
    """
    import hashlib
    
    # Create deterministic hash
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Convert to floats in range [-1, 1]
    embedding = []
    for i in range(dimensions):
        byte_idx = i % len(hash_bytes)
        value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
        embedding.append(value)
    
    return embedding


def assert_memory_equal(mem1: dict[str, Any], mem2: dict[str, Any], ignore_fields: list[str] | None = None):
    """
    Assert two memory dictionaries are equal, ignoring specified fields.
    
    Args:
        mem1: First memory dict
        mem2: Second memory dict
        ignore_fields: Fields to ignore in comparison (e.g., ['id', 'created_at'])
    """
    if ignore_fields is None:
        ignore_fields = []
    
    keys1 = set(mem1.keys()) - set(ignore_fields)
    keys2 = set(mem2.keys()) - set(ignore_fields)
    
    assert keys1 == keys2, f"Memory keys don't match: {keys1} vs {keys2}"
    
    for key in keys1:
        assert mem1[key] == mem2[key], f"Memory field '{key}' doesn't match: {mem1[key]} vs {mem2[key]}"


def wait_for_condition(condition_fn, timeout: float = 5.0, interval: float = 0.1) -> bool:
    """
    Wait for a condition function to return True.
    
    Args:
        condition_fn: Function that returns bool
        timeout: Maximum time to wait in seconds
        interval: Check interval in seconds
        
    Returns:
        True if condition met, False if timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition_fn():
            return True
        time.sleep(interval)
    
    return False
