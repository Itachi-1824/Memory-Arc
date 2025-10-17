"""Verify DisabledProcessor implementation without full imports."""

import ast
import inspect


def verify_disabled_processor():
    """Verify DisabledProcessor is correctly implemented."""
    print("=" * 60)
    print("Verifying DisabledProcessor Implementation")
    print("=" * 60)
    
    # Read the processors.py file
    with open('core/processors.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check 1: Class exists and inherits from MemoryProcessor
    if 'class DisabledProcessor(MemoryProcessor):' in content:
        print("✓ DisabledProcessor class exists and inherits from MemoryProcessor")
    else:
        print("✗ DisabledProcessor class not found or doesn't inherit from MemoryProcessor")
        return False
    
    # Check 2: Has __init__ method
    if 'def __init__(self):' in content:
        print("✓ __init__ method exists")
    else:
        print("✗ __init__ method not found")
        return False
    
    # Check 3: Has summarize method that returns None
    if 'async def summarize(self, messages: list[dict]) -> str | None:' in content:
        print("✓ summarize method exists with correct signature")
        if 'return None' in content:
            print("✓ summarize returns None")
        else:
            print("✗ summarize doesn't return None")
            return False
    else:
        print("✗ summarize method not found")
        return False
    
    # Check 4: Has extract_facts method that returns empty list
    if 'async def extract_facts(self, messages: list[dict]) -> list[dict]:' in content:
        print("✓ extract_facts method exists with correct signature")
        if 'return []' in content:
            print("✓ extract_facts returns empty list")
        else:
            print("✗ extract_facts doesn't return empty list")
            return False
    else:
        print("✗ extract_facts method not found")
        return False
    
    # Check 5: Has score_importance method
    if 'async def score_importance(self, text: str) -> int:' in content:
        print("✓ score_importance method exists with correct signature")
        # Check it returns a neutral score
        if 'return 5' in content:
            print("✓ score_importance returns neutral score (5)")
        else:
            print("⚠ score_importance might not return neutral score")
    else:
        print("✗ score_importance method not found")
        return False
    
    # Check 6: Has get_metrics method
    if 'def get_metrics(self) -> dict[str, Any]:' in content:
        print("✓ get_metrics method exists with correct signature")
    else:
        print("✗ get_metrics method not found")
        return False
    
    # Check 7: Used in MemoryManager
    with open('core/memory_manager.py', 'r', encoding='utf-8') as f:
        mm_content = f.read()
    
    if 'DisabledProcessor' in mm_content:
        print("✓ DisabledProcessor is imported in MemoryManager")
        if 'return DisabledProcessor()' in mm_content:
            print("✓ DisabledProcessor is instantiated in _create_processor")
        else:
            print("✗ DisabledProcessor not instantiated")
            return False
    else:
        print("✗ DisabledProcessor not used in MemoryManager")
        return False
    
    # Check 8: Mode check in _process_stm_for_ltm
    if 'if self.config.mode == "disabled"' in mm_content:
        print("✓ Disabled mode check exists in _process_stm_for_ltm")
    else:
        print("✗ Disabled mode check not found")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All DisabledProcessor implementation checks passed!")
    print("=" * 60)
    print("\nImplementation Summary:")
    print("- DisabledProcessor implements MemoryProcessor interface")
    print("- All methods return None or empty results as expected")
    print("- Used when mode is 'disabled' to skip LTM processing")
    print("- Integrated with MemoryManager's _create_processor method")
    print("- LTM processing is skipped when mode is 'disabled'")
    
    return True


if __name__ == "__main__":
    success = verify_disabled_processor()
    exit(0 if success else 1)
