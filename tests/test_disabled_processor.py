"""Test DisabledProcessor implementation."""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from memory_system.core.processors import DisabledProcessor


async def test_disabled_processor():
    """Test that DisabledProcessor returns None/empty results."""
    processor = DisabledProcessor()
    
    # Test data
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"}
    ]
    
    # Test summarize - should return None
    summary = await processor.summarize(messages)
    assert summary is None, f"Expected None, got {summary}"
    print("✓ summarize() returns None")
    
    # Test extract_facts - should return empty list
    facts = await processor.extract_facts(messages)
    assert facts == [], f"Expected empty list, got {facts}"
    print("✓ extract_facts() returns empty list")
    
    # Test score_importance - should return neutral score (5)
    score = await processor.score_importance("This is important text")
    assert score == 5, f"Expected 5, got {score}"
    print("✓ score_importance() returns neutral score (5)")
    
    # Test get_metrics - should return disabled mode info
    metrics = processor.get_metrics()
    assert "mode" in metrics, "Metrics should contain 'mode' key"
    assert metrics["mode"] == "disabled", f"Expected mode='disabled', got {metrics['mode']}"
    print("✓ get_metrics() returns disabled mode info")
    
    print("\n✅ All DisabledProcessor tests passed!")


if __name__ == "__main__":
    asyncio.run(test_disabled_processor())
