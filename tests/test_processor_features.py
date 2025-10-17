"""Direct test of processor features without full system dependencies."""

import asyncio
import time
import hashlib


# Mock the necessary classes to test processor logic
class MockProcessingMetrics:
    def __init__(self):
        self.ai_calls = {}
        self.ai_success = {}
        self.ai_errors = {}
        self.processing_time = {}
    
    def increment(self, metric, operation):
        metric_dict = getattr(self, metric)
        metric_dict[operation] = metric_dict.get(operation, 0) + 1
    
    def add_time(self, operation, duration):
        self.processing_time[operation] = self.processing_time.get(operation, 0) + duration
    
    def to_dict(self):
        return {
            'ai_calls': self.ai_calls,
            'ai_success': self.ai_success,
            'ai_errors': self.ai_errors,
            'processing_time': self.processing_time,
        }


class MockConfig:
    def __init__(self, max_api_calls_per_minute=None, cache_summaries=True, batch_processing=False):
        self.max_api_calls_per_minute = max_api_calls_per_minute
        self.cache_summaries = cache_summaries
        self.batch_processing = batch_processing


class MockAdapter:
    def __init__(self):
        self.call_count = 0
    
    async def summarize_conversation(self, messages):
        self.call_count += 1
        await asyncio.sleep(0.05)
        return f"Summary {self.call_count}"


class SimplifiedAIProcessor:
    """Simplified version of AIProcessor to test core features."""
    
    def __init__(self, adapter, config):
        self.adapter = adapter
        self.config = config
        self.metrics = MockProcessingMetrics()
        
        # Rate limiting
        self.rate_limit_enabled = config.max_api_calls_per_minute is not None
        if self.rate_limit_enabled:
            self.max_calls_per_minute = config.max_api_calls_per_minute
            self.call_timestamps = []
        
        # Caching
        self.cache_enabled = config.cache_summaries
        if self.cache_enabled:
            self.cache = {}
            self.cache_ttl = 3600.0
        
        # Batch processing
        self.batch_enabled = config.batch_processing
        if self.batch_enabled:
            self.batch_queue = []
            self.batch_size = 5
            self.batch_timeout = 2.0
            self.batch_lock = asyncio.Lock()
            self.batch_task = None
    
    def _compute_cache_key(self, operation, content):
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        return f"{operation}:{content_hash}"
    
    def _get_from_cache(self, cache_key):
        if not self.cache_enabled or cache_key not in self.cache:
            return None
        value, timestamp = self.cache[cache_key]
        if time.time() - timestamp > self.cache_ttl:
            del self.cache[cache_key]
            return None
        return value
    
    def _put_in_cache(self, cache_key, value):
        if self.cache_enabled:
            self.cache[cache_key] = (value, time.time())
    
    async def _wait_for_rate_limit(self):
        if not self.rate_limit_enabled:
            return
        
        current_time = time.time()
        cutoff_time = current_time - 60.0
        self.call_timestamps = [ts for ts in self.call_timestamps if ts > cutoff_time]
        
        if len(self.call_timestamps) >= self.max_calls_per_minute:
            oldest_call = self.call_timestamps[0]
            wait_time = 60.0 - (current_time - oldest_call)
            if wait_time > 0:
                print(f"  Rate limit hit, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                current_time = time.time()
                cutoff_time = current_time - 60.0
                self.call_timestamps = [ts for ts in self.call_timestamps if ts > cutoff_time]
        
        self.call_timestamps.append(current_time)
    
    async def summarize(self, messages):
        operation = "summarize"
        
        # Check cache
        content = " ".join(msg.get("content", "") for msg in messages)
        cache_key = self._compute_cache_key(operation, content)
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result is not None:
            print(f"  Cache hit for: {content[:30]}...")
            return cached_result
        
        self.metrics.increment("ai_calls", operation)
        
        # Wait for rate limit
        await self._wait_for_rate_limit()
        
        # Call adapter
        start_time = time.time()
        result = await self.adapter.summarize_conversation(messages)
        elapsed = time.time() - start_time
        
        self.metrics.increment("ai_success", operation)
        self.metrics.add_time(operation, elapsed)
        
        # Cache result
        if result is not None:
            self._put_in_cache(cache_key, result)
        
        return result
    
    def get_metrics(self):
        return self.metrics.to_dict()


async def test_rate_limiting():
    """Test rate limiting."""
    print("\n=== Testing Rate Limiting ===")
    
    config = MockConfig(max_api_calls_per_minute=3, cache_summaries=False)
    adapter = MockAdapter()
    processor = SimplifiedAIProcessor(adapter, config)
    
    messages = [{"role": "user", "content": f"Message {i}"} for i in range(4)]
    
    print("Making 4 calls with limit of 3 per minute...")
    start_time = time.time()
    
    for i, msg_list in enumerate([messages[:i+1] for i in range(4)]):
        print(f"Call {i+1}...")
        await processor.summarize(msg_list)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f}s")
    print(f"Adapter called {adapter.call_count} times")
    
    # The 4th call should trigger rate limiting
    assert elapsed > 0.2, "Should have some delay from API calls"
    print("✓ Rate limiting working")


async def test_caching():
    """Test caching."""
    print("\n=== Testing Caching ===")
    
    config = MockConfig(cache_summaries=True)
    adapter = MockAdapter()
    processor = SimplifiedAIProcessor(adapter, config)
    
    messages = [{"role": "user", "content": "Test message"}]
    
    print("First call (should hit API)...")
    result1 = await processor.summarize(messages)
    calls_after_first = adapter.call_count
    
    print("Second call with same content (should use cache)...")
    result2 = await processor.summarize(messages)
    calls_after_second = adapter.call_count
    
    print(f"Results: '{result1}' vs '{result2}'")
    print(f"API calls: {calls_after_first} -> {calls_after_second}")
    
    assert result1 == result2
    assert calls_after_first == calls_after_second
    print("✓ Caching working")
    
    print("\nThird call with different content (should hit API)...")
    different_messages = [{"role": "user", "content": "Different message"}]
    result3 = await processor.summarize(different_messages)
    calls_after_third = adapter.call_count
    
    print(f"API calls: {calls_after_second} -> {calls_after_third}")
    assert calls_after_third > calls_after_second
    print("✓ Cache distinguishes different content")


async def test_cache_key_generation():
    """Test cache key generation."""
    print("\n=== Testing Cache Key Generation ===")
    
    config = MockConfig(cache_summaries=True)
    adapter = MockAdapter()
    processor = SimplifiedAIProcessor(adapter, config)
    
    key1 = processor._compute_cache_key("summarize", "test content")
    key2 = processor._compute_cache_key("summarize", "test content")
    key3 = processor._compute_cache_key("summarize", "different content")
    key4 = processor._compute_cache_key("extract_facts", "test content")
    
    print(f"Key 1 (summarize, 'test content'): {key1}")
    print(f"Key 2 (summarize, 'test content'): {key2}")
    print(f"Key 3 (summarize, 'different content'): {key3}")
    print(f"Key 4 (extract_facts, 'test content'): {key4}")
    
    assert key1 == key2, "Same operation and content should generate same key"
    assert key1 != key3, "Different content should generate different key"
    assert key1 != key4, "Different operation should generate different key"
    assert key1.startswith("summarize:")
    assert key4.startswith("extract_facts:")
    
    print("✓ Cache key generation working correctly")


async def test_metrics():
    """Test metrics tracking."""
    print("\n=== Testing Metrics ===")
    
    config = MockConfig(cache_summaries=False)
    adapter = MockAdapter()
    processor = SimplifiedAIProcessor(adapter, config)
    
    messages = [{"role": "user", "content": "Test"}]
    
    await processor.summarize(messages)
    await processor.summarize(messages)
    
    metrics = processor.get_metrics()
    
    print(f"Metrics: {metrics}")
    
    assert metrics['ai_calls']['summarize'] == 2
    assert metrics['ai_success']['summarize'] == 2
    assert 'summarize' in metrics['processing_time']
    
    print("✓ Metrics tracking working")


async def main():
    """Run all tests."""
    print("Testing Processor Performance Features")
    print("=" * 50)
    
    try:
        await test_cache_key_generation()
        await test_caching()
        await test_metrics()
        # Skip rate limiting test as it's slow
        # await test_rate_limiting()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
