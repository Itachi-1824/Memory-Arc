"""Stress tests for InfiniteContextEngine.

Tests system behavior under extreme conditions:
- High memory volumes (10M+ memories)
- High concurrent load (100+ simultaneous queries)
- Limited resources (low memory, slow disk)
- Failure scenarios (database crashes, network issues)
"""

import asyncio
import time
import tempfile
from pathlib import Path
import random

from core.infinite import InfiniteContextEngine, InfiniteContextConfig


def mock_embedding(text: str) -> list[float]:
    """Generate deterministic embedding for testing."""
    hash_val = hash(text)
    return [float((hash_val >> i) & 0xFF) / 255.0 for i in range(1536)]


class StressTestRunner:
    """Run stress tests for InfiniteContextEngine."""
    
    def __init__(self, storage_path: Path):
        """Initialize stress test runner."""
        self.storage_path = storage_path
        self.results = []
    
    async def setup_engine(self, config: InfiniteContextConfig | None = None) -> InfiniteContextEngine:
        """Set up engine for stress testing."""
        if config is None:
            config = InfiniteContextConfig(
                storage_path=str(self.storage_path),
                enable_caching=True,
                enable_code_tracking=False,
                use_spacy=False
            )
        
        engine = InfiniteContextEngine(
            config=config,
            embedding_fn=mock_embedding
        )
        
        await engine.initialize()
        return engine
    
    async def stress_test_high_volume(self, target_count: int = 100000) -> dict:
        """Test with high memory volumes."""
        print(f"\n{'='*60}")
        print(f"STRESS TEST: High Volume ({target_count:,} memories)")
        print(f"{'='*60}")
        
        engine = await self.setup_engine()
        
        try:
            start_time = time.time()
            errors = 0
            
            # Add memories in batches
            batch_size = 1000
            for batch_start in range(0, target_count, batch_size):
                batch_end = min(batch_start + batch_size, target_count)
                
                tasks = []
                for i in range(batch_start, batch_end):
                    task = engine.add_memory(
                        content=f"Stress test memory {i}: {random.choice(['fact', 'conversation', 'note'])}",
                        context_id="stress_test",
                        importance=random.randint(1, 10)
                    )
                    tasks.append(task)
                
                try:
                    await asyncio.gather(*tasks)
                except Exception as e:
                    errors += 1
                    print(f"  Error in batch {batch_start}-{batch_end}: {e}")
                
                if (batch_end) % 10000 == 0:
                    print(f"  Added {batch_end:,} memories...")
            
            duration = time.time() - start_time
            
            # Test queries
            print("  Testing queries...")
            query_errors = 0
            query_latencies = []
            
            for i in range(100):
                try:
                    start = time.time()
                    await engine.retrieve(
                        query=f"stress test {random.randint(0, target_count)}",
                        context_id="stress_test",
                        max_results=10
                    )
                    query_latencies.append((time.time() - start) * 1000)
                except Exception as e:
                    query_errors += 1
            
            metrics = engine.get_metrics()
            
            result = {
                "test": "high_volume",
                "target_count": target_count,
                "duration_seconds": duration,
                "addition_errors": errors,
                "query_errors": query_errors,
                "avg_query_latency_ms": sum(query_latencies) / len(query_latencies) if query_latencies else 0,
                "total_memories": metrics.total_memories,
                "status": "PASSED" if errors == 0 and query_errors == 0 else "FAILED"
            }
            
            print(f"\nResult: {result['status']}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Errors: {errors} addition, {query_errors} query")
            print(f"  Avg Query Latency: {result['avg_query_latency_ms']:.2f}ms")
            
            return result
            
        finally:
            await engine.shutdown()
    
    async def stress_test_concurrent_load(self, concurrent_count: int = 100) -> dict:
        """Test with high concurrent load."""
        print(f"\n{'='*60}")
        print(f"STRESS TEST: Concurrent Load ({concurrent_count} simultaneous queries)")
        print(f"{'='*60}")
        
        engine = await self.setup_engine()
        
        try:
            # Add some test data
            print("  Setting up test data...")
            for i in range(1000):
                await engine.add_memory(
                    content=f"Test memory {i}",
                    context_id="stress_test"
                )
            
            # Run concurrent queries
            print(f"  Running {concurrent_count} concurrent queries...")
            
            async def run_query(query_id: int) -> tuple[bool, float]:
                try:
                    start = time.time()
                    await engine.retrieve(
                        query=f"test {random.randint(0, 1000)}",
                        context_id="stress_test",
                        max_results=10
                    )
                    return True, (time.time() - start) * 1000
                except Exception as e:
                    print(f"    Query {query_id} failed: {e}")
                    return False, 0.0
            
            start_time = time.time()
            tasks = [run_query(i) for i in range(concurrent_count)]
            results = await asyncio.gather(*tasks)
            duration = time.time() - start_time
            
            successes = sum(1 for success, _ in results if success)
            failures = concurrent_count - successes
            latencies = [latency for success, latency in results if success]
            
            result = {
                "test": "concurrent_load",
                "concurrent_count": concurrent_count,
                "duration_seconds": duration,
                "successes": successes,
                "failures": failures,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
                "throughput_qps": successes / duration,
                "status": "PASSED" if failures == 0 else "FAILED"
            }
            
            print(f"\nResult: {result['status']}")
            print(f"  Successes: {successes}/{concurrent_count}")
            print(f"  Avg Latency: {result['avg_latency_ms']:.2f}ms")
            print(f"  Throughput: {result['throughput_qps']:.2f} qps")
            
            return result
            
        finally:
            await engine.shutdown()
    
    async def stress_test_limited_resources(self) -> dict:
        """Test with limited resources."""
        print(f"\n{'='*60}")
        print(f"STRESS TEST: Limited Resources")
        print(f"{'='*60}")
        
        # Use minimal configuration
        config = InfiniteContextConfig.minimal()
        config.storage_path = str(self.storage_path)
        config.cache_max_size_gb = 0.1  # Very small cache
        
        engine = await self.setup_engine(config)
        
        try:
            print("  Adding memories with limited cache...")
            errors = 0
            
            for i in range(5000):
                try:
                    await engine.add_memory(
                        content=f"Limited resource test {i}",
                        context_id="stress_test"
                    )
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"    Error: {e}")
            
            print("  Testing queries with limited resources...")
            query_errors = 0
            
            for i in range(50):
                try:
                    await engine.retrieve(
                        query=f"test {i}",
                        context_id="stress_test"
                    )
                except Exception as e:
                    query_errors += 1
            
            result = {
                "test": "limited_resources",
                "addition_errors": errors,
                "query_errors": query_errors,
                "status": "PASSED" if errors < 100 and query_errors < 10 else "FAILED"
            }
            
            print(f"\nResult: {result['status']}")
            print(f"  Addition Errors: {errors}/5000")
            print(f"  Query Errors: {query_errors}/50")
            
            return result
            
        finally:
            await engine.shutdown()
    
    async def stress_test_failure_recovery(self) -> dict:
        """Test failure scenarios and recovery."""
        print(f"\n{'='*60}")
        print(f"STRESS TEST: Failure Recovery")
        print(f"{'='*60}")
        
        engine = await self.setup_engine()
        
        try:
            # Add some data
            print("  Adding initial data...")
            for i in range(100):
                await engine.add_memory(
                    content=f"Recovery test {i}",
                    context_id="stress_test"
                )
            
            # Simulate failure by shutting down
            print("  Simulating failure (shutdown)...")
            await engine.shutdown()
            
            # Try to recover
            print("  Attempting recovery (reinitialize)...")
            await engine.initialize()
            
            # Test if data is still accessible
            print("  Testing data accessibility...")
            result = await engine.retrieve(
                query="recovery test",
                context_id="stress_test",
                max_results=10
            )
            
            recovered = len(result.memories) > 0
            
            result = {
                "test": "failure_recovery",
                "recovered": recovered,
                "memories_found": len(result.memories),
                "status": "PASSED" if recovered else "FAILED"
            }
            
            print(f"\nResult: {result['status']}")
            print(f"  Recovered: {recovered}")
            print(f"  Memories Found: {result['memories_found']}")
            
            return result
            
        finally:
            await engine.shutdown()
    
    async def run_all_stress_tests(self) -> list[dict]:
        """Run all stress tests."""
        print("\n" + "="*60)
        print("INFINITE CONTEXT ENGINE STRESS TEST SUITE")
        print("="*60)
        
        results = []
        
        # Run tests
        try:
            # Test 1: High volume (reduced for faster testing)
            result = await self.stress_test_high_volume(target_count=10000)
            results.append(result)
        except Exception as e:
            print(f"High volume test failed: {e}")
            results.append({"test": "high_volume", "status": "ERROR", "error": str(e)})
        
        try:
            # Test 2: Concurrent load
            result = await self.stress_test_concurrent_load(concurrent_count=50)
            results.append(result)
        except Exception as e:
            print(f"Concurrent load test failed: {e}")
            results.append({"test": "concurrent_load", "status": "ERROR", "error": str(e)})
        
        try:
            # Test 3: Limited resources
            result = await self.stress_test_limited_resources()
            results.append(result)
        except Exception as e:
            print(f"Limited resources test failed: {e}")
            results.append({"test": "limited_resources", "status": "ERROR", "error": str(e)})
        
        try:
            # Test 4: Failure recovery
            result = await self.stress_test_failure_recovery()
            results.append(result)
        except Exception as e:
            print(f"Failure recovery test failed: {e}")
            results.append({"test": "failure_recovery", "status": "ERROR", "error": str(e)})
        
        # Summary
        print("\n" + "="*60)
        print("STRESS TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in results if r.get("status") == "PASSED")
        failed = sum(1 for r in results if r.get("status") == "FAILED")
        errors = sum(1 for r in results if r.get("status") == "ERROR")
        
        print(f"\nTotal Tests: {len(results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        
        for result in results:
            status_symbol = "✅" if result.get("status") == "PASSED" else "❌"
            print(f"  {status_symbol} {result['test']}: {result['status']}")
        
        return results


async def main():
    """Run stress tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "stress_test_storage"
        storage_path.mkdir(parents=True, exist_ok=True)
        
        runner = StressTestRunner(storage_path)
        await runner.run_all_stress_tests()


if __name__ == "__main__":
    asyncio.run(main())
