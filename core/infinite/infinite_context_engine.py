"""InfiniteContextEngine: Main class integrating all infinite context components.

This is the primary interface for the infinite context system, providing:
- Unlimited memory storage with versioning
- Code change tracking with 1:1 diffs
- Intelligent retrieval with multi-strategy search
- Automatic chunking for any model's context window
- Health monitoring and metrics
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Literal

from .models import (
    Memory,
    MemoryType,
    RetrievalResult,
    QueryAnalysis,
    InfiniteContextConfig,
    SystemMetrics,
)
from .document_store import DocumentStore
from .temporal_index import TemporalIndex
from .vector_store import VectorStore
from .embedding_cache import EmbeddingCache
from .dynamic_memory_store import DynamicMemoryStore
from .code_change_tracker import CodeChangeTracker
from .chunk_manager import ChunkManager, ChunkManagerConfig
from .retrieval_orchestrator import RetrievalOrchestrator

logger = logging.getLogger(__name__)


class InfiniteContextEngine:
    """
    Main class integrating all infinite context components.
    
    This engine provides:
    - Universal memory storage (conversations, documents, preferences, facts, code)
    - Dynamic memory evolution with version tracking
    - Code-specific tracking with AST diffs (optional)
    - Intelligent retrieval with multi-strategy search
    - Automatic chunking for any model's context window
    - Health monitoring and performance metrics
    
    Architecture:
    - Storage Layer: SQLite (documents), Qdrant (vectors), LMDB (cache)
    - Memory Layer: DynamicMemoryStore (versioning), CodeChangeTracker (code)
    - Retrieval Layer: RetrievalOrchestrator (search), ChunkManager (formatting)
    """
    
    def __init__(
        self,
        config: InfiniteContextConfig | None = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
        # Legacy parameters for backward compatibility
        storage_path: str | Path | None = None,
        vector_store_path: str | Path | None = None,
        cache_path: str | Path | None = None,
        model_name: str | None = None,
        max_tokens: int | None = None,
        enable_code_tracking: bool | None = None,
        code_watch_path: str | Path | None = None,
        enable_caching: bool | None = None,
        use_spacy: bool | None = None
    ):
        """
        Initialize InfiniteContextEngine.
        
        Args:
            config: Configuration object (recommended). If None, uses default config.
            embedding_fn: Function to generate embeddings from text
            
            Legacy parameters (for backward compatibility):
            storage_path: Path to SQLite database directory
            vector_store_path: Path to Qdrant vector store
            cache_path: Path to LMDB embedding cache
            model_name: Target model name for chunking
            max_tokens: Maximum tokens for model
            enable_code_tracking: Whether to enable code change tracking
            code_watch_path: Path to watch for code changes
            enable_caching: Whether to enable embedding cache
            use_spacy: Whether to use spaCy for enhanced NLP
        """
        # Handle configuration
        if config is None:
            # Create config from legacy parameters or use defaults
            config = InfiniteContextConfig()
            if storage_path is not None:
                config.storage_path = str(storage_path)
            if vector_store_path is not None:
                config.vector_store_path = str(vector_store_path)
            if cache_path is not None:
                config.cache_path = str(cache_path)
            if model_name is not None:
                config.model_name = model_name
            if max_tokens is not None:
                config.max_tokens = max_tokens
            if enable_code_tracking is not None:
                config.enable_code_tracking = enable_code_tracking
            if code_watch_path is not None:
                config.code_watch_path = str(code_watch_path)
            if enable_caching is not None:
                config.enable_caching = enable_caching
            if use_spacy is not None:
                config.use_spacy = use_spacy
        
        # Validate configuration
        config.validate()
        self.config = config
        
        # Set up paths
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.vector_store_path = (
            Path(config.vector_store_path)
            if config.vector_store_path
            else self.storage_path / "vectors"
        )
        self.cache_path = (
            Path(config.cache_path)
            if config.cache_path
            else self.storage_path / "cache"
        )
        
        self.embedding_fn = embedding_fn
        self.model_name = config.model_name
        self.enable_code_tracking = config.enable_code_tracking
        self.code_watch_path = Path(config.code_watch_path) if config.code_watch_path else None
        
        # Database path
        self.db_path = self.storage_path / "infinite_context.db"
        
        # Initialize storage layer
        self.document_store = DocumentStore(self.db_path)
        self.temporal_index = TemporalIndex(self.db_path)
        self.vector_store = VectorStore(
            path=str(self.vector_store_path),
            embedding_dim=config.vector_embedding_dim
        )
        
        # Initialize embedding cache
        self.embedding_cache: EmbeddingCache | None = None
        if config.enable_caching:
            self.embedding_cache = EmbeddingCache(
                cache_dir=str(self.cache_path),
                max_size_bytes=int(config.cache_max_size_gb * 1024 * 1024 * 1024)
            )
        
        # Initialize memory layer
        self.dynamic_memory = DynamicMemoryStore(
            db_path=self.db_path,
            similarity_threshold=config.similarity_threshold
        )
        
        # Initialize code tracking (optional)
        self.code_tracker: CodeChangeTracker | None = None
        if config.enable_code_tracking and self.code_watch_path:
            self.code_tracker = CodeChangeTracker(
                watch_path=self.code_watch_path,
                db_path=self.db_path,
                auto_track=False  # Manual control
            )
        
        # Initialize chunk manager
        chunk_config = ChunkManagerConfig(
            model_name=config.model_name,
            max_tokens=config.max_tokens
        )
        self.chunk_manager = ChunkManager(
            config=chunk_config,
            embedding_fn=embedding_fn
        )
        
        # Initialize retrieval orchestrator
        self.retrieval_orchestrator = RetrievalOrchestrator(
            document_store=self.document_store,
            temporal_index=self.temporal_index,
            vector_store=self.vector_store,
            code_change_store=self.code_tracker.store if self.code_tracker else None,
            embedding_fn=self._get_embedding,
            use_spacy=config.use_spacy,
            enable_caching=config.enable_query_caching
        )
        
        # Initialization state
        self._initialized = False
        self._start_time = time.time()
        
        # Metrics tracking
        self._query_latencies: list[float] = []
        self._metrics = SystemMetrics(
            last_updated=time.time()
        )
        
        # Error tracking
        self._last_error: str | None = None
        self._error_count: int = 0
    
    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            logger.warning("InfiniteContextEngine already initialized")
            return
        
        logger.info("Initializing InfiniteContextEngine...")
        
        try:
            # Initialize storage layer
            await self.document_store.initialize()
            await self.temporal_index.initialize()
            await self.vector_store.initialize()
            
            if self.embedding_cache:
                await self.embedding_cache.initialize()
            
            # Initialize memory layer
            await self.dynamic_memory.initialize()
            
            if self.code_tracker:
                await self.code_tracker.initialize()
            
            # Update metrics
            self._metrics.total_memories = await self._count_memories()
            if self.code_tracker:
                self._metrics.total_code_changes = await self._count_code_changes()
            
            self._initialized = True
            logger.info("InfiniteContextEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize InfiniteContextEngine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown all components with proper cleanup."""
        if not self._initialized:
            return
        
        logger.info("Shutting down InfiniteContextEngine...")
        
        try:
            # Stop code tracking if active
            if self.code_tracker:
                try:
                    await self.code_tracker.stop_tracking()
                except Exception as e:
                    logger.warning(f"Error stopping code tracker: {e}")
            
            # Close storage connections
            try:
                await self.document_store.close()
            except Exception as e:
                logger.warning(f"Error closing document store: {e}")
            
            try:
                await self.vector_store.close()
            except Exception as e:
                logger.warning(f"Error closing vector store: {e}")
            
            if self.embedding_cache:
                try:
                    await self.embedding_cache.close()
                except Exception as e:
                    logger.warning(f"Error closing embedding cache: {e}")
            
            self._initialized = False
            logger.info("InfiniteContextEngine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            # Don't raise - we want shutdown to complete even with errors
    
    async def add_memory(
        self,
        content: str,
        memory_type: str | MemoryType = MemoryType.CONVERSATION,
        context_id: str = "default",
        importance: int = 5,
        supersedes: str | None = None,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Add a new memory to the system.
        
        Args:
            content: Memory content
            memory_type: Type of memory (conversation, code, fact, preference, etc.)
            context_id: Context identifier for grouping memories
            importance: Importance score (1-10)
            supersedes: ID of memory this supersedes (creates new version)
            thread_id: Optional thread identifier
            metadata: Optional metadata dictionary
            
        Returns:
            ID of the created memory
        """
        if not self._initialized:
            raise RuntimeError("InfiniteContextEngine not initialized. Call initialize() first.")
        
        # Generate embedding
        embedding = await self._get_embedding(content)
        
        # Add to dynamic memory store
        memory_id = await self.dynamic_memory.add_memory(
            content=content,
            memory_type=memory_type,
            context_id=context_id,
            importance=importance,
            supersedes=supersedes,
            thread_id=thread_id,
            metadata=metadata,
            embedding=embedding
        )
        
        # Update metrics
        self._metrics.total_memories += 1
        
        logger.debug(f"Added memory {memory_id} (type: {memory_type})")
        return memory_id
    
    async def retrieve(
        self,
        query: str,
        context_id: str = "default",
        max_results: int = 10,
        memory_types: list[str | MemoryType] | None = None,
        time_range: tuple[float, float] | None = None,
        include_history: bool = False,
        return_chunks: bool = False
    ) -> RetrievalResult:
        """
        Retrieve relevant memories for a query.
        
        Args:
            query: Search query
            context_id: Context identifier to search within
            max_results: Maximum number of results to return
            memory_types: Filter by memory types (None = all types)
            time_range: Optional time range filter (start_time, end_time)
            include_history: Whether to include version history
            return_chunks: Whether to chunk results for model consumption
            
        Returns:
            RetrievalResult with memories, analysis, and optional chunks
        """
        if not self._initialized:
            raise RuntimeError("InfiniteContextEngine not initialized. Call initialize() first.")
        
        # Primary retrieval function
        async def primary_retrieve():
            return await self.retrieval_orchestrator.retrieve(
                query=query,
                context_id=context_id,
                max_results=max_results,
                memory_types=memory_types,
                time_range=time_range,
                include_history=include_history
            )
        
        # Fallback: direct document store query (no vector search)
        async def fallback_retrieve():
            logger.warning("Using fallback retrieval (document store only)")
            memories = await self.document_store.query_memories(
                context_id=context_id,
                limit=max_results
            )
            
            from .models import QueryIntent
            analysis = QueryAnalysis(
                intent=QueryIntent.CONVERSATIONAL,
                confidence=0.5
            )
            
            return RetrievalResult(
                memories=memories,
                total_found=len(memories),
                query_analysis=analysis,
                retrieval_time_ms=0.0,
                metadata={"fallback": True}
            )
        
        # Execute with fallback
        result = await self._execute_with_fallback(
            primary_retrieve,
            fallback_retrieve,
            "retrieve"
        )
        
        # Optionally chunk results
        if return_chunks and result.memories:
            try:
                chunks = self.chunk_manager.chunk_content(
                    content=result.memories,
                    content_type="mixed",
                    query=query
                )
                result.chunks = chunks
            except Exception as e:
                logger.warning(f"Chunking failed: {e}, returning without chunks")
        
        # Update metrics
        self._metrics.total_queries += 1
        self._query_latencies.append(result.retrieval_time_ms)
        
        # Keep only last 1000 latencies for percentile calculation
        if len(self._query_latencies) > 1000:
            self._query_latencies = self._query_latencies[-1000:]
        
        return result
    
    async def query_at_time(
        self,
        query: str,
        timestamp: float,
        context_id: str = "default",
        max_results: int = 10
    ) -> RetrievalResult:
        """
        Query memories as they existed at a specific time.
        
        Args:
            query: Search query
            timestamp: Unix timestamp to query at
            context_id: Context identifier
            max_results: Maximum number of results
            
        Returns:
            RetrievalResult with memories from that time
        """
        if not self._initialized:
            raise RuntimeError("InfiniteContextEngine not initialized. Call initialize() first.")
        
        # Use dynamic memory store for temporal query
        memories = await self.dynamic_memory.query_at_time(
            query=query,
            timestamp=timestamp,
            context_id=context_id,
            top_k=max_results
        )
        
        # Create result with basic analysis
        from .models import QueryIntent
        analysis = QueryAnalysis(
            intent=QueryIntent.TEMPORAL,
            temporal_expressions=[("at_time", timestamp)]
        )
        
        return RetrievalResult(
            memories=memories,
            total_found=len(memories),
            query_analysis=analysis,
            retrieval_time_ms=0.0  # Not measured for temporal queries
        )
    
    async def get_version_history(
        self,
        memory_id: str
    ) -> list[Memory]:
        """
        Get version history for a memory.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            List of memory versions in chronological order
        """
        if not self._initialized:
            raise RuntimeError("InfiniteContextEngine not initialized. Call initialize() first.")
        
        return await self.dynamic_memory.get_version_history(memory_id)
    
    async def detect_contradictions(
        self,
        context_id: str,
        entity_type: str | None = None
    ) -> list[tuple[Memory, Memory, float]]:
        """
        Detect contradictory memories.
        
        Args:
            context_id: Context identifier
            entity_type: Optional entity type filter
            
        Returns:
            List of (memory1, memory2, similarity_score) tuples
        """
        if not self._initialized:
            raise RuntimeError("InfiniteContextEngine not initialized. Call initialize() first.")
        
        return await self.dynamic_memory.detect_contradictions(
            context_id=context_id,
            entity_type=entity_type
        )
    
    async def start_code_tracking(self) -> None:
        """Start automatic code change tracking."""
        if not self.code_tracker:
            raise RuntimeError("Code tracking not enabled. Set enable_code_tracking=True.")
        
        await self.code_tracker.start_tracking()
        logger.info(f"Started code tracking for {self.code_watch_path}")
    
    async def stop_code_tracking(self) -> None:
        """Stop automatic code change tracking."""
        if not self.code_tracker:
            return
        
        await self.code_tracker.stop_tracking()
        logger.info("Stopped code tracking")
    
    def get_health_status(self) -> dict[str, str]:
        """
        Get health status of all components.
        
        Returns:
            Dictionary mapping component names to status ("healthy", "degraded", "down")
        """
        status = {}
        
        # Check initialization
        if not self._initialized:
            return {"engine": "not_initialized"}
        
        # Check storage layer
        status["document_store"] = "healthy" if self.document_store._initialized else "down"
        status["temporal_index"] = "healthy" if self.temporal_index._initialized else "down"
        status["vector_store"] = "healthy" if self.vector_store._initialized else "down"
        
        if self.embedding_cache:
            # EmbeddingCache doesn't have _initialized, check if env is set
            status["embedding_cache"] = "healthy" if self.embedding_cache._env is not None else "down"
        
        # Check memory layer
        status["dynamic_memory"] = "healthy" if self.dynamic_memory._initialized else "down"
        
        if self.code_tracker:
            status["code_tracker"] = "healthy" if self.code_tracker._initialized else "down"
        
        # Overall status
        all_healthy = all(s == "healthy" for s in status.values())
        status["engine"] = "healthy" if all_healthy else "degraded"
        
        return status
    
    def get_metrics(self) -> SystemMetrics:
        """
        Get current system metrics.
        
        Returns:
            SystemMetrics object with current statistics
        """
        # Update uptime
        self._metrics.uptime_seconds = time.time() - self._start_time
        
        # Calculate query latency percentiles
        if self._query_latencies:
            sorted_latencies = sorted(self._query_latencies)
            n = len(sorted_latencies)
            self._metrics.avg_query_latency_ms = sum(sorted_latencies) / n
            self._metrics.p95_query_latency_ms = sorted_latencies[int(n * 0.95)]
            self._metrics.p99_query_latency_ms = sorted_latencies[int(n * 0.99)]
        
        # Update cache metrics if available
        if self.embedding_cache:
            total_requests = self._metrics.cache_hits + self._metrics.cache_misses
            self._metrics.cache_hit_rate = (
                self._metrics.cache_hits / total_requests if total_requests > 0 else 0.0
            )
            # Try to get cache size if method exists
            try:
                if hasattr(self.embedding_cache, 'get_size'):
                    self._metrics.embedding_cache_size_bytes = self.embedding_cache.get_size()
            except Exception:
                pass
        
        # Update storage size (approximate)
        try:
            if self.db_path.exists():
                self._metrics.storage_size_bytes = self.db_path.stat().st_size
        except Exception:
            pass
        
        # Update timestamp
        self._metrics.last_updated = time.time()
        
        return self._metrics
    
    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text, using cache if available."""
        if not self.embedding_fn:
            # Return zero vector if no embedding function
            return [0.0] * 1536
        
        # Check cache first
        if self.embedding_cache:
            cached = await self.embedding_cache.get(text)
            if cached is not None:
                self._metrics.cache_hits += 1
                return cached
            self._metrics.cache_misses += 1
        
        # Compute embedding
        embedding = self.embedding_fn(text)
        
        # Store in cache
        if self.embedding_cache:
            await self.embedding_cache.put(text, embedding)
        
        return embedding
    
    async def _count_memories(self) -> int:
        """Count total memories in the system."""
        try:
            memories = await self.document_store.query_memories(limit=1)
            # This is a simplified count - in production, use COUNT query
            return len(memories)
        except Exception:
            return 0
    
    async def _count_code_changes(self) -> int:
        """Count total code changes tracked."""
        if not self.code_tracker:
            return 0
        
        try:
            changes = await self.code_tracker.store.query_changes(limit=1)
            return len(changes)
        except Exception:
            return 0
    
    async def _execute_with_fallback(
        self,
        primary_fn: Callable,
        fallback_fn: Callable | None,
        operation_name: str
    ) -> Any:
        """
        Execute operation with automatic fallback on failure.
        
        Args:
            primary_fn: Primary function to execute
            fallback_fn: Fallback function to execute on failure (optional)
            operation_name: Name of operation for logging
            
        Returns:
            Result from primary or fallback function
            
        Raises:
            Exception if both primary and fallback fail
        """
        try:
            return await primary_fn()
        except Exception as e:
            self._error_count += 1
            self._last_error = f"{operation_name}: {str(e)}"
            self._metrics.error_count = self._error_count
            self._metrics.last_error = self._last_error
            
            logger.warning(
                f"{operation_name} failed with {type(e).__name__}: {e}"
            )
            
            if fallback_fn is None:
                logger.error(f"No fallback available for {operation_name}")
                raise
            
            try:
                logger.info(f"Attempting fallback for {operation_name}")
                return await fallback_fn()
            except Exception as fallback_error:
                logger.error(
                    f"Fallback for {operation_name} also failed: {fallback_error}"
                )
                raise
    
    async def _recover_from_error(self) -> None:
        """
        Attempt to recover from errors by reinitializing failed components.
        """
        logger.info("Attempting error recovery...")
        
        try:
            # Check and reinitialize storage components
            if not self.document_store._initialized:
                logger.info("Reinitializing document store...")
                await self.document_store.initialize()
            
            if not self.temporal_index._initialized:
                logger.info("Reinitializing temporal index...")
                await self.temporal_index.initialize()
            
            if not self.vector_store._initialized:
                logger.info("Reinitializing vector store...")
                await self.vector_store.initialize()
            
            if self.embedding_cache and not self.embedding_cache._initialized:
                logger.info("Reinitializing embedding cache...")
                await self.embedding_cache.initialize()
            
            logger.info("Error recovery completed successfully")
            
        except Exception as e:
            logger.error(f"Error recovery failed: {e}")
            raise
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
