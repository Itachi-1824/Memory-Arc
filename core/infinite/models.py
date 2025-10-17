"""Data models for infinite context system."""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class MemoryType(Enum):
    """Types of memories that can be stored."""
    CONVERSATION = "conversation"
    CODE = "code"
    FACT = "fact"
    SUMMARY = "summary"
    PREFERENCE = "preference"
    DOCUMENT = "document"


class BoundaryType(Enum):
    """Types of content boundaries for chunking."""
    PARAGRAPH = "paragraph"
    FUNCTION = "function"
    CLASS = "class"
    SECTION = "section"
    SENTENCE = "sentence"


@dataclass
class Memory:
    """Enhanced memory entry with versioning support."""
    id: str
    context_id: str
    content: str
    memory_type: MemoryType
    created_at: float
    importance: int = 5
    updated_at: float | None = None
    version: int = 1
    parent_id: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert memory to dictionary for storage."""
        return {
            "id": self.id,
            "context_id": self.context_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "created_at": self.created_at,
            "importance": self.importance,
            "updated_at": self.updated_at,
            "version": self.version,
            "parent_id": self.parent_id,
            "thread_id": self.thread_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        """Create memory from dictionary."""
        return cls(
            id=data["id"],
            context_id=data["context_id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            created_at=data["created_at"],
            importance=data.get("importance", 5),
            updated_at=data.get("updated_at"),
            version=data.get("version", 1),
            parent_id=data.get("parent_id"),
            thread_id=data.get("thread_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Chunk:
    """A chunk of content with metadata."""
    id: str
    content: str
    chunk_index: int
    total_chunks: int
    token_count: int
    relevance_score: float = 0.0
    start_pos: int = 0
    end_pos: int = 0
    boundary_type: BoundaryType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkBoundary:
    """Represents a detected boundary in content."""
    position: int
    boundary_type: BoundaryType
    confidence: float = 1.0


class QueryIntent(Enum):
    """Types of query intents."""
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    CODE = "code"
    CONVERSATIONAL = "conversational"
    PREFERENCE = "preference"
    MIXED = "mixed"


@dataclass
class QueryAnalysis:
    """Analysis of a user query."""
    intent: QueryIntent
    entities: list[tuple[str, str]] = field(default_factory=list)  # (entity_type, entity_value)
    temporal_expressions: list[tuple[str, float]] = field(default_factory=list)  # (expression, timestamp)
    code_patterns: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of memory retrieval."""
    memories: list[Memory]
    total_found: int
    query_analysis: QueryAnalysis
    retrieval_time_ms: float
    chunks: list[Chunk] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InfiniteContextConfig:
    """Configuration for InfiniteContextEngine.
    
    This dataclass provides comprehensive configuration options for the infinite
    context system with sensible defaults for different use cases.
    """
    # Storage paths
    storage_path: str = "./data/infinite_context"
    vector_store_path: str | None = None  # Default: storage_path/vectors
    cache_path: str | None = None  # Default: storage_path/cache
    
    # Model configuration
    model_name: str = "gpt-4"
    max_tokens: int | None = None  # Auto-detected from model_name
    
    # Code tracking
    enable_code_tracking: bool = False
    code_watch_path: str | None = None
    code_ignore_patterns: list[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", ".git", "node_modules", "venv", ".venv"
    ])
    
    # Caching
    enable_caching: bool = True
    cache_max_size_gb: float = 10.0
    
    # NLP features
    use_spacy: bool = False  # Enhanced entity extraction
    
    # Memory settings
    similarity_threshold: float = 0.7
    default_importance: int = 5
    
    # Retrieval settings
    default_max_results: int = 10
    enable_query_caching: bool = True
    query_cache_ttl_seconds: int = 300
    
    # Performance tuning
    batch_size: int = 100
    max_concurrent_queries: int = 10
    embedding_batch_size: int = 50
    
    # Vector store settings
    vector_embedding_dim: int = 1536  # OpenAI default
    vector_distance_metric: str = "cosine"
    
    # Chunking settings
    chunk_overlap_tokens: int = 100
    preserve_structure: bool = True
    
    def validate(self) -> None:
        """Validate configuration and raise ValueError if invalid."""
        if self.enable_code_tracking and not self.code_watch_path:
            raise ValueError("code_watch_path is required when enable_code_tracking=True")
        
        if self.similarity_threshold < 0.0 or self.similarity_threshold > 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        
        if self.default_importance < 1 or self.default_importance > 10:
            raise ValueError("default_importance must be between 1 and 10")
        
        if self.cache_max_size_gb <= 0:
            raise ValueError("cache_max_size_gb must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.max_concurrent_queries <= 0:
            raise ValueError("max_concurrent_queries must be positive")
        
        if self.vector_embedding_dim <= 0:
            raise ValueError("vector_embedding_dim must be positive")
        
        if self.vector_distance_metric not in ["cosine", "euclidean", "dot"]:
            raise ValueError("vector_distance_metric must be 'cosine', 'euclidean', or 'dot'")
    
    @classmethod
    def minimal(cls) -> "InfiniteContextConfig":
        """Create minimal configuration for basic usage.
        
        Optimized for:
        - Low resource usage
        - Fast startup
        - Simple use cases
        """
        return cls(
            enable_caching=False,
            use_spacy=False,
            enable_code_tracking=False,
            cache_max_size_gb=1.0,
            batch_size=50,
            max_concurrent_queries=5,
            embedding_batch_size=25,
            default_max_results=5
        )
    
    @classmethod
    def balanced(cls) -> "InfiniteContextConfig":
        """Create balanced configuration for general usage.
        
        Optimized for:
        - Good performance
        - Moderate resource usage
        - Most common use cases
        """
        return cls(
            enable_caching=True,
            use_spacy=False,
            enable_code_tracking=False,
            cache_max_size_gb=5.0,
            batch_size=100,
            max_concurrent_queries=10,
            embedding_batch_size=50,
            default_max_results=10
        )
    
    @classmethod
    def performance(cls) -> "InfiniteContextConfig":
        """Create performance configuration for high-throughput usage.
        
        Optimized for:
        - Maximum performance
        - High resource usage acceptable
        - Production workloads
        """
        return cls(
            enable_caching=True,
            use_spacy=True,
            enable_code_tracking=False,
            cache_max_size_gb=20.0,
            batch_size=200,
            max_concurrent_queries=20,
            embedding_batch_size=100,
            default_max_results=20,
            enable_query_caching=True,
            query_cache_ttl_seconds=600
        )
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "InfiniteContextConfig":
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            InfiniteContextConfig instance
        """
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "storage_path": self.storage_path,
            "vector_store_path": self.vector_store_path,
            "cache_path": self.cache_path,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "enable_code_tracking": self.enable_code_tracking,
            "code_watch_path": self.code_watch_path,
            "code_ignore_patterns": self.code_ignore_patterns,
            "enable_caching": self.enable_caching,
            "cache_max_size_gb": self.cache_max_size_gb,
            "use_spacy": self.use_spacy,
            "similarity_threshold": self.similarity_threshold,
            "default_importance": self.default_importance,
            "default_max_results": self.default_max_results,
            "enable_query_caching": self.enable_query_caching,
            "query_cache_ttl_seconds": self.query_cache_ttl_seconds,
            "batch_size": self.batch_size,
            "max_concurrent_queries": self.max_concurrent_queries,
            "embedding_batch_size": self.embedding_batch_size,
            "vector_embedding_dim": self.vector_embedding_dim,
            "vector_distance_metric": self.vector_distance_metric,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "preserve_structure": self.preserve_structure,
        }


@dataclass
class SystemMetrics:
    """System-wide metrics for monitoring and observability.
    
    Tracks performance, resource usage, and operational statistics
    for the infinite context system.
    """
    # Memory statistics
    total_memories: int = 0
    total_code_changes: int = 0
    active_contexts: int = 0
    
    # Storage statistics
    storage_size_bytes: int = 0
    embedding_cache_size_bytes: int = 0
    vector_store_size_bytes: int = 0
    
    # Performance metrics
    avg_query_latency_ms: float = 0.0
    p95_query_latency_ms: float = 0.0
    p99_query_latency_ms: float = 0.0
    total_queries: int = 0
    
    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # System health
    uptime_seconds: float = 0.0
    last_error: str | None = None
    error_count: int = 0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Timestamps
    last_updated: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_memories": self.total_memories,
            "total_code_changes": self.total_code_changes,
            "active_contexts": self.active_contexts,
            "storage_size_bytes": self.storage_size_bytes,
            "embedding_cache_size_bytes": self.embedding_cache_size_bytes,
            "vector_store_size_bytes": self.vector_store_size_bytes,
            "avg_query_latency_ms": self.avg_query_latency_ms,
            "p95_query_latency_ms": self.p95_query_latency_ms,
            "p99_query_latency_ms": self.p99_query_latency_ms,
            "total_queries": self.total_queries,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "uptime_seconds": self.uptime_seconds,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "last_updated": self.last_updated,
        }
