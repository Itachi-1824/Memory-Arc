# Memory-Arc Project Context

## Overview
Memory-Arc is an intelligent memory management system for AI applications. It provides short-term (STM) and long-term (LTM) memory with semantic search capabilities.

## Core Features
- **4 Processing Modes**: AI, Heuristic, Hybrid, Disabled
- **Pluggable AI Adapters**: OpenAI, Anthropic, Ollama, HuggingFace, Pollinations, Custom
- **3 Embedding Models**: Default (all-MiniLM-L6-v2), Enhanced (BAAI/bge-m3), Code (jina-embeddings-v2-base-code)
- **6 Built-in Presets**: chatbot, chatbot-enhanced, coding-agent, coding-agent-enhanced, assistant, offline
- **Infinite Context System**: Unlimited memory with versioning, code tracking, intelligent retrieval
- **Vector Storage**: Qdrant for semantic search
- **Cost Optimization**: Caching, rate limiting, hybrid mode
- **Web Interface**: Test chat interface with live metrics and memory visualization

## Architecture
```
Memory-Arc/
├── adapters/           # AI adapter implementations (OpenAI, Anthropic, Pollinations, etc.)
├── core/
│   ├── infinite/       # Infinite context system (document store, vector store, retrieval)
│   ├── memory_manager.py
│   ├── processors.py
│   └── vector_memory.py
├── website/            # Web chat interface for testing
├── examples/           # Usage examples
├── tests/              # Test suite
├── utils/              # Utilities (model_manager)
├── config.py           # Configuration system
└── presets.py          # Built-in presets
```

## Key Components

### 1. Memory Manager (`core/memory_manager.py`)
- Main interface for memory operations
- Manages STM (JSON) and LTM (Qdrant)
- Methods: `add_message()`, `search_memory()`, `get_recent_messages()`

### 2. Processors (`core/processors.py`)
- **AIProcessor**: Uses AI adapters for processing
- **HeuristicProcessor**: Rule-based processing (offline, fast)
- **HybridProcessor**: Intelligent AI/heuristic switching
- **DisabledProcessor**: No processing

### 3. Configuration (`config.py`)
- **MemoryConfig**: Main configuration class
- **HeuristicConfig**: Heuristic processing settings
- **HybridConfig**: Hybrid mode settings
- Supports JSON/YAML file loading and presets

### 4. AI Adapters (`adapters/`)
- Abstract base: `AIAdapter` interface
- Built-in: OpenAI, Anthropic, Ollama, HuggingFace, Pollinations
- Registry: `AdapterRegistry` for discovery and instantiation
- Custom adapters supported

### 5. Infinite Context System (`core/infinite/`)
- **DocumentStore**: SQLite-based storage for all memories
- **VectorStore**: Qdrant for semantic embeddings
- **TemporalIndex**: Time-based memory queries
- **DynamicMemoryStore**: Memory versioning and evolution
- **RetrievalOrchestrator**: Multi-strategy intelligent retrieval
- **ChunkManager**: Automatic content chunking for any model
- **CodeChangeTracker**: Track code changes with AST diffs (optional)

### 6. Embedding Models
- **Default**: all-MiniLM-L6-v2 (80MB, 384 dim, included)
- **Enhanced**: BAAI/bge-m3 (2.2GB, 1024 dim, auto-download)
- **Code**: jinaai/jina-embeddings-v2-base-code (500MB, 768 dim, auto-download)

### 7. Web Interface (`website/`)
- Flask-based test interface
- Real-time chat with Pollinations AI
- Live metrics dashboard
- Memory retrieval visualization
- Persistent storage across restarts
- Easy setup with `setup.bat` and `start.bat`

## Common Use Cases

### Chatbot
```python
config = MemoryConfig.from_preset("chatbot")
memory = MemoryManager(context_id="user_123", config=config)
```

### Code Assistant
```python
config = MemoryConfig.from_preset("coding-agent")
memory = MemoryManager(context_id="dev_123", config=config)
```

### Task Assistant
```python
config = MemoryConfig.from_preset("assistant")
memory = MemoryManager(context_id="assistant_123", config=config)
```

### Infinite Context Chat
```python
from core.infinite import InfiniteContextEngine, InfiniteContextConfig

config = InfiniteContextConfig(
    storage_path="./data/chat",
    enable_caching=True,
    model_name="gemini"
)

engine = InfiniteContextEngine(config=config, embedding_fn=your_embedding_fn)
await engine.initialize()

# Add memory
await engine.add_memory(content="Hello!", memory_type=MemoryType.CONVERSATION)

# Retrieve with semantic search
result = await engine.retrieve(query="greeting", max_results=10)
```

### Web Interface Testing
```bash
cd website
setup.bat          # First time setup
start.bat          # Start server
# Open index.html in browser
```

## Code Style
- Python 3.8+
- Type hints throughout
- Async-first API
- PEP 8 compliant
- Max line length: 100 characters

## Testing
- pytest for testing
- Coverage tracking
- Tests in `tests/` directory
- Examples in `examples/` directory

## Dependencies
- **Core**: qdrant-client, sentence-transformers, pyyaml, python-dotenv
- **Infinite Context**: lmdb, watchdog, tree-sitter, zstandard
- **Web Interface**: flask, flask-cors, requests
- **Optional AI**: openai, anthropic, ollama, huggingface-hub
- **Optional Heuristics**: keybert, spacy

## Important Patterns

### Configuration
```python
# From preset
config = MemoryConfig.from_preset("chatbot")

# From file
config = MemoryConfig.from_file("config.json")

# Custom
config = MemoryConfig(
    mode="hybrid",
    embedding_model="BAAI/bge-m3",
    ai_adapter_name="openai"
)
```

### Memory Operations
```python
# Add message
await memory.add_message(thread_id, "user", "Hello!")

# Search
results = await memory.search_memory("query", top_k=5)

# Get recent
messages = memory.get_recent_messages(thread_id, limit=10)
```

### Custom Adapter
```python
class MyAdapter(AIAdapter):
    async def summarize_conversation(self, messages): ...
    async def extract_facts(self, messages): ...
    async def score_importance(self, text): ...

AdapterRegistry.register("my_adapter", MyAdapter)
```

## Common Issues & Solutions

### Issue: Model download slow
**Solution**: Pre-download with `python -m utils.model_manager download <model>`

### Issue: AI adapter errors
**Solution**: Check API keys, use hybrid mode with fallback enabled

### Issue: Memory not persisting
**Solution**: Check storage_path permissions, ensure ltm_enabled=True

### Issue: Poor search results
**Solution**: Use enhanced embedding model, increase top_k, check importance scores

## Project Goals
1. **Flexibility**: Support any AI provider, embedding model, or use case
2. **Performance**: Fast heuristic mode, smart hybrid mode
3. **Reliability**: Graceful fallbacks, error handling
4. **Ease of Use**: Presets, simple API, good docs
5. **Production Ready**: Tests, CI/CD, monitoring

## License
Apache License 2.0 - Copyright 2025 Itachi-1824
