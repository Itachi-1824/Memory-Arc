# Infinite Context System & Web Test Interface

## Description

Adds infinite context memory system with unlimited storage, semantic search, and a web-based test interface.

**Key additions:**
- Infinite context engine (`core/infinite/`) - document store, vector store, temporal indexing, retrieval orchestrator
- Pollinations AI adapter for OpenAI-compatible API
- Web chat interface for testing with live metrics
- Fixed all import issues (changed `memory_system` to relative imports)

## Type of change

- [x] New feature
- [x] Documentation update
- [x] Code refactoring (import fixes)

## What's New

### Infinite Context System (`core/infinite/`)
- DocumentStore (SQLite), VectorStore (Qdrant), TemporalIndex
- DynamicMemoryStore with versioning
- RetrievalOrchestrator with multi-strategy search
- ChunkManager for automatic content chunking
- EmbeddingCache (LMDB) for fast lookups

### Pollinations Adapter (`adapters/pollinations_adapter.py`)
- OpenAI-compatible format
- Integrated with adapter registry

### Web Interface (`website/`)
- Flask server with real-time chat
- Live metrics dashboard
- Memory visualization panel
- Persistent storage
- Setup: `setup.bat` and `start.bat`

### Import Fixes
- Fixed all `memory_system` imports to relative imports
- System works without package installation

## Testing

```bash
cd website
setup.bat
start.bat
# Open index.html, enter API token, test chat
```

## Checklist

- [x] Code follows style guidelines
- [x] Self-reviewed
- [x] Documentation updated
- [x] No new warnings
- [x] Tested locally

## Dependencies Added

- `lmdb`, `watchdog`, `tree-sitter`, `zstandard` - Infinite context
- `flask`, `flask-cors` - Web interface

## Notes

Pollinations integration is for demo only. Core memory system is provider-agnostic.
