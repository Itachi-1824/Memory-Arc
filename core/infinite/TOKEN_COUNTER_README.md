# Token Counter Implementation

## Overview

The Token Counter module provides accurate token counting for multiple AI model families with token budget management capabilities. This is essential for managing context windows and ensuring content fits within model limits.

## Features

### 1. Multi-Model Support
- **GPT Models**: GPT-3.5, GPT-4, GPT-4 Turbo (using tiktoken when available)
- **Claude Models**: Claude 2, Claude 3 family (Opus, Sonnet, Haiku)
- **Llama Models**: Llama 2, Llama 3 (using transformers when available)
- **Unknown Models**: Fallback estimation for any model

### 2. Token Counting
- Accurate tokenization using model-specific tokenizers
- Fallback estimation when tokenizers unavailable
- Batch token counting for efficiency
- Support for code, text, and special characters

### 3. Token Budget Management
- Track token usage across components
- Reserve tokens for system prompts and responses
- Allocate and release tokens dynamically
- Safety margins to avoid exceeding limits

### 4. Text Truncation
- Truncate from end, start, or middle
- Binary search for precise token limits
- Preserve important content based on strategy

### 5. Context Window Detection
- Automatic detection of model context windows
- Support for all major model variants
- Fallback to reasonable defaults

## Installation

### Basic Installation (Estimation Only)
No additional dependencies required. The module works with estimation-based token counting.

### Enhanced Installation (Accurate Tokenization)

For GPT models (recommended):
```bash
pip install tiktoken>=0.5.0
```

For Llama models:
```bash
pip install transformers>=4.30.0
```

For Claude models:
```bash
pip install anthropic>=0.18.0
```

## Usage

### Basic Token Counting

```python
from core.infinite import TokenCounter

# Create counter for specific model
counter = TokenCounter("gpt-4")

# Count tokens
text = "Hello, world! This is a test."
tokens = counter.count_tokens(text)
print(f"Tokens: {tokens}")
```

### Multiple Models

```python
from core.infinite import create_token_counter, get_model_context_window

models = ["gpt-4", "claude-3-opus", "llama-3"]

for model_name in models:
    counter = create_token_counter(model_name)
    tokens = counter.count_tokens("Sample text")
    window = get_model_context_window(model_name)
    print(f"{model_name}: {tokens} tokens, {window:,} context window")
```

### Batch Token Counting

```python
counter = TokenCounter("gpt-4")

texts = [
    "First message",
    "Second message with more content",
    "Third message"
]

counts = counter.count_tokens_batch(texts)
total = sum(counts)
print(f"Total tokens: {total}")
```

### Token Budget Management

```python
from core.infinite import TokenBudgetManager

# Create budget for GPT-4
manager = TokenBudgetManager(
    total_tokens=8192,
    system_tokens=500,      # Reserve for system prompt
    response_tokens=1000,   # Reserve for response
    safety_margin=100
)

# Check available tokens
print(f"Available: {manager.available_tokens}")

# Allocate tokens
if manager.allocate(2000):
    print("Allocated 2000 tokens")

# Release tokens when done
manager.release(500)

# Get detailed info
info = manager.get_budget_info()
print(info)
```

### Text Truncation

```python
counter = TokenCounter("gpt-4")

long_text = "Very long text..." * 100
max_tokens = 100

# Truncate from end (keep beginning)
truncated = counter.truncate_to_token_limit(long_text, max_tokens, "end")

# Truncate from start (keep end)
truncated = counter.truncate_to_token_limit(long_text, max_tokens, "start")

# Truncate from middle (keep both ends)
truncated = counter.truncate_to_token_limit(long_text, max_tokens, "middle")
```

## API Reference

### TokenCounter

```python
class TokenCounter:
    def __init__(self, model_name: str = "gpt-4")
    def count_tokens(self, text: str) -> int
    def count_tokens_batch(self, texts: list[str]) -> list[int]
    def truncate_to_token_limit(
        self,
        text: str,
        max_tokens: int,
        truncation_strategy: Literal["end", "start", "middle"] = "end"
    ) -> str
```

### TokenBudgetManager

```python
class TokenBudgetManager:
    def __init__(
        self,
        total_tokens: int,
        system_tokens: int = 0,
        response_tokens: int = 0,
        safety_margin: int = 100
    )
    
    @property
    def available_tokens(self) -> int
    
    @property
    def used_tokens(self) -> int
    
    def allocate(self, tokens: int) -> bool
    def release(self, tokens: int)
    def reset(self)
    def get_budget_info(self) -> dict[str, int]
```

### Helper Functions

```python
def create_token_counter(model_name: str) -> TokenCounter
def get_model_context_window(model_name: str) -> int
```

## Model Context Windows

| Model | Context Window |
|-------|----------------|
| GPT-4 Turbo | 128,000 tokens |
| GPT-4 32k | 32,768 tokens |
| GPT-4 | 8,192 tokens |
| GPT-3.5 Turbo 16k | 16,384 tokens |
| GPT-3.5 Turbo | 4,096 tokens |
| Claude 3 (all) | 200,000 tokens |
| Claude 2 | 100,000 tokens |
| Llama 3 | 8,192 tokens |
| Llama 2 | 4,096 tokens |

## Token Estimation

When model-specific tokenizers are not available, the module uses estimation:

- **Character-based**: 1 token ≈ 4 characters
- **Word-based**: 1 token ≈ 0.75 words
- **Final estimate**: Average of both methods

This provides reasonable accuracy (typically within 10-20% of actual count) without requiring external dependencies.

## Performance

- Token counting: < 1ms for typical text (< 1000 tokens)
- Batch counting: Efficient for multiple texts
- Truncation: Binary search for precise limits
- Budget operations: O(1) time complexity

## Testing

Comprehensive test suite with 54 tests covering:
- Model family detection
- Token counting accuracy
- Batch operations
- Text truncation
- Budget management
- Context window detection
- Edge cases and error handling

Run tests:
```bash
python -m pytest tests/infinite/test_token_counter.py -v
```

## Integration with Chunk Manager

The token counter integrates seamlessly with the semantic chunker:

```python
from core.infinite import SemanticChunker, TokenCounter

counter = TokenCounter("gpt-4")
chunker = SemanticChunker(
    max_chunk_size=1000,
    token_estimator=counter.count_tokens
)

chunks = chunker.chunk_content(content, "text")
```

## Best Practices

1. **Use Model-Specific Counters**: Create separate counters for different models
2. **Install Tokenizers**: Install tiktoken for GPT models for best accuracy
3. **Budget Management**: Always reserve tokens for system prompts and responses
4. **Safety Margins**: Use safety margins to avoid exceeding limits
5. **Batch Operations**: Use batch counting for multiple texts
6. **Truncation Strategy**: Choose appropriate truncation strategy based on use case

## Limitations

1. **Tokenizer Availability**: Some tokenizers require additional packages
2. **Estimation Accuracy**: Fallback estimation is approximate (±10-20%)
3. **Model Updates**: New model variants may need manual context window updates
4. **Language Support**: Estimation works best for English text

## Future Enhancements

- Support for more model families (Gemini, Mistral, etc.)
- Automatic tokenizer downloading
- Caching of token counts
- Streaming token counting
- Multi-language optimization
- Token usage analytics

## Requirements

Requirement 10.2 from design document:
- ✅ Model-specific tokenization rules
- ✅ Token budget management
- ✅ Efficient caching of tokenizers
- ✅ Support for GPT, Claude, Llama models
- ✅ Fallback estimation for unknown models

## Files

- `core/infinite/token_counter.py` - Main implementation
- `tests/infinite/test_token_counter.py` - Comprehensive test suite
- `examples/token_counter_example.py` - Usage examples
- `core/infinite/TOKEN_COUNTER_README.md` - This documentation

## See Also

- [Semantic Chunker](./SEMANTIC_CHUNKER_README.md) - Uses token counter for chunking
- [Design Document](../../.kiro/specs/infinite-context-system/design.md) - Overall system design
- [Requirements](../../.kiro/specs/infinite-context-system/requirements.md) - System requirements
