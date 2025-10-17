"""Token counting for multiple AI models with accurate tokenization."""

import re
from typing import Literal
from enum import Enum


class ModelFamily(Enum):
    """Supported model families for token counting."""
    GPT = "gpt"
    CLAUDE = "claude"
    LLAMA = "llama"
    UNKNOWN = "unknown"


class TokenCounter:
    """
    Accurate token counting for multiple AI model families.
    
    Supports:
    - GPT models (GPT-3.5, GPT-4, GPT-4o)
    - Claude models (Claude 3 family)
    - Llama models (Llama 2, Llama 3)
    - Unknown models (fallback estimation)
    
    Features:
    - Model-specific tokenization rules
    - Token budget management
    - Efficient caching of tokenizers
    """

    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize token counter for a specific model.
        
        Args:
            model_name: Name of the model (e.g., "gpt-4", "claude-3", "llama-3")
        """
        self.model_name = model_name.lower()
        self.model_family = self._detect_model_family(model_name)
        self._tokenizer = None
        self._load_tokenizer()

    def _detect_model_family(self, model_name: str) -> ModelFamily:
        """
        Detect model family from model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelFamily enum value
        """
        model_lower = model_name.lower()
        
        if "gpt" in model_lower or "davinci" in model_lower or "turbo" in model_lower:
            return ModelFamily.GPT
        elif "claude" in model_lower or "anthropic" in model_lower:
            return ModelFamily.CLAUDE
        elif "llama" in model_lower or "meta" in model_lower:
            return ModelFamily.LLAMA
        else:
            return ModelFamily.UNKNOWN

    def _load_tokenizer(self):
        """Load the appropriate tokenizer for the model family."""
        if self.model_family == ModelFamily.GPT:
            self._load_gpt_tokenizer()
        elif self.model_family == ModelFamily.CLAUDE:
            self._load_claude_tokenizer()
        elif self.model_family == ModelFamily.LLAMA:
            self._load_llama_tokenizer()
        else:
            # Unknown model - will use estimation
            self._tokenizer = None

    def _load_gpt_tokenizer(self):
        """Load GPT tokenizer (tiktoken)."""
        try:
            import tiktoken
            
            # Map model names to tiktoken encodings
            if "gpt-4" in self.model_name:
                encoding_name = "cl100k_base"  # GPT-4, GPT-3.5-turbo
            elif "gpt-3.5" in self.model_name or "turbo" in self.model_name:
                encoding_name = "cl100k_base"
            else:
                encoding_name = "cl100k_base"  # Default for modern GPT models
            
            self._tokenizer = tiktoken.get_encoding(encoding_name)
        except ImportError:
            # tiktoken not available, fall back to estimation
            self._tokenizer = None

    def _load_claude_tokenizer(self):
        """Load Claude tokenizer (Anthropic's tokenizer)."""
        try:
            from anthropic import Anthropic
            
            # Claude uses a similar tokenizer to GPT
            # For now, we'll use tiktoken as approximation
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            # Anthropic SDK not available, fall back to estimation
            self._tokenizer = None

    def _load_llama_tokenizer(self):
        """Load Llama tokenizer (SentencePiece)."""
        try:
            from transformers import AutoTokenizer
            
            # Try to load Llama tokenizer
            # Use Llama-2 as default
            model_id = "meta-llama/Llama-2-7b-hf"
            if "llama-3" in self.model_name or "llama3" in self.model_name:
                model_id = "meta-llama/Meta-Llama-3-8B"
            
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            # Transformers not available or model not accessible
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using model-specific tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self._tokenizer is not None:
            return self._count_with_tokenizer(text)
        else:
            return self._estimate_tokens(text)

    def _count_with_tokenizer(self, text: str) -> int:
        """Count tokens using loaded tokenizer."""
        if self.model_family == ModelFamily.GPT or self.model_family == ModelFamily.CLAUDE:
            # tiktoken tokenizer
            return len(self._tokenizer.encode(text))
        elif self.model_family == ModelFamily.LLAMA:
            # Transformers tokenizer
            return len(self._tokenizer.encode(text))
        else:
            return self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count when tokenizer is not available.
        
        Uses heuristics based on character and word counts.
        Formula: average of character-based and word-based estimates.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Character-based estimate (1 token ≈ 4 characters for English)
        char_estimate = len(text) / 4.0
        
        # Word-based estimate (1 token ≈ 0.75 words)
        words = text.split()
        word_estimate = len(words) / 0.75
        
        # Average both methods for better accuracy
        estimate = (char_estimate + word_estimate) / 2
        
        return int(estimate)

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """
        Count tokens for multiple texts efficiently.
        
        Args:
            texts: List of texts to count tokens for
            
        Returns:
            List of token counts
        """
        return [self.count_tokens(text) for text in texts]

    def truncate_to_token_limit(
        self,
        text: str,
        max_tokens: int,
        truncation_strategy: Literal["end", "start", "middle"] = "end"
    ) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            truncation_strategy: Where to truncate ("end", "start", "middle")
            
        Returns:
            Truncated text
        """
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # Binary search for the right length
        if truncation_strategy == "end":
            return self._truncate_end(text, max_tokens)
        elif truncation_strategy == "start":
            return self._truncate_start(text, max_tokens)
        else:  # middle
            return self._truncate_middle(text, max_tokens)

    def _truncate_end(self, text: str, max_tokens: int) -> str:
        """Truncate from the end of text."""
        # Estimate character position
        ratio = max_tokens / self.count_tokens(text)
        estimated_chars = int(len(text) * ratio * 0.95)  # 95% to be safe
        
        # Binary search for exact position
        left, right = 0, len(text)
        result = text[:estimated_chars]
        
        while left < right:
            mid = (left + right + 1) // 2
            candidate = text[:mid]
            tokens = self.count_tokens(candidate)
            
            if tokens <= max_tokens:
                result = candidate
                left = mid
            else:
                right = mid - 1
        
        return result

    def _truncate_start(self, text: str, max_tokens: int) -> str:
        """Truncate from the start of text."""
        # Estimate character position
        ratio = max_tokens / self.count_tokens(text)
        estimated_chars = int(len(text) * (1 - ratio) * 1.05)  # Start position
        
        # Binary search for exact position
        left, right = 0, len(text)
        result = text[estimated_chars:]
        
        while left < right:
            mid = (left + right) // 2
            candidate = text[mid:]
            tokens = self.count_tokens(candidate)
            
            if tokens <= max_tokens:
                result = candidate
                right = mid
            else:
                left = mid + 1
        
        return result

    def _truncate_middle(self, text: str, max_tokens: int) -> str:
        """Truncate from the middle of text, keeping start and end."""
        # Keep first and last portions
        half_tokens = max_tokens // 2
        
        start_part = self._truncate_end(text, half_tokens)
        
        # Find where to start the end part
        remaining_tokens = max_tokens - self.count_tokens(start_part)
        end_part = self._truncate_start(text, remaining_tokens)
        
        return start_part + "\n[...]\n" + end_part


class TokenBudgetManager:
    """
    Manages token budgets for context windows.
    
    Features:
    - Track token usage across multiple components
    - Reserve tokens for system prompts, responses
    - Allocate remaining tokens to context
    """

    def __init__(
        self,
        total_tokens: int,
        system_tokens: int = 0,
        response_tokens: int = 0,
        safety_margin: int = 100
    ):
        """
        Initialize token budget manager.
        
        Args:
            total_tokens: Total context window size
            system_tokens: Tokens reserved for system prompt
            response_tokens: Tokens reserved for response
            safety_margin: Safety margin to avoid exceeding limits
        """
        self.total_tokens = total_tokens
        self.system_tokens = system_tokens
        self.response_tokens = response_tokens
        self.safety_margin = safety_margin
        self._used_tokens = 0

    @property
    def available_tokens(self) -> int:
        """Get available tokens for context."""
        reserved = self.system_tokens + self.response_tokens + self.safety_margin
        return max(0, self.total_tokens - reserved - self._used_tokens)

    @property
    def used_tokens(self) -> int:
        """Get currently used tokens."""
        return self._used_tokens

    def allocate(self, tokens: int) -> bool:
        """
        Try to allocate tokens from budget.
        
        Args:
            tokens: Number of tokens to allocate
            
        Returns:
            True if allocation successful, False otherwise
        """
        if tokens <= self.available_tokens:
            self._used_tokens += tokens
            return True
        return False

    def release(self, tokens: int):
        """
        Release allocated tokens back to budget.
        
        Args:
            tokens: Number of tokens to release
        """
        self._used_tokens = max(0, self._used_tokens - tokens)

    def reset(self):
        """Reset token usage to zero."""
        self._used_tokens = 0

    def get_budget_info(self) -> dict[str, int]:
        """
        Get detailed budget information.
        
        Returns:
            Dictionary with budget details
        """
        return {
            "total_tokens": self.total_tokens,
            "system_tokens": self.system_tokens,
            "response_tokens": self.response_tokens,
            "safety_margin": self.safety_margin,
            "used_tokens": self._used_tokens,
            "available_tokens": self.available_tokens,
            "reserved_tokens": self.system_tokens + self.response_tokens + self.safety_margin
        }


def get_model_context_window(model_name: str) -> int:
    """
    Get the context window size for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Context window size in tokens
    """
    model_lower = model_name.lower()
    
    # GPT models
    if "gpt-4-turbo" in model_lower or "gpt-4-1106" in model_lower:
        return 128000
    elif "gpt-4-32k" in model_lower:
        return 32768
    elif "gpt-4" in model_lower:
        return 8192
    elif "gpt-3.5-turbo-16k" in model_lower:
        return 16384
    elif "gpt-3.5" in model_lower or "turbo" in model_lower:
        return 4096
    
    # Claude models
    elif "claude-3-opus" in model_lower or "claude-3-sonnet" in model_lower:
        return 200000
    elif "claude-3-haiku" in model_lower:
        return 200000
    elif "claude-2" in model_lower:
        return 100000
    elif "claude" in model_lower:
        return 100000
    
    # Llama models
    elif "llama-3" in model_lower or "llama3" in model_lower:
        return 8192
    elif "llama-2" in model_lower or "llama2" in model_lower:
        return 4096
    elif "llama" in model_lower:
        return 4096
    
    # Default fallback
    else:
        return 4096


def create_token_counter(model_name: str) -> TokenCounter:
    """
    Factory function to create a token counter for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        TokenCounter instance
    """
    return TokenCounter(model_name)
