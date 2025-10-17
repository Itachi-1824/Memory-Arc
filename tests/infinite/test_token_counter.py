"""Unit tests for token counting functionality."""

import pytest
from core.infinite import (
    TokenCounter,
    TokenBudgetManager,
    ModelFamily,
    create_token_counter,
    get_model_context_window,
)


class TestTokenCounter:
    """Test suite for TokenCounter class."""

    def test_initialization_gpt(self):
        """Test token counter initialization for GPT models."""
        counter = TokenCounter("gpt-4")
        assert counter.model_name == "gpt-4"
        assert counter.model_family == ModelFamily.GPT

    def test_initialization_claude(self):
        """Test token counter initialization for Claude models."""
        counter = TokenCounter("claude-3-opus")
        assert counter.model_name == "claude-3-opus"
        assert counter.model_family == ModelFamily.CLAUDE

    def test_initialization_llama(self):
        """Test token counter initialization for Llama models."""
        counter = TokenCounter("llama-3")
        assert counter.model_name == "llama-3"
        assert counter.model_family == ModelFamily.LLAMA

    def test_initialization_unknown(self):
        """Test token counter initialization for unknown models."""
        counter = TokenCounter("unknown-model")
        assert counter.model_family == ModelFamily.UNKNOWN

    def test_detect_model_family_gpt_variants(self):
        """Test model family detection for various GPT model names."""
        test_cases = [
            ("gpt-4", ModelFamily.GPT),
            ("gpt-3.5-turbo", ModelFamily.GPT),
            ("GPT-4-turbo", ModelFamily.GPT),
            ("text-davinci-003", ModelFamily.GPT),
        ]
        
        for model_name, expected_family in test_cases:
            counter = TokenCounter(model_name)
            assert counter.model_family == expected_family

    def test_detect_model_family_claude_variants(self):
        """Test model family detection for various Claude model names."""
        test_cases = [
            ("claude-3-opus", ModelFamily.CLAUDE),
            ("claude-3-sonnet", ModelFamily.CLAUDE),
            ("claude-2", ModelFamily.CLAUDE),
            ("anthropic-claude", ModelFamily.CLAUDE),
        ]
        
        for model_name, expected_family in test_cases:
            counter = TokenCounter(model_name)
            assert counter.model_family == expected_family

    def test_detect_model_family_llama_variants(self):
        """Test model family detection for various Llama model names."""
        test_cases = [
            ("llama-3", ModelFamily.LLAMA),
            ("llama-2-7b", ModelFamily.LLAMA),
            ("meta-llama-3", ModelFamily.LLAMA),
        ]
        
        for model_name, expected_family in test_cases:
            counter = TokenCounter(model_name)
            assert counter.model_family == expected_family

    def test_count_tokens_empty_string(self):
        """Test token counting with empty string."""
        counter = TokenCounter("gpt-4")
        assert counter.count_tokens("") == 0

    def test_count_tokens_simple_text(self):
        """Test token counting with simple text."""
        counter = TokenCounter("gpt-4")
        text = "Hello, world!"
        tokens = counter.count_tokens(text)
        
        # Should return a reasonable token count
        assert tokens > 0
        assert tokens < 10  # "Hello, world!" is typically 3-4 tokens

    def test_count_tokens_longer_text(self):
        """Test token counting with longer text."""
        counter = TokenCounter("gpt-4")
        text = "This is a longer piece of text with multiple words and sentences. " * 10
        tokens = counter.count_tokens(text)
        
        # Should return a reasonable token count
        assert tokens > 50
        assert tokens < 500

    def test_count_tokens_consistency(self):
        """Test that token counting is consistent for same text."""
        counter = TokenCounter("gpt-4")
        text = "Consistent token counting test"
        
        count1 = counter.count_tokens(text)
        count2 = counter.count_tokens(text)
        
        assert count1 == count2

    def test_estimate_tokens_fallback(self):
        """Test token estimation when tokenizer is not available."""
        counter = TokenCounter("unknown-model")
        text = "This is a test sentence with several words."
        
        tokens = counter.count_tokens(text)
        
        # Should use estimation
        assert tokens > 0
        # Estimation should be reasonable (roughly 1 token per word)
        word_count = len(text.split())
        assert tokens >= word_count * 0.5
        assert tokens <= word_count * 2

    def test_estimate_tokens_character_based(self):
        """Test character-based token estimation."""
        counter = TokenCounter("unknown-model")
        
        # 100 characters should be roughly 25 tokens (1 token ≈ 4 chars)
        # For repeated single character, word count is 1, so estimate is lower
        text = "a" * 100
        tokens = counter.count_tokens(text)
        
        assert 10 < tokens < 40  # Allow variance for edge case

    def test_count_tokens_batch(self):
        """Test batch token counting."""
        counter = TokenCounter("gpt-4")
        texts = [
            "First text",
            "Second text with more words",
            "Third text"
        ]
        
        counts = counter.count_tokens_batch(texts)
        
        assert len(counts) == 3
        assert all(count > 0 for count in counts)
        # Second text should have more tokens
        assert counts[1] > counts[0]

    def test_count_tokens_batch_empty_list(self):
        """Test batch token counting with empty list."""
        counter = TokenCounter("gpt-4")
        counts = counter.count_tokens_batch([])
        assert counts == []

    def test_truncate_to_token_limit_no_truncation(self):
        """Test truncation when text is within limit."""
        counter = TokenCounter("gpt-4")
        text = "Short text"
        max_tokens = 100
        
        result = counter.truncate_to_token_limit(text, max_tokens)
        
        assert result == text

    def test_truncate_to_token_limit_end(self):
        """Test truncation from end of text."""
        counter = TokenCounter("gpt-4")
        text = "This is a long text that needs to be truncated. " * 20
        max_tokens = 50
        
        result = counter.truncate_to_token_limit(text, max_tokens, "end")
        
        # Result should be shorter
        assert len(result) < len(text)
        # Should start with original text
        assert text.startswith(result[:20])
        # Should be within token limit
        assert counter.count_tokens(result) <= max_tokens

    def test_truncate_to_token_limit_start(self):
        """Test truncation from start of text."""
        counter = TokenCounter("gpt-4")
        text = "This is a long text that needs to be truncated. " * 20
        max_tokens = 50
        
        result = counter.truncate_to_token_limit(text, max_tokens, "start")
        
        # Result should be shorter
        assert len(result) < len(text)
        # Should end with original text
        assert text.endswith(result[-20:])
        # Should be within token limit
        assert counter.count_tokens(result) <= max_tokens

    def test_truncate_to_token_limit_middle(self):
        """Test truncation from middle of text."""
        counter = TokenCounter("gpt-4")
        text = "Start of text. " + ("Middle content. " * 50) + " End of text."
        max_tokens = 50
        
        result = counter.truncate_to_token_limit(text, max_tokens, "middle")
        
        # Result should be shorter
        assert len(result) < len(text)
        # Should contain ellipsis marker
        assert "[...]" in result
        # Should be within token limit (with some margin for ellipsis)
        assert counter.count_tokens(result) <= max_tokens + 5


class TestTokenBudgetManager:
    """Test suite for TokenBudgetManager class."""

    def test_initialization(self):
        """Test budget manager initialization."""
        manager = TokenBudgetManager(
            total_tokens=1000,
            system_tokens=100,
            response_tokens=200,
            safety_margin=50
        )
        
        assert manager.total_tokens == 1000
        assert manager.system_tokens == 100
        assert manager.response_tokens == 200
        assert manager.safety_margin == 50
        assert manager.used_tokens == 0

    def test_available_tokens(self):
        """Test available tokens calculation."""
        manager = TokenBudgetManager(
            total_tokens=1000,
            system_tokens=100,
            response_tokens=200,
            safety_margin=50
        )
        
        # Available = 1000 - 100 - 200 - 50 = 650
        assert manager.available_tokens == 650

    def test_allocate_success(self):
        """Test successful token allocation."""
        manager = TokenBudgetManager(total_tokens=1000)
        
        success = manager.allocate(100)
        
        assert success is True
        assert manager.used_tokens == 100
        assert manager.available_tokens == 900 - 100  # 900 after safety margin

    def test_allocate_failure(self):
        """Test failed token allocation when budget exceeded."""
        manager = TokenBudgetManager(total_tokens=1000)
        
        # Try to allocate more than available
        success = manager.allocate(2000)
        
        assert success is False
        assert manager.used_tokens == 0  # Should not allocate

    def test_allocate_multiple(self):
        """Test multiple token allocations."""
        manager = TokenBudgetManager(total_tokens=1000)
        
        manager.allocate(100)
        manager.allocate(200)
        manager.allocate(150)
        
        assert manager.used_tokens == 450

    def test_release_tokens(self):
        """Test releasing allocated tokens."""
        manager = TokenBudgetManager(total_tokens=1000)
        
        manager.allocate(300)
        assert manager.used_tokens == 300
        
        manager.release(100)
        assert manager.used_tokens == 200

    def test_release_tokens_below_zero(self):
        """Test that releasing tokens doesn't go below zero."""
        manager = TokenBudgetManager(total_tokens=1000)
        
        manager.allocate(100)
        manager.release(200)  # Release more than allocated
        
        assert manager.used_tokens == 0  # Should not go negative

    def test_reset(self):
        """Test resetting token usage."""
        manager = TokenBudgetManager(total_tokens=1000)
        
        manager.allocate(300)
        manager.allocate(200)
        assert manager.used_tokens == 500
        
        manager.reset()
        assert manager.used_tokens == 0

    def test_get_budget_info(self):
        """Test getting detailed budget information."""
        manager = TokenBudgetManager(
            total_tokens=1000,
            system_tokens=100,
            response_tokens=200,
            safety_margin=50
        )
        manager.allocate(150)
        
        info = manager.get_budget_info()
        
        assert info["total_tokens"] == 1000
        assert info["system_tokens"] == 100
        assert info["response_tokens"] == 200
        assert info["safety_margin"] == 50
        assert info["used_tokens"] == 150
        assert info["available_tokens"] == 500  # 650 - 150
        assert info["reserved_tokens"] == 350  # 100 + 200 + 50

    def test_budget_with_no_reserves(self):
        """Test budget manager with no reserved tokens."""
        manager = TokenBudgetManager(total_tokens=1000)
        
        # Only safety margin (100) is reserved by default
        assert manager.available_tokens == 900

    def test_allocate_exact_available(self):
        """Test allocating exactly the available tokens."""
        manager = TokenBudgetManager(total_tokens=1000)
        available = manager.available_tokens
        
        success = manager.allocate(available)
        
        assert success is True
        assert manager.available_tokens == 0


class TestModelContextWindow:
    """Test suite for model context window detection."""

    def test_gpt4_turbo_context_window(self):
        """Test context window for GPT-4 Turbo."""
        window = get_model_context_window("gpt-4-turbo")
        assert window == 128000

    def test_gpt4_32k_context_window(self):
        """Test context window for GPT-4 32k."""
        window = get_model_context_window("gpt-4-32k")
        assert window == 32768

    def test_gpt4_context_window(self):
        """Test context window for GPT-4."""
        window = get_model_context_window("gpt-4")
        assert window == 8192

    def test_gpt35_turbo_16k_context_window(self):
        """Test context window for GPT-3.5 Turbo 16k."""
        window = get_model_context_window("gpt-3.5-turbo-16k")
        assert window == 16384

    def test_gpt35_turbo_context_window(self):
        """Test context window for GPT-3.5 Turbo."""
        window = get_model_context_window("gpt-3.5-turbo")
        assert window == 4096

    def test_claude3_opus_context_window(self):
        """Test context window for Claude 3 Opus."""
        window = get_model_context_window("claude-3-opus")
        assert window == 200000

    def test_claude3_sonnet_context_window(self):
        """Test context window for Claude 3 Sonnet."""
        window = get_model_context_window("claude-3-sonnet")
        assert window == 200000

    def test_claude3_haiku_context_window(self):
        """Test context window for Claude 3 Haiku."""
        window = get_model_context_window("claude-3-haiku")
        assert window == 200000

    def test_claude2_context_window(self):
        """Test context window for Claude 2."""
        window = get_model_context_window("claude-2")
        assert window == 100000

    def test_llama3_context_window(self):
        """Test context window for Llama 3."""
        window = get_model_context_window("llama-3")
        assert window == 8192

    def test_llama2_context_window(self):
        """Test context window for Llama 2."""
        window = get_model_context_window("llama-2")
        assert window == 4096

    def test_unknown_model_context_window(self):
        """Test context window for unknown model."""
        window = get_model_context_window("unknown-model")
        assert window == 4096  # Default fallback

    def test_case_insensitive_detection(self):
        """Test that model detection is case-insensitive."""
        window1 = get_model_context_window("GPT-4")
        window2 = get_model_context_window("gpt-4")
        assert window1 == window2


class TestCreateTokenCounter:
    """Test suite for token counter factory function."""

    def test_create_token_counter_gpt(self):
        """Test creating token counter for GPT model."""
        counter = create_token_counter("gpt-4")
        assert isinstance(counter, TokenCounter)
        assert counter.model_family == ModelFamily.GPT

    def test_create_token_counter_claude(self):
        """Test creating token counter for Claude model."""
        counter = create_token_counter("claude-3-opus")
        assert isinstance(counter, TokenCounter)
        assert counter.model_family == ModelFamily.CLAUDE

    def test_create_token_counter_llama(self):
        """Test creating token counter for Llama model."""
        counter = create_token_counter("llama-3")
        assert isinstance(counter, TokenCounter)
        assert counter.model_family == ModelFamily.LLAMA


class TestTokenCountingAccuracy:
    """Test suite for token counting accuracy."""

    def test_token_count_scales_with_text_length(self):
        """Test that token count increases with text length."""
        counter = TokenCounter("gpt-4")
        
        short_text = "Hello"
        medium_text = "Hello world, this is a test"
        long_text = "Hello world, this is a much longer test with many more words"
        
        short_tokens = counter.count_tokens(short_text)
        medium_tokens = counter.count_tokens(medium_text)
        long_tokens = counter.count_tokens(long_text)
        
        assert short_tokens < medium_tokens < long_tokens

    def test_token_count_code_vs_text(self):
        """Test token counting for code vs natural text."""
        counter = TokenCounter("gpt-4")
        
        code = "def function(x, y): return x + y"
        text = "define a function that takes x and y and returns their sum"
        
        code_tokens = counter.count_tokens(code)
        text_tokens = counter.count_tokens(text)
        
        # Both should have reasonable token counts
        assert code_tokens > 0
        assert text_tokens > 0

    def test_token_count_special_characters(self):
        """Test token counting with special characters."""
        counter = TokenCounter("gpt-4")
        
        text_with_special = "Hello! @#$% ^&*() []{}|\\:;\"'<>,.?/"
        tokens = counter.count_tokens(text_with_special)
        
        # Should handle special characters
        assert tokens > 0

    def test_token_count_unicode(self):
        """Test token counting with unicode characters."""
        counter = TokenCounter("gpt-4")
        
        unicode_text = "Hello 世界 🌍 café"
        tokens = counter.count_tokens(unicode_text)
        
        # Should handle unicode
        assert tokens > 0

    def test_token_count_whitespace(self):
        """Test token counting with various whitespace."""
        counter = TokenCounter("gpt-4")
        
        text_with_spaces = "word1    word2\n\nword3\tword4"
        tokens = counter.count_tokens(text_with_spaces)
        
        # Should handle whitespace appropriately
        assert tokens > 0


class TestTokenBudgetScenarios:
    """Test suite for realistic token budget scenarios."""

    def test_conversation_budget_scenario(self):
        """Test token budget for a conversation scenario."""
        # Typical GPT-4 conversation
        manager = TokenBudgetManager(
            total_tokens=8192,
            system_tokens=500,  # System prompt
            response_tokens=1000,  # Expected response
            safety_margin=100
        )
        
        # Should have tokens available for context
        assert manager.available_tokens > 5000
        
        # Simulate adding conversation history
        manager.allocate(2000)  # Previous messages
        assert manager.available_tokens > 3000

    def test_code_generation_budget_scenario(self):
        """Test token budget for code generation scenario."""
        # Code generation with large context
        manager = TokenBudgetManager(
            total_tokens=128000,  # GPT-4 Turbo
            system_tokens=1000,
            response_tokens=4000,  # Large code response
            safety_margin=500
        )
        
        # Should have plenty of tokens for code context
        assert manager.available_tokens > 100000
        
        # Simulate adding codebase context
        manager.allocate(50000)
        assert manager.available_tokens > 50000

    def test_budget_exhaustion_scenario(self):
        """Test handling of budget exhaustion."""
        manager = TokenBudgetManager(total_tokens=1000)
        
        # Allocate until exhausted
        available = manager.available_tokens
        manager.allocate(available)
        
        # Should not be able to allocate more
        assert manager.allocate(1) is False
        
        # Release some tokens
        manager.release(100)
        
        # Should now be able to allocate again
        assert manager.allocate(50) is True
