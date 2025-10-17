"""Example usage of token counting for multiple models."""

from core.infinite import (
    TokenCounter,
    TokenBudgetManager,
    create_token_counter,
    get_model_context_window,
)


def basic_token_counting():
    """Demonstrate basic token counting."""
    print("=== Basic Token Counting ===\n")
    
    # Create token counter for GPT-4
    counter = TokenCounter("gpt-4")
    
    # Count tokens in text
    text = "Hello, world! This is a test of token counting."
    tokens = counter.count_tokens(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}\n")
    
    # Count tokens in longer text
    long_text = """
    This is a longer piece of text that demonstrates token counting.
    Token counting is essential for managing context windows in AI models.
    Different models have different tokenization schemes.
    """
    tokens = counter.count_tokens(long_text)
    print(f"Long text tokens: {tokens}\n")


def multi_model_comparison():
    """Compare token counting across different models."""
    print("=== Multi-Model Token Counting ===\n")
    
    text = "The quick brown fox jumps over the lazy dog."
    
    models = ["gpt-4", "claude-3-opus", "llama-3", "unknown-model"]
    
    for model_name in models:
        counter = create_token_counter(model_name)
        tokens = counter.count_tokens(text)
        context_window = get_model_context_window(model_name)
        
        print(f"Model: {model_name}")
        print(f"  Tokens: {tokens}")
        print(f"  Context window: {context_window:,}")
        print(f"  Model family: {counter.model_family.value}\n")


def batch_token_counting():
    """Demonstrate batch token counting."""
    print("=== Batch Token Counting ===\n")
    
    counter = TokenCounter("gpt-4")
    
    texts = [
        "First message in conversation",
        "Second message with more content and details",
        "Third message",
        "Fourth message with even more content to demonstrate varying lengths",
    ]
    
    counts = counter.count_tokens_batch(texts)
    
    print("Conversation messages:")
    for i, (text, count) in enumerate(zip(texts, counts), 1):
        print(f"  Message {i}: {count} tokens - {text[:50]}...")
    
    total_tokens = sum(counts)
    print(f"\nTotal conversation tokens: {total_tokens}")


def text_truncation():
    """Demonstrate text truncation to fit token limits."""
    print("=== Text Truncation ===\n")
    
    counter = TokenCounter("gpt-4")
    
    long_text = """
    This is a very long piece of text that needs to be truncated to fit within
    a specific token limit. Token limits are important when working with AI models
    because they have fixed context windows. If your text exceeds the limit,
    you need to truncate it intelligently. This example demonstrates three
    different truncation strategies: truncating from the end, from the start,
    or from the middle while preserving both the beginning and end.
    """ * 5
    
    original_tokens = counter.count_tokens(long_text)
    print(f"Original text: {original_tokens} tokens\n")
    
    max_tokens = 50
    
    # Truncate from end
    truncated_end = counter.truncate_to_token_limit(long_text, max_tokens, "end")
    tokens_end = counter.count_tokens(truncated_end)
    print(f"Truncated from end: {tokens_end} tokens")
    print(f"Preview: {truncated_end[:100]}...\n")
    
    # Truncate from start
    truncated_start = counter.truncate_to_token_limit(long_text, max_tokens, "start")
    tokens_start = counter.count_tokens(truncated_start)
    print(f"Truncated from start: {tokens_start} tokens")
    print(f"Preview: ...{truncated_start[-100:]}\n")
    
    # Truncate from middle
    truncated_middle = counter.truncate_to_token_limit(long_text, max_tokens, "middle")
    tokens_middle = counter.count_tokens(truncated_middle)
    print(f"Truncated from middle: {tokens_middle} tokens")
    print(f"Preview: {truncated_middle[:80]}...\n")


def token_budget_management():
    """Demonstrate token budget management."""
    print("=== Token Budget Management ===\n")
    
    # Create budget manager for GPT-4
    manager = TokenBudgetManager(
        total_tokens=8192,
        system_tokens=500,  # Reserve for system prompt
        response_tokens=1000,  # Reserve for model response
        safety_margin=100
    )
    
    print("Initial budget:")
    info = manager.get_budget_info()
    for key, value in info.items():
        print(f"  {key}: {value:,}")
    print()
    
    # Simulate adding conversation context
    print("Adding conversation context...")
    manager.allocate(2000)
    print(f"  Allocated: 2000 tokens")
    print(f"  Available: {manager.available_tokens:,} tokens\n")
    
    # Add more context
    print("Adding code context...")
    manager.allocate(3000)
    print(f"  Allocated: 3000 tokens")
    print(f"  Available: {manager.available_tokens:,} tokens\n")
    
    # Try to allocate more than available
    print("Trying to allocate 5000 tokens...")
    success = manager.allocate(5000)
    print(f"  Success: {success}")
    print(f"  Available: {manager.available_tokens:,} tokens\n")
    
    # Release some tokens
    print("Releasing 1000 tokens...")
    manager.release(1000)
    print(f"  Available: {manager.available_tokens:,} tokens\n")


def context_window_planning():
    """Demonstrate planning for different model context windows."""
    print("=== Context Window Planning ===\n")
    
    models = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "claude-3-opus",
        "llama-3",
    ]
    
    # Sample content sizes
    system_prompt = 500
    response_reserve = 1000
    conversation_history = 2000
    code_context = 5000
    
    print("Planning token allocation for different models:\n")
    
    for model_name in models:
        context_window = get_model_context_window(model_name)
        
        manager = TokenBudgetManager(
            total_tokens=context_window,
            system_tokens=system_prompt,
            response_tokens=response_reserve,
        )
        
        # Try to allocate conversation and code
        can_fit_conversation = manager.allocate(conversation_history)
        can_fit_code = manager.allocate(code_context) if can_fit_conversation else False
        
        print(f"{model_name}:")
        print(f"  Context window: {context_window:,} tokens")
        print(f"  Can fit conversation: {can_fit_conversation}")
        print(f"  Can fit code context: {can_fit_code}")
        print(f"  Remaining: {manager.available_tokens:,} tokens\n")
        
        manager.reset()


def code_token_counting():
    """Demonstrate token counting for code."""
    print("=== Code Token Counting ===\n")
    
    counter = TokenCounter("gpt-4")
    
    code_samples = {
        "Python function": """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
""",
        "JavaScript class": """
class UserManager {
    constructor() {
        this.users = [];
    }
    
    addUser(user) {
        this.users.push(user);
    }
    
    getUser(id) {
        return this.users.find(u => u.id === id);
    }
}
""",
        "SQL query": """
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2024-01-01'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 5
ORDER BY order_count DESC;
""",
    }
    
    for name, code in code_samples.items():
        tokens = counter.count_tokens(code)
        lines = len(code.strip().split('\n'))
        print(f"{name}:")
        print(f"  Lines: {lines}")
        print(f"  Tokens: {tokens}")
        print(f"  Tokens per line: {tokens/lines:.1f}\n")


def main():
    """Run all examples."""
    basic_token_counting()
    print("\n" + "="*60 + "\n")
    
    multi_model_comparison()
    print("\n" + "="*60 + "\n")
    
    batch_token_counting()
    print("\n" + "="*60 + "\n")
    
    text_truncation()
    print("\n" + "="*60 + "\n")
    
    token_budget_management()
    print("\n" + "="*60 + "\n")
    
    context_window_planning()
    print("\n" + "="*60 + "\n")
    
    code_token_counting()


if __name__ == "__main__":
    main()
