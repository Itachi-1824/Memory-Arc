"""Pollinations AI adapter for Memory-Arc."""

import logging
import requests
from typing import Any

from .ai_adapter import AIAdapter

logger = logging.getLogger(__name__)


class PollinationsAdapter(AIAdapter):
    """Adapter for Pollinations AI API (OpenAI-compatible format)."""
    
    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini",
        base_url: str = "https://text.pollinations.ai/openai",
        timeout: int = 30,
        **kwargs
    ):
        """
        Initialize Pollinations adapter.
        
        Args:
            api_key: API key for Pollinations (required)
            model: Model name (default: "gemini")
            base_url: Base URL for API (default: Pollinations OpenAI endpoint)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        
        logger.info(f"Initialized PollinationsAdapter with model: {model}")
    
    def _call_api(self, messages: list[dict[str, str]]) -> str | None:
        """
        Call Pollinations API with OpenAI format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Response text or None on error
        """
        try:
            headers = {
                "Content-Type": "application/json",
            }
            
            # Add Bearer token if provided
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(
                self.base_url,
                json={
                    "messages": messages,
                    "model": self.model,
                    "stream": False
                },
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    logger.error(f"Unexpected response format: {data}")
                    return None
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to call Pollinations API: {e}")
            return None
    
    async def summarize_conversation(self, messages: list[dict]) -> str | None:
        """
        Summarize a conversation using Pollinations AI.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Summary text or None on error
        """
        if not messages:
            return None
        
        # Build prompt for summarization
        conversation = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages
            if msg.get('content')
        ])
        
        prompt_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates concise summaries of conversations. "
                           "Focus on key points, decisions, and important information."
            },
            {
                "role": "user",
                "content": f"Please summarize this conversation concisely:\n\n{conversation}"
            }
        ]
        
        return self._call_api(prompt_messages)
    
    async def extract_facts(self, messages: list[dict]) -> list[dict]:
        """
        Extract important facts from conversation using Pollinations AI.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            List of extracted facts as dicts
        """
        if not messages:
            return []
        
        # Build prompt for fact extraction
        conversation = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages
            if msg.get('content')
        ])
        
        prompt_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts important facts from conversations. "
                           "Return facts as a simple list, one per line, without numbering."
            },
            {
                "role": "user",
                "content": f"Extract the key facts from this conversation:\n\n{conversation}"
            }
        ]
        
        response = self._call_api(prompt_messages)
        
        if not response:
            return []
        
        # Parse response into fact list
        facts = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove common list markers
            line = line.lstrip('•-*123456789. ')
            if line:
                facts.append({
                    "type": "extracted_fact",
                    "content": line,
                    "source": "pollinations_ai"
                })
        
        return facts
    
    async def score_importance(self, text: str) -> int:
        """
        Score the importance of text using Pollinations AI.
        
        Args:
            text: Text to score
            
        Returns:
            Importance score from 1 (low) to 10 (high)
        """
        if not text or not text.strip():
            return 5  # Default score
        
        prompt_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that rates the importance of information. "
                           "Respond with only a number from 1 (not important) to 10 (very important)."
            },
            {
                "role": "user",
                "content": f"Rate the importance of this information from 1-10:\n\n{text}"
            }
        ]
        
        response = self._call_api(prompt_messages)
        
        if not response:
            return 5  # Default score on error
        
        # Extract number from response
        try:
            # Try to find a number in the response
            import re
            numbers = re.findall(r'\b([1-9]|10)\b', response)
            if numbers:
                score = int(numbers[0])
                return min(max(score, 1), 10)  # Clamp between 1-10
        except Exception as e:
            logger.error(f"Failed to parse importance score: {e}")
        
        return 5  # Default score
