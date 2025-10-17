"""Flask server for Memory-Arc web chat interface."""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import time

# Import from core.infinite package
from core.infinite import (
    InfiniteContextEngine,
    InfiniteContextConfig,
    MemoryType,
)

app = Flask(__name__)
CORS(app)

# Global engine instance
engine = None
api_key = None  # Will be set from user input

# Pollinations AI configuration
POLLINATIONS_BASE_URL = "https://text.pollinations.ai/openai"
POLLINATIONS_MODEL = "openai"  # Use "openai" as the model name for Pollinations


def get_embedding(text: str) -> list[float]:
    """Simple hash-based embedding for demo purposes."""
    # In production, use a real embedding model
    hash_val = hash(text)
    return [float((hash_val >> i) & 0xFF) / 255.0 for i in range(1536)]


async def call_pollinations_ai(messages: list[dict]) -> str:
    """Call Pollinations AI API using OpenAI format."""
    try:
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add Bearer token - REQUIRED for Pollinations
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            print(f"[DEBUG] Using API key: {api_key[:10]}...")
        else:
            print("[ERROR] No API key available!")
            return "Error: No API key configured"
        
        url = POLLINATIONS_BASE_URL
        payload = {
            "messages": messages,
            "model": POLLINATIONS_MODEL,
            "stream": False
        }
        
        print(f"[DEBUG] Calling {url}")
        print(f"[DEBUG] Headers: {headers}")
        
        # Call the API using OpenAI format
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        print(f"[DEBUG] Response status: {response.status_code}")
        print(f"[DEBUG] Response headers: {dict(response.headers)}")
        print(f"[DEBUG] Response body: {response.text[:1000]}")
        
        if response.status_code == 200:
            data = response.json()
            # Extract message from OpenAI format response
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                print(f"[ERROR] Unexpected response format: {data}")
                return "Error: Unexpected response format"
        else:
            print(f"[ERROR] ===== API ERROR =====")
            print(f"[ERROR] Status: {response.status_code}")
            print(f"[ERROR] URL: {url}")
            print(f"[ERROR] Payload: {payload}")
            print(f"[ERROR] Headers: {headers}")
            print(f"[ERROR] Response: {response.text}")
            print(f"[ERROR] =====================")
            return f"Error: API returned status {response.status_code} - {response.text}"
            
    except Exception as e:
        print(f"[ERROR] Exception calling AI: {e}")
        import traceback
        traceback.print_exc()
        return f"Error calling AI: {str(e)}"


@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the memory engine with Pollinations AI adapter."""
    global engine, api_key
    
    try:
        # Get API key from request
        data = request.json
        user_api_key = data.get('api_key', '').strip()
        
        if not user_api_key:
            return jsonify({
                'status': 'error',
                'message': 'API key is required'
            }), 400
        
        # Set global API key
        api_key = user_api_key
        
        # If engine already exists, just return success
        if engine is not None:
            return jsonify({
                'status': 'success',
                'message': 'Engine already initialized'
            })
        
        # Import the adapter registry
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from adapters.registry import AdapterRegistry
        from adapters.pollinations_adapter import PollinationsAdapter
        
        # Register Pollinations adapter if not already registered
        try:
            AdapterRegistry.register("pollinations", PollinationsAdapter)
        except:
            pass  # Already registered
        
        # Use fixed storage path for persistence across restarts
        storage_path = "./data/web_chat"
        
        # Create configuration for infinite context engine
        # The engine handles memory storage and retrieval
        # The Pollinations API (via api_key) is used for chat responses
        config = InfiniteContextConfig(
            storage_path=storage_path,
            enable_caching=True,
            enable_code_tracking=False,
            use_spacy=False,
            model_name="gemini",
        )
        
        # Initialize engine
        engine = InfiniteContextEngine(
            config=config,
            embedding_fn=get_embedding
        )
        
        # Run async initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(engine.initialize())
        loop.close()
        
        return jsonify({
            'status': 'success',
            'message': 'Engine initialized successfully with Pollinations AI'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat message."""
    global engine
    
    if engine is None:
        return jsonify({'error': 'Engine not initialized'}), 400
    
    try:
        data = request.json
        user_message = data.get('message', '')
        context_id = data.get('context_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Run async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Score importance of user message (simple heuristic)
        # Longer messages, questions, and messages with keywords get higher scores
        user_importance = 5  # Default
        if len(user_message) > 100:
            user_importance += 1
        if '?' in user_message:
            user_importance += 1
        if any(word in user_message.lower() for word in ['important', 'remember', 'always', 'never']):
            user_importance += 2
        user_importance = min(user_importance, 10)
        
        # Add user message to memory
        loop.run_until_complete(engine.add_memory(
            content=user_message,
            memory_type=MemoryType.CONVERSATION,
            context_id=context_id,
            importance=user_importance,
            metadata={'role': 'user'}
        ))
        
        # Get last 50 STM messages (recent conversation history)
        stm_memories = loop.run_until_complete(engine.document_store.query_memories(
            context_id=context_id,
            memory_type=MemoryType.CONVERSATION,
            limit=50,
            offset=0
        ))
        
        # Sort STM by time (oldest first) to maintain conversation order
        stm_memories.sort(key=lambda m: m.created_at)
        
        # Build context from STM
        context_messages = []
        for memory in stm_memories:
            role = memory.metadata.get('role', 'user')
            context_messages.append({
                'role': role,
                'content': memory.content
            })
        
        # Also retrieve semantically relevant memories for additional context
        retrieval_result = loop.run_until_complete(engine.retrieve(
            query=user_message,
            context_id=context_id,
            max_results=10
        ))
        
        # Add current message
        context_messages.append({
            'role': 'user',
            'content': user_message
        })
        
        # Call AI
        ai_response = loop.run_until_complete(call_pollinations_ai(context_messages))
        
        # Score importance of AI response (simple heuristic)
        ai_importance = 5  # Default
        if len(ai_response) > 200:
            ai_importance += 1
        if any(word in ai_response.lower() for word in ['important', 'note', 'remember', 'warning', 'error']):
            ai_importance += 1
        ai_importance = min(ai_importance, 10)
        
        # Store AI response in memory
        loop.run_until_complete(engine.add_memory(
            content=ai_response,
            memory_type=MemoryType.CONVERSATION,
            context_id=context_id,
            importance=ai_importance,
            metadata={'role': 'assistant'}
        ))
        
        loop.close()
        
        # Format retrieved memories for display
        retrieved_memories = [
            {
                'content': mem.content,
                'type': mem.memory_type.value,
                'importance': mem.importance,
                'created_at': mem.created_at
            }
            for mem in retrieval_result.memories[:5]
        ]
        
        return jsonify({
            'response': ai_response,
            'retrieved_memories': retrieved_memories,
            'retrieval_time_ms': retrieval_result.retrieval_time_ms
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics."""
    global engine
    
    if engine is None:
        # Return default metrics if not initialized yet
        return jsonify({
            'total_memories': 0,
            'total_queries': 0,
            'avg_query_latency_ms': 0,
            'cache_hit_rate': 0
        })
    
    try:
        metrics = engine.get_metrics()
        return jsonify(metrics.to_dict())
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    global engine
    
    if engine is None:
        return jsonify({'status': 'not_initialized'}), 503
    
    try:
        health_status = engine.get_health_status()
        return jsonify(health_status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*60)
    print("Memory-Arc Web Chat Server")
    print("="*60)
    print("\nStarting server on http://localhost:8000")
    print("Open index.html in your browser to use the chat interface")
    print("\nAPI Configuration:")
    print("  Base URL: https://text.pollinations.ai/")
    print("  Model: gemini")
    print("  You'll be prompted for API key when you open the website")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
