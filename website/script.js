// Configuration
const API_BASE_URL = 'http://localhost:8000';
let apiKey = '';

// Save API configuration
function saveConfig() {
    apiKey = document.getElementById('apiKey').value.trim();
    
    if (!apiKey) {
        alert('Please enter an API token');
        return;
    }
    
    // Store in localStorage
    localStorage.setItem('apiKey', apiKey);
    
    // Hide config, show main content
    document.getElementById('apiConfig').style.display = 'none';
    document.getElementById('mainContent').style.display = 'grid';
    
    // Initialize
    initializeChat();
}

// Initialize chat
async function initializeChat() {
    try {
        const response = await fetch(`${API_BASE_URL}/initialize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ api_key: apiKey })
        });
        
        if (!response.ok) {
            throw new Error('Failed to initialize');
        }
        
        const data = await response.json();
        console.log('Initialized:', data);
        
        // Load initial metrics
        updateMetrics();
        
    } catch (error) {
        console.error('Initialization error:', error);
        alert('Failed to initialize. Please check your API key and try again.');
        document.getElementById('apiConfig').style.display = 'block';
        document.getElementById('mainContent').style.display = 'none';
    }
}

// Send message
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Disable input
    input.disabled = true;
    document.getElementById('sendBtn').disabled = true;
    
    // Add user message to chat
    addMessageToChat('user', message);
    input.value = '';
    
    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                context_id: 'web_user_1'
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to send message');
        }
        
        const data = await response.json();
        
        // Add assistant response
        addMessageToChat('assistant', data.response);
        
        // Update retrieved memories
        if (data.retrieved_memories) {
            displayMemories(data.retrieved_memories);
        }
        
        // Update metrics
        updateMetrics();
        
    } catch (error) {
        console.error('Send message error:', error);
        addMessageToChat('system', 'Error: Failed to send message. Please try again.');
    } finally {
        // Re-enable input
        input.disabled = false;
        document.getElementById('sendBtn').disabled = false;
        input.focus();
    }
}

// Add message to chat UI
function addMessageToChat(role, content) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.textContent = content;
    messageDiv.appendChild(contentDiv);
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString();
    messageDiv.appendChild(timeDiv);
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Display retrieved memories
function displayMemories(memories) {
    const memoryList = document.getElementById('memoryList');
    memoryList.innerHTML = '';
    
    if (!memories || memories.length === 0) {
        memoryList.innerHTML = '<p class="empty-state">No memories retrieved for this query.</p>';
        return;
    }
    
    memories.forEach(memory => {
        const memoryDiv = document.createElement('div');
        memoryDiv.className = 'memory-item';
        
        memoryDiv.innerHTML = `
            <div class="memory-type">${memory.type || 'conversation'}</div>
            <div class="memory-content">${truncate(memory.content, 150)}</div>
            <div class="memory-meta">
                <span>Importance: ${memory.importance || 5}/10</span>
                <span>${formatDate(memory.created_at)}</span>
            </div>
        `;
        
        memoryList.appendChild(memoryDiv);
    });
}

// Update metrics
async function updateMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        if (!response.ok) return;
        
        const metrics = await response.json();
        
        document.getElementById('totalMemories').textContent = metrics.total_memories || 0;
        document.getElementById('totalQueries').textContent = metrics.total_queries || 0;
        document.getElementById('avgLatency').textContent = 
            `${(metrics.avg_query_latency_ms || 0).toFixed(0)}ms`;
        document.getElementById('cacheHitRate').textContent = 
            `${((metrics.cache_hit_rate || 0) * 100).toFixed(1)}%`;
            
    } catch (error) {
        console.error('Failed to update metrics:', error);
    }
}

// Handle Enter key in textarea
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Utility functions
function truncate(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function formatDate(timestamp) {
    if (!timestamp) return 'Unknown';
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

// Check for saved API key on load
window.addEventListener('DOMContentLoaded', () => {
    const savedKey = localStorage.getItem('apiKey');
    if (savedKey) {
        document.getElementById('apiKey').value = savedKey;
        apiKey = savedKey;
        document.getElementById('apiConfig').style.display = 'none';
        document.getElementById('mainContent').style.display = 'grid';
        initializeChat();
    }
});

// Auto-update metrics every 5 seconds
setInterval(updateMetrics, 5000);
