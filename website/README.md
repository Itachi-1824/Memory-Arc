# 🧠 Memory-Arc Web Chat - Test Interface

A simple web interface to test the Memory-Arc infinite context memory system with AI chat capabilities.

## ✨ Features

- 💬 **Real-time AI Chat** - Powered by Pollinations AI (Gemini model)
- 🧠 **Infinite Context Memory** - Stores all conversations with semantic search
- 📊 **Live Metrics Dashboard** - Monitor system performance in real-time
- 🔍 **Memory Retrieval Panel** - See what context the AI is using
- 💾 **Persistent Storage** - Conversations saved across restarts
- ⚡ **Fast Retrieval** - Last 50 messages + semantic search

## 🚀 Quick Start

### 1. Setup (First Time Only)

**Windows:**
```cmd
setup.bat
```

**Manual Setup:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
cd ..
pip install -r requirements.txt
cd website
pip install -r requirements.txt
```

### 2. Start the Server

**Windows:**
```cmd
start.bat
```

**Manual:**
```cmd
venv\Scripts\activate.bat
python server.py
```

### 3. Open the Website

1. Open `index.html` in your web browser
2. Enter your Pollinations API token when prompted
3. Start chatting!

## 🔑 Getting an API Token

1. Visit [https://pollinations.ai/](https://pollinations.ai/)
2. Sign up or log in
3. Get your API token from account settings
4. Enter it in the web interface

## 📖 How It Works

### Architecture

```
Browser (index.html)
    ↓
Flask Server (server.py)
    ↓
Infinite Context Engine
    ├─ Document Store (SQLite) - Stores all messages
    ├─ Vector Store (Qdrant) - Semantic embeddings
    └─ Embedding Cache (LMDB) - Fast lookups
    ↓
Pollinations AI (Gemini)
```

### Memory System

1. **Short-Term Memory (STM)**: Last 50 messages sent to AI
2. **Semantic Retrieval**: Finds relevant past conversations
3. **Persistent Storage**: All data saved in `data/web_chat/`

### Context Building

For each message, the AI receives:
- Last 50 conversation messages (chronological order)
- Current user message

Plus semantic retrieval happens in background for the memory panel.

## 📁 File Structure

```
website/
├── index.html          # Main web interface
├── style.css           # Styling
├── script.js           # Frontend logic
├── server.py           # Flask backend
├── requirements.txt    # Python dependencies
├── setup.bat          # Windows setup script
├── start.bat          # Quick start script
└── README.md          # This file
```

## 🛠️ Configuration

The system uses these settings (in `server.py`):

- **Storage Path**: `./data/web_chat` (persists across restarts)
- **Model**: Gemini via Pollinations
- **STM Size**: 50 messages
- **Retrieval**: 10 most relevant memories
- **Caching**: Enabled for embeddings

## 🐛 Troubleshooting

### "Virtual environment not found"
**Solution**: Run `setup.bat` first

### "Module not found" errors
**Solution**: 
```cmd
venv\Scripts\activate.bat
pip install -r requirements.txt
cd ..
pip install -r requirements.txt
```

### Server won't start
**Solution**: 
- Make sure Python 3.8+ is installed
- Check port 8000 is not in use
- Try running manually: `python server.py`

### "Failed to initialize"
**Solution**: Check your API token is correct

### API returns 402 error
**Solution**: Your API token tier may not support the model. Get a valid token from Pollinations.

### Memories not persisting
**Solution**: Check that `data/web_chat/` folder has write permissions

## 🧪 Testing the System

Try these to test the memory system:

1. **Basic Memory**:
   - "My name is Alice"
   - (later) "What's my name?"

2. **Context Retention**:
   - Have a long conversation
   - Ask about something mentioned 10+ messages ago

3. **Semantic Search**:
   - Talk about different topics
   - Ask a question related to an earlier topic
   - Check the memory panel to see what was retrieved

## 📊 Metrics Explained

- **Total Memories**: Number of stored messages
- **Queries**: Number of retrieval operations
- **Avg Latency**: Average response time
- **Cache Hit Rate**: Efficiency of embedding cache

## 🔒 Security Notes

⚠️ **For Testing Only** - This is a development server:
- API tokens stored in browser localStorage
- No authentication
- CORS enabled for all origins
- Not suitable for production use

## 🤝 Contributing

Found a bug or want to improve the interface? Feel free to:
1. Fork the repository
2. Make your changes
3. Submit a pull request

## 📄 License

Part of the Memory-Arc project - Apache License 2.0

---

**Built with ❤️ using Memory-Arc's Infinite Context System**
