# 🎤 Voice Assistant v4 — Azure OpenAI + Local UI

A production-ready voice assistant built with **LangChain** and **Azure OpenAI**, featuring a **local FastAPI + WebSocket UI** with robust browser-based voice interruption.

## ✨ Features

* **🎙️ Real-time Voice Input** — Browser Speech Recognition with live interim transcripts
* **🔊 Automatic Voice Output** — Web Speech API `speechSynthesis` for natural TTS
* **⚡ Streaming Responses** — Tokens stream to the UI as they're generated via `astream_events`
* **🤝 Hands-free Conversation** — Continuous mode with auto-listen after each response
* **🛑 Voice Interruption** — Speak anytime to interrupt the assistant mid-sentence
  - Dual-recognition system (main + background) for reliable detection
  - Echo filtering to prevent TTS feedback from triggering false interruptions
  - Configurable sensitivity (min words, detection delay)
  - Auto-submit fallback if recognition doesn't restart after interruption
* **🔄 Continuous Mode** — Automatic back-and-forth without clicking anything
* **🛑 Stop Words** — Say "stop", "exit", or "quit" to end conversations naturally
* **💬 Conversation Memory** — Multi-turn context via LangGraph `InMemorySaver`
* **🧠 Azure OpenAI** — Powered by GPT-4 / GPT-4o via `AzureChatOpenAI`
* **🛠️ Built-in Tools** — DuckDuckGo web search, calculator, current time
* **📊 Structured Metrics** — First-token latency, tokens/sec, response time logged as JSON
* **🐛 Debug Mode** — Real-time speech detection logging for tuning interruption thresholds
* **🔌 LangSmith Tracing** — Optional distributed tracing (opt-in via env var)

## 📁 Project Structure

```
voice_agent_app/v4/
├── app.py              # FastAPI server + embedded HTML/JS UI
├── agent.py            # LangChain VoiceAgent with Azure OpenAI
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # This file
```

## 🏗️ Architecture

### Overview

```
Browser (Local)                     FastAPI Server (Local)              Azure OpenAI (Cloud)
─────────────────────────────────   ───────────────────────────────    ──────────────────────
SpeechRecognition API ──ws:msg──→   FastAPI WebSocket ──HTTP─────→     Azure OpenAI Service
        ↓                                   ↓                                    ↓
    Transcript                  VoiceAgent.stream_response()          GPT-4 / GPT-4o
        ↓                                   ↓                                    ↓
    WebSocket ←──────────────   astream_events (tokens)   ←──────────  Streamed tokens
        ↓                           (astream_events v2)
    Display Text + TTS
        ↓
speechSynthesis.speak()
        ↓
    Audio Output

Background SpeechRecognition ──interrupt──→  stop TTS + submit user text
```

### Frontend: Dual-Recognition Voice Interruption

The browser runs **two** `SpeechRecognition` instances simultaneously:

| Instance | When Active | Purpose |
|----------|-------------|---------|
| **Main recognition** | User is speaking / idle | Captures user input, submits to backend |
| **Background recognition** | Assistant TTS is playing | Listens for interruption speech, filters TTS echo |

**Interruption flow:**

1. Assistant TTS starts → background recognition activates after a configurable delay (default 1200ms to avoid TTS echo)
2. User speaks → background accumulates speech with confidence filtering (>0.75)
3. Echo detection filters out speech that matches the TTS output
4. Once 3+ words detected → 500ms capture window → `handleVoiceInterruption()` fires
5. TTS cancelled, background stopped, main recognition starts after 150ms settle delay
6. User can continue speaking → main recognition captures additional text
7. After 1.5s silence (or 2.5s auto-submit), combined text (`interruption + additional`) is sent to backend
8. If main recognition fails to start, interruption text is submitted directly as fallback

**Key settings (adjustable in UI):**

| Setting | Default | Range | Purpose |
|---------|---------|-------|---------|
| Min words to interrupt | 3 | 1–8 | Minimum words to trigger interruption |
| Detection delay | 1200ms | 300–2500ms | Initial silence after TTS start before listening begins |

### Backend: LangChain Agent

The `VoiceAgent` class wraps a LangChain `create_agent` with:

- **Model**: `AzureChatOpenAI` with `streaming=True`
- **Memory**: LangGraph `InMemorySaver` — each WebSocket session gets a unique thread ID
- **Tools**: `get_current_time`, `calculate`, `web_search` (DuckDuckGo)
- **Streaming**: `agent.astream_events(..., version="v2")` — filters `on_chat_model_stream` events, yields token content directly
- **Metrics**: First-token latency, total response time, token count, tokens/sec — logged as structured JSON

**Streaming pipeline:**

```python
async for event in self.agent.astream_events(
    {"messages": [HumanMessage(content=user_input)]},
    {"configurable": {"thread_id": thread_id}},
    version="v2",
):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk.content:
            yield chunk.content  # streamed to UI via WebSocket
```

No manual chunk-joining logic — the LLM emits properly spaced tokens, so chunks are concatenated directly. A light `_normalize_for_voice()` pass collapses multiple whitespace at the end.

### WebSocket Protocol

| Direction | Type | Payload |
|-----------|------|---------|
| Server → Client | `connected` | `{session_id, stats}` |
| Client → Server | `message` | `{type: "message", text: "..."}` |
| Server → Client | `transcript` | `{text: "..."}` (echo of received message) |
| Server → Client | `agent_chunk` | `{text: "..."}` (streamed token) |
| Server → Client | `agent_complete` | `{}` |
| Server → Client | `error` | `{message: "..."}` |
| Client → Server | `clear_history` | `{}` |
| Client → Server | `get_stats` | `{}` |
| Server → Client | `history_cleared` | `{}` |
| Server → Client | `stats` | `{model_type, active_sessions, memory_type}` |

### Key Differences from v2

| Aspect | v2 (Databricks) | v4 (Local + Azure OpenAI) |
|--------|-----------------|---------------------------|
| **UI hosting** | Databricks Apps | Local FastAPI |
| **Model** | Databricks foundation model endpoint | Azure OpenAI (`AzureChatOpenAI`) |
| **Agent** | Simple chain | LangGraph `create_agent` with tools |
| **Interruption** | Dual-recognition (v2 original) | Dual-recognition (ported from v2, improved) |
| **Tracing** | MLflow Unity Catalog | LangSmith (opt-in) |
| **Deployment** | Requires Databricks | Runs locally, no deployment needed |

## 🚀 Quick Start

### Prerequisites

* **Azure OpenAI resource** with a deployed chat model (GPT-4, GPT-4o, etc.)
* **Python 3.10+**
* **Chrome or Edge** (best SpeechRecognition support)
* **Headphones** (essential for reliable voice interruption)

### Step 1: Configure Environment

```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials:
#   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
#   AZURE_OPENAI_DEPLOYMENT=gpt-4
#   AZURE_OPENAI_API_KEY=your-key
#   AZURE_OPENAI_API_VERSION=2025-04-01-preview
```

### Step 2: Install and Run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

### Step 3: Use the Assistant

1. Click 🎤 and speak your question
2. Response streams as text + spoken audio
3. Interrupt anytime by speaking over the TTS
4. Toggle **Continuous Mode** for hands-free conversation
5. Say "stop", "exit", or "quit" to end continuous mode

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | Yes | — |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name | Yes | `gpt-4` |
| `AZURE_OPENAI_API_KEY` | API key | Yes | — |
| `AZURE_OPENAI_API_VERSION` | API version | No | `2024-02-15-preview` |
| `VOICE_TEMPERATURE` | LLM temperature | No | `0.7` |
| `VOICE_MAX_TOKENS` | Max response tokens | No | `1000` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `LOG_FILE` | Path to log file | No | (console only) |
| `LANGSMITH_TRACING` | Enable LangSmith tracing | No | `false` |
| `LANGSMITH_API_KEY` | LangSmith API key | If tracing | — |
| `LANGSMITH_ENDPOINT` | LangSmith endpoint | If tracing | `https://api.smith.langchain.com` |
| `LANGSMITH_PROJECT` | LangSmith project name | If tracing | `voice-agent-v5` |

### VoiceAgent Parameters

```python
agent = VoiceAgent(
    azure_openai_endpoint="...",
    azure_openai_deployment="...",
    azure_openai_api_key="...",
    temperature=0.7,           # LLM creativity (0–1)
    max_tokens=1000,           # Max response length
    max_history_turns=10,      # Conversation memory depth
    system_prompt="...",       # Override default personality
    tools=[...],               # Add custom tools
)
```

### Adding Custom Tools

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72°F in {city}"

agent = VoiceAgent(
    # ... other params
    tools=[get_weather]
)
```

## 🔧 Troubleshooting

### Speech Recognition Not Working
* Use Chrome or Edge (Firefox has limited SpeechRecognition support)
* Check browser microphone permissions
* Test in console: `new (window.SpeechRecognition || window.webkitSpeechRecognition)()`

### False Interruptions During TTS
* **Use headphones** — speakers cause the mic to pick up TTS audio
* Increase **Detection Delay** to 1500–2000ms in the UI
* Increase **Min Words** to 4–5

### Interruption Not Detected
* Decrease **Min Words** to 2
* Decrease **Detection Delay** to 800–1000ms
* Check the debug box (🐞 button) for real-time detection logs
* Speak clearly and distinctly from the TTS voice

### Azure OpenAI Auth Errors
* Verify `AZURE_OPENAI_ENDPOINT` ends with `/`
* Confirm deployment name matches Azure Portal exactly
* Check API key is valid and not expired

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| First-token latency | < 500ms |
| Total response time | < 3s (typical query) |
| Throughput | > 50 tokens/sec |
| Interruption response | < 300ms from detection to TTS stop |

Metrics are logged as structured JSON with the `VOICE_METRICS` prefix.

## 📚 Resources

* [LangChain Agents](https://python.langchain.com/docs/concepts/agents)
* [LangChain Streaming (astream_events)](https://docs.langchain.com/oss/python/langchain/streaming)
* [Azure Chat OpenAI Integration](https://python.langchain.com/docs/integrations/chat/azure_chat_openai)
* [Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)

---

**Built with LangChain, Azure OpenAI, and the Web Speech API**
