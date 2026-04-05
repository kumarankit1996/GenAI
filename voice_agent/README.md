# Voice Agent with LangChain

A production-grade voice assistant optimized for MacBook Pro M4 with 24GB RAM, featuring real-time speech-to-text, conversational AI with LangChain, and native macOS text-to-speech.

## Features

- **Faster Whisper STT**: Optimized speech-to-text for Apple Silicon (M1/M2/M3/M4)
- **Conversational AI**: Powered by LangChain with Ollama (open-source LLMs)
- **Native macOS TTS**: High-quality text-to-speech with native 'say' command
- **Async Architecture**: Asynchronous processing for responsive conversation
- **Apple Silicon Optimized**: 
  - CPU-optimized inference (no GPU required)
  - Multi-threaded Whisper processing (8 CPU threads)
  - Efficient memory management for 24GB RAM systems
- **Production-Ready**: 
  - Comprehensive logging and error handling
  - Environment variable configuration
  - Real-time audio stream processing

## Versions

- **STS_v1.py**: Basic synchronous implementation with OpenAI Whisper
- **STS_v2.py**: Asynchronous with Vosk STT (lower accuracy)
- **STS_v3.py**: ⭐ **RECOMMENDED** - Faster Whisper + async + Apple Silicon optimized

## Prerequisites

- **Python**: 3.10+
- **Hardware**: MacBook Pro M4 (or other Apple Silicon)
- **Ollama**: Running with a CPU-optimized model
- **Microphone**: Working audio input device

## Installation

### 1. Install Dependencies

```bash
cd /path/to/voice_agent
pip install -r requirements.txt
```

### 2. Install Ollama

```bash
# Download from https://ollama.ai or via brew
brew install ollama

# Start Ollama service
ollama serve
```

### 3. Pull a CPU-Optimized Model

For MacBook M4, recommended models:

```bash
# Lightweight (7B, ~4GB RAM)
ollama pull mistral:7b-instruct-q4_0

# Alternative lightweight options
ollama pull neural-chat:7b-v3-q4_0
ollama pull dolphin-mixtral:8x7b-q3_0  # Larger but great quality
```

### 4. Configure Environment (Optional)

Create a `.env` file in the `voice_agent` directory:

```env
# LLM Model
OLLAMA_MODEL=mistral:7b-instruct-q4_0

# Whisper model size (tiny, base, small)
WHISPER_MODEL=base

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
```

## Usage

### Run the Voice Agent (v3 - Recommended)

```bash
python STS_v3.py
```

### Interaction Flow

1. The agent starts listening for speech
2. Speak naturally into your microphone
3. System transcribes your speech
4. LLM generates a response
5. Agent speaks the response back
6. Repeat or say "exit" / "quit" / "stop" / "bye" to quit

### Example Conversation

```
🎤 Recording for 5 seconds... (speak now)
✓ You said: 'What is the capital of France?'
🧠 Thinking...
🔊 Assistant: The capital of France is Paris. It's located in the north-central part of the country and is known for its iconic landmarks like the Eiffel Tower.

🎤 Recording for 5 seconds... (speak now)
...
```

## Architecture

### Component Stack

```
┌─────────────────────┐
│  Audio Input        │ (sounddevice - real-time mic capture)
├─────────────────────┤
│  STT (Faster Whisper)│ (Apple Silicon optimized, ~1-2s per utterance)
├─────────────────────┤
│  LLM (Ollama)       │ (Local, CPU-based LM via LangChain)
├─────────────────────┤
│  TTS (macOS 'say')  │ (Native, high-quality synthesis)
├─────────────────────┤
│  Async Loop         │ (Event-driven architecture)
└─────────────────────┘
```

### Key Optimizations for M4

1. **Whisper**: Faster Whisper library with CPU threads
   - `device="cpu"` for Apple Neural Engine
   - `compute_type="float32"` for M-series compatibility
   - `num_workers=4` for parallel processing
   - `cpu_threads=8` for efficient CPU utilization

2. **LLM**: Ollama with quantized models
   - Q4 quantization (4-bit) for memory efficiency
   - 7B parameter models (4-8GB footprint)
   - CPU inference without GPU overhead

3. **Audio Processing**: Real-time async streaming
   - Non-blocking audio callbacks
   - Silence detection for automatic segmentation
   - Efficient numpy operations

## Configuration

### Model Selection for Different Use Cases

| Model | Size | Speed | Quality | Memory |
|-------|------|-------|---------|--------|
| mistral:7b-q4_0 | ~4GB | Fast | Good | 6GB |
| neural-chat:7b-q4_0 | ~4GB | Fast | Good | 6GB |
| dolphin-mixtral:8x7b-q3_0 | ~6GB | Medium | Excellent | 8GB |
| llama2:7b-chat-q4_0 | ~4GB | Fast | Good | 6GB |

### Environment Variables

In `.env`:

```env
# Model configuration
OLLAMA_MODEL=mistral:7b-instruct-q4_0
WHISPER_MODEL=base              # tiny, base, small
AUDIO_SAMPLE_RATE=16000
RECORD_DURATION=5
LOG_LEVEL=INFO
```

### Whisper Model Sizes

| Model | Accuracy | Speed | Memory |
|-------|----------|-------|--------|
| tiny | ~90% | Very Fast | ~390MB |
| base | ~94% | Fast | ~1.5GB |
| small | ~96% | Medium | ~2.9GB |

## Performance Characteristics

On MacBook Pro M4 with 24GB RAM:

- **STT Latency**: 1-2 seconds (base model)
- **LLM Response Time**: 2-5 seconds (7B models, CPU)
- **TTS Latency**: <500ms (native macOS)
- **Total Round-trip**: ~4-8 seconds

## Logging

Logs are written to:
- **Console**: Real-time output with emoji indicators
- **File**: `/tmp/voice_agent.log` for debugging

View logs:
```bash
tail -f /tmp/voice_agent.log
```

## Troubleshooting

### Ollama Not Running
```
Error: Failed to initialize Ollama
Solution: ollama serve
```

### Model Not Found
```
Error: Failed to generate response
Solution: ollama pull mistral:7b-instruct-q4_0
```

### No Microphone Input
```
Error: Error recording audio
Solution: Check System Preferences → Security & Privacy → Microphone
```

### Slow STT
```
Problem: Transcription takes >5 seconds
Solution: Use tiny model: WHISPER_MODEL=tiny
```

### High Memory Usage
```
Problem: Agent uses >10GB RAM
Solution: Use smaller quantization (q3_0 vs q4_0)
           or smaller model (3B instead of 7B)
```

## Advanced Usage

### Custom Prompt Engineering

Edit the system prompt in STS_v3.py:

```python
system_prompt = PromptTemplate.from_template("""
Your custom system prompt here.
Tailor for specific use cases like:
- Customer service
- Medical assistant
- Code helper
- etc.
...
""")
```

### Changing TTS Voice

In the `speak_text()` function:

```python
# Available macOS voices: Alex, Victoria, Samantha, Victoria, etc.
["say", "-v", "Victoria", text]  # Change 'Alex' to preferred voice
```

## Performance Tips

1. **Faster STT**: Use `WHISPER_MODEL=tiny`
2. **Faster LLM**: Use smaller models or higher quantization (q3_0)
3. **Lower Latency**: Reduce `RECORD_DURATION` to 3 seconds
4. **Better Quality**: Use `WHISPER_MODEL=small` + `mistral` model

## Production Deployment

For production use:

1. ✅ Run Ollama as a background service
2. ✅ Configure logging to file
3. ✅ Use systemd/launchd for auto-restart
4. ✅ Monitor `/tmp/voice_agent.log`
5. ✅ Set appropriate memory limits via Ollama config

## Resources

- **Faster Whisper**: https://github.com/SYSTRAN/faster-whisper
- **Ollama**: https://ollama.ai
- **LangChain**: https://python.langchain.com
- **Apple Silicon ML**: https://developer.apple.com/metal/

## License

MIT License (or as per your project)
