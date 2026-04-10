"""
Voice Assistant v4 with Real-Time Streaming - Azure OpenAI Edition
FastAPI + WebSocket with browser-based voice interruption support.

This version keeps the richer v2-style voice UI and interaction model,
while using the newer Azure/OpenAI VoiceAgent backend interface.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json
import os
import logging
import sys
import uvicorn
from uuid import uuid4
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

from agent import VoiceAgent

load_dotenv()


def setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


setup_logging()
logger = logging.getLogger(__name__)

if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = os.getenv(
        "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
    )
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "voice-agent-v5")

app = FastAPI(title="Voice Assistant v4 - Azure OpenAI")
logger.info("FastAPI application initialized")

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
temperature = float(os.getenv("VOICE_TEMPERATURE", "0.7"))
max_tokens = int(os.getenv("VOICE_MAX_TOKENS", "1000"))

if not azure_openai_endpoint or not azure_openai_deployment:
    raise ValueError(
        "Azure OpenAI configuration not found. Please set "
        "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT in your .env file."
    )

agent = VoiceAgent(
    azure_openai_api_key=azure_openai_api_key,
    azure_openai_endpoint=azure_openai_endpoint,
    azure_openai_deployment=azure_openai_deployment,
    azure_openai_api_version=azure_openai_api_version,
    temperature=temperature,
    max_tokens=max_tokens,
)
logger.info("VoiceAgent initialized successfully")

HTML_CONTENT = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Voice Assistant v4</title>
    <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --panel-2: #1f2937;
            --panel-3: #0b1220;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --primary: #38bdf8;
            --primary-2: #0ea5e9;
            --success: #22c55e;
            --danger: #ef4444;
            --warning: #f59e0b;
            --border: rgba(148, 163, 184, 0.22);
            --shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: radial-gradient(circle at top, #172554 0%, #0f172a 42%, #020617 100%);
            color: var(--text);
            min-height: 100vh;
            padding: 24px;
        }

        .app {
            max-width: 1100px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 20px;
        }

        .card {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
        }

        .main {
            padding: 24px;
        }

        .side {
            padding: 20px;
        }

        .title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 18px;
        }

        .title h1 {
            font-size: 2rem;
            line-height: 1.1;
        }

        .subtitle {
            color: var(--muted);
            margin-top: 8px;
            font-size: 0.98rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.12);
            color: #bae6fd;
            border: 1px solid rgba(56, 189, 248, 0.25);
            font-size: 0.86rem;
            white-space: nowrap;
        }

        .mic-wrap {
            display: grid;
            place-items: center;
            margin: 20px 0 18px;
        }

        #voiceButton {
            width: 132px;
            height: 132px;
            border: none;
            border-radius: 999px;
            color: white;
            cursor: pointer;
            font-size: 3.2rem;
            background: linear-gradient(135deg, var(--primary), #6366f1);
            box-shadow: 0 18px 40px rgba(14, 165, 233, 0.35);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        #voiceButton:hover { transform: translateY(-2px) scale(1.02); }
        #voiceButton.listening { background: linear-gradient(135deg, #fb7185, #f43f5e); animation: pulse 1.4s infinite; }
        #voiceButton.speaking { background: linear-gradient(135deg, #22c55e, #16a34a); animation: pulse 1.4s infinite; }
        #voiceButton.processing { background: linear-gradient(135deg, #f59e0b, #ea580c); animation: pulse 1.4s infinite; }

        @keyframes pulse {
            0%, 100% { transform: scale(1); box-shadow: 0 18px 40px rgba(0,0,0,0.28); }
            50% { transform: scale(1.05); box-shadow: 0 24px 54px rgba(0,0,0,0.38); }
        }

        .status {
            text-align: center;
            min-height: 28px;
            font-weight: 600;
            margin-bottom: 16px;
        }

        .status.idle { color: var(--muted); }
        .status.listening { color: #fda4af; }
        .status.speaking { color: #86efac; }
        .status.processing { color: #fcd34d; }
        .status.error { color: #fca5a5; }
        .status.interrupting { color: #ff5722; font-weight: 700; }

        .transcript {
            background: rgba(30, 41, 59, 0.72);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px;
            min-height: 92px;
            color: #dbeafe;
            margin-bottom: 18px;
        }

        .button-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 18px;
        }

        button.control {
            border: 1px solid var(--border);
            background: rgba(30, 41, 59, 0.88);
            color: var(--text);
            border-radius: 12px;
            padding: 10px 14px;
            cursor: pointer;
            font-size: 0.95rem;
            transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
        }

        button.control:hover {
            transform: translateY(-1px);
            border-color: rgba(56, 189, 248, 0.45);
        }

        button.control.active {
            background: rgba(14, 165, 233, 0.18);
            border-color: rgba(56, 189, 248, 0.45);
            color: #e0f2fe;
        }

        button.control.danger.active,
        button.control.danger {
            background: rgba(239, 68, 68, 0.14);
            border-color: rgba(239, 68, 68, 0.4);
        }

        .conversation {
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: 520px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .message {
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 16px;
            background: rgba(17, 24, 39, 0.94);
        }

        .message.user { border-left: 4px solid var(--primary); }
        .message.assistant { border-left: 4px solid var(--success); }
        .message.system { border-left: 4px solid var(--warning); }
        .message.error { border-left: 4px solid var(--danger); }

        .message .label {
            font-size: 0.82rem;
            color: var(--muted);
            margin-bottom: 7px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .message .text {
            white-space: pre-wrap;
            line-height: 1.55;
        }

        .side h2 {
            font-size: 1.05rem;
            margin-bottom: 14px;
        }

        .setting-group {
            margin-bottom: 18px;
            padding-bottom: 18px;
            border-bottom: 1px solid var(--border);
        }

        .setting-group:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }

        label {
            display: block;
            font-size: 0.92rem;
            color: var(--muted);
            margin-bottom: 8px;
        }

        input[type="range"] {
            width: 100%;
        }

        .stat {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            font-size: 0.95rem;
            margin-top: 8px;
        }

        .hint,
        .debug {
            margin-top: 14px;
            border-radius: 14px;
            padding: 14px;
            line-height: 1.5;
            font-size: 0.92rem;
        }

        .hint {
            background: rgba(59, 130, 246, 0.12);
            border: 1px solid rgba(59, 130, 246, 0.22);
            color: #dbeafe;
        }

        .debug {
            display: none;
            background: rgba(2, 6, 23, 0.88);
            border: 1px solid var(--border);
            color: #cbd5e1;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            white-space: pre-wrap;
        }

        .small {
            color: var(--muted);
            font-size: 0.84rem;
        }

        @media (max-width: 960px) {
            .app { grid-template-columns: 1fr; }
            body { padding: 16px; }
            .title { flex-direction: column; align-items: flex-start; }
        }
    </style>
</head>
<body>
    <div class="app">
        <section class="card main">
            <div class="title">
                <div>
                    <h1>🎤 Voice Assistant v4</h1>
                    <p class="subtitle">Azure OpenAI backend with the richer v2-style voice UI, streaming, continuous mode, and browser-side voice interruption.</p>
                </div>
                <div class="badge"><span>●</span><span id="connectionBadge">Connecting...</span></div>
            </div>

            <div class="mic-wrap">
                <button id="voiceButton" aria-label="Toggle voice input">🎤</button>
            </div>
            <div id="status" class="status idle">Click the microphone to start</div>
            <div id="transcript" class="transcript">Your voice input will appear here...</div>

            <div class="button-row">
                <button id="continuousBtn" class="control" type="button">🔄 Continuous Mode: OFF</button>
                <button id="interruptBtn" class="control active" type="button">🎙️ Voice Interrupt: ON</button>
                <button id="autoSpeakBtn" class="control active" type="button">🔊 Auto-Speak: ON</button>
                <button id="clearBtn" class="control danger" type="button">🗑️ Clear Chat</button>
                <button id="statsBtn" class="control" type="button">📊 Refresh Stats</button>
                <button id="debugBtn" class="control" type="button">🐞 Show Debug Info</button>
            </div>

            <div id="conversation" class="conversation"></div>
        </section>

        <aside class="card side">
            <h2>Voice settings</h2>

            <div class="setting-group">
                <label for="interruptWords">Min words to interrupt: <strong id="interruptWordsValue">3</strong></label>
                <input id="interruptWords" type="range" min="1" max="8" step="1" value="3" />
            </div>

            <div class="setting-group">
                <label for="interruptDelay">Detection delay (ms): <strong id="interruptDelayValue">1200</strong></label>
                <input id="interruptDelay" type="range" min="300" max="2500" step="100" value="1200" />
            </div>

            <div class="setting-group">
                <div class="stat"><span>Session</span><span id="sessionId">Not connected</span></div>
                <div class="stat"><span>Model type</span><span id="modelType">azure_openai</span></div>
                <div class="stat"><span>Active sessions</span><span id="activeSessions">0</span></div>
                <div class="stat"><span>Memory</span><span id="memoryType">InMemorySaver</span></div>
            </div>

            <div class="hint">
                <div><strong>Tips</strong></div>
                <div>- Speak again while the assistant is talking to interrupt it.</div>
                <div>- Headphones help reduce echo and false interruptions.</div>
                <div>- Say “stop”, “exit”, or “quit” to stop continuous mode.</div>
            </div>

            <div id="debugBox" class="debug">Debug: waiting for speech detection...</div>
            <div class="small" style="margin-top: 12px;">Uses browser SpeechRecognition and speechSynthesis, so Chrome or Edge works best.</div>
        </aside>
    </div>

    <script>
        let ws = null;
        let recognition = null;
        let backgroundRecognition = null;
        let isListening = false;
        let isSpeaking = false;
        let isProcessing = false;
        let continuousMode = false;
        let autoSpeak = true;
        let voiceInterruptEnabled = true;
        let showDebug = false;
        let currentAssistantMessage = null;
        let responseBuffer = '';
        let sessionId = null;
        let interruptionTimer = null;

        // Dual-recognition interruption state (from v2)
        let currentTranscript = '';
        let interruptionTranscript = '';
        let lastTTSText = '';
        let speakingStartTime = null;
        let lastInterruptionTime = 0;
        let interruptionCooldown = 2000;
        let waitingForMoreSpeech = false;
        let speechContinuationTimer = null;
        let backgroundAccumulatedText = '';
        let backgroundFinalTranscript = '';
        let interruptionTriggerTimer = null;
        let interruptionAutoSubmitTimer = null;

        let minWordsToInterrupt = 3;
        let detectionDelay = 1200;

        const STOP_WORDS = ['stop', 'exit', 'quit'];

        function byId(id) {
            return document.getElementById(id);
        }

        function setDebug(text) {
            const box = byId('debugBox');
            const timestamp = new Date().toLocaleTimeString();
            box.innerHTML = `[${timestamp}] ${text}<br>` + box.innerHTML;
            box.style.display = showDebug ? 'block' : 'none';
        }

        function updateStatus() {
            const button = byId('voiceButton');
            const status = byId('status');
            button.className = '';
            status.className = 'status';

            if (isSpeaking) {
                button.className = 'speaking';
                status.classList.add('speaking');
                status.textContent = '🔊 Assistant speaking...';
            } else if (isListening) {
                button.className = 'listening';
                status.classList.add('listening');
                status.textContent = '🎙️ Listening...';
            } else if (isProcessing) {
                button.className = 'processing';
                status.classList.add('processing');
                status.textContent = '⚙️ Thinking...';
            } else if (continuousMode) {
                status.classList.add('idle');
                status.textContent = '🔄 Continuous mode ready';
            } else {
                status.classList.add('idle');
                status.textContent = 'Click the microphone to start';
            }
        }

        function updateInterruptingStatus() {
            const button = byId('voiceButton');
            const status = byId('status');
            button.className = 'processing';
            status.className = 'status interrupting';
            status.textContent = '🔴 Interrupted! Continue speaking...';
        }

        function setConnectionState(text) {
            byId('connectionBadge').textContent = text;
        }

        function addMessage(role, text) {
            const conversation = byId('conversation');
            const wrapper = document.createElement('div');
            wrapper.className = `message ${role}`;

            const label = document.createElement('div');
            label.className = 'label';
            label.textContent = role === 'user'
                ? 'You'
                : role === 'assistant'
                ? 'Assistant'
                : role === 'error'
                ? 'Error'
                : 'System';

            const body = document.createElement('div');
            body.className = 'text';
            body.textContent = text;

            wrapper.appendChild(label);
            wrapper.appendChild(body);
            conversation.appendChild(wrapper);
            conversation.scrollTop = conversation.scrollHeight;

            if (role === 'assistant') {
                currentAssistantMessage = wrapper;
            }

            return wrapper;
        }

        function updateAssistantMessage(text) {
            if (!currentAssistantMessage) {
                addMessage('assistant', text);
                return;
            }
            const body = currentAssistantMessage.querySelector('.text');
            if (body) body.textContent = text;
            byId('conversation').scrollTop = byId('conversation').scrollHeight;
        }

        function resetAssistantStream() {
            currentAssistantMessage = null;
            responseBuffer = '';
            // Reset any interrupted styling
            const messages = byId('conversation').querySelectorAll('.message.assistant');
            messages.forEach(msg => {
                msg.style.opacity = '';
                msg.style.borderLeftColor = '';
            });
        }

        function speakText(text) {
            if (!text || !autoSpeak) {
                if (continuousMode && !isListening) {
                    setTimeout(() => startListening(), 350);
                }
                return;
            }

            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';
            utterance.rate = 1.0;

            // Store TTS text for echo detection
            lastTTSText = text.slice(0, 100);

            utterance.onstart = () => {
                isSpeaking = true;
                speakingStartTime = Date.now();
                updateStatus();
                setDebug('Speech synthesis started');

                // Start background recognition after delay to avoid initial TTS echo
                setTimeout(() => {
                    startBackgroundRecognition();
                }, detectionDelay);
            };

            utterance.onend = () => {
                isSpeaking = false;
                speakingStartTime = null;
                stopBackgroundRecognition();
                updateStatus();
                setDebug('Speech synthesis completed');
                if (continuousMode && !isListening) {
                    setTimeout(() => startListening(), 350);
                }
            };

            utterance.onerror = (event) => {
                isSpeaking = false;
                speakingStartTime = null;
                stopBackgroundRecognition();
                updateStatus();
                setDebug('Speech synthesis error: ' + JSON.stringify(event));
                if (continuousMode && !isListening) {
                    setTimeout(() => startListening(), 350);
                }
            };

            window.speechSynthesis.speak(utterance);
        }

        function stopAssistantSpeech() {
            if (window.speechSynthesis.speaking || window.speechSynthesis.pending) {
                window.speechSynthesis.cancel();
            }
            isSpeaking = false;
            speakingStartTime = null;
            stopBackgroundRecognition();
            updateStatus();
        }

        function checkStopWords(text) {
            const lower = text.toLowerCase();
            return STOP_WORDS.some(word => lower.includes(word));
        }

        function submitUserText(text) {
            const cleaned = (text || '').trim();
            if (!cleaned) return;
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                addMessage('error', 'WebSocket is not connected. Refresh the page.');
                return;
            }

            resetAssistantStream();
            isProcessing = true;
            updateStatus();
            ws.send(JSON.stringify({ type: 'message', text: cleaned }));
            setDebug('Sent message to backend: ' + cleaned);
        }

        function startListening() {
            if (!recognition || isListening || isProcessing) return;

            try {
                stopAssistantSpeech();
                currentTranscript = '';
                recognition.start();
            } catch (err) {
                setDebug('Recognition start error: ' + err.message);
            }
        }

        function stopListening() {
            if (!recognition || !isListening) return;
            recognition.stop();

            // Clear auto-submit timer if stopping manually
            if (interruptionAutoSubmitTimer) {
                clearTimeout(interruptionAutoSubmitTimer);
                interruptionAutoSubmitTimer = null;
            }
        }

        function initRecognition() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                addMessage('error', 'SpeechRecognition is unavailable in this browser.');
                return;
            }

            // Main recognition - continuous mode to capture full phrases
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = () => {
                isListening = true;
                updateStatus();
                setDebug('Main recognition started');
            };

            recognition.onresult = (event) => {
                // Clear auto-submit timer when new speech is detected
                if (interruptionAutoSubmitTimer) {
                    clearTimeout(interruptionAutoSubmitTimer);
                    interruptionAutoSubmitTimer = null;
                    setDebug('Auto-submit cancelled - new speech detected');
                }

                let interimTranscript = '';
                let finalTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript + ' ';
                    } else {
                        interimTranscript += transcript;
                    }
                }

                // Store ONLY new speech in currentTranscript (combination happens once in onend)
                currentTranscript = finalTranscript || interimTranscript;

                // Update display
                const transcriptEl = byId('transcript');
                transcriptEl.textContent = (interruptionTranscript ? '[Interrupted] ' : '') + finalTranscript + interimTranscript;

                // Check for stop words
                if (continuousMode && checkStopWords(currentTranscript)) {
                    recognition.stop();
                    continuousMode = false;
                    updateToggleButtons();
                    byId('transcript').textContent = 'Conversation ended: Stop word detected';
                    return;
                }

                // Reset speech continuation timer on new speech
                if (speechContinuationTimer) {
                    clearTimeout(speechContinuationTimer);
                }

                // Wait for 1.5 seconds of silence before submitting
                if (currentTranscript.trim()) {
                    speechContinuationTimer = setTimeout(() => {
                        if (isListening && currentTranscript.trim()) {
                            recognition.stop();
                        }
                    }, 1500);
                }
            };

            recognition.onend = () => {
                const wasListening = isListening;
                isListening = false;
                setDebug('Main recognition ended (wasListening=' + wasListening + ')');

                if (speechContinuationTimer) {
                    clearTimeout(speechContinuationTimer);
                    speechContinuationTimer = null;
                }

                // Clear auto-submit timer since recognition ended
                if (interruptionAutoSubmitTimer) {
                    clearTimeout(interruptionAutoSubmitTimer);
                    interruptionAutoSubmitTimer = null;
                }

                // ALWAYS combine interruption and current transcript
                const combinedText = (interruptionTranscript + ' ' + currentTranscript).trim();
                currentTranscript = '';
                interruptionTranscript = '';

                if (combinedText) {
                    setDebug(`Submitting combined text: "${combinedText}"`);

                    if (continuousMode && checkStopWords(combinedText)) {
                        continuousMode = false;
                        updateToggleButtons();
                        byId('transcript').textContent = 'Conversation ended';
                        updateStatus();
                        return;
                    }

                    submitUserText(combinedText);
                } else {
                    byId('transcript').textContent = 'No speech detected. Try again.';

                    if (continuousMode) {
                        setTimeout(() => startListening(), 1000);
                    }
                }

                updateStatus();
            };

            recognition.onerror = (event) => {
                setDebug('Speech recognition error: ' + event.error);
                isListening = false;

                if (speechContinuationTimer) {
                    clearTimeout(speechContinuationTimer);
                    speechContinuationTimer = null;
                }

                if (interruptionAutoSubmitTimer) {
                    clearTimeout(interruptionAutoSubmitTimer);
                    interruptionAutoSubmitTimer = null;
                }

                updateStatus();

                if (event.error !== 'no-speech' && event.error !== 'aborted') {
                    continuousMode = false;
                    interruptionTranscript = '';
                }

                if (continuousMode && event.error === 'no-speech') {
                    setTimeout(() => startListening(), 1000);
                }
            };

            // Background recognition for interruptions - CONTINUOUS ACCUMULATION
            backgroundRecognition = new SpeechRecognition();
            backgroundRecognition.continuous = true;
            backgroundRecognition.interimResults = true;
            backgroundRecognition.lang = 'en-US';

            backgroundRecognition.onstart = () => {
                backgroundAccumulatedText = '';
                backgroundFinalTranscript = '';
                if (interruptionTriggerTimer) {
                    clearTimeout(interruptionTriggerTimer);
                    interruptionTriggerTimer = null;
                }
                setDebug('Background recognition started - listening for interruption');
            };

            backgroundRecognition.onresult = (event) => {
                if (!isSpeaking || !voiceInterruptEnabled) return;

                const now = Date.now();

                // Initial delay to avoid TTS echo (configurable)
                if (speakingStartTime && (now - speakingStartTime) < detectionDelay) {
                    return;
                }

                // Cooldown between interruptions
                if (!interruptionTriggerTimer && (now - lastInterruptionTime < interruptionCooldown)) {
                    return;
                }

                // PROPERLY ACCUMULATE: Process ALL results to build complete transcript
                let finalText = '';
                let interimText = '';

                // Loop through ALL results (not just new ones) to get complete transcription
                for (let i = 0; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    const confidence = event.results[i][0].confidence || 1.0;
                    const isFinal = event.results[i].isFinal;

                    // HIGHER confidence threshold (0.75) to avoid TTS echo
                    if (!confidence || confidence > 0.75) {
                        if (isFinal) {
                            finalText += transcript + ' ';
                        } else {
                            interimText += transcript + ' ';
                        }
                    }
                }

                // Build complete accumulated text (final + interim)
                backgroundFinalTranscript = finalText.trim();
                backgroundAccumulatedText = (finalText + interimText).trim();

                // Filter out if this matches recent TTS output (echo detection)
                // Only filter when accumulated text is SHORT and closely matches TTS start
                // If user speech is longer than the TTS prefix, it's likely real speech, not echo
                if (lastTTSText && backgroundAccumulatedText.length > 0) {
                    const lowerAccumulated = backgroundAccumulatedText.toLowerCase();
                    const lowerTTS = lastTTSText.toLowerCase().slice(0, 50);

                    // Only filter as echo if accumulated text is shorter than TTS prefix
                    // and is fully contained within it (exact echo match)
                    // If user says something LONGER or DIFFERENT, it's real speech
                    if (backgroundAccumulatedText.length < lowerTTS.length &&
                        lowerTTS.includes(lowerAccumulated)) {
                        setDebug(`FILTERED ECHO: "${backgroundAccumulatedText}" (matches TTS)`);
                        return;
                    }
                }

                // Count words
                const words = backgroundAccumulatedText.split(/\s+/).filter(w => w.length > 1);
                const wordCount = words.length;

                const avgConfidence = event.results[event.results.length-1]?.[0]?.confidence?.toFixed(2) || 'N/A';
                setDebug(`Detected: "${backgroundAccumulatedText}" (${wordCount} words, conf: ${avgConfidence})`);

                // Trigger interruption only if we have enough words AND no active timer
                if (wordCount >= minWordsToInterrupt && !interruptionTriggerTimer) {
                    setDebug(`Voice interruption triggered! (${wordCount} words): ${backgroundAccumulatedText}`);

                    // Keep background running for 500ms MORE to capture additional words
                    interruptionTriggerTimer = setTimeout(() => {
                        const capturedText = backgroundAccumulatedText || backgroundFinalTranscript;
                        setDebug(`Final captured text: "${capturedText}"`);
                        handleVoiceInterruption(capturedText);
                        interruptionTriggerTimer = null;
                    }, 500);
                }
            };

            backgroundRecognition.onerror = (event) => {
                if (event.error !== 'no-speech' && event.error !== 'aborted' && event.error !== 'audio-capture') {
                    setDebug(`Background error: ${event.error}`);
                }
            };

            backgroundRecognition.onend = () => {
                setDebug('Background recognition ended');

                // Don't clear accumulated text if timer is active
                if (!interruptionTriggerTimer) {
                    backgroundAccumulatedText = '';
                    backgroundFinalTranscript = '';
                }

                // Auto-restart if still speaking - with multiple retry attempts
                if (isSpeaking && voiceInterruptEnabled) {
                    setTimeout(() => {
                        if (!isSpeaking || !voiceInterruptEnabled) return;
                        try {
                            backgroundRecognition.start();
                            setDebug('Background recognition auto-restarted');
                        } catch (e) {
                            setDebug(`Background restart failed: ${e.message}, retrying...`);
                            // Second retry after longer delay
                            setTimeout(() => {
                                if (!isSpeaking || !voiceInterruptEnabled) return;
                                try {
                                    backgroundRecognition.start();
                                    setDebug('Background recognition restarted on retry');
                                } catch (e2) {
                                    setDebug(`Background restart retry failed: ${e2.message}`);
                                }
                            }, 300);
                        }
                    }, 100);
                }
            };
        }

        function handleVoiceInterruption(detectedText) {
            lastInterruptionTime = Date.now();
            interruptionTranscript = detectedText;

            // Clear the timer since we're handling now
            if (interruptionTriggerTimer) {
                clearTimeout(interruptionTriggerTimer);
                interruptionTriggerTimer = null;
            }

            setDebug(`INTERRUPTING with: "${detectedText}"`);

            // Stop background recognition FIRST (before TTS cancel, to avoid recognition confusion)
            try {
                backgroundRecognition.stop();
            } catch (e) {}

            // Stop TTS
            stopAssistantSpeech();

            if (currentAssistantMessage) {
                currentAssistantMessage.style.opacity = '0.6';
                currentAssistantMessage.style.borderLeftColor = 'var(--danger)';
            }

            updateInterruptingStatus();

            // Show the captured interruption text
            byId('transcript').innerHTML =
                '<strong style="color: #ff5722;">Interrupted with:</strong> ' +
                '<span>"' + detectedText + '"</span><br>' +
                '<i>Continue speaking or wait 2 seconds to submit</i>';

            // Give the browser a moment to settle after TTS cancel
            // Then try to start main recognition for additional speech
            setTimeout(() => {
                // Reset currentTranscript for any additional speech
                currentTranscript = '';

                // Try to start main recognition (may fail if browser is still settling)
                try {
                    if (recognition && !isListening && !isProcessing) {
                        recognition.start();
                        setDebug('Main recognition started after interruption');

                        // Auto-submit timer: if no new speech in 2.5 seconds, submit what we have
                        interruptionAutoSubmitTimer = setTimeout(() => {
                            setDebug('Auto-submit timer fired');

                            // Always submit whatever we have - either combined or just interruption
                            const textToSubmit = (interruptionTranscript + ' ' + currentTranscript).trim();
                            interruptionTranscript = '';
                            currentTranscript = '';

                            if (textToSubmit) {
                                setDebug(`Auto-submitting: "${textToSubmit}"`);
                                submitUserText(textToSubmit);
                            } else {
                                setDebug('Auto-submit: nothing to submit');
                            }

                            // Stop recognition if still running
                            if (isListening) {
                                try {
                                    recognition.stop();
                                } catch (e) {}
                            }
                            interruptionAutoSubmitTimer = null;
                        }, 2500);
                    } else {
                        // Recognition can't start — submit interruption text directly
                        setDebug(`Cannot start recognition (listening=${isListening}, processing=${isProcessing}). Submitting directly.`);
                        submitUserText(interruptionTranscript);
                        interruptionTranscript = '';
                    }
                } catch (err) {
                    setDebug(`Recognition start failed: ${err.message}. Submitting directly.`);
                    // Fallback: submit the interruption text directly
                    setTimeout(() => {
                        if (interruptionTranscript) {
                            submitUserText(interruptionTranscript);
                            interruptionTranscript = '';
                        }
                    }, 300);
                }
            }, 150);
        }

        function startBackgroundRecognition() {
            if (!backgroundRecognition || !voiceInterruptEnabled || isListening) return;

            try {
                backgroundAccumulatedText = '';
                backgroundFinalTranscript = '';
                backgroundRecognition.start();
                setDebug('Started background listening for interruption');
            } catch (e) {
                setDebug(`Background start failed: ${e.message}`);
            }
        }

        function stopBackgroundRecognition() {
            if (!backgroundRecognition) return;

            try {
                backgroundRecognition.stop();
                backgroundAccumulatedText = '';
                backgroundFinalTranscript = '';
                if (interruptionTriggerTimer) {
                    clearTimeout(interruptionTriggerTimer);
                    interruptionTriggerTimer = null;
                }
                setDebug('Stopped background recognition');
            } catch (e) {}
        }

        function updateToggleButtons() {
            const continuousBtn = byId('continuousBtn');
            const interruptBtn = byId('interruptBtn');
            const autoSpeakBtn = byId('autoSpeakBtn');
            const debugBtn = byId('debugBtn');

            continuousBtn.textContent = continuousMode ? '🔄 Continuous Mode: ON' : '🔄 Continuous Mode: OFF';
            continuousBtn.classList.toggle('active', continuousMode);

            interruptBtn.textContent = voiceInterruptEnabled ? '🎙️ Voice Interrupt: ON' : '🎙️ Voice Interrupt: OFF';
            interruptBtn.classList.toggle('active', voiceInterruptEnabled);

            autoSpeakBtn.textContent = autoSpeak ? '🔊 Auto-Speak: ON' : '🔇 Auto-Speak: OFF';
            autoSpeakBtn.classList.toggle('active', autoSpeak);

            debugBtn.textContent = showDebug ? '🐞 Hide Debug Info' : '🐞 Show Debug Info';
            debugBtn.classList.toggle('active', showDebug);
            byId('debugBox').style.display = showDebug ? 'block' : 'none';
        }

        function clearConversation() {
            byId('conversation').innerHTML = '';
            byId('transcript').textContent = 'Your voice input will appear here...';
            resetAssistantStream();
            stopAssistantSpeech();
            stopBackgroundRecognition();
            currentTranscript = '';
            interruptionTranscript = '';
            lastTTSText = '';
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'clear_history' }));
            }
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                setConnectionState('Connected');
                setDebug('WebSocket connected');
            };

            ws.onclose = () => {
                setConnectionState('Disconnected');
                setDebug('WebSocket closed; retrying in 1.5s');
                setTimeout(connectWebSocket, 1500);
            };

            ws.onerror = () => {
                setConnectionState('Error');
                setDebug('WebSocket error');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'connected') {
                    sessionId = data.session_id;
                    byId('sessionId').textContent = sessionId;
                    if (data.stats) updateStatsPanel(data.stats);
                    return;
                }

                if (data.type === 'transcript') {
                    addMessage('user', data.text || '');
                    byId('transcript').textContent = data.text || '';
                    // Reset assistant message styling for new response
                    if (currentAssistantMessage) {
                        currentAssistantMessage.style.opacity = '';
                        currentAssistantMessage.style.borderLeftColor = '';
                    }
                    return;
                }

                if (data.type === 'agent_chunk') {
                    responseBuffer += data.text || '';
                    updateAssistantMessage(responseBuffer);
                    return;
                }

                if (data.type === 'agent_complete') {
                    isProcessing = false;
                    updateStatus();
                    setDebug('Assistant response completed');
                    if (responseBuffer.trim()) {
                        speakText(responseBuffer.trim());
                    } else if (continuousMode && !isListening) {
                        setTimeout(() => startListening(), 350);
                    }
                    return;
                }

                if (data.type === 'history_cleared') {
                    addMessage('system', 'Chat history cleared.');
                    byId('transcript').textContent = 'Your voice input will appear here...';
                    return;
                }

                if (data.type === 'stats') {
                    updateStatsPanel(data.stats || {});
                    return;
                }

                if (data.type === 'error') {
                    isProcessing = false;
                    updateStatus();
                    addMessage('error', data.message || 'Unknown error');
                    return;
                }
            };
        }

        function updateStatsPanel(stats) {
            byId('modelType').textContent = stats.model_type || 'azure_openai';
            byId('activeSessions').textContent = String(stats.active_sessions ?? 0);
            byId('memoryType').textContent = stats.memory_type || 'InMemorySaver';
            if (stats.session_id) {
                byId('sessionId').textContent = stats.session_id;
            }
        }

        byId('voiceButton').addEventListener('click', () => {
            if (isListening) stopListening();
            else startListening();
        });

        byId('continuousBtn').addEventListener('click', () => {
            continuousMode = !continuousMode;
            updateToggleButtons();
            updateStatus();
            if (continuousMode && !isListening && !isSpeaking && !isProcessing) {
                startListening();
            }
        });

        byId('interruptBtn').addEventListener('click', () => {
            voiceInterruptEnabled = !voiceInterruptEnabled;
            updateToggleButtons();
            if (!voiceInterruptEnabled) {
                stopBackgroundRecognition();
            }
        });

        byId('autoSpeakBtn').addEventListener('click', () => {
            autoSpeak = !autoSpeak;
            if (!autoSpeak) stopAssistantSpeech();
            updateToggleButtons();
        });

        byId('clearBtn').addEventListener('click', clearConversation);

        byId('statsBtn').addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'get_stats' }));
            }
        });

        byId('debugBtn').addEventListener('click', () => {
            showDebug = !showDebug;
            updateToggleButtons();
        });

        byId('interruptWords').addEventListener('input', (event) => {
            minWordsToInterrupt = Number(event.target.value);
            byId('interruptWordsValue').textContent = String(minWordsToInterrupt);
        });

        byId('interruptDelay').addEventListener('input', (event) => {
            detectionDelay = Number(event.target.value);
            byId('interruptDelayValue').textContent = String(detectionDelay);
        });

        window.addEventListener('DOMContentLoaded', () => {
            updateToggleButtons();
            updateStatus();
            connectWebSocket();
            initRecognition();
            setDebug('App initialized');
        });

        window.addEventListener('beforeunload', () => {
            if (ws) ws.close();
            if (recognition && isListening) recognition.stop();
            stopBackgroundRecognition();
            stopAssistantSpeech();
            if (interruptionAutoSubmitTimer) {
                clearTimeout(interruptionAutoSubmitTimer);
            }
            if (interruptionTriggerTimer) {
                clearTimeout(interruptionTriggerTimer);
            }
            if (speechContinuationTimer) {
                clearTimeout(speechContinuationTimer);
            }
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_index() -> HTMLResponse:
    return HTMLResponse(content=HTML_CONTENT)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid4())
    logger.info("WebSocket connected: session_id=%s", session_id)

    try:
        await websocket.send_json(
            {
                "type": "connected",
                "session_id": session_id,
                "stats": agent.get_stats(session_id),
            }
        )

        while True:
            raw_data = await websocket.receive_text()
            message = json.loads(raw_data)
            msg_type = message.get("type")

            if msg_type == "message":
                user_input = (message.get("text") or "").strip()
                if not user_input:
                    continue

                logger.info("Received message for session_id=%s: %s", session_id, user_input[:120])
                await websocket.send_json({"type": "transcript", "text": user_input})

                try:
                    async for chunk in agent.stream_response(user_input, session_id=session_id):
                        await websocket.send_json({"type": "agent_chunk", "text": chunk})
                    await websocket.send_json({"type": "agent_complete"})
                except Exception as exc:
                    logger.exception("Error while streaming agent response")
                    await websocket.send_json(
                        {"type": "error", "message": f"Error processing request: {str(exc)}"}
                    )

            elif msg_type == "clear_history":
                agent.clear_history(session_id)
                await websocket.send_json({"type": "history_cleared"})

            elif msg_type == "get_stats":
                await websocket.send_json({"type": "stats", "stats": agent.get_stats(session_id)})

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unsupported message type: {msg_type}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session_id=%s", session_id)
    except Exception:
        logger.exception("WebSocket endpoint failure for session_id=%s", session_id)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_type": agent.model_type,
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "unknown"),
        "active_sessions": len(agent.thread_store),
        "memory_type": "InMemorySaver (LangGraph)",
    }


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info("Starting Voice Assistant v4 on %s:%s", host, port)
    logger.info("Model type: %s", agent.model_type)

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )