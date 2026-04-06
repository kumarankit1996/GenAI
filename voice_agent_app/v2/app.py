"""
Voice Assistant with Real-Time Streaming
FastAPI + WebSocket with voice interruption support
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import base64
from typing import AsyncIterator
from agent import VoiceAgent
import mlflow
from uuid import uuid4

# Initialize FastAPI app
app = FastAPI(title="Voice Assistant")

# Initialize agent
agent = VoiceAgent(
    model_endpoint="databricks-qwen3-next-80b-a3b-instruct",
    temperature=0.7,
    max_tokens=1000
)

# Enable MLflow tracing
mlflow.langchain.autolog()

# HTML Frontend with voice interruption
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Voice Assistant</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            color: white;
        }
        .container {
            max-width: 800px;
            width: 100%;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            color: #333;
        }
        h1 {
            text-align: center;
            margin-bottom: 10px;
            color: #764ba2;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 0.95em;
        }
        .voice-control {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 30px 0;
        }
        #voiceButton {
            width: 140px;
            height: 140px;
            border-radius: 50%;
            border: 4px solid #667eea;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 3.5em;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        #voiceButton:hover { transform: scale(1.05); }
        #voiceButton.listening {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            animation: pulse 1.5s infinite;
            border-color: #f5576c;
        }
        #voiceButton.speaking {
            background: linear-gradient(135deg, #4CAF50 0%, #2e7d32 100%);
            animation: pulse 1.5s infinite;
            border-color: #2e7d32;
        }
        #voiceButton.continuous {
            background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
            animation: pulse 2s infinite;
            border-color: #f57c00;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
            50% { transform: scale(1.05); box-shadow: 0 8px 30px rgba(0,0,0,0.3); }
        }
        .status {
            text-align: center;
            font-size: 1.2em;
            font-weight: 500;
            min-height: 30px;
            margin-bottom: 10px;
        }
        .status.listening { color: #f5576c; }
        .status.speaking { color: #2e7d32; }
        .status.continuous { color: #ff9800; }
        .status.idle { color: #666; }
        .status.error { color: #c62828; }
        .status.interrupting { color: #ff5722; font-weight: 700; }
        .transcript-box {
            background: #f5f5f5;
            border-radius: 10px;
            padding: 20px;
            min-height: 100px;
            margin: 20px 0;
            border-left: 4px solid #667eea;
            font-size: 1.1em;
            color: #333;
        }
        .conversation {
            margin-top: 30px;
            max-height: 400px;
            overflow-y: auto;
        }
        .message {
            padding: 12px 16px;
            border-radius: 10px;
            margin-bottom: 12px;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
        }
        .message.assistant {
            background: #f5f5f5;
            border-left: 4px solid #4CAF50;
        }
        .message.interrupted {
            opacity: 0.7;
            border-left: 4px solid #ff5722;
        }
        .message-label {
            font-weight: 600;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 500;
        }
        button:hover { transform: translateY(-2px); }
        .clear-btn {
            background: #f44336;
            color: white;
        }
        .clear-btn:hover { background: #d32f2f; }
        .toggle-btn {
            background: #2196F3;
            color: white;
        }
        .toggle-btn:hover { background: #1976D2; }
        .toggle-btn.active {
            background: #4CAF50;
        }
        .continuous-btn {
            background: #ff9800;
            color: white;
        }
        .continuous-btn:hover { background: #f57c00; }
        .continuous-btn.active {
            background: #ff5722;
            animation: glow 1.5s infinite;
        }
        @keyframes glow {
            0%, 100% { box-shadow: 0 2px 8px rgba(255, 87, 34, 0.5); }
            50% { box-shadow: 0 4px 16px rgba(255, 87, 34, 0.8); }
        }
        .info {
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 0.9em;
            border-left: 4px solid #ff9800;
        }
        .error-box {
            background: #ffebee;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            color: #c62828;
            border-left: 4px solid #c62828;
        }
        .hint {
            background: #e3f2fd;
            padding: 12px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 0.9em;
            border-left: 4px solid #2196F3;
            color: #1565c0;
        }
        .debug-box {
            background: #f5f5f5;
            padding: 12px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 0.85em;
            border-left: 4px solid #9c27b0;
            font-family: monospace;
            max-height: 100px;
            overflow-y: auto;
        }
        .settings {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid #9c27b0;
        }
        .setting-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 10px 0;
        }
        .slider {
            width: 200px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Voice Assistant</h1>
        <p class="subtitle">Powered by LangChain + Databricks Qwen 80B</p>
        
        <div class="voice-control">
            <button id="voiceButton" onclick="toggleVoice()">🎤</button>
            <div id="status" class="status idle">Click to start conversation</div>
        </div>
        
        <div id="transcript" class="transcript-box">
            Your voice input will appear here...
        </div>
        
        <div class="controls">
            <button id="continuousBtn" class="continuous-btn" onclick="toggleContinuousMode()">
                🔄 Continuous Mode: OFF
            </button>
            <button id="voiceInterruptBtn" class="toggle-btn active" onclick="toggleVoiceInterrupt()">
                🎙️ Voice Interrupt: ON
            </button>
            <button id="autoSpeakBtn" class="toggle-btn active" onclick="toggleAutoSpeak()">
                🔊 Auto-Speak: ON
            </button>
            <button class="clear-btn" onclick="clearConversation()">
                🗑️ Clear Chat
            </button>
        </div>
        
        <div id="voiceInterruptSettings" class="settings">
            <strong>⚙️ Voice Interruption Sensitivity</strong>
            <div class="setting-item">
                <label>Min Words to Interrupt:</label>
                <input type="range" id="minWordsSlider" class="slider" min="1" max="10" value="3" 
                       oninput="updateMinWords(this.value)">
                <span id="minWordsValue">3 words</span>
            </div>
            <div class="setting-item">
                <label>Detection Delay:</label>
                <input type="range" id="delaySlider" class="slider" min="300" max="2000" step="100" value="700" 
                       oninput="updateDelay(this.value)">
                <span id="delayValue">700ms</span>
            </div>
            <div class="setting-item">
                <label>
                    <input type="checkbox" id="debugCheckbox" onchange="toggleDebug(this.checked)">
                    Show Debug Info
                </label>
            </div>
        </div>
        
        <div id="debugBox" class="debug-box" style="display: none;">
            Debug: Waiting for speech detection...
        </div>
        
        <div id="hint" class="hint">
            💡 <strong>Voice Interrupt Enabled:</strong> Just start speaking while the assistant talks to interrupt! Use headphones for best results. Adjust sensitivity above if too sensitive/not responsive.
        </div>
        
        <div id="conversation" class="conversation"></div>
        
        <div class="info">
            <strong>💡 Features:</strong><br>
            • <strong>Voice Interruption:</strong> Speak anytime to interrupt (enabled by default)<br>
            • <strong>Continuous Mode:</strong> Automatic back-and-forth conversation<br>
            • <strong>Stop Words:</strong> Say "stop", "exit", or "quit" to end<br>
            • <strong>Adjustable:</strong> Tune sensitivity if needed
        </div>
        
        <div id="errorBox" class="error-box" style="display: none;"></div>
    </div>

    <script>
        let ws = null;
        let recognition = null;
        let backgroundRecognition = null;
        let isListening = false;
        let isSpeaking = false;
        let autoSpeak = true;
        let voiceInterrupt = true;  // ON by default
        let continuousMode = false;
        let currentTranscript = '';
        let interruptionTranscript = '';
        let speechSynthesis = window.speechSynthesis;
        let currentUtterance = null;
        let responseBuffer = '';
        let isProcessing = false;
        let showDebug = false;
        
        // More lenient interruption settings
        let minWordsToInterrupt = 3;  // Lower threshold
        let interruptionDelay = 700;  // Shorter delay
        let speakingStartTime = null;
        let lastInterruptionTime = 0;
        let interruptionCooldown = 2000;

        const STOP_WORDS = ['stop', 'exit', 'quit', 'goodbye', 'bye bye', 'end conversation'];

        function updateMinWords(value) {
            minWordsToInterrupt = parseInt(value);
            document.getElementById('minWordsValue').textContent = value + ' words';
            logDebug(`Settings updated: minWords=${minWordsToInterrupt}`);
        }

        function updateDelay(value) {
            interruptionDelay = parseInt(value);
            document.getElementById('delayValue').textContent = value + 'ms';
            logDebug(`Settings updated: delay=${interruptionDelay}ms`);
        }

        function toggleDebug(checked) {
            showDebug = checked;
            document.getElementById('debugBox').style.display = checked ? 'block' : 'none';
        }

        function logDebug(message) {
            if (!showDebug) return;
            const debugBox = document.getElementById('debugBox');
            const timestamp = new Date().toLocaleTimeString();
            debugBox.innerHTML = `[${timestamp}] ${message}<br>` + debugBox.innerHTML;
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                hideError();
            };
            
            ws.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'transcript') {
                    addMessage('user', data.text);
                    document.getElementById('transcript').textContent = 'Processing...';
                    responseBuffer = '';
                    isProcessing = true;
                }
                else if (data.type === 'agent_chunk') {
                    responseBuffer += data.text;
                    updateAssistantMessage(responseBuffer);
                }
                else if (data.type === 'agent_complete') {
                    finalizeAssistantMessage();
                    isProcessing = false;
                    
                    if (autoSpeak && responseBuffer) {
                        speakText(responseBuffer);
                    } else {
                        if (continuousMode) {
                            setTimeout(() => startListening(), 500);
                        }
                    }
                    
                    responseBuffer = '';
                }
                else if (data.type === 'error') {
                    showError(data.message);
                    isProcessing = false;
                    
                    if (continuousMode) {
                        setTimeout(() => startListening(), 1000);
                    }
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                showError('Connection error. Please refresh the page.');
                continuousMode = false;
                updateUI();
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                showError('Disconnected from server. Refresh to reconnect.');
                continuousMode = false;
                updateUI();
            };
        }

        function initSpeechRecognition() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                showError('Speech recognition not supported. Please use Chrome, Edge, or Safari.');
                return false;
            }
            
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            
            // Main recognition
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            recognition.onstart = () => {
                isListening = true;
                updateUI();
                logDebug('Main recognition started');
            };
            
            recognition.onresult = (event) => {
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
                
                if (interruptionTranscript) {
                    currentTranscript = interruptionTranscript + ' ' + (finalTranscript || interimTranscript);
                } else {
                    currentTranscript = finalTranscript || interimTranscript;
                }
                
                document.getElementById('transcript').innerHTML = 
                    (interruptionTranscript ? '<strong style="color: #ff5722;">' + interruptionTranscript + ' </strong>' : '') +
                    finalTranscript + '<i style="color: #999;">' + interimTranscript + '</i>';
                
                if (continuousMode && checkStopWords(currentTranscript)) {
                    recognition.stop();
                    continuousMode = false;
                    updateUI();
                    document.getElementById('transcript').innerHTML = 
                        '<strong style="color: #f44336;">Conversation ended: Stop word detected</strong>';
                    return;
                }
            };
            
            recognition.onend = () => {
                isListening = false;
                logDebug('Main recognition ended');
                
                if (currentTranscript.trim()) {
                    const text = currentTranscript.trim();
                    currentTranscript = '';
                    interruptionTranscript = '';
                    
                    if (continuousMode && checkStopWords(text)) {
                        continuousMode = false;
                        updateUI();
                        document.getElementById('transcript').innerHTML = 
                            '<strong style="color: #f44336;">Conversation ended</strong>';
                        return;
                    }
                    
                    sendMessage(text);
                } else {
                    document.getElementById('transcript').textContent = 'No speech detected. Try again.';
                    interruptionTranscript = '';
                    
                    if (continuousMode) {
                        setTimeout(() => startListening(), 1000);
                    }
                }
                
                updateUI();
            };
            
            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                isListening = false;
                interruptionTranscript = '';
                updateUI();
                
                if (event.error !== 'no-speech' && event.error !== 'aborted') {
                    showError(`Speech error: ${event.error}`);
                    continuousMode = false;
                }
                
                if (continuousMode && event.error === 'no-speech') {
                    setTimeout(() => startListening(), 1000);
                }
            };
            
            // Background recognition with REASONABLE thresholds
            backgroundRecognition = new SpeechRecognition();
            backgroundRecognition.continuous = true;
            backgroundRecognition.interimResults = true;
            backgroundRecognition.lang = 'en-US';
            
            backgroundRecognition.onstart = () => {
                logDebug('Background recognition started');
            };
            
            backgroundRecognition.onresult = (event) => {
                if (!isSpeaking || !voiceInterrupt) return;
                
                const now = Date.now();
                
                // Initial delay to avoid echo
                if (speakingStartTime && (now - speakingStartTime) < interruptionDelay) {
                    return;
                }
                
                // Cooldown between interruptions
                if (now - lastInterruptionTime < interruptionCooldown) {
                    return;
                }
                
                // Collect ANY detected text (more lenient)
                let detectedText = '';
                
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    const confidence = event.results[i][0].confidence || 0;
                    
                    // Accept moderate confidence (0.5+) instead of very high
                    if (!confidence || confidence > 0.5) {
                        detectedText += transcript + ' ';
                    }
                }
                
                detectedText = detectedText.trim();
                
                // Count words (filter very short ones)
                const words = detectedText.split(/\s+/).filter(w => w.length > 1);
                const wordCount = words.length;
                
                logDebug(`Detected: "${detectedText}" (${wordCount} words, conf: ${event.results[event.results.length-1]?.[0]?.confidence?.toFixed(2) || 'N/A'})`);
                
                if (wordCount >= minWordsToInterrupt) {
                    console.log(`✅ Voice interruption triggered! (${wordCount} words):`, detectedText);
                    logDebug(`✅ INTERRUPTING with: "${detectedText}"`);
                    handleVoiceInterruption(detectedText);
                }
            };
            
            backgroundRecognition.onerror = (event) => {
                if (event.error !== 'no-speech' && event.error !== 'aborted' && event.error !== 'audio-capture') {
                    logDebug(`Background error: ${event.error}`);
                }
            };
            
            backgroundRecognition.onend = () => {
                logDebug('Background recognition ended');
                // Auto-restart if still speaking
                if (isSpeaking && voiceInterrupt) {
                    setTimeout(() => {
                        try {
                            backgroundRecognition.start();
                        } catch (e) {
                            logDebug(`Restart failed: ${e.message}`);
                        }
                    }, 100);
                }
            };
            
            return true;
        }

        function handleVoiceInterruption(detectedText) {
            lastInterruptionTime = Date.now();
            interruptionTranscript = detectedText;
            
            stopSpeaking();
            
            if (currentAssistantMessage) {
                currentAssistantMessage.classList.add('interrupted');
            }
            
            try {
                backgroundRecognition.stop();
            } catch (e) {}
            
            const status = document.getElementById('status');
            status.className = 'status interrupting';
            status.textContent = '🎙️ Interrupting... Speak now!';
            
            if (detectedText) {
                document.getElementById('transcript').innerHTML = 
                    '<strong style="color: #ff5722;">Interrupted! Continue speaking: "' + detectedText + '..."</strong>';
            }
            
            setTimeout(() => {
                startListening();
            }, 300);
        }

        function startBackgroundRecognition() {
            if (!backgroundRecognition || !voiceInterrupt || isListening) return;
            
            try {
                backgroundRecognition.start();
                logDebug('Started background listening for interruption');
            } catch (e) {
                logDebug(`Background start failed: ${e.message}`);
            }
        }

        function stopBackgroundRecognition() {
            if (!backgroundRecognition) return;
            
            try {
                backgroundRecognition.stop();
                logDebug('Stopped background recognition');
            } catch (e) {}
        }

        function checkStopWords(text) {
            const lowerText = text.toLowerCase().trim();
            return STOP_WORDS.some(word => {
                return lowerText === word || 
                       lowerText.startsWith(word + ' ') || 
                       lowerText.endsWith(' ' + word) ||
                       lowerText.includes(' ' + word + ' ');
            });
        }

        function speakText(text) {
            if (!speechSynthesis) {
                console.error('Speech synthesis not supported');
                if (continuousMode) {
                    setTimeout(() => startListening(), 500);
                }
                return;
            }
            
            speechSynthesis.cancel();
            
            if (!text || !text.trim()) {
                if (continuousMode) {
                    setTimeout(() => startListening(), 500);
                }
                return;
            }
            
            isSpeaking = true;
            speakingStartTime = Date.now();
            updateUI();
            
            logDebug(`Starting TTS, will listen for interruption after ${interruptionDelay}ms`);
            
            // Start background recognition after delay
            if (voiceInterrupt) {
                setTimeout(() => startBackgroundRecognition(), interruptionDelay);
            }
            
            currentUtterance = new SpeechSynthesisUtterance(text);
            currentUtterance.rate = 1.0;
            currentUtterance.pitch = 1.0;
            currentUtterance.volume = 1.0;
            currentUtterance.lang = 'en-US';
            
            currentUtterance.onstart = () => {
                isSpeaking = true;
                speakingStartTime = Date.now();
                updateUI();
                logDebug('TTS started');
            };
            
            currentUtterance.onend = () => {
                isSpeaking = false;
                currentUtterance = null;
                speakingStartTime = null;
                
                stopBackgroundRecognition();
                updateUI();
                logDebug('TTS ended normally');
                
                if (continuousMode) {
                    setTimeout(() => startListening(), 500);
                }
            };
            
            currentUtterance.onerror = (event) => {
                console.error('Speech synthesis error:', event);
                isSpeaking = false;
                currentUtterance = null;
                speakingStartTime = null;
                
                stopBackgroundRecognition();
                updateUI();
                
                if (continuousMode) {
                    setTimeout(() => startListening(), 1000);
                }
            };
            
            speechSynthesis.speak(currentUtterance);
        }

        function stopSpeaking() {
            if (speechSynthesis) {
                speechSynthesis.cancel();
            }
            isSpeaking = false;
            currentUtterance = null;
            speakingStartTime = null;
            
            stopBackgroundRecognition();
            updateUI();
            logDebug('TTS stopped (interrupted)');
        }

        function startListening() {
            if (!recognition) {
                showError('Speech recognition not initialized');
                return;
            }
            
            if (isListening || isProcessing) {
                return;
            }
            
            stopBackgroundRecognition();
            
            if (!interruptionTranscript) {
                currentTranscript = '';
                document.getElementById('transcript').textContent = 'Listening...';
            }
            
            try {
                recognition.start();
            } catch (e) {
                if (!e.message.includes('already started')) {
                    console.error('Failed to start recognition:', e);
                }
            }
        }

        function toggleVoice() {
            if (!recognition) {
                showError('Speech recognition not initialized');
                return;
            }
            
            if (isSpeaking) {
                handleVoiceInterruption('');
                return;
            }
            
            if (continuousMode) {
                continuousMode = false;
                if (isListening) {
                    recognition.stop();
                }
                stopBackgroundRecognition();
                updateUI();
                document.getElementById('transcript').textContent = 'Continuous mode stopped';
                return;
            }
            
            if (isListening) {
                recognition.stop();
            } else {
                startListening();
            }
        }

        function toggleContinuousMode() {
            continuousMode = !continuousMode;
            
            const btn = document.getElementById('continuousBtn');
            btn.textContent = continuousMode ? '🔄 Continuous Mode: ON' : '🔄 Continuous Mode: OFF';
            btn.className = continuousMode ? 'continuous-btn active' : 'continuous-btn';
            
            if (continuousMode) {
                document.getElementById('transcript').textContent = 'Continuous mode activated. Start speaking...';
                startListening();
            } else {
                if (isListening) {
                    recognition.stop();
                }
                stopSpeaking();
                stopBackgroundRecognition();
                document.getElementById('transcript').textContent = 'Continuous mode deactivated';
            }
            
            updateUI();
        }

        function toggleVoiceInterrupt() {
            voiceInterrupt = !voiceInterrupt;
            const btn = document.getElementById('voiceInterruptBtn');
            const settings = document.getElementById('voiceInterruptSettings');
            
            btn.textContent = voiceInterrupt ? '🎙️ Voice Interrupt: ON' : '🎙️ Voice Interrupt: OFF';
            btn.className = voiceInterrupt ? 'toggle-btn active' : 'toggle-btn';
            
            logDebug(`Voice interrupt: ${voiceInterrupt ? 'ENABLED' : 'DISABLED'}`);
            
            if (!voiceInterrupt) {
                stopBackgroundRecognition();
            } else if (isSpeaking) {
                setTimeout(() => startBackgroundRecognition(), interruptionDelay);
            }
        }

        function sendMessage(text) {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                showError('Not connected to server');
                return;
            }
            
            ws.send(JSON.stringify({
                type: 'user_input',
                text: text
            }));
            
            document.getElementById('transcript').textContent = 'Processing...';
        }

        function addMessage(role, text) {
            const conv = document.getElementById('conversation');
            const msg = document.createElement('div');
            msg.className = `message ${role}`;
            msg.innerHTML = `
                <div class="message-label">${role === 'user' ? '👤 You' : '🤖 Assistant'}</div>
                <div>${text}</div>
            `;
            conv.appendChild(msg);
            conv.scrollTop = conv.scrollHeight;
        }

        let currentAssistantMessage = null;

        function updateAssistantMessage(text) {
            if (!currentAssistantMessage) {
                const conv = document.getElementById('conversation');
                currentAssistantMessage = document.createElement('div');
                currentAssistantMessage.className = 'message assistant';
                currentAssistantMessage.innerHTML = `
                    <div class="message-label">🤖 Assistant</div>
                    <div class="content">${text}</div>
                `;
                conv.appendChild(currentAssistantMessage);
            } else {
                const content = currentAssistantMessage.querySelector('.content');
                content.textContent = text;
            }
            
            const conv = document.getElementById('conversation');
            conv.scrollTop = conv.scrollHeight;
        }

        function finalizeAssistantMessage() {
            currentAssistantMessage = null;
            if (!continuousMode) {
                document.getElementById('transcript').textContent = 'Click to start speaking';
            }
        }

        function toggleAutoSpeak() {
            autoSpeak = !autoSpeak;
            const btn = document.getElementById('autoSpeakBtn');
            btn.textContent = autoSpeak ? '🔊 Auto-Speak: ON' : '🔇 Auto-Speak: OFF';
            btn.className = autoSpeak ? 'toggle-btn active' : 'toggle-btn';
            
            if (!autoSpeak && isSpeaking) {
                stopSpeaking();
            }
        }

        function clearConversation() {
            document.getElementById('conversation').innerHTML = '';
            document.getElementById('transcript').textContent = 'Conversation cleared. Click to start speaking.';
            responseBuffer = '';
            interruptionTranscript = '';
            stopSpeaking();
            stopBackgroundRecognition();
        }

        function updateUI() {
            const button = document.getElementById('voiceButton');
            const status = document.getElementById('status');
            
            button.className = '';
            
            if (continuousMode && !isListening && !isSpeaking) {
                button.classList.add('continuous');
                button.textContent = '🔄';
                status.className = 'status continuous';
                status.textContent = 'Continuous mode active';
            } else if (isSpeaking) {
                button.classList.add('speaking');
                button.textContent = '🔊';
                status.className = 'status speaking';
                status.textContent = voiceInterrupt ? 'Speaking... (Start talking to interrupt)' : 'Speaking...';
            } else if (isListening) {
                button.classList.add('listening');
                button.textContent = '🎙️';
                status.className = 'status listening';
                status.textContent = 'Listening... Speak now!';
            } else {
                button.textContent = '🎤';
                status.className = 'status idle';
                status.textContent = continuousMode ? 'Waiting...' : 'Click to start';
            }
        }

        function showError(message) {
            const errorBox = document.getElementById('errorBox');
            errorBox.textContent = '❌ ' + message;
            errorBox.style.display = 'block';
        }

        function hideError() {
            document.getElementById('errorBox').style.display = 'none';
        }

        window.onload = () => {
            if (initSpeechRecognition()) {
                connectWebSocket();
            }
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the HTML frontend"""
    return HTML_CONTENT

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice interaction
    Supports continuous conversation mode with voice interruption
    """
    await websocket.accept()
    session_id = str(uuid4())
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'user_input':
                user_text = message['text']
                
                # Send transcript back to client
                await websocket.send_text(json.dumps({
                    'type': 'transcript',
                    'text': user_text
                }))
                
                # Stream agent response
                try:
                    full_response = ""
                    async for chunk in agent.stream_response(
                        user_input=user_text,
                        session_id=session_id
                    ):
                        full_response += chunk
                        
                        # Send text chunk to client
                        await websocket.send_text(json.dumps({
                            'type': 'agent_chunk',
                            'text': chunk
                        }))
                    
                    # Signal completion (triggers TTS in browser)
                    await websocket.send_text(json.dumps({
                        'type': 'agent_complete'
                    }))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': str(e)
                    }))
    
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
        except:
            pass

# Run with: python -m uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
