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
import logging
import sys
import os

# Configure production-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Get logger for this module
logger = logging.getLogger(__name__)

# Configure MLflow to use Unity Catalog
try:
    # Set registry URI to use Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    
    # Set experiment in Unity Catalog (or workspace location)
    # For UC: use format "/<workspace_id>/<experiment_name>"
    # For workspace: use format "/Users/<username>/<experiment_name>"
    experiment_name = "/Users/kumarankit1996@gmail.com/voice_agent_experiments"
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow experiment set to: {experiment_name}")
    logger.info(f"MLflow registry URI: databricks-uc")
    
    # Enable MLflow autologging for LangChain (with silent=True to suppress warnings)
    # For inference/serving: only log traces, not models (model doesn't change at runtime)
    mlflow.langchain.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=False,  # Don't log model on every request - only needed once for registration
        log_traces=True,   # Track all requests/responses for observability
        disable=False,
        silent=True
    )
    logger.info("MLflow autologging enabled successfully (traces only, no model logging)")
    
except Exception as e:
    logger.warning(f"Failed to configure MLflow: {e}")

# Initialize FastAPI app
app = FastAPI(title="Voice Assistant")
logger.info("FastAPI application initialized")

# Initialize agent
try:
    agent = VoiceAgent(
        model_endpoint="databricks-qwen3-next-80b-a3b-instruct",
        temperature=0.7,
        max_tokens=1000,
        enable_tracing=False  # Tracing already enabled globally above
    )
    logger.info("VoiceAgent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize VoiceAgent: {e}", exc_info=True)
    raise

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
                <input type="range" id="minWordsSlider" class="slider" min="2" max="10" value="3" 
                       oninput="updateMinWords(this.value)">
                <span id="minWordsValue">3 words</span>
            </div>
            <div class="setting-item">
                <label>Detection Delay (avoid echo):</label>
                <input type="range" id="delaySlider" class="slider" min="800" max="2000" step="100" value="1200" 
                       oninput="updateDelay(this.value)">
                <span id="delayValue">1200ms</span>
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
            💡 <strong>Voice Interrupt:</strong> Start speaking to interrupt. <strong>Use headphones</strong> to prevent echo. The system captures your full phrase after interruption. Increase delay if you hear false triggers.
        </div>
        
        <div id="conversation" class="conversation"></div>
        
        <div class="info">
            <strong>💡 Features:</strong><br>
            • <strong>Voice Interruption:</strong> Speak anytime to interrupt (enabled by default)<br>
            • <strong>Continuous Mode:</strong> Automatic back-and-forth conversation<br>
            • <strong>Stop Words:</strong> Say "stop", "exit", or "quit" to end<br>
            • <strong>Best with headphones:</strong> Prevents assistant voice from triggering false interruptions
        </div>
        
        <div id="errorBox" class="error-box" style="display: none;"></div>
    </div>

    <script>
        let ws = null;
        let recognition = null;
        let backgroundRecognition = null;
        let isListening = false;
        let isSpeaking = false;
        let isProcessing = false;
        let currentTranscript = '';
        let interruptionTranscript = '';
        let responseBuffer = '';
        let currentAssistantMessage = null;
        let continuousMode = false;
        let voiceInterrupt = true;
        let autoSpeak = true;
        let lastTTSText = '';
        let showDebug = false;
        
        // Interruption settings - IMPROVED DEFAULTS
        let minWordsToInterrupt = 3;
        let interruptionDelay = 1200;  // Increased from 700ms to 1200ms to avoid TTS echo
        let speakingStartTime = null;
        let lastInterruptionTime = 0;
        let interruptionCooldown = 2000;
        let waitingForMoreSpeech = false;
        let speechContinuationTimer = null;
        let backgroundAccumulatedText = '';  // Accumulate ALL detected text
        let backgroundFinalTranscript = '';  // Track final transcripts separately
        let interruptionTriggerTimer = null;
        let interruptionAutoSubmitTimer = null;  // NEW: Auto-submit timer

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
            
            // Main recognition - continuous mode to capture full phrases
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            recognition.onstart = () => {
                isListening = true;
                updateUI();
                logDebug('Main recognition started');
            };
            
            recognition.onresult = (event) => {
                // Clear auto-submit timer when new speech is detected
                if (interruptionAutoSubmitTimer) {
                    clearTimeout(interruptionAutoSubmitTimer);
                    interruptionAutoSubmitTimer = null;
                    logDebug('Auto-submit cancelled - new speech detected');
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
                document.getElementById('transcript').innerHTML = 
                    (interruptionTranscript ? '<strong style="color: #ff5722;">[Interrupted] </strong>' : '') +
                    finalTranscript + '<i style="color: #999;">' + interimTranscript + '</i>';
                
                // Check for stop words
                if (continuousMode && checkStopWords(currentTranscript)) {
                    recognition.stop();
                    continuousMode = false;
                    updateUI();
                    document.getElementById('transcript').innerHTML = 
                        '<strong style="color: #f44336;">Conversation ended: Stop word detected</strong>';
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
                isListening = false;
                logDebug('Main recognition ended');
                
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
                    logDebug(`Submitting combined text: "${combinedText}"`);
                    
                    if (continuousMode && checkStopWords(combinedText)) {
                        continuousMode = false;
                        updateUI();
                        document.getElementById('transcript').innerHTML = 
                            '<strong style="color: #f44336;">Conversation ended</strong>';
                        return;
                    }
                    
                    sendMessage(combinedText);
                } else {
                    document.getElementById('transcript').textContent = 'No speech detected. Try again.';
                    
                    if (continuousMode) {
                        setTimeout(() => startListening(), 1000);
                    }
                }
                
                updateUI();
            };
            
            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                isListening = false;
                
                if (speechContinuationTimer) {
                    clearTimeout(speechContinuationTimer);
                    speechContinuationTimer = null;
                }
                
                if (interruptionAutoSubmitTimer) {
                    clearTimeout(interruptionAutoSubmitTimer);
                    interruptionAutoSubmitTimer = null;
                }
                
                updateUI();
                
                if (event.error !== 'no-speech' && event.error !== 'aborted') {
                    showError(`Speech error: ${event.error}`);
                    continuousMode = false;
                    interruptionTranscript = '';
                }
                
                if (continuousMode && event.error === 'no-speech') {
                    setTimeout(() => startListening(), 1000);
                }
            };
            
            // Background recognition for interruptions - FIXED: CONTINUOUS ACCUMULATION
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
                logDebug('Background recognition started - listening for interruption');
            };
            
            backgroundRecognition.onresult = (event) => {
                // REMOVED interruptionPending from this check to allow continuous accumulation
                if (!isSpeaking || !voiceInterrupt) return;
                
                const now = Date.now();
                
                // LONGER initial delay to avoid TTS echo (1200ms default, configurable)
                if (speakingStartTime && (now - speakingStartTime) < interruptionDelay) {
                    return;
                }
                
                // Cooldown between interruptions (only check if no active timer)
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
                    
                    // HIGHER confidence threshold (0.75 instead of 0.5) to avoid TTS echo
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
                if (lastTTSText && backgroundAccumulatedText.length > 0) {
                    const lowerAccumulated = backgroundAccumulatedText.toLowerCase();
                    const lowerTTS = lastTTSText.toLowerCase().slice(0, 50); // First 50 chars of TTS
                    
                    // If accumulated text is similar to TTS start, it's echo
                    if (lowerTTS.includes(lowerAccumulated) || lowerAccumulated.includes(lowerTTS.slice(0, 20))) {
                        logDebug(`FILTERED ECHO: "${backgroundAccumulatedText}" (matches TTS)`);
                        return;
                    }
                }
                
                // Count words
                const words = backgroundAccumulatedText.split(/\s+/).filter(w => w.length > 1);
                const wordCount = words.length;
                
                const avgConfidence = event.results[event.results.length-1]?.[0]?.confidence?.toFixed(2) || 'N/A';
                logDebug(`Detected: "${backgroundAccumulatedText}" (${wordCount} words, conf: ${avgConfidence})`);
                
                // Trigger interruption only if we have enough words AND no active timer
                if (wordCount >= minWordsToInterrupt && !interruptionTriggerTimer) {
                    console.log(`Voice interruption triggered! (${wordCount} words):`, backgroundAccumulatedText);
                    logDebug(`INTERRUPTING with: "${backgroundAccumulatedText}"`);
                    
                    // Keep background running for 500ms MORE to capture additional words
                    // This extended delay allows user to complete their phrase
                    // NOTE: We DON'T set interruptionPending here, allowing continued accumulation
                    interruptionTriggerTimer = setTimeout(() => {
                        // Capture the LATEST accumulated text at the moment of handoff
                        const capturedText = backgroundAccumulatedText || backgroundFinalTranscript;
                        logDebug(`Final captured text: "${capturedText}"`);
                        handleVoiceInterruption(capturedText);
                        interruptionTriggerTimer = null;
                    }, 500);  // Reduced to 500ms for faster response
                }
            };
            
            backgroundRecognition.onerror = (event) => {
                if (event.error !== 'no-speech' && event.error !== 'aborted' && event.error !== 'audio-capture') {
                    logDebug(`Background error: ${event.error}`);
                }
            };
            
            backgroundRecognition.onend = () => {
                logDebug('Background recognition ended');
                
                // Don't clear accumulated text if timer is active
                if (!interruptionTriggerTimer) {
                    backgroundAccumulatedText = '';
                    backgroundFinalTranscript = '';
                }
                
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
            
            // Clear the timer since we're handling now
            if (interruptionTriggerTimer) {
                clearTimeout(interruptionTriggerTimer);
                interruptionTriggerTimer = null;
            }
            
            stopSpeaking();
            
            if (currentAssistantMessage) {
                currentAssistantMessage.classList.add('interrupted');
            }
            
            try {
                backgroundRecognition.stop();
            } catch (e) {}
            
            const status = document.getElementById('status');
            status.className = 'status interrupting';
            status.textContent = 'Interrupted! Continue speaking...';
            
            // Show the captured interruption text
            document.getElementById('transcript').innerHTML = 
                '<strong style="color: #ff5722;">Interrupted with:</strong> ' +
                '<span style="color: #333;">"' + detectedText + '"</span><br>' +
                '<i style="color: #666;">Continue speaking or wait 2 seconds to submit</i>';
            
            logDebug(`Interruption captured: "${detectedText}". Starting main recognition immediately.`);
            
            // Start main recognition IMMEDIATELY (50ms delay)
            setTimeout(() => {
                startListening();
                
                // NEW: Set auto-submit timer - if no new speech in 2 seconds, submit what we have
                interruptionAutoSubmitTimer = setTimeout(() => {
                    logDebug('Auto-submit triggered - no additional speech detected');
                    
                    // If still listening and we have interruption text, submit it
                    if (isListening && interruptionTranscript) {
                        logDebug(`Auto-submitting: "${interruptionTranscript}"`);
                        recognition.stop(); // This will trigger onend which sends the message
                    }
                    interruptionAutoSubmitTimer = null;
                }, 2000);  // 2 second timeout
            }, 50);
        }

        function startBackgroundRecognition() {
            if (!backgroundRecognition || !voiceInterrupt || isListening) return;
            
            try {
                backgroundAccumulatedText = '';
                backgroundFinalTranscript = '';
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
                backgroundAccumulatedText = '';
                backgroundFinalTranscript = '';
                if (interruptionTriggerTimer) {
                    clearTimeout(interruptionTriggerTimer);
                    interruptionTriggerTimer = null;
                }
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
            
            stopSpeaking();
            
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            
            lastTTSText = text.slice(0, 100);
            
            utterance.onstart = () => {
                isSpeaking = true;
                speakingStartTime = Date.now();
                updateUI();
                
                setTimeout(() => {
                    startBackgroundRecognition();
                }, interruptionDelay);
            };
            
            utterance.onend = () => {
                isSpeaking = false;
                speakingStartTime = null;
                stopBackgroundRecognition();
                updateUI();
                
                if (continuousMode) {
                    setTimeout(() => startListening(), 500);
                }
            };
            
            utterance.onerror = (event) => {
                console.error('Speech synthesis error:', event);
                isSpeaking = false;
                speakingStartTime = null;
                stopBackgroundRecognition();
                updateUI();
                
                if (continuousMode) {
                    setTimeout(() => startListening(), 1000);
                }
            };
            
            speechSynthesis.speak(utterance);
        }

        function stopSpeaking() {
            if (speechSynthesis && speechSynthesis.speaking) {
                speechSynthesis.cancel();
            }
            isSpeaking = false;
            speakingStartTime = null;
            stopBackgroundRecognition();
        }

        function startListening() {
            if (!recognition || isListening || isProcessing) return;
            
            try {
                stopSpeaking();
                currentTranscript = '';
                recognition.start();
            } catch (e) {
                console.error('Failed to start recognition:', e);
            }
        }

        function stopListening() {
            if (recognition && isListening) {
                recognition.stop();
            }
            
            // Clear auto-submit timer if stopping manually
            if (interruptionAutoSubmitTimer) {
                clearTimeout(interruptionAutoSubmitTimer);
                interruptionAutoSubmitTimer = null;
            }
        }

        function toggleVoice() {
            if (isListening) {
                stopListening();
            } else {
                startListening();
            }
        }

        function toggleContinuousMode() {
            continuousMode = !continuousMode;
            updateUI();
            
            if (continuousMode && !isListening && !isSpeaking && !isProcessing) {
                startListening();
            }
        }

        function toggleVoiceInterrupt() {
            voiceInterrupt = !voiceInterrupt;
            updateUI();
            
            if (!voiceInterrupt) {
                stopBackgroundRecognition();
            }
        }

        function toggleAutoSpeak() {
            autoSpeak = !autoSpeak;
            updateUI();
        }

        function updateUI() {
            const voiceBtn = document.getElementById('voiceButton');
            const status = document.getElementById('status');
            const continuousBtn = document.getElementById('continuousBtn');
            const voiceInterruptBtn = document.getElementById('voiceInterruptBtn');
            const autoSpeakBtn = document.getElementById('autoSpeakBtn');
            
            voiceBtn.className = '';
            status.className = 'status';
            
            if (isSpeaking) {
                voiceBtn.classList.add('speaking');
                status.classList.add('speaking');
                status.textContent = 'Speaking...';
            } else if (isListening) {
                voiceBtn.classList.add('listening');
                status.classList.add('listening');
                status.textContent = 'Listening...';
            } else if (continuousMode) {
                voiceBtn.classList.add('continuous');
                status.classList.add('continuous');
                status.textContent = 'Continuous Mode Active';
            } else {
                status.classList.add('idle');
                status.textContent = 'Click to start';
            }
            
            continuousBtn.className = 'continuous-btn' + (continuousMode ? ' active' : '');
            continuousBtn.textContent = '🔄 Continuous: ' + (continuousMode ? 'ON' : 'OFF');
            
            voiceInterruptBtn.className = 'toggle-btn' + (voiceInterrupt ? ' active' : '');
            voiceInterruptBtn.textContent = '🎙️ Voice Interrupt: ' + (voiceInterrupt ? 'ON' : 'OFF');
            
            autoSpeakBtn.className = 'toggle-btn' + (autoSpeak ? ' active' : '');
            autoSpeakBtn.textContent = '🔊 Auto-Speak: ' + (autoSpeak ? 'ON' : 'OFF');
        }

        function sendMessage(text) {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                showError('Not connected to server');
                return;
            }
            
            ws.send(JSON.stringify({
                type: 'user_message',
                text: text
            }));
        }

        function addMessage(role, content) {
            const conversation = document.getElementById('conversation');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const label = document.createElement('div');
            label.className = 'message-label';
            label.textContent = role === 'user' ? 'You' : 'Assistant';
            
            const text = document.createElement('div');
            text.textContent = content;
            
            messageDiv.appendChild(label);
            messageDiv.appendChild(text);
            conversation.appendChild(messageDiv);
            conversation.scrollTop = conversation.scrollHeight;
        }

        function updateAssistantMessage(content) {
            if (!currentAssistantMessage) {
                const conversation = document.getElementById('conversation');
                currentAssistantMessage = document.createElement('div');
                currentAssistantMessage.className = 'message assistant';
                
                const label = document.createElement('div');
                label.className = 'message-label';
                label.textContent = 'Assistant';
                
                const text = document.createElement('div');
                text.className = 'message-text';
                text.textContent = content;
                
                currentAssistantMessage.appendChild(label);
                currentAssistantMessage.appendChild(text);
                conversation.appendChild(currentAssistantMessage);
            } else {
                const textDiv = currentAssistantMessage.querySelector('.message-text');
                if (textDiv) {
                    textDiv.textContent = content;
                }
            }
            
            const conversation = document.getElementById('conversation');
            conversation.scrollTop = conversation.scrollHeight;
        }

        function finalizeAssistantMessage() {
            currentAssistantMessage = null;
        }

        function clearConversation() {
            document.getElementById('conversation').innerHTML = '';
            document.getElementById('transcript').textContent = 'Conversation cleared.';
            currentAssistantMessage = null;
            responseBuffer = '';
        }

        function showError(message) {
            const errorBox = document.getElementById('errorBox');
            errorBox.textContent = message;
            errorBox.style.display = 'block';
        }

        function hideError() {
            document.getElementById('errorBox').style.display = 'none';
        }

        // Initialize
        window.onload = () => {
            if (!initSpeechRecognition()) {
                showError('Speech recognition not available');
                return;
            }
            
            connectWebSocket();
        };

        // Cleanup
        window.onbeforeunload = () => {
            stopListening();
            stopSpeaking();
            stopBackgroundRecognition();
            if (interruptionAutoSubmitTimer) {
                clearTimeout(interruptionAutoSubmitTimer);
            }
            if (ws) {
                ws.close();
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
    """Handle WebSocket connections"""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'user_message':
                user_input = message.get('text', '')
                logger.info(f"Received user message: {user_input[:100]}...")
                
                # Echo transcript back
                await websocket.send_text(json.dumps({
                    'type': 'transcript',
                    'text': user_input
                }))
                
                try:
                    # Stream agent response
                    logger.debug("Starting agent response stream")
                    async for chunk in agent.stream_response(user_input):
                        await websocket.send_text(json.dumps({
                            'type': 'agent_chunk',
                            'text': chunk
                        }))
                    
                    # Send completion signal
                    await websocket.send_text(json.dumps({
                        'type': 'agent_complete'
                    }))
                    logger.info("Agent response completed successfully")
                    
                except Exception as e:
                    logger.error(f"Error in agent response: {e}", exc_info=True)
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': f'Error: {str(e)}'
                    }))
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket connection closed")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Voice Assistant server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
