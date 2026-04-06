"""Databricks Voice Agent - FastAPI Application"""
import os
import base64
from typing import AsyncIterator
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from databricks.sdk import WorkspaceClient
from langchain_community.chat_models import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration from environment
LLM_ENDPOINT =  "databricks-qwen3-next-80b-a3b-instruct"

# Initialize Databricks client
w = WorkspaceClient()

# ============================================================================
# Event Classes
# ============================================================================

@dataclass
class VoiceAgentEvent:
    """Base class for all voice agent events."""
    type: str

@dataclass
class AgentChunkEvent(VoiceAgentEvent):
    """Partial agent response chunk (streaming)."""
    type: str = "agent_chunk"
    text: str = ""

@dataclass
class AgentOutputEvent(VoiceAgentEvent):
    """Final agent response output."""
    type: str = "agent_output"
    text: str = ""

@dataclass
class StatusEvent(VoiceAgentEvent):
    """Status update event for UI feedback."""
    type: str = "status"
    message: str = ""

# ============================================================================
# Agent (LangChain + Databricks Foundation Models)
# ============================================================================

async def agent_stream(transcript: str) -> AsyncIterator[VoiceAgentEvent]:
    """Process transcript through LangChain agent."""
    print(f"🤖 [AGENT] Processing transcript: '{transcript}'")
    
    if not transcript.strip():
        print("⚠️ [AGENT] Empty transcript!")
        yield StatusEvent(message="⚠️ No transcript detected")
        return
    
    try:
        # Initialize Databricks LLM with pay-per-token endpoint
        llm = ChatDatabricks(
            endpoint=LLM_ENDPOINT,
            temperature=0.7,
            max_tokens=500
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful voice assistant. Provide concise, conversational responses."),
            ("human", "{input}")
        ])
        
        # Create chain
        chain = prompt | llm | StrOutputParser()
        
        print("🤖 [AGENT] Calling LLM...")
        yield StatusEvent(message="🤖 Agent thinking...")
        
        response_text = ""
        async for chunk in chain.astream({"input": transcript}):
            response_text += chunk
            print(f"🤖 [AGENT] Chunk: '{chunk}'")
            yield AgentChunkEvent(text=chunk)
        
        print(f"🤖 [AGENT] Full response: '{response_text}'")
        yield AgentOutputEvent(text=response_text)
        yield StatusEvent(message="✅ Agent response complete!")
        
    except Exception as e:
        print(f"❌ [AGENT] Error: {e}")
        import traceback
        traceback.print_exc()
        yield StatusEvent(message=f"❌ Error: {str(e)}")

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Databricks Voice Agent")

# HTML Client with Browser Web Speech API
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Databricks Voice Agent</title>
    <style>
        body { font-family: Arial; padding: 20px; max-width: 800px; margin: 0 auto; }
        h1 { color: #FF3621; }
        .info { background: #E3F2FD; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #2196F3; }
        button {
            padding: 15px 30px;
            font-size: 18px;
            margin: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
        }
        #startBtn { background: #FF3621; color: white; }
        #startBtn:hover { background: #CC2B1A; }
        #startBtn:disabled { background: #ccc; }
        #stopBtn { background: #1B3139; color: white; }
        #stopBtn:hover { background: #0D1A1F; }
        #stopBtn:disabled { background: #ccc; }
        #status {
            margin: 20px 0;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 5px;
            font-weight: bold;
        }
        #stages {
            margin: 10px 0;
            padding: 10px;
            background: #E8EAF6;
            border-radius: 5px;
            font-size: 14px;
            border-left: 4px solid #3F51B5;
            max-height: 400px;
            overflow-y: auto;
        }
        #transcript {
            margin-top: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
            min-height: 200px;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .user { background: #E3F2FD; border-left: 4px solid #2196F3; }
        .agent { background: #FFF3E0; border-left: 4px solid #FF9800; }
        .error { background: #FFCDD2; border-left: 4px solid #F44336; }
        .speaking { 
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <h1>🏔️ Databricks Voice Agent</h1>
    <p>Powered by Browser Web Speech API (STT + TTS) + Qwen3 Next 80B</p>
    
    <div class="info">
        <strong>ℹ️ Note:</strong> This uses your browser's built-in speech recognition and synthesis (Chrome/Edge recommended). 
        Works offline and requires no additional setup!
    </div>
    
    <div>
        <button id="startBtn" onclick="startRecording()">🎤 Start Recording</button>
        <button id="stopBtn" onclick="stopRecording()" disabled>⏹️ Stop Recording</button>
    </div>
    
    <div id="status">Status: Ready to connect</div>
    <div id="stages"></div>
    <div id="transcript"></div>
    
    <script>
        let ws = null;
        let recognition = null;
        let synthesis = window.speechSynthesis;
        let finalTranscript = '';
        let interimTranscript = '';
        
        function updateStage(message) {
            const stagesDiv = document.getElementById('stages');
            const timestamp = new Date().toLocaleTimeString();
            stagesDiv.innerHTML = '<div class="stage-active">[' + timestamp + '] ' + message + '</div>' + stagesDiv.innerHTML;
        }
        
        function speak(text) {
            // Cancel any ongoing speech
            synthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;  // Normal speed
            utterance.pitch = 1.0; // Normal pitch
            utterance.volume = 1.0; // Max volume
            
            utterance.onstart = () => {
                updateStage('🔊 Speaking response...');
                document.getElementById('status').textContent = 'Status: 🔊 Speaking...';
                document.getElementById('status').style.background = '#FFE082';
                document.getElementById('status').classList.add('speaking');
            };
            
            utterance.onend = () => {
                updateStage('✅ Speech complete!');
                document.getElementById('status').textContent = 'Status: Ready to connect';
                document.getElementById('status').style.background = '#f0f0f0';
                document.getElementById('status').classList.remove('speaking');
            };
            
            utterance.onerror = (event) => {
                console.error('Speech synthesis error:', event);
                updateStage('❌ TTS error: ' + event.error);
            };
            
            synthesis.speak(utterance);
        }
        
        function getWebSocketUrl() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            return protocol + '//' + window.location.host + '/ws';
        }
        
        function startRecording() {
            // Check browser support
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                alert('Speech recognition not supported in this browser. Please use Chrome or Edge.');
                return;
            }
            
            // Clear previous stages and transcript
            document.getElementById('stages').innerHTML = '';
            finalTranscript = '';
            interimTranscript = '';
            
            // Stop any ongoing speech
            synthesis.cancel();
            
            // Connect WebSocket
            const wsUrl = getWebSocketUrl();
            console.log('Connecting to:', wsUrl);
            updateStage('🔌 Connecting to WebSocket...');
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                updateStage('✅ WebSocket connected');
                
                // Initialize speech recognition
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;
                recognition.lang = 'en-US';
                
                recognition.onstart = () => {
                    document.getElementById('status').textContent = 'Status: 🟢 Listening - Speak now!';
                    document.getElementById('status').style.background = '#C8E6C9';
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    updateStage('🎤 Speech recognition started');
                };
                
                recognition.onresult = (event) => {
                    interimTranscript = '';
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        const transcript = event.results[i][0].transcript;
                        if (event.results[i].isFinal) {
                            finalTranscript += transcript + ' ';
                            updateStage('📝 Recognized: "' + transcript + '"');
                        } else {
                            interimTranscript += transcript;
                        }
                    }
                    console.log('Final:', finalTranscript, 'Interim:', interimTranscript);
                };
                
                recognition.onerror = (event) => {
                    console.error('Speech recognition error:', event.error);
                    updateStage('❌ Recognition error: ' + event.error);
                    if (event.error === 'no-speech') {
                        updateStage('ℹ️ No speech detected - please speak clearly');
                    }
                };
                
                recognition.onend = () => {
                    console.log('Recognition ended');
                };
                
                recognition.start();
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                const transcriptDiv = document.getElementById('transcript');
                
                console.log('Received event:', data.type, data);
                
                if (data.type === 'status') {
                    updateStage(data.message);
                    document.getElementById('status').textContent = 'Status: ' + data.message;
                    
                    // Show errors in transcript area
                    if (data.message.startsWith('❌')) {
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'message error';
                        errorDiv.innerHTML = '<strong>⚠️ Error:</strong> ' + data.message;
                        transcriptDiv.appendChild(errorDiv);
                        transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
                    }
                }
                else if (data.type === 'agent_chunk') {
                    let agentDiv = document.getElementById('current-agent-response');
                    if (!agentDiv) {
                        agentDiv = document.createElement('div');
                        agentDiv.id = 'current-agent-response';
                        agentDiv.className = 'message agent';
                        agentDiv.innerHTML = '<strong>🤖 Agent:</strong> ';
                        transcriptDiv.appendChild(agentDiv);
                        updateStage('🤖 Agent streaming response...');
                    }
                    const currentText = agentDiv.textContent.replace('🤖 Agent: ', '');
                    agentDiv.innerHTML = '<strong>🤖 Agent:</strong> ' + currentText + data.text;
                    transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
                }
                else if (data.type === 'agent_output') {
                    const agentDiv = document.getElementById('current-agent-response');
                    if (agentDiv) {
                        // Get the full agent response text
                        const agentText = agentDiv.textContent.replace('🤖 Agent: ', '');
                        agentDiv.id = '';
                        
                        updateStage('✅ Agent response complete');
                        
                        // Speak the response using TTS
                        if (agentText.trim()) {
                            speak(agentText);
                        }
                    }
                    
                    // Close connection after speech completes
                    setTimeout(() => {
                        if (ws) {
                            updateStage('🔌 Closing connection');
                            ws.close();
                            cleanup();
                        }
                    }, 2000); // Wait 2 seconds to allow speech to start
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                document.getElementById('status').textContent = 'Status: ⚠️ Error';
                document.getElementById('status').style.background = '#FFCDD2';
                updateStage('❌ WebSocket error occurred');
            };
            
            ws.onclose = () => {
                updateStage('🔌 WebSocket closed');
                cleanup();
            };
        }
        
        function stopRecording() {
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('status').textContent = 'Status: ⏳ Processing...';
            document.getElementById('status').style.background = '#FFF9C4';
            updateStage('⏹️ Stopping recording...');
            
            if (recognition) {
                recognition.stop();
                recognition = null;
            }
            
            // Add user transcript to UI
            if (finalTranscript.trim()) {
                const transcriptDiv = document.getElementById('transcript');
                const userDiv = document.createElement('div');
                userDiv.className = 'message user';
                userDiv.innerHTML = '<strong>🗣️ You:</strong> ' + finalTranscript;
                transcriptDiv.appendChild(userDiv);
                transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
                
                updateStage('✅ Final transcript: "' + finalTranscript + '"');
                
                // Send transcript to server for agent processing
                if (ws && ws.readyState === WebSocket.OPEN) {
                    updateStage('📤 Sending transcript to agent...');
                    ws.send(JSON.stringify({
                        type: 'transcript',
                        text: finalTranscript.trim()
                    }));
                }
            } else {
                updateStage('⚠️ No speech detected');
                cleanup();
            }
        }
        
        function cleanup() {
            if (recognition) {
                recognition.stop();
                recognition = null;
            }
            
            document.getElementById('status').textContent = 'Status: Ready to connect';
            document.getElementById('status').style.background = '#f0f0f0';
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            ws = null;
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get_client():
    """Serve HTML client."""
    return HTMLResponse(content=HTML_CLIENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections."""
    await websocket.accept()
    print("✅ Client connected")
    
    try:
        while True:
            # Receive transcript from browser
            data = await websocket.receive_json()
            
            if data.get("type") == "transcript":
                transcript = data.get("text", "")
                print(f"📝 Received transcript: '{transcript}'")
                
                # Process through agent
                async for event in agent_stream(transcript):
                    event_dict = {"type": event.type}
                    
                    if hasattr(event, "text"):
                        event_dict["text"] = event.text
                    elif hasattr(event, "message"):
                        event_dict["message"] = event.message
                    
                    print(f"📤 [WS] Sending to client: {event_dict}")
                    await websocket.send_json(event_dict)
                
                # Close after agent completes
                break
        
        print("✅ [WS] Pipeline finished, closing WebSocket")
    
    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
