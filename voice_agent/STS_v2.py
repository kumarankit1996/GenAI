import asyncio
import logging
import os
import subprocess
import threading
from vosk import Model, KaldiRecognizer
import json
import sounddevice as sd
import numpy as np
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "vosk-model-en-us-0.22-lgraph")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Set logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

# Global loop
loop = None

def set_loop(l):
    global loop
    loop = l

# Download and load Vosk model
try:
    if not os.path.exists(VOSK_MODEL_PATH):
        logging.info("Downloading Vosk model...")
        import urllib.request
        import zipfile
        url = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
        zip_path = "vosk-model.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)
        logging.info("Vosk model downloaded successfully.")
    vosk_model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(vosk_model, 16000)
    logging.info("Vosk model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Vosk model: {e}")
    raise

# Initialize LLM
try:
    llm = ChatOllama(model=OLLAMA_MODEL)
    memory = ConversationBufferMemory()
    custom_prompt = PromptTemplate.from_template(
        "You are a helpful voice assistant. Keep your answers short, precise, and to the point.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAssistant:"
    )
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=custom_prompt,
        verbose=False,
    )
    logging.info(f"LangChain conversation chain initialized with model {OLLAMA_MODEL}.")
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}")
    raise

# Global variables
input_queue = asyncio.Queue()
speaking = False
tts_process = None

def audio_callback(indata, frames, time, status):
    """Callback for audio input stream."""
    try:
        if status:
            logging.warning(f"Audio status: {status}")
        # Convert float32 to int16 bytes for Vosk
        indata_int16 = (indata * 32767).astype(np.int16)
        if rec.AcceptWaveform(indata_int16.tobytes()):
            result = json.loads(rec.Result())
            text = result.get("text", "").strip()
            if text:
                asyncio.run_coroutine_threadsafe(input_queue.put(text), loop)
    except Exception as e:
        logging.error(f"Error in audio callback: {e}")

def speak(text):
    global speaking, tts_process
    try:
        speaking = True
        logging.info(f"Speaking: {text}")
        # Use macOS 'say' command for TTS
        if tts_process and tts_process.poll() is None:
            tts_process.terminate()
        tts_process = subprocess.Popen(["say", text])
        tts_process.wait()
        speaking = False
    except Exception as e:
        logging.error(f"Error in TTS: {e}")
        speaking = False

async def main():
    global speaking
    try:
        set_loop(asyncio.get_event_loop())
        logging.info("Voice Assistant started. Say 'stop', 'exit', 'quit', 'bye', 'goodbye', 'end', 'terminate', or 'halt' to exit.")
        with sd.InputStream(device=0, samplerate=16000, channels=1, dtype='float32', callback=audio_callback):
            while True:
                user_input = await input_queue.get()
                logging.info(f"You said: '{user_input}'")

                if user_input.lower() in ['stop', 'exit', 'quit', 'bye', 'goodbye', 'end', 'terminate', 'halt']:
                    if speaking and tts_process:
                        tts_process.terminate()
                    speak("Goodbye!")
                    break

                # Interrupt current speaking if any
                if speaking and tts_process:
                    tts_process.terminate()
                    speaking = False

                # Generate response
                try:
                    response = conversation.invoke({"input": user_input})
                    if isinstance(response, dict):
                        response_text = response.get("response", "").strip()
                    else:
                        response_text = str(response).strip()
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    response_text = "Sorry, there was an error processing your request."

                if response_text:
                    threading.Thread(target=speak, args=(response_text,)).start()
                else:
                    threading.Thread(target=speak, args=("Sorry, I couldn't generate a response.",)).start()
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    finally:
        logging.info("Voice Assistant shutting down.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
