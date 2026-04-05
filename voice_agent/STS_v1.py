import sounddevice as sd  # Import sounddevice library for audio recording
import numpy as np  # Import numpy for numerical operations on audio data
import whisper  # Import OpenAI Whisper for speech-to-text transcription
import logging  # Import logging module for detailed logging
from langchain_ollama import ChatOllama  # Import ChatOllama for chat-based interactions with Ollama models
from langchain.chains import ConversationChain  # Import ConversationChain from LangChain for managing conversations
from langchain.memory import ConversationBufferMemory  # Import memory buffer for conversation history
from langchain.prompts import PromptTemplate  # Import PromptTemplate for custom prompts
import subprocess  # Import subprocess for running commands
import threading  # Import threading for parallel execution

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Set logging level to INFO and format

# Initialize Whisper model for STT
whisper_model = whisper.load_model("small")  # Load the small Whisper model for efficient speech recognition
logging.info("Whisper model 'small' loaded successfully.")  # Log model loading

# Initialize LangChain with ChatOllama and message history
model = ChatOllama(model="gemma4")  # Create an instance of ChatOllama using the Gemma 4 model
memory = ConversationBufferMemory()  # Initialize memory buffer for conversation history
# Custom prompt to make responses short and precise
custom_prompt = PromptTemplate.from_template(
    "You are a helpful voice assistant. Keep your answers short, precise, and to the point. Do not provide long explanations.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAssistant:"
)
conversation = ConversationChain(
    llm=model,  # The language model
    memory=memory,  # The memory buffer
    prompt=custom_prompt,  # Custom prompt for concise responses
    verbose=False,  # Disable verbose logging for cleaner output
)
logging.info("LangChain ConversationChain initialized with Gemma 4 model.")  # Log setup

def record_audio(duration=3, fs=16000):  # Reduced duration for faster interaction
    """Record audio from microphone for given duration."""  # Docstring explaining the function
    logging.info(f"Starting audio recording for {duration} seconds at {fs} Hz.")  # Log recording start
    logging.info("Recording...")  # Indicate recording has started
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')  # Start recording audio for specified duration
    sd.wait()  # Wait for the recording to complete
    logging.info("Audio recording completed.")  # Log recording end
    return np.squeeze(recording)  # Return the recorded audio as a 1D numpy array

def transcribe_audio(audio):  # Define function to transcribe audio to text
    """Transcribe audio using Whisper."""  # Docstring
    logging.info("Starting audio transcription with Whisper.")  # Log transcription start
    # Preprocess audio
    audio = whisper.pad_or_trim(audio)  # Pad or trim audio to fit Whisper's expected length
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)  # Convert audio to mel spectrogram on the model's device
    options = whisper.DecodingOptions(language='en')  # Set decoding options to English language
    result = whisper.decode(whisper_model, mel, options)  # Decode the spectrogram to text using Whisper
    transcribed_text = result.text.strip()  # Get transcribed text
    logging.info(f"Audio transcription completed: '{transcribed_text}'")  # Log transcription result
    return transcribed_text  # Return the transcribed text, stripped of whitespace

def speak_text(text):  # Define function to speak text using TTS
    """Speak the text using TTS."""  # Docstring
    logging.info(f"Starting TTS for text: '{text}'")  # Log TTS start
    # Use Mac's say command for TTS and wait for the process to complete
    process = subprocess.Popen(['say', text])  # Start the say command as a subprocess
    process.wait()  # Wait for the TTS process to complete
    logging.info("TTS completed.")  # Log TTS end

def generate_response(user_input, response_result):
    """Generate response from LLM in a separate thread."""
    try:
        response = conversation.invoke({"input": user_input})
        if isinstance(response, dict):
            response_text = response.get("response", "").strip()
        else:
            response_text = str(response).strip()
        response_result[0] = response_text
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        response_result[0] = "Sorry, there was an error processing your request."

def main():  # Define the main function for the voice assistant loop
    logging.info("Voice Assistant application started.")  # Log app start
    logging.info("Voice Assistant started. Say 'stop', 'exit', 'quit', 'bye', 'goodbye', 'end', 'terminate', or 'halt' to exit.")  # Print startup message
    while True:  # Start infinite loop for continuous interaction
        logging.info("Starting new interaction cycle.")  # Log cycle start
        # Record audio
        audio = record_audio(3)  # Record 3 seconds of audio for faster interaction

        # Transcribe
        user_input = transcribe_audio(audio)  # Transcribe the recorded audio to text
        logging.info(f"You said: {user_input}")  # Log the transcribed user input

        if user_input.lower() in ['stop', 'exit', 'quit', 'bye', 'goodbye', 'end', 'terminate', 'halt']:  # Check if user wants to stop
            logging.info("User requested to stop the assistant.")  # Log stop request
            speak_text("Goodbye!")  # Speak goodbye message
            break  # Exit the loop

        if user_input:  # If there is valid user input
            logging.info("Valid user input detected, starting response generation.")  # Log response generation start
            response_result = [None]  # Initialize response holder
            response_thread = threading.Thread(target=generate_response, args=(user_input, response_result))  # Create thread for response generation
            response_thread.start()  # Start the response generation thread in parallel
            speak_text(f"You said: {user_input}")  # Speak the user's input for confirmation while response generates
            response_thread.join()  # Wait for the response generation to complete
            response_text = response_result[0]  # Get the generated response
            if response_text:
                logging.info(f"Response generated: '{response_text}'")  # Log response
                logging.info(f"Assistant: {response_text}")  # Log the assistant's response
                speak_text(f"Assistant: {response_text}")  # Speak the response
            else:
                logging.warning("Empty response from model.")  # Log empty response
                speak_text("Sorry, I couldn't generate a response.")  # Speak error message
        else:  # If no speech was detected
            logging.warning("No speech detected in audio, retrying.")  # Log no speech
            logging.info("No speech detected, trying again...")  # Print message and continue loop
            speak_text("No speech detected, please try again.")  # Speak retry message

if __name__ == "__main__":  # Check if script is run directly
    main()  # Call the main function
