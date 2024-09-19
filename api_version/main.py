import os
import pyaudio
import threading
from pynput.keyboard import Controller
import queue
import numpy as np
import logging
import signal
import sys
from openai import OpenAI
import wave
import tempfile
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
BUFFER_DURATION = 5  # Increased buffer duration for API calls
OVERLAP_DURATION = 0.5

# Flag to indicate shutdown
shutdown_flag = threading.Event()

def signal_handler(sig, frame):
    logging.info('Interrupt received, shutting down...')
    shutdown_flag.set()

signal.signal(signal.SIGINT, signal_handler)

load_dotenv()  # Load environment variables from .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Initialize OpenAI client with API key

def record_audio_stream(audio_queue, duration, sample_rate=SAMPLE_RATE, channels=CHANNELS, chunk=CHUNK_SIZE):
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=chunk)
        logging.info("Recording started.")
    except Exception as e:
        logging.error(f"Error opening audio stream: {e}")
        shutdown_flag.set()
        return

    total_chunks = int(sample_rate / chunk * duration)
    for _ in range(total_chunks):
        if shutdown_flag.is_set():
            break
        try:
            data = stream.read(chunk, exception_on_overflow=False)
            audio_queue.put(data)
        except Exception as e:
            logging.error(f"Error reading audio stream: {e}")
            continue

    logging.info("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    audio_queue.put(None)  # Sentinel to indicate end of recording

def transcribe_audio(audio_data, translate=False):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filename = temp_file.name
        
    # Save audio data as WAV file
    with wave.open(temp_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for 'int16' type
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)
    
    try:
        with open(temp_filename, "rb") as audio_file:
            if translate:
                response = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            else:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
        return response  # The response is already a string
    except Exception as e:
        logging.error(f"Error in transcription: {e}")
        return ""
    finally:
        os.remove(temp_filename)

def correct_transcript(transcript):
    system_prompt = "You are a helpful transcription assistant. Your task is to correct any spelling discrepancies, or convert semantically incorrect text to semantically correct text for LLMs or humans to understand. Add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ]
    )
    return response.choices[0].message.content

def process_audio_stream(audio_queue, translate=False):
    keyboard = Controller()
    audio_buffer = b""
    buffer_size = int(SAMPLE_RATE * BUFFER_DURATION) * 2  # *2 for 16-bit audio

    logging.info("Transcription thread started.")
    while not shutdown_flag.is_set():
        try:
            data = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        if data is None:
            break  # Exit signal received

        audio_buffer += data

        if len(audio_buffer) >= buffer_size:
            transcription = transcribe_audio(audio_buffer, translate)
            if transcription:
                corrected_transcription = correct_transcript(transcription)
                
                logging.info(f"Transcription: {corrected_transcription}")
                keyboard.type(corrected_transcription + " ")

            # Keep the overlap
            audio_buffer = audio_buffer[int(-OVERLAP_DURATION * SAMPLE_RATE * 2):]

    logging.info("Transcription thread terminating.")

def main():
    duration = 100  # Recording duration in seconds (5 minutes)
    audio_queue = queue.Queue()

    # Ask user if they want translation
    translate = input("Do you want to translate the audio to English? (y/n): ").lower() == 'y'

    # Start the recording thread
    recording_thread = threading.Thread(target=record_audio_stream, args=(audio_queue, duration))
    recording_thread.start()

    # Start the transcription thread
    transcription_thread = threading.Thread(target=process_audio_stream, args=(audio_queue, translate))
    transcription_thread.start()

    # Wait for both threads to complete
    recording_thread.join()
    transcription_thread.join()
    logging.info("Transcription and typing completed.")

if __name__ == "__main__":
    main()