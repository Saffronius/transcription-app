import pyaudio
import os
import whisper
import threading
from pynput.keyboard import Controller
import queue
import numpy as np
import logging
import signal
import sys
import webrtcvad
import noisereduce as nr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1
CHUNK_SIZE = 1024   # Number of frames per buffer
BUFFER_DURATION = 1.5  # Increased from 1.0 to 1.5 seconds
OVERLAP_DURATION = 0.5  # 0.5-second overlap
VAD_MODE = 1  # WebRTC VAD aggressiveness mode (0-3)
ENERGY_THRESHOLD = 0.0  # Not used anymore with WebRTC VAD

# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

# Flag to indicate shutdown
shutdown_flag = threading.Event()

def signal_handler(sig, frame):
    logging.info('Interrupt received, shutting down...')
    shutdown_flag.set()

signal.signal(signal.SIGINT, signal_handler)

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

def is_speech(audio, sample_rate=SAMPLE_RATE, frame_duration_ms=30, vad=vad):
    """Determine if the audio chunk contains speech using WebRTC VAD."""
    # Convert float32 audio back to int16 for VAD
    audio_int16 = (audio * 32768).astype(np.int16).tobytes()
    
    # Frame duration must be 10, 20, or 30 ms
    num_samples = int(sample_rate * frame_duration_ms / 1000)
    for i in range(0, len(audio_int16), num_samples * 2):  # 2 bytes per int16 sample
        frame = audio_int16[i:i + num_samples * 2]
        if len(frame) < num_samples * 2:
            break
        if vad.is_speech(frame, sample_rate):
            return True
    return False

def reduce_noise(audio_chunk):
    """Apply noise reduction to the audio chunk."""
    reduced_noise = nr.reduce_noise(y=audio_chunk, sr=SAMPLE_RATE)
    return reduced_noise

def transcribe_audio_stream(audio_queue, model, sample_rate=SAMPLE_RATE, channels=CHANNELS, buffer_duration=BUFFER_DURATION, overlap_duration=OVERLAP_DURATION):
    keyboard = Controller()
    buffer_size = int(sample_rate * buffer_duration)
    overlap_size = int(sample_rate * overlap_duration)
    audio_buffer = np.array([], dtype=np.float32)

    logging.info("Transcription thread started.")
    while not shutdown_flag.is_set():
        try:
            data = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        if data is None:
            break  # Exit signal received

        # Convert bytes data to numpy array
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio_chunk = np.mean(audio_chunk.reshape(-1, channels), axis=1)

        # Voice Activity Detection
        if not is_speech(audio_chunk):
            continue  # Skip transcription if no speech detected

        # Apply noise reduction
        audio_chunk = reduce_noise(audio_chunk)

        # Accumulate audio data with overlap
        audio_buffer = np.concatenate((audio_buffer, audio_chunk))
        
        # Transcribe when buffer is full
        if len(audio_buffer) >= buffer_size:
            audio_to_transcribe = audio_buffer[:buffer_size]
            # Retain overlap for the next buffer
            audio_buffer = audio_buffer[buffer_size - overlap_size:]
            
            # Transcribe the audio
            try:
                result = model.transcribe(audio_to_transcribe, fp16=False)
                transcription = result["text"].strip()

                if transcription:
                    logging.info(f"Transcription: {transcription}")
                    # Simulate typing the transcription
                    keyboard.type(transcription + " ")
            except Exception as e:
                logging.error(f"Error during transcription: {e}")

    logging.info("Transcription thread terminating.")

def main():
    output_directory = "recordings"
    os.makedirs(output_directory, exist_ok=True)

    duration = 60  # Recording duration in seconds

    audio_queue = queue.Queue()

    # Load the Whisper model once
    logging.info("Loading Whisper model...")
    try:
        model = whisper.load_model("base")  # Using "base" for better accuracy
        logging.info("Whisper model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading Whisper model: {e}")
        sys.exit(1)

    # Start the recording thread
    recording_thread = threading.Thread(target=record_audio_stream, args=(audio_queue, duration))
    recording_thread.start()

    # Start the transcription thread
    transcription_thread = threading.Thread(target=transcribe_audio_stream, args=(audio_queue, model))
    transcription_thread.start()
# Wait for both threads to complete
    recording_thread.join()
    transcription_thread.join()

    logging.info("Transcription and typing completed.")

if __name__ == "__main__":
    main()
