# Real-Time Audio Transcription App

This project implements a real-time audio transcription system using OpenAI's Whisper model. It offers two versions: one that runs locally and another that uses the OpenAI API.

## Features

- Real-time audio recording and transcription
- Voice Activity Detection (VAD) for speech detection
- Noise reduction
- Simulated typing of transcribed text
- API version with translation and GPT-4 correction

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/transcription-app.git
   cd transcription-app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env`:
   ```
   cp .env.example .env
   ```

5. Edit `.env` and add your actual OpenAI API key.

## Usage

### Local Version (main.py)

This version runs the Whisper model locally on your machine.

1. Run the script:
   ```
   python main.py
   ```

2. Speak into your microphone. The transcribed text will be typed automatically.

3. Press Ctrl+C to stop the program.

### API Version (main_api.py)

This version uses the OpenAI API for transcription and offers additional features.

1. Ensure your OpenAI API key is set in the `.env` file.

2. Run the script:
   ```
   python main_api.py
   ```

3. Choose whether you want to translate the audio to English when prompted.

4. Speak into your microphone. The transcribed (and optionally translated) text will be typed automatically.

5. Press Ctrl+C to stop the program.

## How It Works

Both versions of the app follow these general steps:

1. Record audio in real-time using PyAudio.
2. Process the audio in chunks (local version uses VAD and noise reduction).
3. Transcribe the audio using Whisper (locally or via API).
4. Simulate typing the transcribed text using pynput.

The API version adds these features:
- Option to translate audio to English
- GPT-4 based correction of transcriptions for improved accuracy

## Configuration

You can adjust various parameters in the scripts:

- `SAMPLE_RATE`: Audio sample rate (default: 16000)
- `CHUNK_SIZE`: Size of audio chunks (default: 1024)
- `BUFFER_DURATION`: Duration of audio to process at once (default: 5 seconds for API version, shorter for local version)

## Requirements

- Python 3.7+
- OpenAI API key (for API version)
- See `requirements.txt` for Python package dependencies

## Limitations

- Local version requires significant computational resources for larger Whisper models.
- API version requires an internet connection and incurs API usage costs.
- Maximum audio file size for API version is 25 MB.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
