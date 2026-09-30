# ReelTalk Studio

Local AI video translation and dubbing with OpenAI and FFmpeg. Upload a video, transcribe and detect its spoken language, translate timed subtitle segments, generate target-language speech, and render a new MP4. The uploaded original is never modified.

## Requirements

- Python 3.10 or newer
- An OpenAI API key with access to audio transcription, chat completions, and text-to-speech
- Internet access for OpenAI requests

Install the Python dependencies:

```powershell
py -m pip install -r requirements.txt
```

Set the API key in the process environment, or enter it in the app's advanced settings:

```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
```

Do not commit API keys to source control.

## Run

```powershell
py app.py
```

Open <http://127.0.0.1:3000>. Override the port with the `PORT` environment variable. The local server binds to loopback and is intended for one-user local use.

## Workflow

1. Upload an MP4, MOV, MKV, WEBM, AVI, MPEG, or MPG file (up to 1 GB).
2. Select a target language, OpenAI voice and delivery style, and optional embedded subtitle track.
3. Start the job. FFmpeg extracts audio, Whisper transcribes speech with timestamps and source-language detection, OpenAI translates timed segments, and OpenAI TTS generates speech.
4. FFmpeg mixes the new voice with low-volume original audio, preserves the video stream, and writes a new H.264/AAC MP4 with an optional selectable subtitle track.
5. Preview and download the original, translated video, or SRT subtitles. Edit subtitle text in the editor to regenerate speech and video.

Failed jobs can be retried. Completed transcription, translation, and voice files are reused where possible. Job status and API keys are held in process memory; a server restart clears job status. Uploaded originals remain under `uploads/`, and generated files remain under `outputs/` until manually removed.

## Notes

- OpenAI requests can incur usage charges and are subject to account limits. Khmer TTS is supported, though OpenAI voices are optimized for English.
- Dubbing is generated per subtitle segment. The renderer fits slightly overlong lines when possible; long translations can still differ in pacing from the source.
- Original video audio is retained quietly under the generated voice. The source file itself is never overwritten.
- The editor and job endpoints have no user authentication. Keep this server bound to localhost unless access control is added.
