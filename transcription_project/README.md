# Video Transcriber

A simple local web tool for transcribing Google Drive videos and optionally translating the transcript between Indian languages, using Google Gemini 2.5 Pro.

## Features

- Paste up to 3 public Google Drive video links at once
- Automatically split each video into 5-minute batches with 15-second overlap
- Pick source language and target translation language from 11 options
- If source and target are the same, translation is skipped automatically
- Sequential background processing; browser tab can be closed while jobs run
- Per-video status, live log, segment progress, transcript CSV and XLSX download links
- Retry button for failed videos (reuses the already-downloaded file)
- Pipe-delimited transcript output with timestamps and speaker columns
- Three retries with exponential backoff on transient API errors

## Supported languages

Hindi, Telugu, Bengali, Tamil, English, Marathi, Gujarati, Kannada, Malayalam, Urdu, Punjabi.

## Prerequisites

- macOS with Python 3.10 or newer (`python3 --version` to check)
- FFmpeg installed and available on your `PATH` (`ffmpeg -version` and `ffprobe -version`)
- A Gemini API key from https://aistudio.google.com/apikey
- Google Drive video links shared as "Anyone with the link"

## Setup

```bash
cd video-transcriber

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Now edit .env and paste your real Gemini API key
```

## Run

```bash
# With the venv activated:
python app.py
```

Then open http://localhost:5000 in your browser.

Keep the terminal window open for the duration of the batch. To stop the server, press Ctrl+C.

## How it works

For each video in the batch, sequentially:

1. Download the video from Google Drive using `gdown` into `./downloads/`
2. Split the requested time range into 5-minute segments with 15-second overlap
3. Render each segment to its own temporary video file with FFmpeg
4. Upload each segment to the Gemini Files API
5. Wait until Gemini marks the uploaded segment as `ACTIVE`
6. Call `gemini-2.5-flash` with the same transcription prompt for every segment
7. Shift each segment transcript back onto the original video timeline
8. Join all segment transcripts into one final CSV and one XLSX in `./outputs/`
9. Delete temporary local video files and uploaded Gemini files

All Gemini calls are wrapped in a retry loop (3 attempts, exponential backoff) that retries on HTTP 408, 429, 500, 502, 503, 504, and unknown-status errors.

## Cost notes

Gemini 2.5 Pro is a premium model. A 60-minute video of roughly 500k input tokens plus a few thousand output tokens will typically cost a few USD per video, plus a small additional cost for the translation call. Check current pricing at https://ai.google.dev/pricing before running many batches.

## Files

```
video-transcriber/
    app.py              - Flask server, batch queue, background worker
    transcribe.py       - Gemini API calls with retries
    templates/
        index.html      - The one and only web page
    requirements.txt    - Python dependencies
    .env.example        - Template for your API key
    downloads/          - Temporary video files (auto-cleaned on success)
    outputs/            - Generated .txt files (persist across batches)
```

## Troubleshooting

**"gdown failed to download the file"**
Make sure the Google Drive link is shared as "Anyone with the link". Right-click the file in Drive, choose Share, change General access to "Anyone with the link", copy the link.

**"GEMINI_API_KEY is not set"**
You either did not create the `.env` file, or the virtual environment is not active. Activate the venv (`source venv/bin/activate`) before running `python app.py`.

**"Missing required video tools: ffmpeg, ffprobe"**
Install FFmpeg and make sure both `ffmpeg` and `ffprobe` are on your `PATH`. On macOS with Homebrew, `brew install ffmpeg` is the usual fix.

**Transcript is suspiciously short (warning in the log)**
Gemini can still occasionally return short output for an individual segment. Use the Retry button on the failed or short video.

**The page stops updating**
Check the terminal where `app.py` is running. If the process died, restart it. Batch state is in-memory only, so restarting the server clears any in-progress state, but already-saved `.txt` files in `outputs/` are preserved.

**Computer went to sleep mid-batch**
Processing will have paused and Gemini may time out. When you wake it up, the worker thread may or may not recover. Safest: stop the server, restart, and re-run the failed videos. Consider `caffeinate -i python app.py` on macOS to prevent sleep during a batch.

## Known limitations (v1)

- Single user, single batch at a time
- In-memory state: server restart loses batch status (but not output files)
- Segment overlap may create duplicate lines near batch boundaries; this is expected and can be cleaned manually
- Only public Google Drive links (no OAuth, no private files)
- Computer sleep or terminal close kills in-progress jobs
