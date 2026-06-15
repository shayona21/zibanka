# Video Transcriber

A simple local web tool for transcribing Google Drive videos and optionally translating the transcript between Indian languages, using Google Gemini 2.5 Flash.

## Features

- Paste up to 3 public Google Drive video links at once
- Pick source language and target translation language from 11 options
- If source and target are the same, translation is skipped automatically
- Sequential background processing; browser tab can be closed while jobs run
- Optional per-video time range in `mm:ss` format
- Per-video status, live log, transcript CSV and XLSX download links
- Retry button for failed videos (reuses the already-downloaded file)
- Pipe-delimited transcript output with timestamps and speaker columns
- Three retries with exponential backoff on transient API errors

## Supported languages

Hindi, Telugu, Bengali, Tamil, English, Marathi, Gujarati, Kannada, Malayalam, Urdu, Punjabi.

## Prerequisites

- macOS with Python 3.10 or newer (`python3 --version` to check)
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
python transcribe_app.py
```

Then open http://127.0.0.1:5001 in your browser.

Keep the terminal window open for the duration of the batch. To stop the server, press Ctrl+C.

## How it works

For each video in the batch, sequentially:

1. Download the video from Google Drive using `gdown` into `./downloads/`
2. Upload the downloaded video to the Gemini Files API
3. Wait until Gemini marks the uploaded file as `ACTIVE`
4. Call Gemini once to produce the transcript CSV
5. Save the transcript to `./outputs/` as CSV and XLSX
6. Delete the local video file and the uploaded Gemini file

All Gemini calls are wrapped in a retry loop (3 attempts, exponential backoff) that retries on HTTP 408, 429, 500, 502, 503, 504, and unknown-status errors.

## Cost notes

Gemini 2.5 Pro is a premium model. A 60-minute video of roughly 500k input tokens plus a few thousand output tokens will typically cost a few USD per video, plus a small additional cost for the translation call. Check current pricing at https://ai.google.dev/pricing before running many batches.

## Files

```
video-transcriber/
    transcribe_app.py   - Flask server, batch queue, background worker
    transcribe.py       - Gemini API calls with retries
    templates/
        index.html      - The one and only web page
    requirements.txt    - Python dependencies
    .env.example        - Template for your API key
    downloads/          - Temporary video files (auto-cleaned on success)
    outputs/            - Generated CSV/XLSX files (persist across batches)
```

## Troubleshooting

**"gdown failed to download the file"**
Make sure the Google Drive link is shared as "Anyone with the link". Right-click the file in Drive, choose Share, change General access to "Anyone with the link", copy the link.

**"GEMINI_API_KEY is not set"**
You either did not create the `.env` file, or the virtual environment is not active. Activate the venv (`source venv/bin/activate`) before running `python transcribe_app.py`.

**Transcript is suspiciously short (warning in the log)**
Gemini can occasionally return short output for a video. Use the Retry button on the failed or short video.

**The page stops updating**
Check the terminal where `transcribe_app.py` is running. If the process died, restart it. Batch state is in-memory only, so restarting the server clears any in-progress state, but already-saved output files in `outputs/` are preserved.

**Computer went to sleep mid-batch**
Processing will have paused and Gemini may time out. When you wake it up, the worker thread may or may not recover. Safest: stop the server, restart, and re-run the failed videos. Consider `caffeinate -i python transcribe_app.py` on macOS to prevent sleep during a batch.

## Known limitations (v1)

- Single user, single batch at a time
- In-memory state: server restart loses batch status (but not output files)
- Only public Google Drive links (no OAuth, no private files)
- Computer sleep or terminal close kills in-progress jobs
