# Video Transcriber

A simple local web tool for transcribing Google Drive videos (up to 60 minutes each) and optionally translating the transcript between Indian languages, using Google Gemini 2.5 Pro.

## Features

- Paste up to 3 public Google Drive video links at once
- Pick source language and target translation language from 11 options
- If source and target are the same, translation is skipped automatically
- Sequential background processing; browser tab can be closed while jobs run
- Per-video status, live log, transcript and translation download links
- Retry button for failed videos (reuses the already-downloaded file)
- Plain prose output, no timestamps or speaker labels
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
python app.py
```

Then open http://localhost:5000 in your browser.

Keep the terminal window open for the duration of the batch. To stop the server, press Ctrl+C.

## How it works

For each video in the batch, sequentially:

1. Download the video from Google Drive using `gdown` into `./downloads/`
2. Upload the video to the Gemini Files API
3. Wait until Gemini marks the file as `ACTIVE`
4. Call `gemini-2.5-pro` to produce a plain prose transcript in the source language
5. Save the transcript to `./outputs/videoN_<sourcelang>_transcript.txt`
6. If source and target differ, call Gemini again to translate the transcript
7. Save the translation to `./outputs/videoN_<targetlang>_translation.txt`
8. Delete the local video file and the uploaded Gemini file

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

**Transcript is suspiciously short (warning in the log)**
Gemini occasionally truncates output on very long videos. Use the Retry button on the failed or short video. If it keeps happening, the video may need to be split into shorter chunks (not currently implemented).

**The page stops updating**
Check the terminal where `app.py` is running. If the process died, restart it. Batch state is in-memory only, so restarting the server clears any in-progress state, but already-saved `.txt` files in `outputs/` are preserved.

**Computer went to sleep mid-batch**
Processing will have paused and Gemini may time out. When you wake it up, the worker thread may or may not recover. Safest: stop the server, restart, and re-run the failed videos. Consider `caffeinate -i python app.py` on macOS to prevent sleep during a batch.

## Known limitations (v1)

- Single user, single batch at a time
- In-memory state: server restart loses batch status (but not output files)
- No chunking of long videos; one Gemini call per video
- Only public Google Drive links (no OAuth, no private files)
- Computer sleep or terminal close kills in-progress jobs
