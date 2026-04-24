"""
Flask web server for the video transcription tool.

Run with:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import gdown
from flask import Flask, render_template, request, jsonify, send_from_directory, abort, redirect, url_for
from dotenv import load_dotenv

import transcribe


# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

SUPPORTED_LANGUAGES = [
    "Hindi", "Telugu", "Bengali", "Tamil", "English",
    "Marathi", "Gujarati", "Kannada", "Malayalam", "Urdu", "Punjabi",
]

MAX_LINKS_PER_BATCH = 3

app = Flask(__name__)


# -------------------------------------------------------------------
# In-memory batch state
# -------------------------------------------------------------------
# Only one batch runs at a time. This dict holds the current batch's state.
# Fields:
#   id: str
#   status: "running" | "done"
#   source_language: str
#   target_language: str
#   prompt: str               # the transcription prompt used for this batch
#   started_at: iso timestamp
#   videos: list of dicts, each with:
#       index, link, status, step, elapsed_seconds, started_at, finished_at,
#       error, transcript_file, translation_file, log (list of strings)
BATCH_LOCK = threading.Lock()
BATCH = None  # type: dict | None


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
GDRIVE_PATTERNS = [
    re.compile(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)"),
    re.compile(r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)"),
    re.compile(r"drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)"),
    re.compile(r"[?&]id=([a-zA-Z0-9_-]+)"),
]


def extract_drive_id(link: str) -> str | None:
    for pat in GDRIVE_PATTERNS:
        m = pat.search(link)
        if m:
            return m.group(1)
    return None


def safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_") or "file"


def download_drive_video(link: str, target_path: Path, log) -> Path:
    """Download a public Google Drive video to target_path. Returns the actual path used."""
    file_id = extract_drive_id(link)
    if not file_id:
        raise RuntimeError(
            f"Could not extract a Google Drive file ID from the link: {link}"
        )
    url = f"https://drive.google.com/uc?id={file_id}"
    log(f"Downloading from Google Drive (file id: {file_id})...")
    # gdown returns the output path on success, None on failure.
    result = gdown.download(url=url, output=str(target_path), quiet=True)
    if not result:
        raise RuntimeError(
            "gdown failed to download the file. "
            "Make sure the link is shared as 'Anyone with the link'."
        )
    actual = Path(result)
    size_mb = actual.stat().st_size / (1024 * 1024)
    log(f"Download complete: {actual.name} ({size_mb:.1f} MB)")
    return actual


def write_output_file(content: str, filename: str) -> Path:
    path = OUTPUTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    return path


# -------------------------------------------------------------------
# Background worker
# -------------------------------------------------------------------
def run_batch(batch_id: str):
    """Process all videos in the current batch sequentially."""
    global BATCH

    with BATCH_LOCK:
        if not BATCH or BATCH["id"] != batch_id:
            return
        videos = BATCH["videos"]
        source_lang = BATCH["source_language"]
        target_lang = BATCH["target_language"]
        prompt = BATCH["prompt"]
        needs_translation = source_lang != target_lang

    for video in videos:
        idx = video["index"]

        def log(msg: str, v=video):
            stamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{stamp}] {msg}"
            with BATCH_LOCK:
                v["log"].append(line)
                # Keep log bounded so the UI doesn't get huge.
                if len(v["log"]) > 200:
                    v["log"] = v["log"][-200:]

        with BATCH_LOCK:
            video["status"] = "running"
            video["step"] = "Downloading"
            video["started_at"] = datetime.now().isoformat(timespec="seconds")

        start_time = time.time()
        local_video_path: Path | None = None

        try:
            # Step 1: Download from Drive
            tmp_name = f"batch_{batch_id}_video_{idx}.mp4"
            tmp_path = DOWNLOADS_DIR / tmp_name
            local_video_path = download_drive_video(video["link"], tmp_path, log)

            # Step 2: Transcribe
            with BATCH_LOCK:
                video["step"] = "Transcribing"
            transcript = transcribe.transcribe_video(
                str(local_video_path),
                source_language=source_lang,
                prompt=prompt,
                log_callback=log,
            )
            transcript_filename = f"video{idx}_{safe_slug(source_lang)}_transcript.txt"
            write_output_file(transcript, transcript_filename)
            with BATCH_LOCK:
                video["transcript_file"] = transcript_filename
            log(f"Transcript saved: {transcript_filename} ({len(transcript)} chars)")

            # Sanity check: flag suspiciously short transcripts.
            if len(transcript) < 500:
                log(
                    "WARNING: transcript is under 500 characters, which is short for a "
                    "60-minute video. You may want to retry."
                )

            # Step 3: Translate (if needed)
            if needs_translation:
                with BATCH_LOCK:
                    video["step"] = "Translating"
                translation = transcribe.translate_text(
                    transcript,
                    source_language=source_lang,
                    target_language=target_lang,
                    log_callback=log,
                )
                translation_filename = f"video{idx}_{safe_slug(target_lang)}_translation.txt"
                write_output_file(translation, translation_filename)
                with BATCH_LOCK:
                    video["translation_file"] = translation_filename
                log(f"Translation saved: {translation_filename} ({len(translation)} chars)")
            else:
                log("Source and target language are the same; skipping translation.")

            # Success: clean up the local video file.
            try:
                if local_video_path and local_video_path.exists():
                    local_video_path.unlink()
                    log("Cleaned up local video file.")
            except Exception as e:
                log(f"Warning: could not delete local video file: {e}")

            with BATCH_LOCK:
                video["status"] = "done"
                video["step"] = "Done"
                video["finished_at"] = datetime.now().isoformat(timespec="seconds")
                video["elapsed_seconds"] = int(time.time() - start_time)

        except Exception as e:
            log(f"ERROR: {e}")
            with BATCH_LOCK:
                video["status"] = "failed"
                video["step"] = "Failed"
                video["error"] = str(e)
                video["finished_at"] = datetime.now().isoformat(timespec="seconds")
                video["elapsed_seconds"] = int(time.time() - start_time)
            # We keep the downloaded video on failure so the user can retry
            # without re-downloading. It will be overwritten on retry.
            # Continue with the next video.

    with BATCH_LOCK:
        if BATCH and BATCH["id"] == batch_id:
            BATCH["status"] = "done"
            BATCH["finished_at"] = datetime.now().isoformat(timespec="seconds")


def retry_single_video(video: dict, source_lang: str, target_lang: str, prompt: str, batch_id: str):
    """Retry a single failed video. Runs in its own thread."""
    needs_translation = source_lang != target_lang
    idx = video["index"]

    def log(msg: str, v=video):
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        with BATCH_LOCK:
            v["log"].append(line)
            if len(v["log"]) > 200:
                v["log"] = v["log"][-200:]

    with BATCH_LOCK:
        video["status"] = "running"
        video["step"] = "Downloading"
        video["error"] = None
        video["started_at"] = datetime.now().isoformat(timespec="seconds")
        video["finished_at"] = None
        video["elapsed_seconds"] = 0
        video["transcript_file"] = None
        video["translation_file"] = None

    start_time = time.time()
    local_video_path: Path | None = None

    try:
        tmp_name = f"batch_{batch_id}_video_{idx}.mp4"
        tmp_path = DOWNLOADS_DIR / tmp_name

        # If the downloaded file from the previous failed attempt still exists
        # and looks healthy, reuse it.
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            log(f"Reusing previously downloaded file: {tmp_path.name}")
            local_video_path = tmp_path
        else:
            local_video_path = download_drive_video(video["link"], tmp_path, log)

        with BATCH_LOCK:
            video["step"] = "Transcribing"
        transcript = transcribe.transcribe_video(
            str(local_video_path),
            source_language=source_lang,
            prompt=prompt,
            log_callback=log,
        )
        transcript_filename = f"video{idx}_{safe_slug(source_lang)}_transcript.txt"
        write_output_file(transcript, transcript_filename)
        with BATCH_LOCK:
            video["transcript_file"] = transcript_filename
        log(f"Transcript saved: {transcript_filename} ({len(transcript)} chars)")

        if needs_translation:
            with BATCH_LOCK:
                video["step"] = "Translating"
            translation = transcribe.translate_text(
                transcript, source_lang, target_lang, log_callback=log,
            )
            translation_filename = f"video{idx}_{safe_slug(target_lang)}_translation.txt"
            write_output_file(translation, translation_filename)
            with BATCH_LOCK:
                video["translation_file"] = translation_filename
            log(f"Translation saved: {translation_filename} ({len(translation)} chars)")

        try:
            if local_video_path and local_video_path.exists():
                local_video_path.unlink()
        except Exception:
            pass

        with BATCH_LOCK:
            video["status"] = "done"
            video["step"] = "Done"
            video["finished_at"] = datetime.now().isoformat(timespec="seconds")
            video["elapsed_seconds"] = int(time.time() - start_time)

    except Exception as e:
        log(f"ERROR: {e}")
        with BATCH_LOCK:
            video["status"] = "failed"
            video["step"] = "Failed"
            video["error"] = str(e)
            video["finished_at"] = datetime.now().isoformat(timespec="seconds")
            video["elapsed_seconds"] = int(time.time() - start_time)


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    with BATCH_LOCK:
        batch_snapshot = BATCH.copy() if BATCH else None
        if batch_snapshot:
            batch_snapshot["videos"] = [v.copy() for v in BATCH["videos"]]
    return render_template(
        "index.html",
        languages=SUPPORTED_LANGUAGES,
        max_links=MAX_LINKS_PER_BATCH,
        default_prompt=transcribe.DEFAULT_TRANSCRIPTION_PROMPT,
        batch=batch_snapshot,
    )


@app.route("/start", methods=["POST"])
def start():
    global BATCH

    # Block starting a new batch while one is running.
    with BATCH_LOCK:
        if BATCH and BATCH["status"] == "running":
            return "A batch is already running. Wait for it to finish.", 400

    links_raw = request.form.get("links", "").strip()
    source_lang = request.form.get("source_language", "").strip()
    target_lang = request.form.get("target_language", "").strip()
    prompt_raw = request.form.get("prompt", "")

    if source_lang not in SUPPORTED_LANGUAGES:
        return f"Invalid source language: {source_lang}", 400
    if target_lang not in SUPPORTED_LANGUAGES:
        return f"Invalid target language: {target_lang}", 400

    # Fall back to the default if the user blanked out the prompt box.
    prompt = prompt_raw.strip() or transcribe.DEFAULT_TRANSCRIPTION_PROMPT

    links = [ln.strip() for ln in links_raw.splitlines() if ln.strip()]
    if not links:
        return "Please paste at least one Google Drive link.", 400
    if len(links) > MAX_LINKS_PER_BATCH:
        return f"Please paste at most {MAX_LINKS_PER_BATCH} links.", 400

    # Validate each link parses.
    for ln in links:
        if not extract_drive_id(ln):
            return f"Could not parse a Google Drive file ID from: {ln}", 400

    batch_id = uuid.uuid4().hex[:8]
    videos = []
    for i, ln in enumerate(links, start=1):
        videos.append({
            "index": i,
            "link": ln,
            "status": "queued",
            "step": "Queued",
            "elapsed_seconds": 0,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "transcript_file": None,
            "translation_file": None,
            "log": [],
        })

    with BATCH_LOCK:
        BATCH = {
            "id": batch_id,
            "status": "running",
            "source_language": source_lang,
            "target_language": target_lang,
            "prompt": prompt,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "videos": videos,
        }

    worker = threading.Thread(target=run_batch, args=(batch_id,), daemon=True)
    worker.start()

    return redirect(url_for("index"))


@app.route("/status")
def status():
    """Return the current batch state as JSON. Used by the auto-refresh in the UI."""
    with BATCH_LOCK:
        if not BATCH:
            return jsonify({"batch": None})
        snapshot = BATCH.copy()
        snapshot["videos"] = [v.copy() for v in BATCH["videos"]]
    return jsonify({"batch": snapshot})


@app.route("/retry/<int:video_index>", methods=["POST"])
def retry(video_index):
    with BATCH_LOCK:
        if not BATCH:
            return "No active batch.", 400
        # Don't allow retry while the batch is actively processing another video.
        any_running = any(v["status"] == "running" for v in BATCH["videos"])
        if any_running:
            return "Cannot retry while another video is running.", 400
        video = next((v for v in BATCH["videos"] if v["index"] == video_index), None)
        if not video:
            return "Video not found.", 404
        if video["status"] != "failed":
            return "Only failed videos can be retried.", 400
        source_lang = BATCH["source_language"]
        target_lang = BATCH["target_language"]
        prompt = BATCH["prompt"]
        batch_id = BATCH["id"]

    worker = threading.Thread(
        target=retry_single_video,
        args=(video, source_lang, target_lang, prompt, batch_id),
        daemon=True,
    )
    worker.start()
    return redirect(url_for("index"))


@app.route("/new", methods=["POST"])
def new_batch():
    """Clear the current batch (only allowed when not running)."""
    global BATCH
    with BATCH_LOCK:
        if BATCH and BATCH["status"] == "running":
            return "Cannot clear while a batch is running.", 400
        BATCH = None
    return redirect(url_for("index"))


@app.route("/outputs/<path:filename>")
def download_output(filename):
    # Basic safety: only allow plain filenames, no traversal.
    if "/" in filename or ".." in filename:
        abort(404)
    return send_from_directory(OUTPUTS_DIR, filename, as_attachment=True)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY is not set. Create a .env file with:")
        print("    GEMINI_API_KEY=your-key-here")
    # threaded=True is essential so the /status endpoint stays responsive
    # while the background worker is running.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
