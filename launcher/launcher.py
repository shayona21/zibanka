import os
from pathlib import Path

from flask import Flask, render_template, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
LOGO_DIR = BASE_DIR.parent / "fixing_srt_delay"

TOOLS = (
    {
        "name": "Video Transcription",
        "description": "Turn video content into a clear, ready-to-use transcript.",
        "url": "http://35.225.88.95:5000",
        "category": "Transcription",
        "icon": "video",
    },
    {
        "name": "Video Transcription (in batches)",
        "description": "Transcribe multiple video files together in one streamlined workflow.",
        "url": "http://35.225.88.95:5006",
        "category": "Transcription",
        "icon": "batch",
        "available": False,
        "status": "Under construction",
    },
    {
        "name": "English to German Translation",
        "description": "Translate English subtitles into German, with additional context input for more accurate subtitle translation.",
        "url": "http://35.225.88.95:5003",
        "category": "Translation",
        "icon": "language",
    },
    {
        "name": "English to Malay Translation",
        "description": "Translate English subtitles into Malay, with additional context input for more accurate subtitle translation.",
        "url": "http://35.225.88.95:5040",
        "category": "Translation",
        "icon": "language",
    },
    {
        "name": "Subtitle Translation (Without Context)",
        "description": "Translate subtitle text directly when supporting context is unavailable.",
        "url": "http://35.225.88.95:5030",
        "category": "Translation",
        "icon": "subtitles",
    },
    {
        "name": "SRT Conversion - From Roman to Native Script",
        "description": "Convert romanized SRT subtitle text into its native script.",
        "url": "http://35.225.88.95:5001",
        "category": "Conversion",
        "icon": "script",
    },
    {
        "name": "Audio Description Text to Speech",
        "description": "Convert audio description text into natural-sounding speech for accessible media.",
        "url": "http://35.225.88.95:5070/",
        "category": "Audio Description",
        "icon": "audio",
    },
)

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", tools=TOOLS)


@app.get("/assets/zibanka-logo.jpeg")
def logo():
    return send_from_directory(LOGO_DIR, "zibanka_logo.jpeg", max_age=86400)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5040)
