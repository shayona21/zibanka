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
    },
    {
        "name": "English to German Subtitle Translation (With Context)",
        "description": "Translate English subtitles into German while preserving context.",
        "url": "http://35.225.88.95:5003",
        "category": "Translation",
        "icon": "language",
    },
    {
        "name": "English to Malay Subtitle Translation (With Context)",
        "description": "Translate English subtitles into Malay while preserving context.",
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
)

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", tools=TOOLS)


@app.get("/assets/zibanka-logo.jpeg")
def logo():
    return send_from_directory(LOGO_DIR, "zibanka_logo.jpeg", max_age=86400)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False)
