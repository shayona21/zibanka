import os
from pathlib import Path

from flask import Flask, render_template, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
LOGO_DIR = BASE_DIR.parent / "fixing_srt_delay"

TOOLS = (
    {
        "name": "Dubbing Tool",
        "description": "Turn dialogue spreadsheets into synchronized voice tracks for each episode and character.",
        "url": "http://136.113.166.224:5071",
        "category": "Dubbing",
        "icon": "audio",
    },
    {
        "name": "Dubbing Quality Check",
        "description": "Preview each character's dubbed audio against the video and regenerate individual lines before delivery.",
        "url": "http://136.113.166.224:5072",
        "category": "Dubbing",
        "icon": "video",
    },
    {
        "name": "Video Transcription",
        "description": "Turn video content into a clear, ready-to-use transcript.",
        "url": "http://136.113.166.224:5000",
        "category": "Transcription",
        "icon": "video",
    },
    {
        "name": "German Subtitling",
        "description": "Translate English subtitles into German, with additional context input for more accurate subtitle translation.",
        "url": "http://136.113.166.224:5003",
        "category": "Subtitling",
        "icon": "language",
    },
    {
        "name": "Malay Subtitling",
        "description": "Translate English subtitles into Malay, with additional context input for more accurate subtitle translation.",
        "url": "http://136.113.166.224:5040",
        "category": "Subtitling",
        "icon": "language",
    },
    {
        "name": "Subtitle Translation (Without Context)",
        "description": "Translate subtitle text directly when supporting context is unavailable.",
        "url": "http://136.113.166.224:5030",
        "category": "Translation",
        "icon": "subtitles",
    },
    {
        "name": "SRT Conversion - From Roman to Native Script",
        "description": "Convert romanized SRT subtitle text into its native script.",
        "url": "http://136.113.166.224:5001",
        "category": "Conversion",
        "icon": "script",
    },
    {
        "name": "Audio Description",
        "description": "Convert audio description text into natural-sounding speech for accessible media.",
        "url": "http://136.113.166.224:5070/",
        "category": "Audio Description",
        "icon": "audio",
    },
)

# Order and icon of each product group on the page.
GROUPS = (
    ("Dubbing", "audio"),
    ("Audio Description", "audio"),
    ("Subtitling", "language"),
    ("Translation", "subtitles"),
    ("Transcription", "video"),
    ("Conversion", "script"),
)

# Address shown under Contact. Leave empty to hide it.
CONTACT_EMAIL = "manoj.gupta@zibanka.com"


def grouped_tools():
    groups = []
    for name, icon in GROUPS:
        tools = [t for t in TOOLS if t["category"] == name]
        if tools:
            groups.append({"name": name, "icon": icon, "tools": tools})
    return groups


app = Flask(__name__)


@app.get("/")
def index():
    return render_template(
        "index.html",
        groups=grouped_tools(),
        contact_email=CONTACT_EMAIL,
        qc_url=next((t["url"] for t in TOOLS if t["name"] == "Dubbing Quality Check"), None),
    )


@app.get("/assets/zibanka-logo.jpeg")
def logo():
    return send_from_directory(LOGO_DIR, "zibanka_logo.jpeg", max_age=86400)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5040)
