from pathlib import Path

from flask import Flask, render_template, send_from_directory


FIXING_SRT_DELAY_URL = "http://35.225.88.95:5001"
TRANSCRIPTION_PROJECT_URL = "http://35.225.88.95:5000/"
BASE_DIR = Path(__file__).resolve().parent
LOGO_DIR = BASE_DIR.parent / "fixing_srt_delay"

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        fixing_srt_delay_url=FIXING_SRT_DELAY_URL,
        transcription_project_url=TRANSCRIPTION_PROJECT_URL,
    )


@app.route("/assets/<path:filename>", methods=["GET"])
def asset(filename):
    return send_from_directory(LOGO_DIR, filename)


if __name__ == "__main__":
    app.run(debug=False, port=5050)
