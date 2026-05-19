from flask import Flask, render_template


FIXING_SRT_DELAY_URL = "https://example.com/fixing-srt-delay"
TRANSCRIPTION_PROJECT_URL = "http://35.225.88.95:5000/"

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        fixing_srt_delay_url=FIXING_SRT_DELAY_URL,
        transcription_project_url=TRANSCRIPTION_PROJECT_URL,
    )


if __name__ == "__main__":
    app.run(debug=False, port=5050)
