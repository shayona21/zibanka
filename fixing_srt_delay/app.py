import os
import threading
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory, url_for
from werkzeug.utils import secure_filename

from main import process_srt_file


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
ALLOWED_EXTENSIONS = {".srt"}


def load_local_env(env_path):
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()

        if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
            continue

        key, value = stripped_line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


load_local_env(BASE_DIR / ".env")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
jobs = {}
jobs_lock = threading.Lock()


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def create_job(filename, input_language, output_file_name, batch_size):
    job_id = uuid4().hex
    job = {
        "id": job_id,
        "filename": filename,
        "input_language": input_language,
        "output_file_name": output_file_name,
        "batch_size": batch_size,
        "status": "queued",
        "progress": 0,
        "current_batch": 0,
        "total_batches": 0,
        "message": "Waiting to start.",
        "logs": [],
        "download_name": None,
        "error": None,
    }
    with jobs_lock:
        jobs[job_id] = job
    return job


def append_log(job, message):
    job["logs"].append(message)
    job["logs"] = job["logs"][-200:]


def update_job_progress(job, payload):
    message = payload.get("message", "")
    job["progress"] = payload.get("percent", job["progress"])
    job["current_batch"] = payload.get("current_batch", job["current_batch"])
    job["total_batches"] = payload.get("total_batches", job["total_batches"])
    job["message"] = message or job["message"]
    if message:
        append_log(job, message)


def run_job(job_id, upload_path):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        return

    def progress_callback(payload):
        with jobs_lock:
            live_job = jobs.get(job_id)
            if live_job is None:
                return
            update_job_progress(live_job, payload)

    try:
        with jobs_lock:
            job["status"] = "running"
            job["message"] = "Starting processing."
            append_log(job, "Starting processing.")

        output_path = process_srt_file(
            file_path=upload_path,
            input_language=job["input_language"],
            batch_size=job["batch_size"],
            output_file_name=job["output_file_name"],
            output_dir=OUTPUT_DIR,
            progress_callback=progress_callback
        )

        with jobs_lock:
            job["status"] = "completed"
            job["progress"] = 100
            job["message"] = "Processing complete."
            job["output_file_name"] = output_path.name
            job["download_name"] = output_path.name
            append_log(job, "Processing complete.")
    except Exception as exc:
        with jobs_lock:
            job["status"] = "failed"
            job["error"] = str(exc)
            job["message"] = "Processing failed."
            append_log(job, f"Error: {exc}")


@app.route("/", methods=["GET"])
def index():
    job_id = request.args.get("job")
    active_job = None

    if job_id:
        with jobs_lock:
            active_job = jobs.get(job_id)

    return render_template("index.html", active_job=active_job)


@app.route("/assets/<path:filename>", methods=["GET"])
def asset(filename):
    return send_from_directory(BASE_DIR, filename)


@app.route("/process", methods=["POST"])
def process():
    uploaded_file = request.files.get("srt_file")
    input_language = request.form.get("input_language", "").strip()
    output_file_name = request.form.get("output_file_name", "").strip()
    batch_size_raw = request.form.get("batch_size", "").strip()

    if not uploaded_file or uploaded_file.filename == "":
        return render_template(
            "index.html",
            active_job=None,
            form_error="Please upload an SRT file."
        ), 400

    if not allowed_file(uploaded_file.filename):
        return render_template(
            "index.html",
            active_job=None,
            form_error="Only .srt files are supported."
        ), 400

    if not input_language:
        return render_template(
            "index.html",
            active_job=None,
            form_error="Please enter the input language."
        ), 400

    if not output_file_name:
        return render_template(
            "index.html",
            active_job=None,
            form_error="Please enter the output file name."
        ), 400

    if not batch_size_raw:
        return render_template(
            "index.html",
            active_job=None,
            form_error="Please enter a batch size between 1 and 100."
        ), 400

    try:
        batch_size = int(batch_size_raw)
    except ValueError:
        return render_template(
            "index.html",
            active_job=None,
            form_error="Batch size must be a whole number between 1 and 100."
        ), 400

    if batch_size < 1 or batch_size > 100:
        return render_template(
            "index.html",
            active_job=None,
            form_error="Batch size must be between 1 and 100."
        ), 400

    safe_name = secure_filename(uploaded_file.filename)
    unique_name = f"{uuid4().hex}_{safe_name}"
    upload_path = UPLOAD_DIR / unique_name
    uploaded_file.save(upload_path)

    job = create_job(
        filename=safe_name,
        input_language=input_language,
        output_file_name=output_file_name,
        batch_size=batch_size)

    worker = threading.Thread(target=run_job, args=(job["id"], upload_path), daemon=True)
    worker.start()

    return render_template("index.html", active_job=job, started=True)


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        return jsonify({"error": "Job not found."}), 404

    response = {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "current_batch": job["current_batch"],
        "total_batches": job["total_batches"],
        "message": job["message"],
        "logs": job["logs"],
        "error": job["error"],
        "filename": job["filename"],
        "input_language": job["input_language"],
        "output_file_name": job["output_file_name"],
        "batch_size": job["batch_size"],
        "download_url": url_for("download_file", filename=job["download_name"]) if job["download_name"] else None,
    }

    return jsonify(response)


@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        return "File not found.", 404

    return send_file(file_path, as_attachment=True, download_name=file_path.name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(debug=False, port=port)
