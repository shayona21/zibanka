import os
import threading
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory, url_for
from werkzeug.utils import secure_filename

from main import CHARACTER_LIST_OPTIONS, SERIES_PROMPTS, get_character_reference_path, process_srt_file


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
ALLOWED_EXTENSIONS = {".srt"}
SUPPORTED_SERIES = [
    {"key": series_key, "label": config["label"]}
    for series_key, config in SERIES_PROMPTS.items()
]
SUPPORTED_CHARACTER_LISTS = [
    {"key": option_key, "label": option["label"]}
    for option_key, option in CHARACTER_LIST_OPTIONS.items()
]


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


def create_job(files, series_key, batch_size, character_reference_key="none", character_reference_name=None, character_reference_path=None):
    job_id = uuid4().hex
    job = {
        "id": job_id,
        "title": f"{len(files)} file{'s' if len(files) != 1 else ''} queued",
        "series_key": series_key,
        "series_label": SERIES_PROMPTS[series_key]["label"],
        "character_reference_key": character_reference_key,
        "character_reference_name": character_reference_name,
        "character_reference_path": character_reference_path,
        "files": files,
        "current_file_number": 0,
        "total_files": len(files),
        "current_file_name": "",
        "current_output_file_name": "",
        "batch_size": batch_size,
        "status": "queued",
        "progress": 0,
        "current_batch": 0,
        "total_batches": 0,
        "message": "Waiting to start.",
        "logs": [],
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


def run_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        return

    total_files = len(job["files"])

    try:
        with jobs_lock:
            job["status"] = "running"
            job["message"] = "Starting processing."
            append_log(job, "Starting processing.")

        for file_index, file_job in enumerate(job["files"], start=1):
            with jobs_lock:
                live_job = jobs.get(job_id)
                if live_job is None:
                    return

                live_job["current_file_number"] = file_index
                live_job["current_file_name"] = file_job["filename"]
                live_job["current_output_file_name"] = file_job["requested_output_file_name"]
                live_job["current_batch"] = 0
                live_job["total_batches"] = 0
                file_job["status"] = "running"
                start_message = f"Starting file {file_index} of {total_files}: {file_job['filename']}"
                live_job["message"] = start_message
                append_log(live_job, start_message)

            def progress_callback(payload, current_file_index=file_index, current_file=file_job):
                file_percent = payload.get("percent", 0)
                overall_percent = int(
                    (((current_file_index - 1) * 100) + file_percent) / total_files
                )
                message = payload.get("message", "")
                if message:
                    message = f"File {current_file_index}/{total_files} ({current_file['filename']}): {message}"

                progress_payload = {
                    "current_batch": payload.get("current_batch", 0),
                    "total_batches": payload.get("total_batches", 0),
                    "percent": overall_percent,
                    "message": message,
                }

                with jobs_lock:
                    live_job = jobs.get(job_id)
                    if live_job is None:
                        return
                    update_job_progress(live_job, progress_payload)

            output_path = process_srt_file(
                file_path=file_job["upload_path"],
                series_key=job["series_key"],
                batch_size=job["batch_size"],
                output_file_name=file_job["requested_output_file_name"],
                output_dir=OUTPUT_DIR,
                character_reference_path=job["character_reference_path"],
                progress_callback=progress_callback,
            )

            with jobs_lock:
                live_job = jobs.get(job_id)
                if live_job is None:
                    return

                file_job["status"] = "completed"
                file_job["output_file_name"] = output_path.name
                file_job["download_name"] = output_path.name
                live_job["current_output_file_name"] = output_path.name
                live_job["progress"] = int((file_index / total_files) * 100)
                completion_message = f"Completed file {file_index} of {total_files}: {output_path.name}"
                live_job["message"] = completion_message
                append_log(live_job, completion_message)

        with jobs_lock:
            live_job = jobs.get(job_id)
            if live_job is None:
                return

            live_job["status"] = "completed"
            live_job["progress"] = 100
            live_job["message"] = "Processing complete."
            append_log(live_job, "Processing complete.")
    except Exception as exc:
        with jobs_lock:
            live_job = jobs.get(job_id)
            if live_job is None:
                return

            current_file_number = live_job.get("current_file_number", 0)
            if 1 <= current_file_number <= len(live_job["files"]):
                live_job["files"][current_file_number - 1]["status"] = "failed"
            live_job["status"] = "failed"
            live_job["error"] = str(exc)
            live_job["message"] = "Processing failed."
            append_log(live_job, f"Error: {exc}")


@app.route("/", methods=["GET"])
def index():
    job_id = request.args.get("job")
    active_job = None

    if job_id:
        with jobs_lock:
            active_job = jobs.get(job_id)

    return render_template(
        "index.html",
        active_job=active_job,
        series_options=SUPPORTED_SERIES,
        character_list_options=SUPPORTED_CHARACTER_LISTS,
    )


@app.route("/assets/<path:filename>", methods=["GET"])
def asset(filename):
    return send_from_directory(BASE_DIR, filename)


@app.route("/process", methods=["POST"])
def process():
    series_key = request.form.get("series_key", "").strip()
    batch_size_raw = request.form.get("batch_size", "").strip()
    character_reference_key = request.form.get("character_reference_key", "none").strip() or "none"
    pending_files = []

    for index in range(1, 4):
        uploaded_file = request.files.get(f"srt_file_{index}")
        output_file_name = request.form.get(f"output_file_name_{index}", "").strip()

        has_file = bool(uploaded_file and uploaded_file.filename)
        has_output_name = bool(output_file_name)

        if not has_file and not has_output_name:
            continue

        if has_file and not allowed_file(uploaded_file.filename):
            return render_template(
                "index.html",
                active_job=None,
                form_error=f"File {index} must be an .srt file.",
                series_options=SUPPORTED_SERIES,
            ), 400

        if has_file and not has_output_name:
            return render_template(
                "index.html",
                active_job=None,
                form_error=f"Please enter an output file name for file {index}.",
                series_options=SUPPORTED_SERIES,
            ), 400

        if has_output_name and not has_file:
            return render_template(
                "index.html",
                active_job=None,
                form_error=f"Please upload an SRT file for file {index}.",
                series_options=SUPPORTED_SERIES,
            ), 400

        pending_files.append(
            {
                "uploaded_file": uploaded_file,
                "requested_output_file_name": output_file_name,
            }
        )

    if not pending_files:
        return render_template(
            "index.html",
            active_job=None,
            form_error="Please upload at least one SRT file.",
            series_options=SUPPORTED_SERIES,
            character_list_options=SUPPORTED_CHARACTER_LISTS,
        ), 400

    if not series_key:
        return render_template(
            "index.html",
            active_job=None,
            form_error="Please select the subtitle project.",
            series_options=SUPPORTED_SERIES,
            character_list_options=SUPPORTED_CHARACTER_LISTS,
        ), 400

    if series_key not in SERIES_PROMPTS:
        return render_template(
            "index.html",
            active_job=None,
            form_error="Please select a valid subtitle project.",
            series_options=SUPPORTED_SERIES,
            character_list_options=SUPPORTED_CHARACTER_LISTS,
        ), 400

    if character_reference_key not in CHARACTER_LIST_OPTIONS:
        return render_template(
            "index.html",
            active_job=None,
            form_error="Please select a valid character list.",
            series_options=SUPPORTED_SERIES,
            character_list_options=SUPPORTED_CHARACTER_LISTS,
        ), 400

    character_reference_path = get_character_reference_path(character_reference_key)
    character_reference_name = CHARACTER_LIST_OPTIONS[character_reference_key]["label"]

    if not batch_size_raw:
        return render_template(
            "index.html",
            active_job=None,
            form_error="Please enter a batch size between 1 and 100.",
            series_options=SUPPORTED_SERIES,
            character_list_options=SUPPORTED_CHARACTER_LISTS,
        ), 400

    try:
        batch_size = int(batch_size_raw)
    except ValueError:
        return render_template(
            "index.html",
            active_job=None,
            form_error="Batch size must be a whole number between 1 and 100.",
            series_options=SUPPORTED_SERIES,
            character_list_options=SUPPORTED_CHARACTER_LISTS,
        ), 400

    if batch_size < 1 or batch_size > 100:
        return render_template(
            "index.html",
            active_job=None,
            form_error="Batch size must be between 1 and 100.",
            series_options=SUPPORTED_SERIES,
            character_list_options=SUPPORTED_CHARACTER_LISTS,
        ), 400

    file_entries = []
    for pending_file in pending_files:
        safe_name = secure_filename(pending_file["uploaded_file"].filename)
        unique_name = f"{uuid4().hex}_{safe_name}"
        upload_path = UPLOAD_DIR / unique_name
        pending_file["uploaded_file"].save(upload_path)

        file_entries.append(
            {
                "filename": safe_name,
                "requested_output_file_name": pending_file["requested_output_file_name"],
                "upload_path": str(upload_path),
                "status": "queued",
                "output_file_name": "",
                "download_name": None,
            }
        )

    job = create_job(
        files=file_entries,
        series_key=series_key,
        batch_size=batch_size,
        character_reference_key=character_reference_key,
        character_reference_name=character_reference_name,
        character_reference_path=str(character_reference_path) if character_reference_path else None,
    )

    worker = threading.Thread(target=run_job, args=(job["id"],), daemon=True)
    worker.start()

    return render_template(
        "index.html",
        active_job=job,
        started=True,
        series_options=SUPPORTED_SERIES,
        character_list_options=SUPPORTED_CHARACTER_LISTS,
    )


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        return jsonify({"error": "Job not found."}), 404

    response = {
        "id": job["id"],
        "title": job["title"],
        "status": job["status"],
        "progress": job["progress"],
        "current_batch": job["current_batch"],
        "total_batches": job["total_batches"],
        "message": job["message"],
        "logs": job["logs"],
        "error": job["error"],
        "series_key": job["series_key"],
        "series_label": job["series_label"],
        "character_reference_key": job["character_reference_key"],
        "character_reference_name": job["character_reference_name"],
        "files": [
            {
                "filename": file_job["filename"],
                "requested_output_file_name": file_job["requested_output_file_name"],
                "output_file_name": file_job["output_file_name"],
                "status": file_job["status"],
                "download_url": (
                    url_for("download_file", filename=file_job["download_name"])
                    if file_job["download_name"]
                    else None
                ),
            }
            for file_job in job["files"]
        ],
        "current_file_number": job["current_file_number"],
        "total_files": job["total_files"],
        "current_file_name": job["current_file_name"],
        "current_output_file_name": job["current_output_file_name"],
        "batch_size": job["batch_size"],
    }

    return jsonify(response)


@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        return "File not found.", 404

    return send_file(file_path, as_attachment=True, download_name=file_path.name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5003"))
    app.run(debug=False, port=port)
