# main.py
# This is the main orchestrator script.
# Cloud Scheduler calls this every 10 minutes via HTTP.

import os
import json
import requests
from flask import Flask, jsonify
from drive_helper import get_drive_service, list_videos_in_folder, upload_txt_to_drive

app = Flask(__name__)

# ── Your Configuration ───────────────────────────────────────────────────────

INPUT_FOLDER_ID  = "1xm-X1EExsWfmRObuSlBH0X4XnH9xrQfZ"
OUTPUT_FOLDER_ID = "17ND7nIC6_O1TdPUi8WnLcPra2JwRPZw4"

# The URL of your existing transcription app on GCP
TRANSCRIPTION_APP_URL = "http://35.225.88.95:5000/"

# A simple file that tracks which video IDs we already processed
PROCESSED_IDS_FILE = "processed_ids.json"


# ── Helper: Load Processed IDs ───────────────────────────────────────────────

def load_processed_ids():
    """
    Read the list of already-processed video IDs from our tracking file.
    If the file doesn't exist yet, return an empty list.
    """
    if os.path.exists(PROCESSED_IDS_FILE):
        with open(PROCESSED_IDS_FILE, "r") as f:
            return json.load(f)   # load the list from the file
    return []  # first run — nothing processed yet


# ── Helper: Save Processed IDs ───────────────────────────────────────────────

def save_processed_ids(id_list):
    """
    Save the updated list of processed video IDs back to the tracking file.
    """
    with open(PROCESSED_IDS_FILE, "w") as f:
        json.dump(id_list, f)  # write the list as JSON


# ── Helper: Call Your Transcription App ─────────────────────────────────────

def call_transcription_app(drive_url, language, context, video_name):
    """
    Submit a video to the transcription app via the /start endpoint.
    The app uses form data (not JSON), so we send it that way.

    Parameters:
        drive_url  : the shareable Google Drive link to the video
        language   : spoken language e.g. "English"
        context    : short description of the video content
        video_name : used as the title field

    Returns:
        True if submitted successfully, False if something went wrong
    """

    # Build the form data exactly as the app expects it
    form_data = {
        "link_1":           drive_url,
        "title_1":          video_name,
        "context_1":        context,
        "source_language":  language,
        # leave out "translate" since we don't need translation
    }

    try:
        # Send as form data using 'data=' not 'json='
        response = requests.post(
            TRANSCRIPTION_APP_URL + "start",
            data=form_data,       # <-- form data, not JSON
            timeout=30,
            allow_redirects=True  # the app redirects to / after submitting
        )

        # A 200 response means the job was accepted and is now running
        if response.status_code == 200:
            print(f"Successfully submitted to transcription app: {video_name}")
            return True
        else:
            print(f"Submission failed with status {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"Error submitting to transcription app: {e}")
        return False


# ── Helper: Wait for Transcription to Finish ────────────────────────────────

def wait_for_transcription(video_name, timeout_minutes=30):
    """
    Poll the /status endpoint every 15 seconds until the transcription
    is done, then return the output filename.

    Parameters:
        video_name      : used to match the right video in the status response
        timeout_minutes : give up after this many minutes (default 30)

    Returns:
        The transcript text as a string, or None if it failed/timed out
    """

    import time

    max_attempts = (timeout_minutes * 60) // 15  # how many 15-second checks
    attempt = 0

    print(f"Waiting for transcription to finish...")

    while attempt < max_attempts:

        try:
            # Check the current status of the batch
            response = requests.get(TRANSCRIPTION_APP_URL + "status", timeout=10)
            data = response.json()

            batch = data.get("batch")

            # If there's no active batch yet, wait a moment
            if not batch:
                print("No active batch yet, waiting...")
                time.sleep(15)
                attempt += 1
                continue

            # Look through the videos in the batch for ours
            for video in batch.get("videos", []):

                status = video.get("status")
                step   = video.get("step", "")

                print(f"  Status: {status} — {step}")

                # If it finished successfully, download the transcript
                if status == "done":
                    transcript_file = video.get("transcript_file")

                    if transcript_file:
                        # Download the actual .txt file from /outputs/<filename>
                        txt_url = TRANSCRIPTION_APP_URL + "outputs/" + transcript_file
                        txt_response = requests.get(txt_url, timeout=30)

                        if txt_response.status_code == 200:
                            print(f"Transcript downloaded successfully!")
                            return txt_response.text  # return the text content
                        else:
                            print(f"Could not download transcript file: {txt_url}")
                            return None

                # If it failed, stop waiting
                elif status == "failed":
                    error = video.get("error", "unknown error")
                    print(f"Transcription failed: {error}")
                    return None

            # Still running — wait 15 seconds and check again
            time.sleep(15)
            attempt += 1

        except Exception as e:
            print(f"Error checking status: {e}")
            time.sleep(15)
            attempt += 1

    print(f"Timed out waiting for transcription after {timeout_minutes} minutes")
    return None


# ── Main Route: Called by Cloud Scheduler ────────────────────────────────────

@app.route("/run", methods=["POST", "GET"])
def run_orchestrator():
    """
    This function runs every time Cloud Scheduler calls /run.
    It checks Drive, finds new videos, and processes them.
    """

    print("Orchestrator started...")

    # Step 1: Connect to Google Drive
    service = get_drive_service()

    # Step 2: Get the list of videos currently in the input folder
    videos = list_videos_in_folder(service, INPUT_FOLDER_ID)
    print(f"Found {len(videos)} video(s) in the input folder")

    # Step 3: Load the list of videos we already processed
    processed_ids = load_processed_ids()

    # Step 4: Loop through each video found in the folder
    for video in videos:

        video_id   = video["id"]
        video_name = video["name"]

        # Skip this video if we already processed it
        if video_id in processed_ids:
            print(f"Skipping (already processed): {video_name}")
            continue

        print(f"Processing new video: {video_name}")

        # Step 5: Build the shareable Drive URL for the video
        # Using confirm=t helps gdown bypass the large file warning page
        drive_url = f"https://drive.google.com/uc?id={video_id}&confirm=t"

        # Step 6: For now, use placeholder language and context
        # (In Step 3, Gemini will fill these in automatically)
        language = "Hindi"
        context  = "This is a hindi language"

        # Step 7: Submit the video to the transcription app
        submitted = call_transcription_app(drive_url, language, context, video_name)

        # Step 8: If submitted successfully, wait for it to finish
        if submitted:
            transcription_text = wait_for_transcription(video_name)
        else:
            transcription_text = None

        # Step 9: If we got a transcription, upload it to the output Drive folder
        if transcription_text:
            output_file_name = video_name.rsplit(".", 1)[0] + "_transcript.txt"
            # e.g. "interview.mp4" → "interview_transcript.txt"

            upload_txt_to_drive(
                service,
                OUTPUT_FOLDER_ID,
                output_file_name,
                transcription_text
            )

            # Step 10: Mark this video as processed so we don't run it again
            processed_ids.append(video_id)
            save_processed_ids(processed_ids)

            print(f"Done: {video_name} → {output_file_name}")

        else:
            print(f"Transcription failed for: {video_name}. Will retry next run.")

    # This return is OUTSIDE the for loop — runs after ALL videos are processed
    return jsonify({"status": "ok", "message": "Orchestrator run complete"})


# ── Run the App ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Port 8080 is the default Cloud Run port
    app.run(host="0.0.0.0", port=8080)