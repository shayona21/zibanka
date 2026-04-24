"""
Gemini API calls for transcription and translation.

Two public functions:
  - transcribe_video(video_path, source_language, prompt) -> str (transcript text)
  - translate_text(text, source_language, target_language) -> str (translated text)

Both functions retry transient errors up to 3 times with exponential backoff.
"""

import os
import time
import mimetypes
from google import genai
from google.genai import errors as genai_errors


MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 10  # doubled each retry

# How long we'll wait for Gemini to finish processing an uploaded video.
# A 60-min video typically takes 1-3 minutes. We give a generous ceiling.
FILE_ACTIVE_TIMEOUT_SECONDS = 600
FILE_POLL_INTERVAL_SECONDS = 5

# Default transcription prompt. The {source_language} placeholder (if present in
# a user-supplied prompt) is replaced with the chosen source language at call time.
DEFAULT_TRANSCRIPTION_PROMPT = (
    "Transcribe the spoken audio in this video. "
    "The video is in {source_language}. "
    "Output ONLY the transcribed text in {source_language}, as flowing natural prose. "
    "Do not include timestamps, speaker labels, stage directions, commentary, "
    "section headers, bullet points, or any formatting. "
    "Do not translate. Do not summarise. Transcribe the full audio from start to finish."
)


def _get_client():
    """Create a Gemini client using the GEMINI_API_KEY env var."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Put it in your .env file."
        )
    return genai.Client(api_key=api_key)


def _guess_mime_type(path: str) -> str:
    """Best-effort mime type for the video file."""
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("video/"):
        return mime
    # Default to mp4 if we can't tell.
    return "video/mp4"


def _wait_until_active(client, file_obj, log_callback=None):
    """Poll an uploaded file until its state is ACTIVE, or raise on FAILED/timeout."""
    elapsed = 0
    while elapsed < FILE_ACTIVE_TIMEOUT_SECONDS:
        refreshed = client.files.get(name=file_obj.name)
        state = str(refreshed.state) if refreshed.state is not None else ""
        # state is an enum; str() gives something like "FileState.ACTIVE"
        if "ACTIVE" in state:
            return refreshed
        if "FAILED" in state:
            raise RuntimeError(
                f"Gemini file processing failed (state={state})."
            )
        if log_callback:
            log_callback(f"Waiting for Gemini to process the video (state={state})...")
        time.sleep(FILE_POLL_INTERVAL_SECONDS)
        elapsed += FILE_POLL_INTERVAL_SECONDS
    raise RuntimeError(
        f"Timed out waiting for Gemini file to become ACTIVE after {FILE_ACTIVE_TIMEOUT_SECONDS}s."
    )


def _with_retries(fn, description, log_callback=None):
    """Run fn() with retries on transient errors. Raises after MAX_RETRIES failures."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except genai_errors.APIError as e:
            # Some APIErrors are not worth retrying (e.g. 400 bad request).
            # We retry on 429 / 500 / 503 / network-ish errors.
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            transient = status in (408, 429, 500, 502, 503, 504) or status is None
            last_error = e
            if not transient or attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            if log_callback:
                log_callback(
                    f"{description} failed (attempt {attempt}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {wait}s..."
                )
            time.sleep(wait)
        except Exception as e:
            # Non-API errors: retry once or twice then bubble up.
            last_error = e
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            if log_callback:
                log_callback(
                    f"{description} failed (attempt {attempt}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {wait}s..."
                )
            time.sleep(wait)
    # Should not reach here, but just in case:
    raise last_error if last_error else RuntimeError(f"{description} failed.")


def _render_prompt(prompt: str, source_language: str) -> str:
    """
    Substitute {source_language} in the prompt if present. If the user didn't
    include the placeholder, we return the prompt unchanged so their wording wins.
    """
    if "{source_language}" in prompt:
        return prompt.replace("{source_language}", source_language)
    return prompt


def transcribe_video(
    video_path: str,
    source_language: str,
    prompt: str | None = None,
    log_callback=None,
) -> str:
    """
    Upload a local video file to Gemini and return the transcribed text.

    Args:
        video_path: absolute path to the local video file
        source_language: human-readable language name (e.g. "Hindi")
        prompt: optional custom prompt string. If omitted or blank, uses the default.
                May include the placeholder {source_language} which will be substituted.
        log_callback: optional callable(str) for status updates

    Returns:
        The transcribed text as flowing prose.
    """
    client = _get_client()
    mime = _guess_mime_type(video_path)

    # Resolve the prompt: use the caller's if non-blank, else the default.
    effective_prompt = (prompt or "").strip() or DEFAULT_TRANSCRIPTION_PROMPT
    effective_prompt = _render_prompt(effective_prompt, source_language)

    if log_callback:
        log_callback(f"Uploading video to Gemini (mime={mime})...")

    def _upload():
        return client.files.upload(
            file=video_path,
            config={"mime_type": mime},
        )

    uploaded = _with_retries(_upload, "Uploading video", log_callback)

    if log_callback:
        log_callback(f"Upload complete. Gemini file name: {uploaded.name}")

    try:
        uploaded = _wait_until_active(client, uploaded, log_callback)

        if log_callback:
            log_callback("Transcribing (this can take several minutes for a 60-min video)...")

        def _generate():
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=[uploaded, effective_prompt],
            )

        response = _with_retries(_generate, "Transcription", log_callback)
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty transcript.")
        return text

    finally:
        # Always try to clean up the uploaded file from Gemini.
        try:
            client.files.delete(name=uploaded.name)
            if log_callback:
                log_callback(f"Deleted uploaded file from Gemini: {uploaded.name}")
        except Exception as e:
            if log_callback:
                log_callback(f"Warning: could not delete Gemini file {uploaded.name}: {e}")


def translate_text(text: str, source_language: str, target_language: str, log_callback=None) -> str:
    """
    Translate a block of text from source_language to target_language using Gemini.
    """
    if source_language.strip().lower() == target_language.strip().lower():
        # Caller should have checked this, but be defensive.
        return text

    client = _get_client()

    prompt = (
        f"Translate the following {source_language} text into {target_language}. "
        f"Output ONLY the translation, as flowing natural prose. "
        f"Do not include the original text, headers, notes, commentary, or formatting. "
        f"Preserve meaning faithfully.\n\n"
        f"--- TEXT TO TRANSLATE ---\n"
        f"{text}"
    )

    if log_callback:
        log_callback(f"Translating transcript to {target_language}...")

    def _generate():
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
        )

    response = _with_retries(_generate, "Translation", log_callback)
    translated = (response.text or "").strip()
    if not translated:
        raise RuntimeError("Gemini returned an empty translation.")
    return translated
