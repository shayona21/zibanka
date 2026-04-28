"""
Gemini API calls for video transcription.

Public function:
  - transcribe_video(video_path, source_language, video_title, prompt) -> str

Retries transient errors up to 3 times with exponential backoff.

NOTE: translate_text() below is currently UNUSED. It is preserved as dead code
because translation will be re-introduced later (likely to translate only the
final dialogue column of the new CSV transcript). Do not delete.
"""

import os
import re
import time
import mimetypes
from google import genai
from google.genai import errors as genai_errors


MODEL_NAME = "gemini-2.5-pro"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 10  # doubled each retry

# How long we'll wait for Gemini to finish processing an uploaded video.
# A 60-min video typically takes 1-3 minutes. We give a generous ceiling.
FILE_ACTIVE_TIMEOUT_SECONDS = 600
FILE_POLL_INTERVAL_SECONDS = 5

# Default transcription prompt. Two placeholders are substituted at call time:
#   {source_language}     - e.g. "Hindi"
#   {video_title_param}   - the user-provided title, or "untitled video" if blank
#
# The prompt asks Gemini for a pipe-delimited CSV so the output can be opened
# in Google Sheets via "custom delimiter = pipe". The strict-format instruction
# is intentionally placed at the very end of the prompt because Gemini tends
# to honour the most recent instruction when there is any ambiguity.
DEFAULT_TRANSCRIPTION_PROMPT = (
    "I am uploading a {source_language} video ({video_title_param}). "
    "Please transcribe all the dialogues of this episode verbatim in {source_language} Script "
    "from the attached file. Complete the full file. "
    "Also write the same script in the Roman {source_language} after each sentence. "
    "You have to write each sentence in a new line. "
    "Also write the timestamp when the sentence begins and when the sentence ends. "
    "Time stamp should be in the format hh:mm:ss:ff.\n"
    "Don't skip any dialogue. Transcribe the dialogue, even if there is no lip-movement on the screen. "
    "If you cannot identify the speaker, mention the speaker name as \"Unknown\". "
    "Even Don't shorten or summarize. Also mention time-stamp and speaker names if you can.\n"
    "If a pipe character | appears in a dialogue, replace it with a forward slash / . "
    "Do not include stage directions, commentary, section headers, bullet points. "
    "Transcribe the full audio from start to finish.\n"
    "Speaker name should be both in {source_language} and Roman {source_language}. "
    "Speaker name in {source_language} should be put within square brackets [ ].\n"
    "Time-stamp: For time-stamp, follow the video time, and not the time-code printed on the video. "
    "Also, video is in PAL format, that is 25 frames per second. "
    "The time-stamp should be in the format hh:mm:ss:ff.\n"
    "The output should be in the tabular format, with the following column names and sequence:\n"
    "1. Start Timestamp\n"
    "2. End Timestamp\n"
    "3. Speaker name in Roman {source_language}\n"
    "4. Dialogue \u2013 {source_language} Script\n"
    "5. [Speaker name in {source_language}]\n"
    "6. Dialogue \u2013 Roman {source_language} Script\n"
    "These will be the headers: \"start_timestamp\", \"end_timestamp\", \"speaker_roman\", "
    "\"dialogue_source\", \"speaker_source\", \"dialogue_roman\". "
    "Output ONLY the pipe-delimited CSV. Use the pipe character | to separate columns. "
    "Do not wrap in markdown code blocks. Do not add explanations before or after. "
    "Do not include a separator row. The first row must be the header row exactly as specified above. "
    "Each subsequent row is one sentence."
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


def _render_prompt(prompt: str, source_language: str, video_title: str) -> str:
    """
    Substitute {source_language} and {video_title_param} in the prompt.
    Either may be absent in custom prompts; we leave non-placeholders alone.
    """
    rendered = prompt
    if "{source_language}" in rendered:
        rendered = rendered.replace("{source_language}", source_language)
    if "{video_title_param}" in rendered:
        rendered = rendered.replace("{video_title_param}", video_title)
    return rendered


def _clean_csv_output(text: str) -> str:
    """
    Defensive post-processing for Gemini's CSV output.

    Gemini sometimes wraps responses in markdown code fences, adds preamble
    like "Here is the transcript:", or includes a markdown separator row.
    We strip these out best-effort. We also fix pipe collisions inside fields:
    for any row that has more than 6 pipe-separated columns, the extra pipes
    must have come from inside a dialogue, so we collapse the trailing columns
    back into the last one (preserving content over structure is the right
    default; alternatively, we replace stray pipes with '/').

    Returns the cleaned CSV text. Always preserves a trailing newline.
    """
    text = text.strip()

    # Strip markdown code fences if Gemini wrapped the output.
    # Handles ```csv\n...\n``` and ```\n...\n``` patterns.
    if text.startswith("```"):
        # Remove first line (the opening fence, possibly with language hint)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove trailing closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    # Drop any empty leading/trailing lines.
    lines = [ln for ln in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    # Drop a markdown separator row if Gemini added one (e.g. "|---|---|---|").
    cleaned = []
    for ln in lines:
        stripped = ln.strip()
        # A separator row is dashes, pipes, colons, and spaces only.
        if stripped and re.fullmatch(r"[\s|:\-]+", stripped) and "-" in stripped:
            continue
        cleaned.append(ln)

    # Fix rows with too many columns by joining trailing extras into the last
    # column. Header row has 6 fields, so we expect 6 fields per row (5 pipes).
    EXPECTED_FIELDS = 6
    fixed = []
    for ln in cleaned:
        if "|" not in ln:
            fixed.append(ln)
            continue
        parts = ln.split("|")
        if len(parts) > EXPECTED_FIELDS:
            # Merge the surplus into the last column, replacing the merged
            # pipes with '/' so the row is still valid pipe-delimited CSV.
            head = parts[:EXPECTED_FIELDS - 1]
            tail = "/".join(parts[EXPECTED_FIELDS - 1:])
            parts = head + [tail]
        fixed.append("|".join(parts))

    result = "\n".join(fixed)
    if not result.endswith("\n"):
        result += "\n"
    return result


def transcribe_video(
    video_path: str,
    source_language: str,
    video_title: str = "untitled video",
    prompt: str | None = None,
    log_callback=None,
) -> str:
    """
    Upload a local video file to Gemini and return the transcribed text.

    Args:
        video_path: absolute path to the local video file
        source_language: human-readable language name (e.g. "Hindi")
        video_title: the user-provided title (raw, not slugged). Substituted
                     into the prompt's {video_title_param} placeholder.
                     Defaults to "untitled video" if blank.
        prompt: optional custom prompt string. If omitted or blank, uses the default.
                May include the placeholders {source_language} and {video_title_param}
                which will be substituted.
        log_callback: optional callable(str) for status updates

    Returns:
        The transcribed text as pipe-delimited CSV (post-processed and cleaned).
    """
    client = _get_client()
    mime = _guess_mime_type(video_path)

    # Resolve the prompt: use the caller's if non-blank, else the default.
    effective_prompt = (prompt or "").strip() or DEFAULT_TRANSCRIPTION_PROMPT
    title_for_prompt = (video_title or "").strip() or "untitled video"
    effective_prompt = _render_prompt(effective_prompt, source_language, title_for_prompt)

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
            log_callback("Transcribing (this can take several minutes)...")

        def _generate():
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=[uploaded, effective_prompt],
            )

        response = _with_retries(_generate, "Transcription", log_callback)
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty transcript.")
        # Post-process to strip code fences, separator rows, and over-pipe rows.
        text = _clean_csv_output(text)
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


# ---------------------------------------------------------------------------
# DEAD CODE BELOW: translate_text() is currently unused. The UI no longer
# offers translation, and run_batch / retry_single_video do not call this.
# Preserved for an upcoming change that will translate only the dialogue
# columns of the new CSV transcript. Do not delete.
# TODO: when translation is re-enabled, this function will likely need to
# operate per-row or per-column rather than on the whole transcript blob.
# ---------------------------------------------------------------------------
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
