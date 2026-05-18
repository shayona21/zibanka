"""
Roman-to-native-script conversion for subtitle files (.srt and .json).

The point of this module is to guarantee that timestamps in the output are
identical to the input. Gemini only ever sees the dialogue text; the indexes
and timestamps are managed entirely in Python.

Public function:
    convert_file(input_path, output_path, target_language, chunk_size=50,
                 log_callback=None) -> dict

Returns a summary dict on success. Raises ConversionError if Gemini's output
cannot be safely stitched back to the original timestamps.
"""

import json
import re
import time
from pathlib import Path

import pysrt

import transcribe  # reuse the Gemini client and retry helpers


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 50
MAX_ATTEMPTS_PER_CHUNK = 2          # before falling back to split-in-half
MAX_ATTEMPTS_PER_HALF_CHUNK = 2     # second-level retry on each half

# Sentinel used to preserve newlines inside a single subtitle. Chosen because
# it is extremely unlikely to appear in dialogue and contains no characters
# that conflict with our pipe-delimited format.
NL_SENTINEL = "<<NL>>"

SUPPORTED_LANGUAGES = [
    "Hindi", "Telugu", "Bengali", "Tamil",
    "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi",
]


class ConversionError(RuntimeError):
    """Raised when Gemini's output cannot be safely reconciled with the input."""

# -------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------
def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    target_language: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    log_callback=None,
) -> dict:
    """
    Convert a subtitle file from Roman script to `target_language` native script.

    Args:
        input_path: path to a .srt or .json file
        output_path: path where the converted file is written. Format inferred
                     from extension (must match the input's format).
        target_language: one of SUPPORTED_LANGUAGES
        chunk_size: number of subtitle entries sent to Gemini per API call
        log_callback: optional callable(str) for progress messages

    Returns:
        Summary dict with keys: total_entries, converted_entries, skipped_empty,
        chunks_processed, elapsed_seconds.

    Raises:
        ConversionError: if Gemini's output cannot be safely validated against
        the input. No output file is written in this case.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if target_language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{target_language}'. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
        )

    input_format = _detect_format(input_path)
    output_format = _detect_format(output_path)
    if input_format != output_format:
        raise ValueError(
            f"Input format ({input_format}) and output format ({output_format}) "
            f"must match."
        )

    def log(msg: str):
        if log_callback:
            log_callback(msg)

    start_time = time.time()

    entries = _parse_input(input_path, input_format)
    log(f"Loaded {len(entries)} subtitle entries from {input_path.name}.")

    # Separate entries with empty dialogue from the conversion set; we'll
    # restore them as empty in the final output, keeping their timestamps.
    non_empty_entries = [e for e in entries if e["dialogue"].strip()]
    skipped_empty = len(entries) - len(non_empty_entries)
    if skipped_empty:
        log(f"Skipping {skipped_empty} entries with empty dialogue.")

    chunks = _split_into_chunks(non_empty_entries, chunk_size)
    log(f"Split into {len(chunks)} chunk(s) of up to {chunk_size} entries each.")

    converted_by_index: dict[int, str] = {}
    for i, chunk in enumerate(chunks, start=1):
        log(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} entries)...")
        chunk_result = _process_chunk_with_fallback(chunk, target_language, log)
        converted_by_index.update(chunk_result)

    # Stitch: rebuild every entry using its original timestamps. Entries with
    # empty dialogue stay empty.
    final_entries = []
    for entry in entries:
        new_entry = dict(entry)
        if entry["dialogue"].strip():
            # Defensive — should never fire because _process_chunk_with_fallback
            # raises if any expected index is missing.
            if entry["index"] not in converted_by_index:
                raise ConversionError(
                    f"Internal error: index {entry['index']} missing from "
                    f"converted results after successful chunk processing."
                )
            new_entry["dialogue"] = converted_by_index[entry["index"]]
        final_entries.append(new_entry)

    _write_output(final_entries, output_path, output_format)
    elapsed = time.time() - start_time
    log(f"Wrote {len(final_entries)} entries to {output_path.name} in {elapsed:.1f}s.")

    return {
        "total_entries": len(entries),
        "converted_entries": len(non_empty_entries),
        "skipped_empty": skipped_empty,
        "chunks_processed": len(chunks),
        "elapsed_seconds": round(elapsed, 1),
    }


# -------------------------------------------------------------------
# Parsing
# -------------------------------------------------------------------
def _detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".srt":
        return "srt"
    if ext == ".json":
        return "json"
    raise ValueError(f"Unsupported file extension '{ext}'. Use .srt or .json.")


def _parse_input(path: Path, fmt: str) -> list[dict]:
    if fmt == "srt":
        return _parse_srt(path)
    return _parse_json(path)


def _parse_srt(path: Path) -> list[dict]:
    subs = pysrt.open(str(path), encoding="utf-8")
    entries = []
    seen = set()
    for sub in subs:
        if sub.index in seen:
            raise ValueError(f"Duplicate index {sub.index} in SRT file.")
        seen.add(sub.index)
        entries.append({
            "index": sub.index,
            "start_time": str(sub.start),  # e.g. "00:00:01,000"
            "end_time": str(sub.end),
            "dialogue": sub.text,           # may contain '\n'
        })
    if not entries:
        raise ValueError("SRT file contains no subtitle entries.")
    return entries


def _parse_json(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("JSON file must contain a list of subtitle entries.")
    if not raw:
        raise ValueError("JSON file contains no entries.")

    required_keys = {"index", "start_time", "end_time", "dialogue"}
    seen = set()
    entries = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"JSON entry at position {i} is not an object.")
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(
                f"JSON entry at position {i} is missing keys: {sorted(missing)}"
            )
        if not isinstance(item["index"], int):
            raise ValueError(f"JSON entry at position {i} has non-integer index.")
        if item["index"] in seen:
            raise ValueError(f"Duplicate index {item['index']} in JSON file.")
        seen.add(item["index"])
        entries.append({
            "index": item["index"],
            "start_time": str(item["start_time"]),
            "end_time": str(item["end_time"]),
            "dialogue": str(item["dialogue"]),
        })
    return entries


# -------------------------------------------------------------------
# Chunking & Gemini call
# -------------------------------------------------------------------
def _split_into_chunks(entries: list[dict], size: int) -> list[list[dict]]:
    return [entries[i : i + size] for i in range(0, len(entries), size)]


def _process_chunk_with_fallback(
    chunk: list[dict], target_language: str, log,
) -> dict[int, str]:
    """
    Try a chunk up to MAX_ATTEMPTS_PER_CHUNK times. If still failing AND the
    chunk has more than one entry, split it in half and try each half
    independently (each with its own retries). If a half-chunk still fails,
    raise ConversionError.
    """
    last_problems: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS_PER_CHUNK + 1):
        try:
            return _process_chunk_once(chunk, target_language, log)
        except ConversionError as e:
            last_problems = e.args[0] if e.args else [str(e)]
            log(f"  Attempt {attempt} failed: {_summarise_problems(last_problems)}")
            if attempt < MAX_ATTEMPTS_PER_CHUNK:
                log("  Retrying with the same chunk...")

    # Both top-level attempts failed. Try splitting the chunk in half if we can.
    if len(chunk) <= 1:
        raise ConversionError(
            f"Chunk of {len(chunk)} entry failed after {MAX_ATTEMPTS_PER_CHUNK} "
            f"attempts. Problems: {last_problems}"
        )

    log(f"  Splitting chunk of {len(chunk)} into halves and retrying each...")
    mid = len(chunk) // 2
    first_half, second_half = chunk[:mid], chunk[mid:]

    result: dict[int, str] = {}
    for label, half in (("first", first_half), ("second", second_half)):
        log(f"  Processing {label} half ({len(half)} entries)...")
        half_problems: list[str] = []
        succeeded = False
        for attempt in range(1, MAX_ATTEMPTS_PER_HALF_CHUNK + 1):
            try:
                half_result = _process_chunk_once(half, target_language, log)
                result.update(half_result)
                succeeded = True
                break
            except ConversionError as e:
                half_problems = e.args[0] if e.args else [str(e)]
                log(f"    {label.capitalize()} half attempt {attempt} failed: "
                    f"{_summarise_problems(half_problems)}")
        if not succeeded:
            raise ConversionError(
                f"{label.capitalize()} half of {len(chunk)}-entry chunk failed "
                f"after {MAX_ATTEMPTS_PER_HALF_CHUNK} attempts. "
                f"Problems: {half_problems}"
            )

    return result


def _process_chunk_once(
    chunk: list[dict], target_language: str, log,
) -> dict[int, str]:
    """
    Single attempt: build the input table, call Gemini, parse, validate.
    Raises ConversionError on any validation failure.
    """
    input_table = _build_input_table(chunk)
    expected_indexes = {e["index"] for e in chunk}

    prompt = _render_prompt(input_table, target_language)
    raw_response = _call_gemini(prompt)
    cleaned = _strip_code_fences(raw_response)
    output_rows = _parse_pipe_table(cleaned)

    result, problems = _validate_chunk_output(output_rows, expected_indexes)
    if problems:
        raise ConversionError(problems)

    # Restore newlines from sentinels before returning.
    return {idx: text.replace(NL_SENTINEL, "\n") for idx, text in result.items()}


def _build_input_table(chunk: list[dict]) -> str:
    lines = ["index|dialogue"]
    for entry in chunk:
        d = entry["dialogue"]
        d = d.replace("|", "/")                         # delimiter collision
        d = d.replace("\r\n", "\n").replace("\r", "\n") # normalise line endings
        d = d.replace("\n", NL_SENTINEL)                # preserve line breaks
        lines.append(f"{entry['index']}|{d}")
    return "\n".join(lines)


def _call_gemini(prompt: str) -> str:
    """Send the prompt to Gemini with the same retry wrapper as the main app."""
    client = transcribe._get_client()

    def _generate():
        return client.models.generate_content(
            model=transcribe.MODEL_NAME,
            contents=[prompt],
        )

    response = transcribe._with_retries(_generate, "Transliteration", log_callback=None)
    return (response.text or "").strip()


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences and surrounding whitespace if present."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text


def _parse_pipe_table(text: str) -> list[tuple[int, str]]:
    """
    Parse Gemini's output into (index, dialogue) tuples. Tolerant: skips
    blank lines, the header row, and markdown separator rows.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("index|"):
            continue
        # Markdown separator rows like "|---|---|"
        if re.fullmatch(r"[\s|:\-]+", line) and "-" in line:
            continue
        if "|" not in line:
            continue
        index_part, _, dialogue_part = line.partition("|")
        index_str = index_part.strip()
        dialogue = dialogue_part.strip()
        if not index_str.isdigit():
            continue
        rows.append((int(index_str), dialogue))
    return rows


def _validate_chunk_output(
    output_rows: list[tuple[int, str]],
    expected_indexes: set[int],
) -> tuple[dict[int, str], list[str]]:
    """Check that every expected index appears exactly once with non-empty text."""
    seen = set()
    result: dict[int, str] = {}
    problems: list[str] = []

    for index, dialogue in output_rows:
        if index in seen:
            problems.append(f"duplicate index in output: {index}")
            continue
        if index not in expected_indexes:
            problems.append(f"unexpected index in output: {index}")
            continue
        if not dialogue.strip():
            problems.append(f"empty dialogue for index {index}")
            continue
        seen.add(index)
        result[index] = dialogue

    missing = expected_indexes - seen
    if missing:
        # Show at most the first 10 missing indexes to keep messages readable.
        sample = sorted(missing)[:10]
        suffix = "..." if len(missing) > 10 else ""
        problems.append(f"missing indexes ({len(missing)}): {sample}{suffix}")

    return result, problems


def _summarise_problems(problems: list[str]) -> str:
    if not problems:
        return "no problems reported"
    if len(problems) <= 2:
        return "; ".join(problems)
    return f"{problems[0]}; {problems[1]}; ...({len(problems) - 2} more)"


# -------------------------------------------------------------------
# Prompt
# -------------------------------------------------------------------
def _render_prompt(input_table: str, target_language: str) -> str:
    return (
        f"I am giving you a pipe-delimited table of subtitle dialogues in "
        f"Roman {target_language}. Convert each dialogue from Roman script to "
        f"{target_language} native script.\n"
        f"\n"
        f"Rules:\n"
        f"- Keep the index column EXACTLY as given. Do not change any number.\n"
        f"- Convert ONLY the dialogue column from Roman {target_language} to "
        f"{target_language} native script.\n"
        f"- If a dialogue contains \"XXX\", keep it as \"XXX\" in the output.\n"
        f"- If a dialogue starts with \"SDH \" followed by content in brackets, "
        f"translate the bracketed content from English to {target_language}. "
        f"Leave the \"SDH\" text in English.\n"
        f"- The placeholder {NL_SENTINEL} represents a line break inside a dialogue. "
        f"Keep it as {NL_SENTINEL} in your output, in the same position.\n"
        f"- Do not include reference links, citations, or extra commentary.\n"
        f"\n"
        f"CRITICAL RULES ABOUT OUTPUT STRUCTURE:\n"
        f"- Output exactly the same number of rows as input.\n"
        f"- Output every input index, in the same order.\n"
        f"- Do not add new indexes or drop any indexes.\n"
        f"- Each output row must have exactly one pipe character separating "
        f"the index from the dialogue.\n"
        f"- Do not wrap the output in markdown code blocks.\n"
        f"- Do not add any text before or after the table.\n"
        f"- Do not include a markdown separator row (e.g. |---|---|).\n"
        f"- The first row of your output must be the header: index|dialogue\n"
        f"\n"
        f"Input table:\n"
        f"{input_table}"
    )


# -------------------------------------------------------------------
# Writing
# -------------------------------------------------------------------
def _write_output(entries: list[dict], path: Path, fmt: str) -> None:
    if fmt == "srt":
        _write_srt(entries, path)
    else:
        _write_json(entries, path)


def _write_srt(entries: list[dict], path: Path) -> None:
    subs = pysrt.SubRipFile()
    for entry in entries:
        item = pysrt.SubRipItem(
            index=entry["index"],
            start=pysrt.SubRipTime.from_string(entry["start_time"]),
            end=pysrt.SubRipTime.from_string(entry["end_time"]),
            text=entry["dialogue"],
        )
        subs.append(item)
    subs.save(str(path), encoding="utf-8")


def _write_json(entries: list[dict], path: Path) -> None:
    path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
