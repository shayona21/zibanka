# %% libraries
import json
import os
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types
from werkzeug.utils import secure_filename


MODEL_NAME = "gemini-2.5-flash"
BATCH_SIZE = 50
MAX_CHARACTER_REFERENCE_CHARS = 12000
CHAR_LISTS_DIR = Path(__file__).resolve().parent / "char_lists"

SERIES_PROMPTS = {
    "qubool_hai": {
        "label": "Qubool Hai",
        "prompt": """
            GERMAN SUBTITLES - Qubool Hai

            I am uploading English Subtitle for a TV serial. You need to translate this into German.
            Please keep the translation natural and fluent and concise.
            Don't translate in fragments.
            If a sentence is across more than one subtitle, please translate the full sentence to keep proper sentence structure, and then divide across the original timestamps.
            I want the same number of subtitles as in the source.
            Also, do not change the timestamps.
            Donot change single quotes to doucle quotes.

            Use the attached characterlist in the source area (QUBOOL HAI CHARACTER LIST. xlsx) to know the gender of the characters and their relationship with other characters.
            Give the output in SRT file format.
        """,
    },
    "doli_armano_ki": {
        "label": "Doli Armano Ki",
        "prompt": """
            GERMAN SUBTITLES - DOLI ARMANO KI

            I am uploading English Subtitle for Zee TV Hindi drama serial DOLI ARMANO KI.
            You need to translate this into German.
            Please keep the translation natural and fluent and concise.
            Don't expand the translation by putting extraneous facts.
            Don't translate in fragments.
            If a sentence is across more than one subtitle, please translate the full sentence to keep proper sentence structure, and then divide across the original timestamps.
            I want the same number of subtitles as in the source.
            Also, do not change the timestamps.
            Donot change single quotes to doucle quotes.

            Use the attached characterlist in the source area (DOLI ARMANO KI CHARACTER LIST. xlsx) to know the gender of the characters and their relationship with other characters.
            Don't keep any reference link in your output.
            Give the output in SRT file format.
        """,
    },
}

CHARACTER_LIST_OPTIONS = {
    "none": {
        "label": "No character list",
        "path": None,
    },
    "qubool_hai": {
        "label": "Qubool Hai character list",
        "path": CHAR_LISTS_DIR / "QUBOOL_HAI_CHARACTER_LIST.json",
    },
    "doli_armano_ki": {
        "label": "Doli Armano Ki character list",
        "path": CHAR_LISTS_DIR / "DOLI_ARMANO_KI_CHARACTER_LIST.json",
    },
}


def srt_to_dataframe(file_path):
    """
    Convert SRT subtitle file into pandas DataFrame.

    Output columns:
    index, start_time, end_time, dialogue

    Multi-line subtitles are preserved using <n>
    """

    with open(file_path, "r", encoding="utf-8-sig") as file:
        content = file.read()

    blocks = content.strip().split("\n\n")
    rows = []

    for block in blocks:
        lines = block.strip().split("\n")

        if len(lines) < 3:
            continue

        subtitle_index = lines[0].strip()
        time_line = lines[1].strip()
        start_time, end_time = time_line.split(" --> ")
        dialogue = "<n>".join(lines[2:]).strip()

        rows.append(
            {
                "index": int(subtitle_index),
                "start_time": start_time,
                "end_time": end_time,
                "dialogue": dialogue,
            }
        )

    return pd.DataFrame(rows)


def trim_df(df):
    return df[["index", "dialogue"]].copy()


def split_dataframe_into_batches(df, batch_size=BATCH_SIZE):
    batches = []

    for start_idx in range(0, len(df), batch_size):
        batch_df = df.iloc[start_idx:start_idx + batch_size]
        batches.append(batch_df)

    return batches


def dataframe_batch_to_json(batch_df):
    records = batch_df.to_dict(orient="records")
    return json.dumps(records, ensure_ascii=False, indent=2)


def get_character_reference_path(character_reference_key):
    option = CHARACTER_LIST_OPTIONS.get(character_reference_key or "none")
    if not option:
        raise ValueError(f"Unsupported character list: {character_reference_key}")
    return option["path"]


def _format_json_character_reference(reference_data):
    if isinstance(reference_data, list):
        lines = []
        for entry in reference_data:
            if isinstance(entry, dict):
                name = str(entry.get("NAME", "")).strip()
                gender = str(entry.get("GENDER", "")).strip()
                description = str(entry.get("DESCRIPTION", "")).strip()
                parts = [part for part in (name, gender, description) if part]
                if parts:
                    lines.append(" | ".join(parts))
            else:
                lines.append(str(entry))
        return "\n".join(lines)

    if isinstance(reference_data, dict):
        return json.dumps(reference_data, ensure_ascii=False, indent=2)

    return str(reference_data)


def load_character_reference(character_reference_path):
    if not character_reference_path:
        return ""

    reference_path = Path(character_reference_path)
    if not reference_path.exists():
        raise ValueError(f"Character list file was not found: {reference_path.name}")

    suffix = reference_path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        try:
            sheets = pd.read_excel(reference_path, sheet_name=None, dtype=str)
        except ImportError as exc:
            raise RuntimeError(
                "Reading Excel character lists requires openpyxl. "
                "Install it with `python3 -m pip install openpyxl`."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Could not read the character list Excel file {reference_path.name}: {exc}"
            ) from exc

        chunks = []
        for sheet_name, sheet_df in sheets.items():
            normalized_df = sheet_df.fillna("").astype(str)
            chunks.append(f"[Sheet: {sheet_name}]")
            chunks.append(normalized_df.to_csv(index=False, sep="|").strip())

        reference_text = "\n\n".join(chunk for chunk in chunks if chunk.strip())
    elif suffix == ".csv":
        reference_text = reference_path.read_text(encoding="utf-8-sig")
    elif suffix == ".txt":
        reference_text = reference_path.read_text(encoding="utf-8")
    elif suffix == ".json":
        try:
            reference_data = json.loads(reference_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Could not read the character list JSON file {reference_path.name}: {exc}"
            ) from exc
        reference_text = _format_json_character_reference(reference_data)
    else:
        raise ValueError(
            "Character list must be an .xlsx, .xls, .csv, .txt, or .json file."
        )

    reference_text = reference_text.strip()
    if not reference_text:
        return ""

    if len(reference_text) > MAX_CHARACTER_REFERENCE_CHARS:
        reference_text = (
            reference_text[:MAX_CHARACTER_REFERENCE_CHARS]
            + "\n\n[Character list truncated for prompt length.]"
        )

    return reference_text


def build_translation_prompt(series_key, character_reference_text=""):
    """
    Build the instruction prompt for Gemini.
    """

    if series_key not in SERIES_PROMPTS:
        raise ValueError(f"Unsupported series key: {series_key}")

    base_prompt = SERIES_PROMPTS[series_key]["prompt"].strip()

    character_reference_block = ""
    if character_reference_text:
        character_reference_block = f"""
            Character list reference:
            {character_reference_text}
        """

    prompt = f"""
        You are given a JSON array containing subtitle dialogue from an English SRT file.

        Each object contains:
        - "index"
        - "dialogue"

        {base_prompt}

        {character_reference_block}

        Technical instructions for this task:
        1. Preserve the exact JSON structure.
        2. Preserve the exact "index" value.
        3. Return ONE output object for every input object.
        4. Do not reorder rows.
        5. Do not omit any rows.
        6. Do not add explanations or commentary.
        7. Do not wrap the output in markdown.
        8. Output STRICT VALID JSON ONLY.
        9. Do not output raw SRT blocks. The system will rebuild the final SRT after your JSON response.

        For each object:
        - Keep the original "dialogue" unchanged.
        - Add a new field called "translated_dialogue".

        Translation rules:
        - Translate from English into natural, fluent, concise German.
        - Preserve any "<n>" markers exactly as they appear.
        - Preserve single quotes exactly. Do not convert them to double quotes.
        - Keep the same subtitle count as the source.
        - If a sentence spans multiple subtitle rows, use the surrounding rows for context so the German reads naturally, but still return one translated subtitle row per input row.
        - Do not add reference links.
        - Do not add extra facts or explanations.

        Return only a JSON array.
    """
    return prompt.strip()


def call_gemini_for_batch(
    client,
    batch_json,
    series_key,
    character_reference_text="",
    model_name=MODEL_NAME,
):
    prompt = build_translation_prompt(
        series_key=series_key,
        character_reference_text=character_reference_text,
    )

    full_prompt = f"""
        {prompt}

        Here is the input JSON array:
        {batch_json}
    """

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    response_text = response.text.strip()
    return json.loads(response_text)


def normalize_batch_result(batch_result, batch_num):
    if isinstance(batch_result, dict):
        batch_result = [batch_result]

    if not isinstance(batch_result, list):
        raise ValueError(
            f"Batch {batch_num} returned {type(batch_result).__name__}, expected a JSON array."
        )

    normalized_rows = []

    for row_num, row in enumerate(batch_result, start=1):
        if not isinstance(row, dict):
            raise ValueError(
                f"Batch {batch_num}, row {row_num} returned {type(row).__name__}, expected an object."
            )

        if "index" not in row or "translated_dialogue" not in row:
            raise ValueError(
                f"Batch {batch_num}, row {row_num} is missing required keys. "
                f"Found keys: {sorted(row.keys())}"
            )

        normalized_rows.append(row)

    return normalized_rows


def emit_progress(progress_callback, current_batch, total_batches, message):
    if progress_callback is None:
        return

    progress_callback(
        {
            "current_batch": current_batch,
            "total_batches": total_batches,
            "percent": int((current_batch / total_batches) * 100) if total_batches else 100,
            "message": message,
        }
    )


def get_output_file_name(file_path, requested_name=None):
    input_path = Path(file_path)

    if requested_name:
        requested_stem = Path(requested_name).stem
        safe_stem = secure_filename(requested_stem)
        if safe_stem:
            return f"{safe_stem}.srt"

    return f"{input_path.stem}_german.srt"


def main(
    file_path,
    series_key,
    batch_size=BATCH_SIZE,
    character_reference_path=None,
    progress_callback=None,
):
    df = srt_to_dataframe(file_path)
    trimmed_df = trim_df(df)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    character_reference_text = load_character_reference(character_reference_path)

    batch_list = split_dataframe_into_batches(trimmed_df, batch_size=batch_size)
    total_batches = len(batch_list)
    all_results = []

    for batch_num, batch_df in enumerate(batch_list):
        batch_message = f"Processing batch {batch_num + 1} of {total_batches}"
        print(batch_message)
        emit_progress(progress_callback, batch_num, total_batches, batch_message)

        batch_json = dataframe_batch_to_json(batch_df)
        batch_result = call_gemini_for_batch(
            client=client,
            batch_json=batch_json,
            series_key=series_key,
            character_reference_text=character_reference_text,
        )

        response_message = f"Gemini batch {batch_num + 1} response received."
        print(response_message)

        normalized_batch_result = normalize_batch_result(
            batch_result=batch_result,
            batch_num=batch_num + 1,
        )

        all_results.extend(normalized_batch_result)
        emit_progress(progress_callback, batch_num + 1, total_batches, response_message)

        time.sleep(1)

    gemini_output_df = pd.DataFrame(all_results)
    final_df = df.merge(
        gemini_output_df[["index", "translated_dialogue"]],
        on="index",
        how="left",
    )

    emit_progress(progress_callback, total_batches, total_batches, "Merging batch results.")

    return gemini_output_df, final_df


def process_srt_file(
    file_path,
    series_key,
    batch_size=BATCH_SIZE,
    output_file_name=None,
    output_dir="output",
    character_reference_path=None,
    progress_callback=None,
):
    """
    Run German subtitle translation for an SRT file and write the processed SRT to disk.
    """

    _, final_df = main(
        file_path=file_path,
        series_key=series_key,
        batch_size=batch_size,
        character_reference_path=character_reference_path,
        progress_callback=progress_callback,
    )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_filename = get_output_file_name(
        file_path=file_path,
        requested_name=output_file_name,
    )

    output_path = output_dir_path / output_filename
    dataframe_to_srt(final_df, output_path)
    emit_progress(progress_callback, 1, 1, f"Output ready: {output_filename}")

    return output_path


def dataframe_to_srt(final_df, output_path):
    srt_blocks = []

    for _, row in final_df.iterrows():
        subtitle_index = int(row["index"])
        start_time = row["start_time"]
        end_time = row["end_time"]

        if pd.notna(row["translated_dialogue"]):
            subtitle_text = row["translated_dialogue"]
        else:
            subtitle_text = row["dialogue"]

        subtitle_text = subtitle_text.replace("<n>", "\n")

        block = f"{subtitle_index}\n{start_time} --> {end_time}\n{subtitle_text}"
        srt_blocks.append(block)

    srt_content = "\n\n".join(srt_blocks)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(srt_content)

    print(f"SRT file saved to: {output_path}")
