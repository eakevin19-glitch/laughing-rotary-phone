import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from openai import OpenAI


DEFAULT_TONE = "រឿងបុរាណ/យុទ្ធសិល្ប៍"


def parse_glossary(value: str) -> dict[str, str]:
    glossary = {}
    for line in value.splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            raise ValueError("សូមប្រើទម្រង់ ឈ្មោះដើម = ឈ្មោះខ្មែរ សម្រាប់ Glossary នីមួយៗ។")
        original, translated = line.split("=", maxsplit=1)
        if not original.strip() or not translated.strip():
            raise ValueError("ធាតុ Glossary ត្រូវមានទាំងឈ្មោះដើម និងឈ្មោះខ្មែរ។")
        glossary[original.strip()] = translated.strip()
    return glossary


def translate_story_chapter(
    text: str,
    glossary: dict[str, str],
    tone: str = DEFAULT_TONE,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    selected_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not selected_api_key:
        raise ValueError("សូមបញ្ចូល API key ឬកំណត់ OPENAI_API_KEY ជាមុនសិន។")
    if not text.strip():
        raise ValueError("សូមបញ្ចូលអត្ថបទរឿងដែលត្រូវបកប្រែ។")

    glossary_text = "\n".join(
        f"- {original}: {translated}" for original, translated in glossary.items()
    )
    system_prompt = f"""
អ្នកជាអ្នកបកប្រែរឿងប្រលោមលោកអាជីពទៅជាភាសាខ្មែរ។

គោលការណ៍ណែនាំ៖
១. បកប្រែឱ្យមានលក្ខណៈរលូន ស័ក្តិសមជារចនាប័ទ្ម៖ {tone}
២. ប្រើប្រាស់ពាក្យសព្វនាមឱ្យស៊ីគ្នានឹងទំនាក់ទំនងតួអង្គក្នុងបរិបទ។
៣. រក្សាឈ្មោះតួអង្គ ទីកន្លែង និងពាក្យក្នុងតារាង Glossary ឱ្យដូចដែលបានកំណត់។

[តារាង GLOSSARY]
{glossary_text}

បញ្ចេញមកតែអត្ថបទបកប្រែជាភាសាខ្មែរប៉ុណ្ណោះ មិនបន្ថែមការពន្យល់ឡើយ។
""".strip()

    client = OpenAI(
        api_key=selected_api_key,
        base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
    )
    response = client.chat.completions.create(
        model=model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"សូមបកប្រែអត្ថបទខាងក្រោម៖\n\n{text}"},
        ],
        temperature=0.3,
    )
    translated_text = response.choices[0].message.content
    if not translated_text:
        raise RuntimeError("អ្នកផ្តល់សេវា AI មិនបានត្រឡប់អត្ថបទបកប្រែទេ។")
    return translated_text


def transcribe_video(
    video_bytes: bytes,
    filename: str,
    *,
    api_key: str,
    source_language: str | None = None,
) -> list[dict[str, Any]]:
    if not api_key:
        raise ValueError("សូមបញ្ចូល OpenAI API key សម្រាប់ស្គាល់សំឡេង។")
    if not video_bytes:
        raise ValueError("ឯកសារវីដេអូទទេ។")

    chunk_seconds = 600
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        video_extension = Path(filename).suffix or ".mp4"
        video_path = temporary_path / f"input{video_extension}"
        video_path.write_bytes(video_bytes)
        audio_pattern = temporary_path / "audio_%03d.mp3"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "32k",
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            "-segment_format",
            "mp3",
            str(audio_pattern),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            details = result.stderr.strip().splitlines()
            reason = details[-1] if details else "FFmpeg could not read the video."
            raise ValueError(f"មិនអាចអានសំឡេងពីវីដេអូបានទេ៖ {reason}")

        audio_chunks = sorted(temporary_path.glob("audio_*.mp3"))
        if not audio_chunks:
            raise ValueError("វីដេអូនេះមិនមានសំឡេងសម្រាប់បកប្រែទេ។")

        client = OpenAI(api_key=api_key)
        timed_segments = []
        for chunk_index, audio_path in enumerate(audio_chunks):
            if audio_path.stat().st_size > 25 * 1024 * 1024:
                raise ValueError("ផ្នែកសំឡេងធំពេកសម្រាប់ transcription API។")
            transcription_options = {
                "model": "whisper-1",
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
            }
            if source_language:
                transcription_options["language"] = source_language
            with audio_path.open("rb") as audio_file:
                response = client.audio.transcriptions.create(
                    file=audio_file,
                    **transcription_options,
                )

            segments = getattr(response, "segments", None)
            if segments is None and isinstance(response, dict):
                segments = response.get("segments", [])
            for segment in segments or []:
                if isinstance(segment, dict):
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    text = segment.get("text", "")
                else:
                    start = segment.start
                    end = segment.end
                    text = segment.text
                if text.strip():
                    timed_segments.append(
                        {
                            "start": float(start) + chunk_index * chunk_seconds,
                            "end": float(end) + chunk_index * chunk_seconds,
                            "source_text": text.strip(),
                        }
                    )

    if not timed_segments:
        raise ValueError("មិនរកឃើញការនិយាយក្នុងវីដេអូនេះទេ។")
    return timed_segments


def translate_subtitle_segments(
    segments: list[dict[str, Any]],
    glossary: dict[str, str],
    tone: str,
    *,
    api_key: str,
    base_url: str | None = None,
    model: str = "gpt-4o-mini",
) -> list[dict[str, Any]]:
    if not api_key:
        raise ValueError("សូមបញ្ចូល API key សម្រាប់បកប្រែ subtitle។")
    if not segments:
        raise ValueError("សូមបង្កើត transcription មុនពេលបកប្រែ។")

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    glossary_text = "\n".join(
        f"- {original}: {translated}" for original, translated in glossary.items()
    )
    translated_segments = []
    batch_size = 40
    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start : batch_start + batch_size]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"""
អ្នកជាអ្នកបកប្រែ subtitle ទៅជាភាសាខ្មែរ។
បកប្រែឃ្លានីមួយៗឱ្យខ្លី រលូន សមនឹងរចនាប័ទ្ម {tone} និងរក្សាន័យឱ្យត្រឹមត្រូវ។
រក្សាឈ្មោះ និងពាក្យ Glossary ទាំងនេះឱ្យដដែល៖
{glossary_text}
ត្រឡប់ JSON តែមួយគត់ក្នុងទម្រង់ {{"translations": ["..."]}}។
ត្រូវមានចំនួនធាតុដូច input និងរក្សាលំដាប់ដើម។
""".strip(),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        [segment["source_text"] for segment in batch],
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("អ្នកផ្តល់សេវា AI មិនបានត្រឡប់ subtitle ទេ។")
        translations = json.loads(content).get("translations")
        if not isinstance(translations, list) or len(translations) != len(batch):
            raise RuntimeError("ចំនួន subtitle ដែលបានបកប្រែមិនត្រូវនឹងអត្ថបទដើមទេ។")
        for segment, translated_text in zip(batch, translations):
            if not isinstance(translated_text, str) or not translated_text.strip():
                raise RuntimeError("មាន subtitle ទទេក្នុងលទ្ធផលបកប្រែ។")
            translated_segments.append(
                {
                    **segment,
                    "translated_text": translated_text.strip(),
                }
            )
    return translated_segments


def format_srt(segments: list[dict[str, Any]]) -> str:
    def timestamp(seconds: float) -> str:
        milliseconds = max(0, round(seconds * 1000))
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        whole_seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"

    entries = []
    for index, segment in enumerate(segments, start=1):
        entries.append(
            f"{index}\n{timestamp(segment['start'])} --> {timestamp(segment['end'])}"
            f"\n{segment['translated_text']}"
        )
    return "\n\n".join(entries)
