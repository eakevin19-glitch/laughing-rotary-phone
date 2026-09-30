import subprocess
import wave
from pathlib import Path
from typing import Any, Callable

import imageio_ffmpeg
from local_models import VOICE_STYLE_SPEED


AUDIO_CHUNK_SECONDS = 600
LANGUAGE_CODES = {
    "Khmer": "khm",
    "English": "eng",
    "Thai": "tha",
    "Chinese": "zho",
    "Japanese": "jpn",
    "Korean": "kor",
    "Vietnamese": "vie",
    "French": "fra",
    "Spanish": "spa",
}
ProgressCallback = Callable[[float, str], None]


def _run_ffmpeg(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", *arguments]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        details = result.stderr.strip().splitlines()
        reason = details[-1] if details else "FFmpeg could not process this media file."
        raise RuntimeError(reason)
    return result


def extract_audio_chunks(video_path: Path, work_dir: Path) -> list[Path]:
    audio_dir = work_dir / "audio_chunks"
    audio_dir.mkdir(parents=True, exist_ok=True)
    existing_chunks = sorted(audio_dir.glob("audio_*.mp3"))
    if existing_chunks:
        return existing_chunks

    output_pattern = audio_dir / "audio_%04d.mp3"
    try:
        _run_ffmpeg(
            [
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
                "48k",
                "-f",
                "segment",
                "-segment_time",
                str(AUDIO_CHUNK_SECONDS),
                "-reset_timestamps",
                "1",
                "-segment_format",
                "mp3",
                str(output_pattern),
            ]
        )
    except RuntimeError as error:
        message = str(error)
        if "matches no streams" in message or "Stream map" in message:
            raise ValueError("វីដេអូនេះមិនមាន audio track សម្រាប់ស្គាល់ពាក្យនិយាយទេ។") from error
        raise ValueError(f"មិនអាចដកសំឡេងពីវីដេអូបានទេ៖ {message}") from error

    chunks = sorted(audio_dir.glob("audio_*.mp3"))
    if not chunks:
        raise ValueError("មិនអាចបង្កើត audio file ពីវីដេអូបានទេ។")
    return chunks


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def _timestamp(seconds: float, decimal_separator: str) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}{decimal_separator}{milliseconds:03}"


def format_srt(segments: list[dict[str, Any]]) -> str:
    entries = []
    for index, segment in enumerate(segments, start=1):
        entries.append(
            f"{index}\n{_timestamp(segment['start'], ',')} --> {_timestamp(segment['end'], ',')}"
            f"\n{segment['translated_text']}"
        )
    return "\n\n".join(entries) + "\n"


def format_vtt(segments: list[dict[str, Any]]) -> str:
    entries = ["WEBVTT", ""]
    for segment in segments:
        entries.append(
            f"{_timestamp(segment['start'], '.')} --> {_timestamp(segment['end'], '.')}"
            f"\n{segment['translated_text']}\n"
        )
    return "\n".join(entries)


def _voice_filter(segment: dict[str, Any], voice_path: Path, input_index: int, label: str) -> str:
    try:
        voice_duration = _wav_duration(voice_path)
    except (wave.Error, OSError, ZeroDivisionError) as error:
        raise ValueError(f"សំឡេង subtitle លេខ {segment['id']} ខូច ឬមិនមែនជា WAV ត្រឹមត្រូវ។") from error
    segment_duration = max(0.5, segment["end"] - segment["start"])
    style_speed = VOICE_STYLE_SPEED.get(segment.get("voice_style", "Natural"), 1.0)
    tempo = voice_duration / segment_duration * style_speed if segment_duration else 1.0
    filters = []
    if 0.5 <= tempo <= 2.0 and abs(tempo - 1.0) > 0.08:
        filters.append(f"atempo={tempo:.3f}")
    delay = max(0, int(segment["start"] * 1000))
    filters.extend([f"adelay={delay}|{delay}", "volume=1.6"])
    return f"[{input_index}:a]{','.join(filters)}[{label}]"


def render_dubbed_audio(
    output_path: Path,
    voice_segments: list[tuple[dict[str, Any], Path]],
    work_dir: Path,
) -> None:
    if not voice_segments:
        raise ValueError("គ្មានសំឡេងបកប្រែសម្រាប់បង្កើត audio file ទេ។")

    input_arguments = ["-y"]
    filters = []
    mix_inputs = []
    for index, (segment, voice_path) in enumerate(voice_segments):
        input_arguments.extend(["-i", str(voice_path)])
        label = f"voice{index}"
        filters.append(_voice_filter(segment, voice_path, index, label))
        mix_inputs.append(f"[{label}]")

    filters.append(
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest:normalize=0,alimiter=limit=0.95[dubbed]"
    )
    filter_script = work_dir / "dubbed_audio_filter.txt"
    filter_script.write_text(";\n".join(filters), encoding="utf-8")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        *input_arguments,
        "-filter_complex_script",
        str(filter_script),
        "-map",
        "[dubbed]",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "24000",
        "-ac",
        "1",
        str(output_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise RuntimeError(f"មិនអាចចាប់ផ្តើម FFmpeg audio export បានទេ៖ {error}") from error
    if result.returncode:
        details = result.stderr.strip().splitlines()
        reason = details[-1] if details else "FFmpeg could not render the dubbed audio."
        raise RuntimeError(f"FFmpeg audio export បរាជ័យ៖ {reason}")


def render_mp3_audio(wav_path: Path, mp3_path: Path) -> None:
    if not wav_path.is_file():
        raise ValueError("មិនមាន WAV voice track សម្រាប់បង្កើត MP3 ទេ។")
    try:
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(wav_path),
                "-c:a",
                "libmp3lame",
                "-q:a",
                "3",
                str(mp3_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise RuntimeError(f"មិនអាចចាប់ផ្តើម FFmpeg MP3 export បានទេ៖ {error}") from error
    if result.returncode:
        details = result.stderr.strip().splitlines()
        reason = details[-1] if details else "FFmpeg could not encode MP3 audio."
        raise RuntimeError(f"FFmpeg MP3 export បរាជ័យ៖ {reason}")


def render_translated_video(
    original_path: Path,
    output_path: Path,
    subtitles_path: Path,
    voice_segments: list[tuple[dict[str, Any], Path]],
    subtitles_enabled: bool,
    target_language: str,
    work_dir: Path,
) -> None:
    if not voice_segments:
        raise ValueError("គ្មានសំឡេងបកប្រែសម្រាប់ render វីដេអូទេ។")

    filters = ["[0:a:0]aformat=channel_layouts=stereo,volume=0.14[background]"]
    mix_inputs = ["[background]"]
    input_arguments = ["-y", "-i", str(original_path)]
    for index, (segment, voice_path) in enumerate(voice_segments):
        input_index = index + 1
        input_arguments.extend(["-i", str(voice_path)])
        output_label = f"voice{index}"
        filters.append(_voice_filter(segment, voice_path, input_index, output_label))
        mix_inputs.append(f"[{output_label}]")

    filters.append(
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first:normalize=0,alimiter=limit=0.95[mixed]"
    )
    filter_script = work_dir / "audio_mix_filter.txt"
    filter_script.write_text(";\n".join(filters), encoding="utf-8")

    subtitle_input_index = len(voice_segments) + 1
    if subtitles_enabled:
        input_arguments.extend(["-i", str(subtitles_path)])
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        *input_arguments,
        "-filter_complex_script",
        str(filter_script),
        "-map",
        "0:v:0",
        "-map",
        "[mixed]",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
    ]
    if subtitles_enabled:
        language_code = LANGUAGE_CODES.get(target_language, "und")
        command.extend(
            [
                "-map",
                f"{subtitle_input_index}:s:0",
                "-c:s",
                "mov_text",
                "-metadata:s:s:0",
                f"language={language_code}",
            ]
        )
    else:
        command.append("-sn")
    command.append(str(output_path))

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise RuntimeError(f"មិនអាចចាប់ផ្តើម FFmpeg render បានទេ៖ {error}") from error
    if result.returncode:
        details = result.stderr.strip().splitlines()
        reason = details[-1] if details else "FFmpeg could not render the translated video."
        raise RuntimeError(f"FFmpeg render បរាជ័យ៖ {reason}")
