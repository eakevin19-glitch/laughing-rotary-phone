import json
import mimetypes
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from local_models import (
    LOCAL_VOICE,
    VOICE_STYLES,
    WHISPER_MODEL_SIZES,
    generate_voice_segments,
    transcribe_audio_chunks,
    translate_segments,
)
from video_pipeline import (
    extract_audio_chunks,
    format_srt,
    format_vtt,
    render_dubbed_audio,
    render_mp3_audio,
    render_translated_video,
)


load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT_DIR / "uploads"
OUTPUT_DIR = ROOT_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

LANGUAGES = {
    "Khmer": "ខ្មែរ",
}
VOICES = {LOCAL_VOICE}
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".mpeg", ".mpg"}
STAGES = ["Upload", "Transcribe", "Translate", "Voice", "Render", "Complete"]
jobs: dict[str, dict] = {}
jobs_lock = threading.RLock()
job_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video-job")


def _parse_glossary(value: str) -> dict[str, str]:
    glossary = {}
    for line in value.splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            raise ValueError("ប្រើទម្រង់ ឈ្មោះដើម = ឈ្មោះគោលដៅ សម្រាប់ Glossary នីមួយៗ។")
        source, target = line.split("=", maxsplit=1)
        if not source.strip() or not target.strip():
            raise ValueError("ធាតុ Glossary ត្រូវមានទាំងពាក្យដើម និងពាក្យបកប្រែ។")
        glossary[source.strip()] = target.strip()
    return glossary


def _update_job(job_id: str, **values) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            job.update(values)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def _friendly_error(error: Exception) -> str:
    message = str(error)
    normalized = message.lower()
    if "out of memory" in normalized or "cannot allocate memory" in normalized:
        return "RAM មិនគ្រប់សម្រាប់ local model ទេ។ បិទកម្មវិធីផ្សេងៗ រួច Retry ឬជ្រើស Whisper base។"
    if "connection" in normalized or "download" in normalized or "offline" in normalized:
        return "មិនអាចទាញ local model បានទេ។ ពិនិត្យ internet សម្រាប់ការប្រើលើកដំបូង រួច Retry។"
    return message[:900] or "ដំណើរការបរាជ័យដោយមិនមានព័ត៌មានលម្អិត។"


def _process_job(job_id: str) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job["status"] = "processing"

    work_dir = Path(job["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = work_dir / "transcript.json"
    translations_path = work_dir / "translations.json"
    srt_path = work_dir / "translated.srt"
    vtt_path = work_dir / "translated.vtt"
    translated_video_path = Path(job["translated_video_path"])
    dubbed_audio_path = Path(job["dubbed_audio_path"])
    dubbed_mp3_path = Path(job["dubbed_mp3_path"])

    try:
        if transcript_path.exists():
            transcript = _read_json(transcript_path)
            source_language = transcript["source_language"]
            source_segments = transcript["segments"]
        else:
            _update_job(job_id, stage="Transcribe", progress=10, message="FFmpeg extracting audio · loading local Whisper model")
            audio_chunks = extract_audio_chunks(Path(job["original_path"]), work_dir)

            def transcription_progress(fraction: float, message: str) -> None:
                _update_job(job_id, progress=10 + round(fraction * 23), message=message)

            source_language, source_segments = transcribe_audio_chunks(
                audio_chunks,
                model_size=job["whisper_model"],
                progress=transcription_progress,
            )
            transcript = {"source_language": source_language, "segments": source_segments}
            _write_json(transcript_path, transcript)
        _update_job(
            job_id,
            stage="Translate",
            progress=35,
            source_language=source_language,
            message=f"Detected source language: {source_language} · loading local NLLB int8",
        )

        if translations_path.exists():
            translated_segments = _read_json(translations_path)
        else:
            def translation_progress(fraction: float, message: str) -> None:
                _update_job(job_id, progress=35 + round(fraction * 18), message=message)

            translated_segments = translate_segments(
                source_segments,
                source_language,
                job["target_language"],
                job["glossary"],
                progress=translation_progress,
            )
            _write_json(translations_path, translated_segments)
        _update_job(job_id, stage="Voice", progress=55, message="Loading local MMS Khmer TTS voice")

        def voice_progress(fraction: float, message: str) -> None:
            _update_job(job_id, progress=55 + round(fraction * 25), message=message)

        voice_segments = generate_voice_segments(
            translated_segments,
            work_dir,
            job["target_language"],
            job["voice"],
            job.get("voice_style", "Natural"),
            progress=voice_progress,
        )
        if not dubbed_audio_path.exists() or dubbed_audio_path.stat().st_size <= 44:
            _update_job(job_id, stage="Voice", progress=80, message="Assembling synchronized Khmer voice WAV")
            render_dubbed_audio(dubbed_audio_path, voice_segments, work_dir)
        if not dubbed_mp3_path.exists() or dubbed_mp3_path.stat().st_size == 0:
            render_mp3_audio(dubbed_audio_path, dubbed_mp3_path)
        srt_path.write_text(format_srt(translated_segments), encoding="utf-8-sig")
        vtt_path.write_text(format_vtt(translated_segments), encoding="utf-8-sig")

        _update_job(job_id, stage="Render", progress=82, message="Rendering new MP4 with FFmpeg")
        render_translated_video(
            Path(job["original_path"]),
            translated_video_path,
            srt_path,
            voice_segments,
            job["subtitles_enabled"],
            job["target_language"],
            work_dir,
        )
        _update_job(
            job_id,
            status="complete",
            stage="Complete",
            progress=100,
            message="Translation complete",
            translated_segments=translated_segments,
            source_language=source_language,
            srt_path=str(srt_path),
            vtt_path=str(vtt_path),
            error=None,
        )
    except Exception as error:
        safe_message = _friendly_error(error)
        current = jobs.get(job_id, {})
        app.logger.error("Video job %s failed during %s: %s", job_id, current.get("stage"), safe_message)
        _update_job(
            job_id,
            status="failed",
            error=safe_message,
            message="Processing failed",
        )


def _get_job_or_404(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        abort(404, description="រកមិនឃើញ video job នេះទេ។")
    return job


def _public_job(job: dict) -> dict:
    job_id = job["id"]
    return {
        "id": job_id,
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job.get("error"),
        "filename": job["filename"],
        "source_language": job.get("source_language"),
        "target_language": job["target_language"],
        "voice": job["voice"],
        "voice_style": job.get("voice_style", "Natural"),
        "available_voices": sorted(VOICES),
        "available_voice_styles": list(VOICE_STYLES),
        "whisper_model": job.get("whisper_model", "base"),
        "subtitles_enabled": job["subtitles_enabled"],
        "translated_segments": job.get("translated_segments", []),
        "stages": STAGES,
        "original_url": url_for("stream_media", job_id=job_id, asset="original"),
        "translated_url": url_for("stream_media", job_id=job_id, asset="translated"),
        "original_download_url": url_for("download_media", job_id=job_id, asset="original"),
        "translated_download_url": url_for("download_media", job_id=job_id, asset="translated"),
        "dubbed_audio_url": url_for("stream_media", job_id=job_id, asset="voice"),
        "dubbed_audio_download_url": url_for("download_media", job_id=job_id, asset="voice"),
        "dubbed_mp3_download_url": url_for("download_media", job_id=job_id, asset="voice_mp3"),
        "subtitle_download_url": url_for("download_media", job_id=job_id, asset="subtitles"),
        "captions_url": url_for("captions", job_id=job_id),
        "editor_url": url_for("editor", job_id=job_id),
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        languages=LANGUAGES,
        voices=sorted(VOICES),
        voice_styles=VOICE_STYLES,
        whisper_models=sorted(WHISPER_MODEL_SIZES),
    )


@app.post("/api/jobs")
def create_job():
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify(error="សូមជ្រើសរើសឯកសារវីដេអូ។"), 400

    suffix = Path(video.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify(error="ទម្រង់វីដេអូមិនគាំទ្រទេ។ ប្រើ MP4, MOV, MKV, WEBM ឬ AVI។"), 400

    target_language = request.form.get("target_language", "Khmer")
    if target_language not in LANGUAGES:
        return jsonify(error="Local Khmer TTS បច្ចុប្បន្នគាំទ្រភាសាខ្មែរតែប៉ុណ្ណោះ។"), 400
    voice = request.form.get("voice", LOCAL_VOICE)
    if voice not in VOICES:
        return jsonify(error="Local MMS voice មិនត្រឹមត្រូវទេ។"), 400
    whisper_model = request.form.get("whisper_model", "base")
    if whisper_model not in WHISPER_MODEL_SIZES:
        return jsonify(error="Whisper model ត្រូវជា base ឬ small។"), 400
    voice_style = request.form.get("voice_style", "Natural")
    if voice_style not in VOICE_STYLES:
        return jsonify(error="រចនាប័ទ្មសំឡេងដែលបានជ្រើសមិនត្រឹមត្រូវទេ។"), 400
    try:
        glossary = _parse_glossary(request.form.get("glossary", ""))
    except ValueError as error:
        return jsonify(error=str(error)), 400

    job_id = uuid.uuid4().hex
    original_name = secure_filename(video.filename) or f"video{suffix}"
    original_path = UPLOAD_DIR / f"{job_id}_{original_name}"
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    video.save(original_path)
    if original_path.stat().st_size == 0:
        return jsonify(error="ឯកសារវីដេអូទទេ។"), 400

    job = {
        "id": job_id,
        "status": "queued",
        "stage": "Upload",
        "progress": 4,
        "message": "Video uploaded and original saved",
        "error": None,
        "filename": original_name,
        "original_path": str(original_path),
        "translated_video_path": str(output_dir / f"{Path(original_name).stem}.translated.mp4"),
        "dubbed_audio_path": str(output_dir / f"{Path(original_name).stem}.{target_language.lower()}.voice.wav"),
        "dubbed_mp3_path": str(output_dir / f"{Path(original_name).stem}.{target_language.lower()}.voice.mp3"),
        "work_dir": str(output_dir),
        "target_language": target_language,
        "voice": voice,
        "whisper_model": whisper_model,
        "voice_style": voice_style,
        "subtitles_enabled": request.form.get("subtitles_enabled", "off").lower() in {"on", "true", "1"},
        "glossary": glossary,
        "translated_segments": [],
    }
    with jobs_lock:
        jobs[job_id] = job
    job_executor.submit(_process_job, job_id)
    return jsonify(job=_public_job(job)), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    return jsonify(job=_public_job(_get_job_or_404(job_id)))


@app.post("/api/jobs/<job_id>/retry")
def retry_job(job_id: str):
    job = _get_job_or_404(job_id)
    with jobs_lock:
        if job["status"] == "processing" or job["status"] == "queued":
            return jsonify(error="ការងារនេះកំពុងដំណើរការរួចហើយ។"), 409
        if job["status"] != "failed":
            return jsonify(error="អាច Retry បានតែការងារដែលបរាជ័យ។"), 409
        job.update(status="queued", error=None, message="Retrying from saved intermediate files")
    job_executor.submit(_process_job, job_id)
    return jsonify(job=_public_job(job)), 202


@app.post("/api/jobs/<job_id>/edit")
def edit_subtitles(job_id: str):
    job = _get_job_or_404(job_id)
    if job["status"] != "complete":
        return jsonify(error="វីដេអូត្រូវបញ្ចប់ដំណើរការមុនពេលកែ subtitle។"), 409
    edits = request.get_json(silent=True) or {}
    translated_texts = edits.get("translated_texts")
    voice = edits.get("voice", job["voice"])
    voice_style = edits.get("voice_style", job.get("voice_style", "Natural"))
    subtitles_enabled = edits.get("subtitles_enabled", job["subtitles_enabled"])
    segments = job.get("translated_segments", [])
    if not isinstance(translated_texts, list) or len(translated_texts) != len(segments):
        return jsonify(error="ទិន្នន័យ subtitle មិនត្រឹមត្រូវ ឬចំនួនមិនត្រូវគ្នា។"), 400
    if voice not in VOICES or voice_style not in VOICE_STYLES or not isinstance(subtitles_enabled, bool):
        return jsonify(error="ការកំណត់ voice ឬ subtitle មិនត្រឹមត្រូវទេ។"), 400
    if any(not isinstance(text, str) or not text.strip() for text in translated_texts):
        return jsonify(error="Subtitle នីមួយៗត្រូវមានអត្ថបទ។"), 400

    updated_segments = [
        {**segment, "translated_text": text.strip()}
        for segment, text in zip(segments, translated_texts)
    ]
    work_dir = Path(job["work_dir"])
    _write_json(work_dir / "translations.json", updated_segments)
    (work_dir / "translated.srt").write_text(format_srt(updated_segments), encoding="utf-8-sig")
    (work_dir / "translated.vtt").write_text(format_vtt(updated_segments), encoding="utf-8-sig")
    voice_dir = work_dir / "voice_segments"
    if voice_dir.exists():
        for voice_file in voice_dir.glob("voice_*.wav"):
            voice_file.unlink(missing_ok=True)
    Path(job["translated_video_path"]).unlink(missing_ok=True)
    Path(job["dubbed_audio_path"]).unlink(missing_ok=True)
    Path(job["dubbed_mp3_path"]).unlink(missing_ok=True)
    with jobs_lock:
        job.update(
            status="queued",
            stage="Voice",
            progress=55,
            error=None,
            message="Updating voice and render from edited subtitles",
            translated_segments=updated_segments,
            voice=voice,
            voice_style=voice_style,
            subtitles_enabled=subtitles_enabled,
        )
    job_executor.submit(_process_job, job_id)
    return jsonify(job=_public_job(job)), 202


@app.get("/jobs/<job_id>")
def editor(job_id: str):
    return render_template("editor.html", task=_public_job(_get_job_or_404(job_id)))


@app.get("/media/<job_id>/<asset>")
def stream_media(job_id: str, asset: str):
    job = _get_job_or_404(job_id)
    if asset == "original":
        path = Path(job["original_path"])
    elif asset == "translated" and job["status"] == "complete":
        path = Path(job["translated_video_path"])
    elif asset == "voice" and job["status"] == "complete":
        path = Path(job["dubbed_audio_path"])
    elif asset == "voice_mp3" and job["status"] == "complete":
        path = Path(job["dubbed_mp3_path"])
    else:
        abort(404)
    if not path.is_file():
        abort(404)
    mimetype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return send_file(path, mimetype=mimetype, conditional=True, as_attachment=False)


@app.get("/api/jobs/<job_id>/captions.vtt")
def captions(job_id: str):
    job = _get_job_or_404(job_id)
    path = Path(job["work_dir"]) / "translated.vtt"
    if job["status"] != "complete" or not path.is_file():
        abort(404)
    return send_file(path, mimetype="text/vtt; charset=utf-8", conditional=True)


@app.get("/download/<job_id>/<asset>")
def download_media(job_id: str, asset: str):
    job = _get_job_or_404(job_id)
    if asset == "original":
        path = Path(job["original_path"])
        download_name = job["filename"]
        mimetype = "application/octet-stream"
    elif asset == "translated":
        path = Path(job["translated_video_path"])
        download_name = Path(job["translated_video_path"]).name
        mimetype = "video/mp4"
    elif asset == "voice":
        path = Path(job["dubbed_audio_path"])
        download_name = Path(job["dubbed_audio_path"]).name
        mimetype = "audio/wav"
    elif asset == "voice_mp3":
        path = Path(job["dubbed_mp3_path"])
        download_name = Path(job["dubbed_mp3_path"]).name
        mimetype = "audio/mpeg"
    elif asset == "subtitles":
        path = Path(job["work_dir"]) / "translated.srt"
        download_name = f"{Path(job['filename']).stem}.{job['target_language'].lower()}.srt"
        mimetype = "application/x-subrip"
    else:
        abort(404)
    if not path.is_file() or (asset != "original" and job["status"] != "complete"):
        abort(404)
    return send_file(path, mimetype=mimetype, as_attachment=True, download_name=download_name, conditional=True)


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify(error="វីដេអូធំពេក។ កំណត់ទំហំ upload អតិបរមា 1 GB។"), 413


@app.errorhandler(404)
def not_found(error):
    description = getattr(error, "description", "រកមិនឃើញទំព័រនេះទេ។")
    if request.path.startswith("/api/") or request.path.startswith("/media/") or request.path.startswith("/download/"):
        return jsonify(error=description), 404
    return description, 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "3000")), debug=False, threaded=True)
