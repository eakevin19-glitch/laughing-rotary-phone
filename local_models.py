import gc
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable


MODEL_CACHE_DIR = Path(os.environ.get("LOCAL_MODEL_CACHE", Path.home() / ".cache" / "reeltalk-models"))
NLLB_MODEL_ID = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
NLLB_TOKENIZER_ID = "facebook/nllb-200-distilled-600M"
KHMER_TTS_MODEL_ID = "facebook/mms-tts-khm"
LOCAL_VOICE = "mms-khmer"
WHISPER_MODEL_SIZES = {"base", "small"}
VOICE_STYLES = {
    "Natural": "Natural pace",
    "Warm narrator": "Warm, slightly slower pace",
    "Dramatic": "Dramatic, slower pace",
    "Calm": "Calm, measured pace",
}
VOICE_STYLE_SPEED = {
    "Natural": 1.0,
    "Warm narrator": 0.96,
    "Dramatic": 0.92,
    "Calm": 0.9,
}
SOURCE_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "km": "khm_Khmr",
    "nl": "nld_Latn",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "uk": "ukr_Cyrl",
}
SOURCE_LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
    "vi": "Vietnamese",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "id": "Indonesian",
    "ms": "Malay",
    "km": "Khmer",
    "nl": "Dutch",
    "tr": "Turkish",
    "pl": "Polish",
    "uk": "Ukrainian",
}
ProgressCallback = Callable[[float, str], None]


def _cpu_threads() -> int:
    return max(1, min(os.cpu_count() or 2, 4))


def transcribe_audio_chunks(
    audio_chunks: list[Path],
    model_size: str = "base",
    progress: ProgressCallback | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    if model_size not in WHISPER_MODEL_SIZES:
        raise ValueError("Whisper model ត្រូវជា base ឬ small។")
    try:
        from faster_whisper import WhisperModel

        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=_cpu_threads(),
            num_workers=1,
            download_root=str(MODEL_CACHE_DIR / "whisper"),
        )
    except Exception as error:
        raise RuntimeError(f"មិនអាចផ្ទុក local Whisper {model_size} បានទេ។ ពិនិត្យ disk/internet សម្រាប់ការទាញ model លើកដំបូង៖ {error}") from error

    detected_languages = []
    transcript_segments = []
    try:
        for chunk_index, audio_path in enumerate(audio_chunks):
            try:
                segments, info = model.transcribe(
                    str(audio_path),
                    beam_size=3,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                if info.language:
                    detected_languages.append(info.language)
                for segment in segments:
                    text = segment.text.strip()
                    if not text or segment.end <= segment.start:
                        continue
                    transcript_segments.append(
                        {
                            "id": len(transcript_segments) + 1,
                            "start": float(segment.start) + chunk_index * 600,
                            "end": float(segment.end) + chunk_index * 600,
                            "source_text": text,
                        }
                    )
            except Exception as error:
                raise RuntimeError(f"Local Whisper មិនអាចស្គាល់ audio part {chunk_index + 1} បានទេ៖ {error}") from error
            if progress:
                progress((chunk_index + 1) / len(audio_chunks), f"Local Whisper · {chunk_index + 1}/{len(audio_chunks)}")
    finally:
        del model
        gc.collect()

    if not transcript_segments:
        raise ValueError("Local Whisper មិនរកឃើញពាក្យនិយាយក្នុងវីដេអូទេ។")
    source_language = Counter(detected_languages).most_common(1)[0][0] if detected_languages else "unknown"
    if source_language not in SOURCE_LANGUAGE_CODES:
        raise ValueError(f"Local NLLB មិនទាន់មាន language mapping សម្រាប់ Whisper language code: {source_language}។")
    return source_language, transcript_segments


def _protect_glossary(source_text: str, translated_text: str, glossary: dict[str, str]) -> str:
    for source_term, target_term in glossary.items():
        if source_term.casefold() not in source_text.casefold():
            continue
        start = translated_text.casefold().find(source_term.casefold())
        if start >= 0:
            translated_text = translated_text[:start] + target_term + translated_text[start + len(source_term):]
    return translated_text


def translate_segments(
    segments: list[dict[str, Any]],
    source_language: str,
    target_language: str,
    glossary: dict[str, str],
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    if not segments:
        raise ValueError("គ្មាន subtitle សម្រាប់បកប្រែទេ។")
    if target_language != "Khmer":
        raise ValueError("Local MMS TTS ក្នុង build នេះបង្កើតសំឡេងភាសាខ្មែរតែប៉ុណ្ណោះ។")

    source_code = SOURCE_LANGUAGE_CODES.get(source_language)
    target_code = "khm_Khmr"
    if not source_code:
        raise ValueError(f"Local NLLB មិនគាំទ្រ source language: {source_language}។")

    try:
        import ctranslate2
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        translation_path = snapshot_download(
            repo_id=NLLB_MODEL_ID,
            cache_dir=str(MODEL_CACHE_DIR / "nllb"),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            NLLB_TOKENIZER_ID,
            src_lang=source_code,
            cache_dir=str(MODEL_CACHE_DIR / "tokenizer"),
        )
        translator = ctranslate2.Translator(
            translation_path,
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=_cpu_threads(),
        )
    except Exception as error:
        raise RuntimeError(f"មិនអាចផ្ទុក local NLLB int8 បានទេ។ ពិនិត្យ disk/internet សម្រាប់ការទាញ model លើកដំបូង៖ {error}") from error

    glossary_texts = dict(glossary)
    translated_segments = []
    batch_size = 8
    batches = [segments[index:index + batch_size] for index in range(0, len(segments), batch_size)]
    try:
        for batch_index, batch in enumerate(batches):
            source_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(segment["source_text"])) for segment in batch]
            results = translator.translate_batch(
                source_tokens,
                target_prefix=[[target_code] for _ in batch],
                beam_size=2,
                max_decoding_length=192,
            )
            for segment, result in zip(batch, results):
                tokens = result.hypotheses[0]
                if tokens and tokens[0] == target_code:
                    tokens = tokens[1:]
                translated_text = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(tokens),
                    skip_special_tokens=True,
                ).strip()
                if not translated_text:
                    raise RuntimeError("Local NLLB បានត្រឡប់ subtitle ទទេ។")
                translated_text = _protect_glossary(segment["source_text"], translated_text, glossary_texts)
                translated_segments.append({**segment, "translated_text": translated_text})
            if progress:
                progress((batch_index + 1) / len(batches), f"Local NLLB · {batch_index + 1}/{len(batches)}")
    finally:
        del translator, tokenizer
        gc.collect()

    return translated_segments


def generate_voice_segments(
    segments: list[dict[str, Any]],
    work_dir: Path,
    target_language: str,
    voice: str,
    voice_style: str = "Natural",
    progress: ProgressCallback | None = None,
) -> list[tuple[dict[str, Any], Path]]:
    if target_language != "Khmer":
        raise ValueError("Local MMS TTS ក្នុង build នេះបង្កើតសំឡេងភាសាខ្មែរតែប៉ុណ្ណោះ។")
    if voice != LOCAL_VOICE:
        raise ValueError("Voice ដែលបានជ្រើសមិនមែនជា local MMS Khmer voice ទេ។")
    if voice_style not in VOICE_STYLES:
        raise ValueError("Voice style មិនត្រឹមត្រូវទេ។")

    voice_dir = work_dir / "voice_segments"
    voice_dir.mkdir(parents=True, exist_ok=True)
    voice_segments = [segment for segment in segments if segment["translated_text"].strip()]
    voice_paths = [voice_dir / f"voice_{segment['id']:05d}.wav" for segment in voice_segments]
    pending = [(segment, path) for segment, path in zip(voice_segments, voice_paths) if not path.exists() or path.stat().st_size <= 44]
    tokenizer = None
    model = None
    if pending:
        try:
            import numpy as np
            import torch
            from huggingface_hub import snapshot_download
            from scipy.io import wavfile
            from transformers import AutoTokenizer, VitsModel

            torch.set_num_threads(_cpu_threads())
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tts_path = snapshot_download(
                repo_id=KHMER_TTS_MODEL_ID,
                cache_dir=str(MODEL_CACHE_DIR / "mms-tts"),
            )
            tokenizer = AutoTokenizer.from_pretrained(tts_path, local_files_only=True)
            model = VitsModel.from_pretrained(tts_path, local_files_only=True).to("cpu").eval()
        except Exception as error:
            raise RuntimeError(f"មិនអាចផ្ទុក local MMS Khmer TTS បានទេ។ ពិនិត្យ disk/RAM/internet សម្រាប់ការទាញ model លើកដំបូង៖ {error}") from error

    generated = []
    try:
        for index, (segment, voice_path) in enumerate(zip(voice_segments, voice_paths)):
            if voice_path not in [path for _, path in pending]:
                generated.append(({**segment, "voice_style": voice_style}, voice_path))
            else:
                inputs = tokenizer(segment["translated_text"], return_tensors="pt")
                with torch.inference_mode():
                    waveform = model(**inputs).waveform[0].cpu().numpy()
                pcm = np.asarray(np.clip(waveform, -1.0, 1.0) * 32767, dtype=np.int16)
                wavfile.write(voice_path, model.config.sampling_rate, pcm)
                generated.append(({**segment, "voice_style": voice_style}, voice_path))
            if progress:
                progress((index + 1) / max(1, len(voice_segments)), f"Local MMS TTS · {index + 1}/{len(voice_segments)}")
    except Exception as error:
        raise RuntimeError(f"Local MMS TTS មិនអាចបង្កើតសំឡេងសម្រាប់ subtitle លេខ {segment['id']} បានទេ៖ {error}") from error
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()

    if not generated:
        raise ValueError("គ្មាន subtitle សម្រាប់បង្កើតសំឡេងទេ។")
    return generated
