"""faster-whisper transcription of SD-QA recordings (ORP_SCOPE.md, Phase 1 step 2).

faster-whisper is Linux-only here (its onnxruntime dep has no Intel-macOS wheel) and its model
weights download from the Hugging Face Hub, which is blocked in this cloud container — so this
runs on IMS (or any Linux host with HF access), not the Mac and not here. The import is lazy so
the module loads anywhere (CI, Mac) without faster-whisper installed.

Transcripts are cached to a JSON file keyed by example_id, so an interrupted run resumes and a
finished one is never recomputed.
"""

import json
from collections.abc import Callable
from pathlib import Path

LANGUAGE = "sw"
MODEL_SIZE = "large-v3"


def load_model(size: str = MODEL_SIZE, device: str = "cpu", compute_type: str = "int8"):
    """Load a faster-whisper model (int8 CPU by default). Needs HF access to fetch weights."""
    from faster_whisper import WhisperModel  # lazy: Linux-only, weights from HF Hub

    return WhisperModel(size, device=device, compute_type=compute_type)


def transcribe_file(model, wav_path: str | Path, language: str = LANGUAGE) -> str:
    """Transcribe one audio file to text."""
    segments, _info = model.transcribe(str(wav_path), language=language)
    return "".join(segment.text for segment in segments).strip()


def transcribe_all(
    wav_paths: dict[int, Path],
    cache_path: str | Path,
    model,
    language: str = LANGUAGE,
    transcriber: Callable[[object, str | Path, str], str] = transcribe_file,
) -> dict[int, str]:
    """Transcribe each recording, caching to disk; already-cached ids are skipped.

    `transcriber` is injected so the orchestration is testable without a real model.
    """
    cache_path = Path(cache_path)
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for example_id, wav in wav_paths.items():
        if str(example_id) in cache:
            continue
        cache[str(example_id)] = transcriber(model, wav, language)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return {int(k): v for k, v in cache.items()}
