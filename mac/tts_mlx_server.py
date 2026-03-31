#!/usr/bin/env python3
"""Chatterbox Turbo MLX TTS server with pre-computed voice embeddings.

Uses mlx_audio directly (no subprocess proxy) and loads voice conditionals
from .safetensors files for accurate voice cloning.

On startup, scans voices/clips/ for audio files that lack a corresponding
.safetensors file and extracts conditionals using the fp16 model so the
quantised serving model gets high-quality voice embeddings.

Exposes:
  POST /v1/audio/speech  — OpenAI-compatible TTS endpoint (returns WAV)
  GET  /health           — health check
  GET  /voices           — list available voices
"""

import io
import os
import struct
import threading
from pathlib import Path

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

MODEL_ID = os.getenv("CHATTERBOX_MLX_MODEL", "mlx-community/chatterbox-turbo-8bit")
EXTRACT_MODEL_ID = os.getenv(
    "CHATTERBOX_MLX_EXTRACT_MODEL", "mlx-community/chatterbox-turbo-fp16"
)
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "4123"))
VOICES_DIR = Path(os.getenv("VOICES_DIR", Path(__file__).parent.parent / "voices"))
CLIPS_DIR = VOICES_DIR / "clips"

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")

app = FastAPI(title="Chatterbox MLX TTS Server")
_model = None
_model_lock = threading.Lock()


def _load_model(model_id: str = MODEL_ID):
    from mlx_audio.tts.utils import load_model
    print(f"Loading TTS model: {model_id}", flush=True)
    model = load_model(model_id)
    print(f"TTS model loaded: {model_id}", flush=True)
    return model


def _extract_missing_safetensors():
    """Find clips without .safetensors and extract conditionals using the fp16 model."""
    if not CLIPS_DIR.is_dir():
        return

    missing = []
    for f in sorted(CLIPS_DIR.iterdir()):
        if f.suffix in AUDIO_EXTENSIONS:
            st_path = VOICES_DIR / f"{f.stem}.safetensors"
            if not st_path.is_file():
                missing.append(f)

    if not missing:
        return

    print(
        f"Found {len(missing)} clip(s) without safetensors: "
        f"{[f.name for f in missing]}",
        flush=True,
    )
    print(f"Loading fp16 model for extraction: {EXTRACT_MODEL_ID}", flush=True)
    extract_model = _load_model(EXTRACT_MODEL_ID)

    for clip_path in missing:
        st_path = VOICES_DIR / f"{clip_path.stem}.safetensors"
        print(f"  Extracting conditionals: {clip_path.name} -> {st_path.name}", flush=True)
        extract_model.prepare_conditionals(str(clip_path))
        conds = extract_model._conds

        arrays = {}
        for attr in ("cond_prompt_speech_tokens", "speaker_emb"):
            val = getattr(conds.t3, attr, None)
            if val is not None:
                arrays[f"t3_{attr}"] = val
        for key, val in conds.gen.items():
            arrays[f"gen_{key}"] = val

        mx.save_safetensors(str(st_path), arrays)
        print(f"  Saved {st_path.name}", flush=True)

    # Free the fp16 model
    del extract_model
    print("Extraction complete, fp16 model unloaded", flush=True)


def _load_voice_conditionals(voice_name: str):
    """Load pre-computed voice conditionals from .safetensors."""
    from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import (
        Conditionals,
        T3Cond,
    )

    st_path = VOICES_DIR / f"{voice_name}.safetensors"
    if st_path.is_file():
        arrays = mx.load(str(st_path))
        t3_kwargs = {}
        gen = {}
        for k, v in arrays.items():
            if k.startswith("t3_"):
                t3_kwargs[k[3:]] = v
            elif k.startswith("gen_"):
                gen[k[4:]] = v
        return Conditionals(t3=T3Cond(**t3_kwargs), gen=gen)

    # Fall back to extracting from clip at runtime (lower quality with quantised model).
    for ext in AUDIO_EXTENSIONS:
        wav_path = CLIPS_DIR / f"{voice_name}{ext}"
        if wav_path.is_file():
            print(
                f"Warning: no safetensors for '{voice_name}', "
                f"falling back to runtime extraction (lower quality)",
                flush=True,
            )
            _model.prepare_conditionals(str(wav_path))
            return _model._conds

    return None


class SpeechRequest(BaseModel):
    model: str = "chatterbox"
    input: str = ""
    voice: str = "default"
    response_format: str = "wav"
    speed: float = 1.0


def _make_wav(pcm_float: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert float32 audio to WAV bytes."""
    pcm16 = np.clip(pcm_float * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    data = pcm16.tobytes()
    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(data)
    return buf.getvalue()


@app.on_event("startup")
async def startup():
    global _model
    _extract_missing_safetensors()
    _model = _load_model()


@app.get("/health")
async def health():
    if _model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "model": MODEL_ID}


@app.get("/voices")
async def list_voices():
    """List available voices (one entry per safetensors file)."""
    voices = []
    if VOICES_DIR.is_dir():
        for f in sorted(VOICES_DIR.iterdir()):
            if f.suffix == ".safetensors":
                voices.append({"id": f.stem, "description": f.stem})
    return {"voices": voices}


@app.post("/v1/audio/speech")
async def speech(request: Request):
    body = await request.json()
    payload = SpeechRequest(**body)

    if not payload.input.strip():
        return JSONResponse({"error": "Empty input"}, status_code=400)

    with _model_lock:
        # Load voice conditionals.
        voice = payload.voice or "default"
        conds = _load_voice_conditionals(voice)
        if conds is not None:
            _model._conds = conds

        # Generate audio.
        all_audio = []
        for result in _model.generate(
            payload.input,
            voice=None,
            ref_audio=None,
            speed=payload.speed,
        ):
            if hasattr(result, "audio") and result.audio is not None:
                audio = result.audio
                if isinstance(audio, mx.array):
                    audio = np.array(audio)
                all_audio.append(audio.flatten())

    if not all_audio:
        return JSONResponse({"error": "No audio generated"}, status_code=500)

    audio = np.concatenate(all_audio)
    wav_bytes = _make_wav(audio, sample_rate=_model.sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model"}] if _model else [],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
