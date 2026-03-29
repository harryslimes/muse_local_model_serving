#!/usr/bin/env python3
"""Chatterbox Turbo MLX TTS server with pre-computed voice embeddings.

Uses mlx_audio directly (no subprocess proxy) and loads voice conditionals
from .safetensors files for accurate voice cloning.

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
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "4123"))
VOICES_DIR = Path(os.getenv("VOICES_DIR", Path(__file__).parent.parent / "voices"))

app = FastAPI(title="Chatterbox MLX TTS Server")
_model = None
_model_lock = threading.Lock()


def _load_model():
    from mlx_audio.tts.utils import load_model
    print(f"Loading TTS model: {MODEL_ID}", flush=True)
    model = load_model(MODEL_ID)
    print("TTS model loaded", flush=True)
    return model


def _load_voice_conditionals(voice_name: str):
    """Load pre-computed voice conditionals from .safetensors, or extract from .wav."""
    from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import (
        Conditionals,
        T3Cond,
    )

    # Prefer safetensors (pre-computed embeddings).
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

    # Fall back to extracting from wav at runtime.
    for ext in (".wav", ".mp3", ".flac", ".ogg"):
        wav_path = VOICES_DIR / f"{voice_name}{ext}"
        if wav_path.is_file():
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
    _model = _load_model()


@app.get("/health")
async def health():
    if _model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "model": MODEL_ID}


@app.get("/voices")
async def list_voices():
    voices = []
    if VOICES_DIR.is_dir():
        for f in sorted(VOICES_DIR.iterdir()):
            if f.suffix in (".wav", ".mp3", ".flac", ".ogg", ".safetensors"):
                voices.append({"id": f.stem, "file": f.name})
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
