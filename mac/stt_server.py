#!/usr/bin/env python3
"""Parakeet TDT 0.6B v3 MLX — FastAPI STT server (Apple Silicon).

Exposes:
  POST /transcribe — audio → {"text": "...", "duration_seconds": ...}
  GET  /health     — health check

Accepts raw PCM16 16kHz mono or WAV in the request body.

Model: mlx-community/parakeet-tdt-0.6b-v3

Requirements:
  pip install parakeet-mlx fastapi uvicorn soundfile numpy
  (parakeet-mlx provides the MLX-native Parakeet TDT inference)
"""

import io
import os
import tempfile
import wave

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Parakeet TDT v3 MLX STT Server")

MODEL_ID = os.getenv("PARAKEET_MLX_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
_sample_rate = 16000
_model = None


def _load_model():
    import parakeet_mlx

    print(f"Loading Parakeet TDT MLX model: {MODEL_ID}", flush=True)
    model = parakeet_mlx.load(MODEL_ID)
    print("Parakeet TDT MLX loaded successfully", flush=True)
    return model


@app.on_event("startup")
async def startup():
    global _model
    _model = _load_model()


@app.get("/health")
async def health():
    if _model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


def _pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _parse_wav_or_raw(data: bytes) -> np.ndarray:
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if wf.getnchannels() == 2:
                audio = audio[::2]
            return audio
    return _pcm16_bytes_to_float32(data)


@app.post("/transcribe")
async def transcribe(request: Request):
    """Transcribe audio to text.

    Accepts raw binary body: PCM16 16kHz mono or WAV.
    Returns: {"text": "...", "duration_seconds": ...}
    """
    body = await request.body()
    if not body:
        return JSONResponse({"error": "Empty audio data"}, status_code=400)

    audio = _parse_wav_or_raw(body)

    if len(audio) < 160:  # less than 10ms
        return {"text": "", "duration_seconds": 0.0}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio, _sample_rate)
        result = _model.transcribe(tmp.name)

    text = result if isinstance(result, str) else str(result)
    return {"text": text.strip(), "duration_seconds": len(audio) / _sample_rate}


if __name__ == "__main__":
    port = int(os.getenv("LISTEN_PORT", "4124"))
    uvicorn.run(app, host="0.0.0.0", port=port)
