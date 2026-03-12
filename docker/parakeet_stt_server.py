#!/usr/bin/env python3
"""Parakeet TDT 0.6B v2 — FastAPI STT server (ONNX Runtime).

Exposes POST /transcribe accepting raw PCM16 16 kHz mono audio (or WAV)
and returning {"text": "..."}.
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

app = FastAPI(title="Parakeet STT Server")

_model = None
_sample_rate = 16000


def _load_model():
    """Load the Parakeet TDT 0.6B v2 ONNX model via onnx-asr."""
    import onnx_asr

    model_name = os.getenv("PARAKEET_ONNX_MODEL", "nemo-parakeet-tdt-0.6b-v2")
    quantization = os.getenv("PARAKEET_QUANTIZATION", None) or None
    print(f"Loading ASR model: {model_name} (quantization={quantization})", flush=True)
    model = onnx_asr.load_model(model_name, quantization=quantization)
    print("Model loaded successfully", flush=True)
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
    """Convert raw PCM16 little-endian bytes to float32 numpy array."""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _parse_wav_or_raw(data: bytes) -> np.ndarray:
    """Parse WAV (extracting PCM data) or treat as raw PCM16 16kHz mono."""
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

    Accepts:
    - Raw binary body: PCM16 16kHz mono or WAV
    - Content-Type: application/octet-stream or audio/wav
    """
    body = await request.body()
    if not body:
        return JSONResponse({"error": "Empty audio data"}, status_code=400)

    audio = _parse_wav_or_raw(body)

    if len(audio) < 160:  # Less than 10ms of audio
        return {"text": "", "duration_seconds": 0.0}

    # onnx-asr expects a file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio, _sample_rate)
        result = _model.recognize(tmp.name)

    text = result if isinstance(result, str) else str(result)

    return {"text": text.strip(), "duration_seconds": len(audio) / _sample_rate}


if __name__ == "__main__":
    port = int(os.getenv("LISTEN_PORT", "4124"))
    uvicorn.run(app, host="0.0.0.0", port=port)
