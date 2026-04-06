#!/usr/bin/env python3
"""Chatterbox Turbo MLX — FastAPI TTS server (Apple Silicon).

Exposes:
  POST /synthesize       — text → raw PCM16 24kHz mono (application/octet-stream)
  POST /v1/audio/speech  — OpenAI-compatible TTS endpoint (returns WAV)
  GET  /voices           — list available voice reference clips
  POST /voices/upload    — upload a new voice reference clip
  GET  /health           — health check

Model: mlx-community/chatterbox-turbo-8bit

Requirements:
  pip install chatterbox-audio mlx fastapi uvicorn soundfile numpy
  (chatterbox-audio >=0.2 includes MLX backend support)
"""

import io
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="Chatterbox Turbo MLX TTS Server")

MODEL_ID = os.getenv("CHATTERBOX_MLX_MODEL", "mlx-community/chatterbox-turbo-8bit")
_voices_dir: Path = Path(os.getenv("VOICES_DIR", str(Path(__file__).parent.parent / "voices")))
_output_sample_rate = 24000
_model = None
_conds_cache: dict[str, object] = {}


def _load_model():
    from chatterbox.tts import ChatterboxTTS

    print(f"Loading Chatterbox Turbo MLX model: {MODEL_ID}", flush=True)
    # MLX backend is selected automatically on Apple Silicon when loading an MLX-format repo.
    # Pass repo_id to override the default pretrained weights.
    model = ChatterboxTTS.from_pretrained(repo_id=MODEL_ID)
    print("Chatterbox Turbo MLX loaded successfully", flush=True)
    return model


@app.on_event("startup")
async def startup():
    global _model
    _voices_dir.mkdir(parents=True, exist_ok=True)
    _model = _load_model()


@app.get("/health")
async def health():
    if _model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


def _get_voices_config() -> dict:
    config_path = _voices_dir / "voices.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


def _resolve_voice_path(voice_id: str) -> Path | None:
    voices = _get_voices_config()
    if voice_id in voices:
        file_name = voices[voice_id].get("file", f"{voice_id}.wav")
        path = _voices_dir / file_name
        if path.exists():
            return path
    for ext in (".wav", ".mp3", ".flac"):
        path = _voices_dir / f"{voice_id}{ext}"
        if path.exists():
            return path
    return None


@app.get("/voices")
async def list_voices():
    voices = _get_voices_config()
    result = []
    for voice_id, meta in voices.items():
        file_path = _voices_dir / meta.get("file", f"{voice_id}.wav")
        result.append({
            "id": voice_id,
            "description": meta.get("description", ""),
            "file": meta.get("file", ""),
            "available": file_path.exists(),
        })
    return {"voices": result}


@app.post("/voices/upload")
async def upload_voice(
    voice_id: str = Form(...),
    description: str = Form(default=""),
    file: UploadFile = File(...),
):
    safe_name = "".join(c for c in voice_id if c.isalnum() or c in "-_")
    if not safe_name:
        return JSONResponse({"error": "Invalid voice_id"}, status_code=400)

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    dest = _voices_dir / f"{safe_name}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    voices = _get_voices_config()
    voices[safe_name] = {"file": f"{safe_name}{suffix}", "description": description}
    (_voices_dir / "voices.json").write_text(json.dumps(voices, indent=2))

    _conds_cache.pop(safe_name, None)
    return {"status": "ok", "voice_id": safe_name, "file": f"{safe_name}{suffix}"}


def _synthesize(text: str, voice_id: str) -> np.ndarray:
    audio_prompt_path = _resolve_voice_path(voice_id)

    if audio_prompt_path:
        if voice_id not in _conds_cache:
            _model.prepare_conditionals(str(audio_prompt_path))
            _conds_cache[voice_id] = _model.conds
        else:
            _model.conds = _conds_cache[voice_id]

    wav = _model.generate(text)

    import mlx.core as mx
    if isinstance(wav, mx.array):
        audio_np = np.array(wav).flatten().astype(np.float32)
    elif hasattr(wav, "cpu"):
        audio_np = wav.squeeze().cpu().numpy()
    else:
        audio_np = np.array(wav, dtype=np.float32).flatten()

    peak = np.abs(audio_np).max()
    if peak > 0:
        audio_np = audio_np / peak * 0.95
    return audio_np


def _to_pcm16(audio_np: np.ndarray) -> bytes:
    return (audio_np * 32767).astype(np.int16).tobytes()


def _to_wav(pcm_data: bytes, sample_rate: int) -> bytes:
    data_size = len(pcm_data)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_data)
    return buf.getvalue()


@app.post("/synthesize")
async def synthesize(request: Request):
    """JSON body: {"text": "...", "voice_id": "default"}
    Returns raw PCM16 24kHz mono.
    """
    body = await request.json()
    text = body.get("text", "")
    voice_id = body.get("voice_id", "default")

    if not text.strip():
        return JSONResponse({"error": "Empty text"}, status_code=400)

    import time as _time
    t0 = _time.monotonic()
    try:
        audio_np = _synthesize(text, voice_id)
    except Exception as e:
        return JSONResponse({"error": f"Generation failed: {e}"}, status_code=500)

    pcm16 = _to_pcm16(audio_np)
    gen_ms = round((_time.monotonic() - t0) * 1000, 1)
    audio_dur_ms = round(len(audio_np) / _output_sample_rate * 1000, 1)
    print(f"[TTS] text={len(text)}ch gen={gen_ms}ms audio={audio_dur_ms}ms voice={voice_id}", flush=True)

    return Response(
        content=pcm16,
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(_output_sample_rate),
            "X-Channels": "1",
            "X-Sample-Width": "2",
        },
    )


@app.post("/v1/audio/speech")
async def openai_speech(request: Request):
    """OpenAI-compatible TTS.
    JSON body: {"model": "chatterbox", "input": "...", "voice": "default"}
    Returns WAV audio.
    """
    body = await request.json()
    text = body.get("input", "")
    voice_id = body.get("voice", "default")

    if not text.strip():
        return JSONResponse({"error": {"message": "Empty input"}}, status_code=400)

    import time as _time
    t0 = _time.monotonic()
    try:
        audio_np = _synthesize(text, voice_id)
    except Exception as e:
        return JSONResponse({"error": {"message": f"Generation failed: {e}"}}, status_code=500)

    pcm16 = _to_pcm16(audio_np)
    gen_ms = round((_time.monotonic() - t0) * 1000, 1)
    print(f"[TTS] text={len(text)}ch gen={gen_ms}ms voice={voice_id} endpoint=v1/audio/speech", flush=True)

    return Response(content=_to_wav(pcm16, _output_sample_rate), media_type="audio/wav")


if __name__ == "__main__":
    port = int(os.getenv("LISTEN_PORT", "4123"))
    uvicorn.run(app, host="0.0.0.0", port=port)
