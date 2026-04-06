#!/usr/bin/env python3
"""Chatterbox-Turbo — FastAPI TTS server with voice cloning support.

Exposes:
  POST /v1/audio/speech — OpenAI-compatible TTS endpoint (returns WAV)
  POST /synthesize      — text → PCM16 24kHz mono audio
  GET  /voices          — list available voice reference clips
  POST /voices/upload   — upload a new voice reference clip
  GET  /health          — health check
"""

import io
import json
import os
import shutil
import struct
from pathlib import Path

# Must be set before importing torch — GB10 (sm_121) needs NVFuser disabled
# and CUDA arch forced to 12.0 to avoid NVRTC JIT compilation errors.
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

import numpy as np
import torch

# Workaround for GB10 (sm_121): torch.abs() on complex CUDA tensors triggers
# NVRTC JIT compilation that fails with "invalid value for --gpu-architecture".
# Patch complex abs to compute sqrt(real² + imag²) manually, avoiding the JIT kernel.
_orig_abs = torch.abs


def _safe_abs(input, *, out=None):
    if input.is_cuda and input.is_complex():
        real = input.real
        imag = input.imag
        result = torch.sqrt(real * real + imag * imag)
        if out is not None:
            out.copy_(result)
            return out
        return result
    return _orig_abs(input, out=out) if out is not None else _orig_abs(input)


torch.abs = _safe_abs
# Also patch Tensor.abs and the method form
_orig_tensor_abs = torch.Tensor.abs


def _safe_tensor_abs(self):
    return _safe_abs(self)


torch.Tensor.abs = _safe_tensor_abs
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="Chatterbox TTS Server")

_model = None
_voices_dir: Path = Path(os.getenv("VOICES_DIR", "/voices"))
_output_sample_rate = 24000
_conds_cache: dict[str, object] = {}  # voice_id -> cached Conditionals


def _load_model():
    """Load Chatterbox-Turbo model."""
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("Loading Chatterbox-Turbo model...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    print("Chatterbox-Turbo loaded successfully", flush=True)
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
    """Load voices.json registry."""
    config_path = _voices_dir / "voices.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


def _resolve_voice_path(voice_id: str) -> Path | None:
    """Resolve a voice_id to an audio file path."""
    voices = _get_voices_config()
    if voice_id in voices:
        file_name = voices[voice_id].get("file", f"{voice_id}.wav")
        path = _voices_dir / file_name
        if path.exists():
            return path
    # Fallback: try direct file name
    for ext in (".wav", ".mp3", ".flac"):
        path = _voices_dir / f"{voice_id}{ext}"
        if path.exists():
            return path
    return None


@app.get("/voices")
async def list_voices():
    """List available voice reference clips."""
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
    """Upload a new voice reference clip."""
    safe_name = "".join(c for c in voice_id if c.isalnum() or c in "-_")
    if not safe_name:
        return JSONResponse({"error": "Invalid voice_id"}, status_code=400)

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    dest = _voices_dir / f"{safe_name}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Update voices.json
    voices = _get_voices_config()
    voices[safe_name] = {
        "file": f"{safe_name}{suffix}",
        "description": description,
    }
    (_voices_dir / "voices.json").write_text(json.dumps(voices, indent=2))

    # Invalidate cached conditionals for re-uploaded voice
    for key in [k for k in _conds_cache if k == safe_name]:
        del _conds_cache[key]

    return {"status": "ok", "voice_id": safe_name, "file": f"{safe_name}{suffix}"}


@app.post("/synthesize")
async def synthesize(request: Request):
    """Synthesize text to speech.

    JSON body:
      {"text": "Hello world", "voice_id": "default"}

    Returns: raw PCM16 24kHz mono audio as application/octet-stream.
    """
    body = await request.json()
    text = body.get("text", "")
    voice_id = body.get("voice_id", "default")

    if not text.strip():
        return JSONResponse({"error": "Empty text"}, status_code=400)

    # Resolve voice reference for cloning, with cached conditionals
    audio_prompt_path = _resolve_voice_path(voice_id)
    if audio_prompt_path:
        if voice_id in _conds_cache:
            _model.conds = _conds_cache[voice_id]
        else:
            _model.prepare_conditionals(str(audio_prompt_path))
            _conds_cache[voice_id] = _model.conds

    import time as _time
    t0 = _time.monotonic()
    cache_hit = voice_id in _conds_cache

    try:
        wav = _model.generate(text)
    except Exception as e:
        return JSONResponse({"error": f"Generation failed: {e}"}, status_code=500)

    gen_ms = round((_time.monotonic() - t0) * 1000, 1)

    # wav is a torch tensor — convert to PCM16 bytes
    if isinstance(wav, torch.Tensor):
        audio_np = wav.squeeze().cpu().numpy()
    else:
        audio_np = np.array(wav, dtype=np.float32)

    # Normalize to [-1, 1] range if needed
    peak = np.abs(audio_np).max()
    if peak > 0:
        audio_np = audio_np / peak * 0.95

    pcm16 = (audio_np * 32767).astype(np.int16)
    audio_dur_ms = round(len(pcm16) / _output_sample_rate * 1000, 1)

    print(
        f"[TTS] text={len(text)}ch gen={gen_ms}ms audio={audio_dur_ms}ms "
        f"cache={'hit' if cache_hit else 'miss'} voice={voice_id}",
        flush=True,
    )

    return Response(
        content=pcm16.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(_output_sample_rate),
            "X-Channels": "1",
            "X-Sample-Width": "2",
        },
    )


def _pcm16_to_wav(pcm_data: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM16 bytes in a WAV container."""
    data_size = len(pcm_data)
    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<HHIIHH", 1, channels, sample_rate,
                          sample_rate * channels * sample_width,
                          channels * sample_width, sample_width * 8))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_data)
    return buf.getvalue()


@app.post("/v1/audio/speech")
async def openai_speech(request: Request):
    """OpenAI-compatible TTS endpoint.

    JSON body:
      {"model": "chatterbox", "input": "Hello", "voice": "default", "response_format": "wav"}

    Returns: WAV audio (audio/wav).
    """
    body = await request.json()
    text = body.get("input", "")
    voice_id = body.get("voice", "default")

    if not text.strip():
        return JSONResponse({"error": {"message": "Empty input"}}, status_code=400)

    audio_prompt_path = _resolve_voice_path(voice_id)
    if audio_prompt_path:
        if voice_id in _conds_cache:
            _model.conds = _conds_cache[voice_id]
        else:
            _model.prepare_conditionals(str(audio_prompt_path))
            _conds_cache[voice_id] = _model.conds

    import time as _time
    t0 = _time.monotonic()
    cache_hit = voice_id in _conds_cache

    try:
        wav = _model.generate(text)
    except Exception as e:
        return JSONResponse({"error": {"message": f"Generation failed: {e}"}}, status_code=500)

    gen_ms = round((_time.monotonic() - t0) * 1000, 1)

    if isinstance(wav, torch.Tensor):
        audio_np = wav.squeeze().cpu().numpy()
    else:
        audio_np = np.array(wav, dtype=np.float32)

    peak = np.abs(audio_np).max()
    if peak > 0:
        audio_np = audio_np / peak * 0.95

    pcm16 = (audio_np * 32767).astype(np.int16)
    audio_dur_ms = round(len(pcm16) / _output_sample_rate * 1000, 1)

    print(
        f"[TTS] text={len(text)}ch gen={gen_ms}ms audio={audio_dur_ms}ms "
        f"cache={'hit' if cache_hit else 'miss'} voice={voice_id} endpoint=v1/audio/speech",
        flush=True,
    )

    wav_bytes = _pcm16_to_wav(pcm16.tobytes(), _output_sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


if __name__ == "__main__":
    port = int(os.getenv("LISTEN_PORT", "4123"))
    uvicorn.run(app, host="0.0.0.0", port=port)
