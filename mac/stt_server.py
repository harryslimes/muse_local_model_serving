#!/usr/bin/env python3
"""Parakeet TDT v3 Core ML — FastAPI STT server (Apple Silicon).

Exposes:
  POST /transcribe — audio → {"text": "...", "duration_seconds": ...}
  GET  /health     — health check

Accepts raw PCM16 16kHz mono or WAV in the request body.

Uses FluidAudio CLI (fluidaudiocli) for Core ML inference on the Neural Engine,
avoiding GPU contention with LLM/TTS workloads.

Requirements:
  pip install fastapi uvicorn numpy
  fluidaudiocli binary in PATH or FLUIDAUDIO_CLI env var
"""

import asyncio
import io
import json
import os
import struct
import tempfile
import time
import wave

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Parakeet TDT v3 Core ML STT Server")

_FLUIDAUDIO_CLI = os.getenv(
    "FLUIDAUDIO_CLI",
    os.path.expanduser("~/.local/bin/fluidaudiocli"),
)
_sample_rate = 16000


@app.on_event("startup")
async def startup():
    # Verify the CLI binary exists
    if not os.path.isfile(_FLUIDAUDIO_CLI):
        print(f"WARNING: fluidaudiocli not found at {_FLUIDAUDIO_CLI}", flush=True)
    else:
        print(f"Using fluidaudiocli at {_FLUIDAUDIO_CLI}", flush=True)
        # Warm up: run a trivial transcription to trigger model download/load
        print("Warming up FluidAudio Core ML model...", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            _write_wav(tmp.name, np.zeros(1600, dtype=np.float32))
            proc = await asyncio.create_subprocess_exec(
                _FLUIDAUDIO_CLI, "transcribe", tmp.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
        print("FluidAudio Core ML model ready", flush=True)


@app.get("/health")
async def health():
    if not os.path.isfile(_FLUIDAUDIO_CLI):
        return JSONResponse({"status": "no_cli"}, status_code=503)
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


def _write_wav(path: str, audio: np.ndarray):
    """Write float32 audio to a 16-bit PCM WAV file."""
    pcm16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        wf.writeframes(pcm16.tobytes())


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

    duration_seconds = len(audio) / _sample_rate

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        _write_wav(tmp_path, audio)

    try:
        t0 = time.monotonic()
        json_out = tmp_path + ".json"
        proc = await asyncio.create_subprocess_exec(
            _FLUIDAUDIO_CLI, "transcribe", tmp_path,
            "--output-json", json_out,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        if proc.returncode != 0:
            err = stderr.decode().strip() if stderr else "unknown error"
            print(f"fluidaudiocli error ({elapsed_ms}ms): {err}", flush=True)
            return JSONResponse(
                {"error": f"STT CLI error: {err}"},
                status_code=500,
            )

        # Parse JSON output if available, fall back to stdout text
        text = ""
        if os.path.isfile(json_out):
            with open(json_out) as f:
                result = json.load(f)
            text = result.get("text", "").strip()
            os.unlink(json_out)
        else:
            text = stdout.decode().strip()

        print(
            f"STT: {text!r} ({elapsed_ms}ms, audio={round(duration_seconds * 1000)}ms)",
            flush=True,
        )
        return {"text": text, "duration_seconds": duration_seconds}
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.getenv("LISTEN_PORT", "4124"))
    uvicorn.run(app, host="0.0.0.0", port=port)
