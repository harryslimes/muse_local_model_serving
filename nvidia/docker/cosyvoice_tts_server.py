#!/usr/bin/env python3
"""CosyVoice 3.0 FastAPI server with streaming endpoints.

Exposes:
  GET  /health                 - health check + model metadata
  GET  /voices                 - list registered prompt voices
  POST /voices/upload          - upload prompt voice metadata + audio
  POST /synthesize             - voice_id-based synthesis, returns full PCM16 audio
  POST /synthesize/stream      - voice_id-based synthesis, streams PCM16 audio
  POST /inference_zero_shot    - direct CosyVoice zero-shot streaming endpoint
  POST /inference_cross_lingual - direct CosyVoice cross-lingual streaming endpoint
  POST /inference_instruct2    - direct CosyVoice instruct2 streaming endpoint
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

LOGGER = logging.getLogger("cosyvoice_tts")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

Mode = Literal["zero_shot", "cross_lingual", "instruct2"]

app = FastAPI(title="CosyVoice 3.0 TTS Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_lock = threading.Lock()
_voices_dir = Path(os.getenv("VOICES_DIR", "/voices"))
_repo_dir = Path(os.getenv("COSYVOICE_REPO_DIR", "/opt/CosyVoice/repo"))
_model_dir = os.getenv("COSYVOICE_MODEL_DIR", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
_default_mode: Mode = os.getenv("COSYVOICE_DEFAULT_MODE", "zero_shot")  # type: ignore[assignment]
_default_speed = float(os.getenv("COSYVOICE_DEFAULT_SPEED", "1.0"))
_default_text_frontend = os.getenv("COSYVOICE_TEXT_FRONTEND", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_output_sample_rate = 24000


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    voice_id: str = "default"
    mode: Mode | None = None
    prompt_text: str | None = None
    instruct_text: str | None = None
    speed: float = Field(default=_default_speed, gt=0.0, le=3.0)
    text_frontend: bool | None = None


@dataclass(slots=True)
class SynthesisSpec:
    text: str
    mode: Mode
    prompt_audio: Any
    prompt_text: str
    instruct_text: str
    speed: float
    text_frontend: bool
    stream: bool


def _ensure_repo_imports() -> None:
    repo_path = str(_repo_dir)
    matcha_path = str(_repo_dir / "third_party" / "Matcha-TTS")
    if repo_path not in sys.path:
        sys.path.append(repo_path)
    if matcha_path not in sys.path:
        sys.path.append(matcha_path)


def _load_model():
    _ensure_repo_imports()
    from cosyvoice.cli.cosyvoice import AutoModel

    kwargs = {
        "model_dir": _model_dir,
        "load_trt": os.getenv("COSYVOICE_LOAD_TRT", "false").strip().lower() in {"1", "true", "yes", "on"},
        "load_vllm": os.getenv("COSYVOICE_LOAD_VLLM", "false").strip().lower() in {"1", "true", "yes", "on"},
        "fp16": os.getenv("COSYVOICE_LOAD_FP16", "false").strip().lower() in {"1", "true", "yes", "on"},
        "trt_concurrent": int(os.getenv("COSYVOICE_TRT_CONCURRENT", "1")),
    }
    LOGGER.info("Loading CosyVoice model_dir=%s", _model_dir)
    model = AutoModel(**kwargs)
    LOGGER.info("CosyVoice loaded sample_rate=%s", getattr(model, "sample_rate", "unknown"))
    return model


@app.on_event("startup")
async def startup() -> None:
    global _model, _output_sample_rate
    _voices_dir.mkdir(parents=True, exist_ok=True)
    _model = _load_model()
    _output_sample_rate = int(getattr(_model, "sample_rate", 24000))


@app.get("/health", response_model=None)
async def health() -> Any:
    if _model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {
        "status": "ok",
        "model_dir": _model_dir,
        "sample_rate": _output_sample_rate,
        "default_mode": _default_mode,
    }


def _voices_config_path() -> Path:
    return _voices_dir / "voices.json"


def _load_voices_config() -> dict[str, dict[str, Any]]:
    config_path = _voices_config_path()
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid voices.json: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="voices.json must contain an object")
    result: dict[str, dict[str, Any]] = {}
    for voice_id, meta in data.items():
        if isinstance(voice_id, str) and isinstance(meta, dict):
            result[voice_id] = meta
    return result


def _save_voices_config(config: dict[str, dict[str, Any]]) -> None:
    _voices_config_path().write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")


def _safe_voice_id(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in "-_")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid voice_id")
    return cleaned


def _resolve_voice_path(voice_id: str, meta: dict[str, Any]) -> Path:
    file_name = str(meta.get("file") or f"{voice_id}.wav")
    path = _voices_dir / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Voice audio not found for '{voice_id}'")
    return path


def _as_pcm16_bytes(tts_speech: Any) -> bytes:
    if hasattr(tts_speech, "detach"):
        audio_np = tts_speech.detach().cpu().float().numpy()
    else:
        audio_np = np.asarray(tts_speech, dtype=np.float32)
    audio_np = np.squeeze(audio_np)
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return (audio_np * 32767.0).astype(np.int16).tobytes()


def _validate_mode_inputs(mode: Mode, prompt_text: str, instruct_text: str) -> None:
    if mode == "zero_shot" and not prompt_text.strip():
        raise HTTPException(status_code=400, detail="prompt_text is required for zero_shot mode")
    if mode == "instruct2" and not instruct_text.strip():
        raise HTTPException(status_code=400, detail="instruct_text is required for instruct2 mode")


def _build_spec_from_voice_request(payload: SynthesizeRequest, *, stream: bool) -> SynthesisSpec:
    voices = _load_voices_config()
    voice_meta = voices.get(payload.voice_id)
    if voice_meta is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown voice_id '{payload.voice_id}'. Upload a prompt clip or use a direct inference endpoint.",
        )
    prompt_audio = str(_resolve_voice_path(payload.voice_id, voice_meta))
    mode = payload.mode or voice_meta.get("mode") or _default_mode
    if mode not in {"zero_shot", "cross_lingual", "instruct2"}:
        raise HTTPException(status_code=400, detail=f"Unsupported mode '{mode}'")
    prompt_text = payload.prompt_text if payload.prompt_text is not None else str(voice_meta.get("prompt_text", ""))
    instruct_text = (
        payload.instruct_text if payload.instruct_text is not None else str(voice_meta.get("instruct_text", ""))
    )
    _validate_mode_inputs(mode, prompt_text, instruct_text)
    return SynthesisSpec(
        text=payload.text.strip(),
        mode=mode,
        prompt_audio=prompt_audio,
        prompt_text=prompt_text,
        instruct_text=instruct_text,
        speed=payload.speed,
        text_frontend=_default_text_frontend if payload.text_frontend is None else payload.text_frontend,
        stream=stream,
    )


def _build_spec_for_upload(
    *,
    text: str,
    mode: Mode,
    prompt_audio: Any,
    prompt_text: str,
    instruct_text: str,
    speed: float,
    text_frontend: bool,
    stream: bool,
) -> SynthesisSpec:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    _validate_mode_inputs(mode, prompt_text, instruct_text)
    return SynthesisSpec(
        text=text,
        mode=mode,
        prompt_audio=prompt_audio,
        prompt_text=prompt_text,
        instruct_text=instruct_text,
        speed=speed,
        text_frontend=text_frontend,
        stream=stream,
    )


def _model_output_iter(spec: SynthesisSpec):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    with _model_lock:
        if spec.mode == "zero_shot":
            yield from _model.inference_zero_shot(
                spec.text,
                spec.prompt_text,
                spec.prompt_audio,
                stream=spec.stream,
                speed=spec.speed,
                text_frontend=spec.text_frontend,
            )
            return
        if spec.mode == "cross_lingual":
            yield from _model.inference_cross_lingual(
                spec.text,
                spec.prompt_audio,
                stream=spec.stream,
                speed=spec.speed,
                text_frontend=spec.text_frontend,
            )
            return
        if spec.mode == "instruct2":
            yield from _model.inference_instruct2(
                spec.text,
                spec.instruct_text,
                spec.prompt_audio,
                stream=spec.stream,
                speed=spec.speed,
                text_frontend=spec.text_frontend,
            )
            return
        raise HTTPException(status_code=400, detail=f"Unsupported mode '{spec.mode}'")


def _pcm_stream(spec: SynthesisSpec):
    try:
        for item in _model_output_iter(spec):
            tts_speech = item.get("tts_speech")
            if tts_speech is None:
                continue
            chunk = _as_pcm16_bytes(tts_speech)
            if chunk:
                yield chunk
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("CosyVoice synthesis failed")
        raise RuntimeError(f"Generation failed: {exc}") from exc


def _pcm_headers(mode: Mode) -> dict[str, str]:
    return {
        "X-Sample-Rate": str(_output_sample_rate),
        "X-Channels": "1",
        "X-Sample-Width": "2",
        "X-CosyVoice-Mode": mode,
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
    }


def _collect_pcm_bytes(spec: SynthesisSpec) -> bytes:
    return b"".join(_pcm_stream(spec))


@app.get("/voices")
async def list_voices() -> dict[str, list[dict[str, Any]]]:
    voices = _load_voices_config()
    result: list[dict[str, Any]] = []
    for voice_id, meta in voices.items():
        path = _voices_dir / str(meta.get("file", f"{voice_id}.wav"))
        result.append(
            {
                "id": voice_id,
                "description": meta.get("description", ""),
                "file": meta.get("file", f"{voice_id}.wav"),
                "available": path.exists(),
                "mode": meta.get("mode", _default_mode),
                "prompt_text": meta.get("prompt_text", ""),
                "instruct_text": meta.get("instruct_text", ""),
            }
        )
    return {"voices": result}


@app.post("/voices/upload")
async def upload_voice(
    voice_id: str = Form(...),
    description: str = Form(default=""),
    prompt_text: str = Form(default=""),
    mode: Mode = Form(default="zero_shot"),
    instruct_text: str = Form(default=""),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    safe_voice_id = _safe_voice_id(voice_id)
    suffix = Path(file.filename or "").suffix or ".wav"
    file_name = f"{safe_voice_id}{suffix}"
    destination = _voices_dir / file_name
    with destination.open("wb") as handle:
        handle.write(await file.read())

    voices = _load_voices_config()
    voices[safe_voice_id] = {
        "file": file_name,
        "description": description,
        "prompt_text": prompt_text,
        "mode": mode,
        "instruct_text": instruct_text,
    }
    _save_voices_config(voices)
    return {"status": "ok", "voice_id": safe_voice_id, "file": file_name}


@app.post("/synthesize")
async def synthesize(payload: SynthesizeRequest) -> Response:
    spec = _build_spec_from_voice_request(payload, stream=False)
    try:
        content = await run_in_threadpool(_collect_pcm_bytes, spec)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    return Response(content=content, media_type="application/octet-stream", headers=_pcm_headers(spec.mode))


@app.post("/synthesize/stream")
async def synthesize_stream(payload: SynthesizeRequest) -> StreamingResponse:
    spec = _build_spec_from_voice_request(payload, stream=True)
    return StreamingResponse(_pcm_stream(spec), media_type="application/octet-stream", headers=_pcm_headers(spec.mode))


def _load_prompt_audio(upload: UploadFile):
    _ensure_repo_imports()
    from cosyvoice.utils.file_utils import load_wav

    return load_wav(upload.file, 16000)


@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_wav: UploadFile = File(...),
    speed: float = Form(default=_default_speed),
    text_frontend: bool = Form(default=_default_text_frontend),
) -> StreamingResponse:
    prompt_audio = await run_in_threadpool(_load_prompt_audio, prompt_wav)
    spec = _build_spec_for_upload(
        text=tts_text,
        mode="zero_shot",
        prompt_audio=prompt_audio,
        prompt_text=prompt_text,
        instruct_text="",
        speed=speed,
        text_frontend=text_frontend,
        stream=True,
    )
    return StreamingResponse(_pcm_stream(spec), media_type="application/octet-stream", headers=_pcm_headers(spec.mode))


@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Form(...),
    prompt_wav: UploadFile = File(...),
    speed: float = Form(default=_default_speed),
    text_frontend: bool = Form(default=_default_text_frontend),
) -> StreamingResponse:
    prompt_audio = await run_in_threadpool(_load_prompt_audio, prompt_wav)
    spec = _build_spec_for_upload(
        text=tts_text,
        mode="cross_lingual",
        prompt_audio=prompt_audio,
        prompt_text="",
        instruct_text="",
        speed=speed,
        text_frontend=text_frontend,
        stream=True,
    )
    return StreamingResponse(_pcm_stream(spec), media_type="application/octet-stream", headers=_pcm_headers(spec.mode))


@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(...),
    instruct_text: str = Form(...),
    prompt_wav: UploadFile = File(...),
    speed: float = Form(default=_default_speed),
    text_frontend: bool = Form(default=_default_text_frontend),
) -> StreamingResponse:
    prompt_audio = await run_in_threadpool(_load_prompt_audio, prompt_wav)
    spec = _build_spec_for_upload(
        text=tts_text,
        mode="instruct2",
        prompt_audio=prompt_audio,
        prompt_text="",
        instruct_text=instruct_text,
        speed=speed,
        text_frontend=text_frontend,
        stream=True,
    )
    return StreamingResponse(_pcm_stream(spec), media_type="application/octet-stream", headers=_pcm_headers(spec.mode))


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_request: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=500)


if __name__ == "__main__":
    port = int(os.getenv("LISTEN_PORT", "4126"))
    uvicorn.run(app, host="0.0.0.0", port=port)
