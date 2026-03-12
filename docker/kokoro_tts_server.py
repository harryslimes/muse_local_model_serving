#!/usr/bin/env python3
"""Kokoro — FastAPI TTS server.

Exposes:
  POST /synthesize  — text → PCM16 24kHz mono audio
  GET  /voices      — list available Kokoro voices
  GET  /health      — health check

Kokoro is a lightweight 82M-parameter TTS model using ONNX.
Supports American English, British English, Spanish, French, Hindi,
Italian, Japanese, Brazilian Portuguese, and Mandarin Chinese.
"""

import io
import os

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="Kokoro TTS Server")

_pipeline = None
_output_sample_rate = 24000

# "cpu" forces CPU-only mode; "cuda" or unset uses GPU with CPU fallback
_device = os.getenv("KOKORO_DEVICE", "").lower() or None  # None = auto (GPU)

# Kokoro voice catalog — maps voice_id to (lang_code, kokoro_voice_name)
# Full list: https://github.com/hexgrad/kokoro
KOKORO_VOICES = {
    # American English
    "af_heart": ("a", "af_heart", "Heart (American, Female)"),
    "af_alloy": ("a", "af_alloy", "Alloy (American, Female)"),
    "af_aoede": ("a", "af_aoede", "Aoede (American, Female)"),
    "af_bella": ("a", "af_bella", "Bella (American, Female)"),
    "af_jessica": ("a", "af_jessica", "Jessica (American, Female)"),
    "af_kore": ("a", "af_kore", "Kore (American, Female)"),
    "af_nicole": ("a", "af_nicole", "Nicole (American, Female)"),
    "af_nova": ("a", "af_nova", "Nova (American, Female)"),
    "af_river": ("a", "af_river", "River (American, Female)"),
    "af_sarah": ("a", "af_sarah", "Sarah (American, Female)"),
    "af_sky": ("a", "af_sky", "Sky (American, Female)"),
    "am_adam": ("a", "am_adam", "Adam (American, Male)"),
    "am_echo": ("a", "am_echo", "Echo (American, Male)"),
    "am_eric": ("a", "am_eric", "Eric (American, Male)"),
    "am_liam": ("a", "am_liam", "Liam (American, Male)"),
    "am_michael": ("a", "am_michael", "Michael (American, Male)"),
    "am_onyx": ("a", "am_onyx", "Onyx (American, Male)"),
    # British English
    "bf_emma": ("b", "bf_emma", "Emma (British, Female)"),
    "bf_isabella": ("b", "bf_isabella", "Isabella (British, Female)"),
    "bm_george": ("b", "bm_george", "George (British, Male)"),
    "bm_lewis": ("b", "bm_lewis", "Lewis (British, Male)"),
    "bm_fable": ("b", "bm_fable", "Fable (British, Male)"),
    # Spanish
    "ef_dora": ("e", "ef_dora", "Dora (Spanish, Female)"),
    "em_alex": ("e", "em_alex", "Alex (Spanish, Male)"),
    "em_santa": ("e", "em_santa", "Santa (Spanish, Male)"),
    # French
    "ff_siwis": ("f", "ff_siwis", "Siwis (French, Female)"),
    # Hindi
    "hf_alpha": ("h", "hf_alpha", "Alpha (Hindi, Female)"),
    "hm_omega": ("h", "hm_omega", "Omega (Hindi, Male)"),
    # Italian
    "if_sara": ("i", "if_sara", "Sara (Italian, Female)"),
    "im_nicola": ("i", "im_nicola", "Nicola (Italian, Male)"),
    # Japanese
    "jf_alpha": ("j", "jf_alpha", "Alpha (Japanese, Female)"),
    "jf_gongitsune": ("j", "jf_gongitsune", "Gongitsune (Japanese, Female)"),
    "jm_kumo": ("j", "jm_kumo", "Kumo (Japanese, Male)"),
    # Brazilian Portuguese
    "pf_dora": ("p", "pf_dora", "Dora (Portuguese BR, Female)"),
    "pm_alex": ("p", "pm_alex", "Alex (Portuguese BR, Male)"),
    "pm_santa": ("p", "pm_santa", "Santa (Portuguese BR, Male)"),
    # Mandarin Chinese
    "zf_xiaobei": ("z", "zf_xiaobei", "Xiaobei (Mandarin, Female)"),
    "zf_xiaoni": ("z", "zf_xiaoni", "Xiaoni (Mandarin, Female)"),
    "zf_xiaoxiao": ("z", "zf_xiaoxiao", "Xiaoxiao (Mandarin, Female)"),
    "zf_xiaoyi": ("z", "zf_xiaoyi", "Xiaoyi (Mandarin, Female)"),
    "zm_yunjian": ("z", "zm_yunjian", "Yunjian (Mandarin, Male)"),
    "zm_yunxi": ("z", "zm_yunxi", "Yunxi (Mandarin, Male)"),
    "zm_yunxia": ("z", "zm_yunxia", "Yunxia (Mandarin, Male)"),
    "zm_yunyang": ("z", "zm_yunyang", "Yunyang (Mandarin, Male)"),
}


def _load_pipeline():
    """Load Kokoro pipeline on the configured device."""
    from kokoro import KPipeline

    default_lang = os.getenv("KOKORO_DEFAULT_LANG", "a")
    kwargs = {"lang_code": default_lang}
    if _device == "cpu":
        kwargs["device"] = "cpu"
    print(f"Loading Kokoro pipeline (lang={default_lang}, device={_device or 'auto'})...", flush=True)
    pipeline = KPipeline(**kwargs)
    print("Kokoro pipeline loaded successfully", flush=True)
    return pipeline


# Cache pipelines by language code and device
_pipelines: dict[str, object] = {}
_cpu_pipelines: dict[str, object] = {}


def _get_pipeline(lang_code: str):
    """Get or create a pipeline for the given language code on the configured device."""
    global _pipelines
    if lang_code not in _pipelines:
        from kokoro import KPipeline
        kwargs = {"lang_code": lang_code}
        if _device == "cpu":
            kwargs["device"] = "cpu"
        print(f"Loading Kokoro pipeline for lang={lang_code} (device={_device or 'auto'})...", flush=True)
        _pipelines[lang_code] = KPipeline(**kwargs)
        print(f"Kokoro pipeline for lang={lang_code} loaded", flush=True)
    return _pipelines[lang_code]


def _get_cpu_pipeline(lang_code: str):
    """Get or create a CPU fallback pipeline for the given language code."""
    global _cpu_pipelines
    if lang_code not in _cpu_pipelines:
        from kokoro import KPipeline
        print(f"Loading Kokoro CPU fallback pipeline for lang={lang_code}...", flush=True)
        _cpu_pipelines[lang_code] = KPipeline(lang_code=lang_code, device="cpu")
        print(f"Kokoro CPU fallback pipeline for lang={lang_code} loaded", flush=True)
    return _cpu_pipelines[lang_code]


@app.on_event("startup")
async def startup():
    global _pipeline
    _pipeline = _load_pipeline()
    _pipelines["a"] = _pipeline


@app.get("/health")
async def health():
    if _pipeline is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.get("/voices")
async def list_voices():
    """List available Kokoro voices."""
    result = []
    for voice_id, (lang_code, kokoro_name, description) in KOKORO_VOICES.items():
        result.append({
            "id": voice_id,
            "lang_code": lang_code,
            "description": description,
            "available": True,
        })
    return {"voices": result}


@app.post("/synthesize")
async def synthesize(request: Request):
    """Synthesize text to speech.

    JSON body:
      {"text": "Hello world", "voice_id": "af_heart", "speed": 1.0}

    Returns: raw PCM16 24kHz mono audio as application/octet-stream.
    """
    body = await request.json()
    text = body.get("text", "")
    voice_id = body.get("voice_id", "af_heart")
    speed = float(body.get("speed", 1.0))

    if not text.strip():
        return JSONResponse({"error": "Empty text"}, status_code=400)

    # Resolve voice
    if voice_id in KOKORO_VOICES:
        lang_code, kokoro_voice, _ = KOKORO_VOICES[voice_id]
    else:
        # Default to American English heart voice
        lang_code, kokoro_voice = "a", "af_heart"

    def _synthesize(pipe):
        chunks = []
        for _graphemes, _phonemes, audio in pipe(text, voice=kokoro_voice, speed=speed):
            if audio is not None:
                chunks.append(audio)
        return chunks

    try:
        audio_chunks = _synthesize(_get_pipeline(lang_code))
    except Exception as e:
        if _device == "cpu":
            # Already on CPU — no fallback available
            return JSONResponse({"error": f"Generation failed: {e}"}, status_code=500)
        # GPU errors (cuFFT, OOM) — fall back to CPU
        print(f"GPU synthesis failed ({e}), retrying on CPU...", flush=True)
        try:
            audio_chunks = _synthesize(_get_cpu_pipeline(lang_code))
        except Exception as e2:
            return JSONResponse({"error": f"Generation failed: {e2}"}, status_code=500)

    if not audio_chunks:
        return JSONResponse({"error": "No audio generated"}, status_code=500)

    audio_np = np.concatenate(audio_chunks)

    # Normalize to [-1, 1] range if needed
    peak = np.abs(audio_np).max()
    if peak > 0:
        audio_np = audio_np / peak * 0.95

    pcm16 = (audio_np * 32767).astype(np.int16)

    return Response(
        content=pcm16.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(_output_sample_rate),
            "X-Channels": "1",
            "X-Sample-Width": "2",
        },
    )


if __name__ == "__main__":
    port = int(os.getenv("LISTEN_PORT", "4125"))
    uvicorn.run(app, host="0.0.0.0", port=port)
