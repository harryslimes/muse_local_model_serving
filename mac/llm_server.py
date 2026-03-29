#!/usr/bin/env python3
"""Qwen3.5-4B MLX — OpenAI-compatible LLM server (Apple Silicon).

Wraps mlx_lm.server with a thin management layer.
Exposes the standard mlx_lm OpenAI-compatible endpoints:
  GET  /v1/models
  POST /v1/chat/completions
  GET  /health  (added by this wrapper)

Usage:
  python llm_server.py
  # or directly:
  python -m mlx_lm.server --model mlx-community/Qwen3.5-4B-MLX-4bit --port 12434

Requirements:
  pip install mlx-lm fastapi uvicorn
"""

import os
import subprocess
import sys
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MODEL_ID = os.getenv("MLX_LLM_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit")
UPSTREAM_PORT = int(os.getenv("MLX_LLM_UPSTREAM_PORT", "12435"))
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "12434"))

app = FastAPI(title="MLX LLM Server (Qwen3.5-4B)")
_upstream_proc: subprocess.Popen | None = None
_client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{UPSTREAM_PORT}", timeout=120.0)


def _start_upstream():
    global _upstream_proc
    cmd = [
        sys.executable, "-m", "mlx_lm.server",
        "--model", MODEL_ID,
        "--port", str(UPSTREAM_PORT),
        "--host", "127.0.0.1",
    ]
    print(f"Starting mlx_lm.server: {' '.join(cmd)}", flush=True)
    _upstream_proc = subprocess.Popen(cmd)

    # Wait for upstream to be ready
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            import httpx as _httpx
            r = _httpx.get(f"http://127.0.0.1:{UPSTREAM_PORT}/v1/models", timeout=2)
            if r.status_code < 500:
                print("mlx_lm.server ready", flush=True)
                return
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError("mlx_lm.server did not become ready in time")


@app.on_event("startup")
async def startup():
    _start_upstream()


@app.on_event("shutdown")
async def shutdown():
    if _upstream_proc:
        _upstream_proc.terminate()
    await _client.aclose()


@app.get("/health")
async def health():
    try:
        r = await _client.get("/v1/models")
        if r.status_code < 500:
            return {"status": "ok", "model": MODEL_ID}
    except Exception:
        pass
    return JSONResponse({"status": "loading"}, status_code=503)


async def _proxy(request: Request, path: str):
    body = await request.body()
    upstream_url = f"/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    rsp = await _client.request(
        method=request.method,
        url=upstream_url,
        headers=headers,
        content=body,
    )

    if "text/event-stream" in rsp.headers.get("content-type", ""):
        return StreamingResponse(
            rsp.aiter_bytes(),
            status_code=rsp.status_code,
            media_type="text/event-stream",
        )
    return JSONResponse(rsp.json(), status_code=rsp.status_code)


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_v1(path: str, request: Request):
    return await _proxy(request, f"v1/{path}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
