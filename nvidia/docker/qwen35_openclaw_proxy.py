#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI(title="Qwen3.5 OpenClaw-style Proxy")

UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "http://qwen35-35b-a3b-llama:8080").rstrip("/")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
PROXY_MODEL_NAME = os.getenv("PROXY_MODEL_NAME", "Qwen3.5-35B-A3B")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))
FORCE_MODEL_NAME = os.getenv("FORCE_MODEL_NAME", "true").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_THINKING_DEFAULT = os.getenv("ENABLE_THINKING_DEFAULT", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _flatten_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts: list[str] = []
        for block in value:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        texts.append(text)
            elif isinstance(block, str):
                texts.append(block)
        if texts:
            return "\n".join(texts)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _normalize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        msg = dict(item)
        role = str(msg.get("role", "user"))
        content = msg.get("content")

        if role == "developer":
            role = "system"
        elif role == "tool":
            role = "user"
            content = f"[tool result]\n{_flatten_content(content)}"
        elif role not in {"system", "user", "assistant"}:
            role = "user"

        msg["role"] = role
        msg["content"] = _flatten_content(content)
        normalized.append(msg)
    return normalized


def _process_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    processed = dict(body)
    messages = processed.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="'messages' must be an array.")

    processed["messages"] = _normalize_messages(messages)

    requested_model = processed.get("model")
    if FORCE_MODEL_NAME or not requested_model:
        processed["model"] = PROXY_MODEL_NAME

    template_kwargs = processed.get("chat_template_kwargs")
    if not isinstance(template_kwargs, dict):
        template_kwargs = {}
    template_kwargs.setdefault("enable_thinking", ENABLE_THINKING_DEFAULT)
    processed["chat_template_kwargs"] = template_kwargs
    return processed


def _forward_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in {"host", "content-length"}:
            continue
        headers[key] = value
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    return headers


def _patch_json_response(path: str, payload: Any) -> Any:
    if not FORCE_MODEL_NAME:
        return payload
    if path == "models" and isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for model_info in data:
                if isinstance(model_info, dict) and "id" in model_info:
                    model_info["id"] = PROXY_MODEL_NAME
    if isinstance(payload, dict) and "model" in payload:
        payload["model"] = PROXY_MODEL_NAME
    return payload


@app.get("/healthz")
def healthz() -> dict[str, str]:
    url = f"{UPSTREAM_BASE_URL}/v1/models"
    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Upstream unreachable: {exc}") from exc

    if response.status_code >= 500:
        raise HTTPException(status_code=503, detail=f"Upstream error status={response.status_code}")
    return {"status": "ok"}


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_v1(path: str, request: Request) -> Response:
    method = request.method.upper()
    url = f"{UPSTREAM_BASE_URL}/v1/{path}"
    headers = _forward_headers(request)

    body: dict[str, Any] | None = None
    stream = False
    if method == "POST":
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

        if path == "chat/completions":
            body = _process_chat_request(body)
            stream = bool(body.get("stream", False))

    try:
        if stream:
            upstream = requests.request(
                method,
                url,
                headers=headers,
                json=body,
                timeout=REQUEST_TIMEOUT_SECONDS,
                stream=True,
            )

            def iter_chunks():
                try:
                    # Forward upstream bytes as soon as they arrive; large chunk sizes
                    # can visibly "de-stream" SSE in downstream clients.
                    for chunk in upstream.iter_content(chunk_size=None):
                        if chunk:
                            yield chunk
                finally:
                    upstream.close()

            content_type = upstream.headers.get("content-type", "text/event-stream")
            return StreamingResponse(
                iter_chunks(),
                status_code=upstream.status_code,
                media_type=content_type,
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        upstream = requests.request(
            method,
            url,
            headers=headers,
            json=body,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    content_type = upstream.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            payload = upstream.json()
            payload = _patch_json_response(path, payload)
            return JSONResponse(status_code=upstream.status_code, content=payload)
        except Exception:
            pass

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=content_type or None,
    )
