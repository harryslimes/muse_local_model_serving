#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT_DIR/docker-compose.atlas-qwen35-35b-a3b-nvfp4.yml}"
SERVICE_NAME="${SERVICE_NAME:-atlas-qwen35-35b-a3b-nvfp4}"
BASE_URL="${BASE_URL:-http://127.0.0.1:18888/v1}"
MODEL="${MODEL:-Kbenkhaled/Qwen3.5-35B-A3B-NVFP4}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-slai}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-1024}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-180}"
CONTAINER_NAME="${CONTAINER_NAME:-muse_local_model_serving-atlas-qwen35-35b-a3b-nvfp4-1}"
PROMPT_REPEAT="${PROMPT_REPEAT:-650}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Missing required command: docker"
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "Missing required command: curl"
  exit 1
fi

tmp_payload="$(mktemp)"
tmp_resp1="$(mktemp)"
tmp_resp2="$(mktemp)"
trap 'rm -f "$tmp_payload" "$tmp_resp1" "$tmp_resp2"' EXIT

echo "Restarting Atlas with policy=${SCHEDULING_POLICY} max_prefill_tokens=${MAX_PREFILL_TOKENS}..."
(
  cd "$ROOT_DIR"
  ATLAS_SCHEDULING_POLICY="$SCHEDULING_POLICY" \
  ATLAS_MAX_PREFILL_TOKENS="$MAX_PREFILL_TOKENS" \
    docker compose -f "$COMPOSE_FILE" up -d "$SERVICE_NAME" >/dev/null
)

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if curl -fsS "${BASE_URL%/}/models" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl -fsS "${BASE_URL%/}/models" >/dev/null 2>&1; then
  echo "Atlas did not become ready within ${READY_TIMEOUT_SECONDS}s"
  exit 1
fi

python3 - <<'PY' >"$tmp_payload"
import json
import os

repeat = int(os.environ.get("PROMPT_REPEAT", "650"))
long_text = " ".join(["muse cache probe context"] * repeat)
payload = {
    "model": os.environ.get("MODEL", "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4"),
    "messages": [
        {"role": "system", "content": "Reply with one short sentence."},
        {
            "role": "user",
            "content": (
                "Use this context and then say quick probe ok.\n"
                f"{long_text}"
            ),
        },
    ],
    "temperature": 0,
    "max_tokens": 16,
}
print(json.dumps(payload))
PY

echo "Running probe requests..."
curl -sS "${BASE_URL%/}/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "@$tmp_payload" >"$tmp_resp1"
curl -sS "${BASE_URL%/}/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "@$tmp_payload" >"$tmp_resp2"

python3 - <<'PY' "$tmp_resp1" "$tmp_resp2"
import json
import sys

def read(path):
    data = json.load(open(path))
    usage = data.get("usage", {})
    return {
        "ttft_ms": usage.get("time_to_first_token_ms"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }

r1 = read(sys.argv[1])
r2 = read(sys.argv[2])
print("run1", r1)
print("run2", r2)
PY

echo
echo "Recent Atlas log lines:"
docker logs --tail 80 "$CONTAINER_NAME" 2>&1 | grep -E "Prefix cache hit|Chunked prefill start|Done:|TTFT=" | tail -n 12 || true
