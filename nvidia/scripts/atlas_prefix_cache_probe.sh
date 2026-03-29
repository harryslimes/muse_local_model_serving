#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:18888/v1}"
MODEL="${MODEL:-Kbenkhaled/Qwen3.5-35B-A3B-NVFP4}"
CONTAINER_NAME="${CONTAINER_NAME:-muse_local_model_serving-atlas-qwen35-35b-a3b-nvfp4-1}"

if ! command -v curl >/dev/null 2>&1; then
  echo "Missing required command: curl"
  exit 1
fi

tmp_payload="$(mktemp)"
tmp_resp1="$(mktemp)"
tmp_resp2="$(mktemp)"
trap 'rm -f "$tmp_payload" "$tmp_resp1" "$tmp_resp2"' EXIT

python3 - <<'PY' >"$tmp_payload"
import json

long_text = " ".join(["cache probe prefix text"] * 1200)
payload = {
    "model": "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4",
    "messages": [
        {"role": "system", "content": "Reply with one short sentence."},
        {
            "role": "user",
            "content": (
                "Use this context and then say cache probe ok.\n"
                f"{long_text}"
            ),
        },
    ],
    "temperature": 0,
    "max_tokens": 16,
}
print(json.dumps(payload))
PY

# Inject model override if requested.
python3 - <<'PY' "$tmp_payload" "$MODEL"
import json
import sys

path, model = sys.argv[1], sys.argv[2]
data = json.load(open(path))
data["model"] = model
open(path, "w").write(json.dumps(data))
PY

echo "Calling Atlas twice with the same long prefix..."
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
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return {
        "ttft_ms": usage.get("time_to_first_token_ms"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "text": text[:120],
    }

r1 = read(sys.argv[1])
r2 = read(sys.argv[2])
print("Run 1:", r1)
print("Run 2:", r2)
PY

echo
echo "Recent Atlas cache-hit log lines:"
docker logs --tail 200 "$CONTAINER_NAME" 2>&1 | grep -E "Prefix cache hit|Chunked prefill start|TTFT=" | tail -n 20 || true

echo
echo "If you see 'Prefix cache hit: ... reused', prefix caching is active."
