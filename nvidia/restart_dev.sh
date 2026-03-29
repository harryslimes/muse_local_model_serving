#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_MODEL_SERVING_DIR="$ROOT_DIR"
ENV_FILE="$ROOT_DIR/.env"

# Read MUSE_BACKEND_DIR / MUSE_SVELTE_DIR from .env; resolve relative paths.
_read_env_path() {
  local key="$1" default="$2"
  local val=""
  if [[ -f "$ENV_FILE" ]]; then
    val="$(grep -E "^${key}=" "$ENV_FILE" | tail -n 1 | cut -d'=' -f2- || true)"
    val="$(echo "$val" | sed 's/^ *//;s/ *$//')"
    val="${val%\"}" ; val="${val#\"}" ; val="${val%\'}" ; val="${val#\'}"
  fi
  val="${val:-$default}"
  if [[ "$val" != /* ]]; then
    val="$ROOT_DIR/$val"
  fi
  echo "$val"
}

BACKEND_DIR="$(_read_env_path MUSE_BACKEND_DIR ../../muse_backend)"
FRONTEND_DIR="$(_read_env_path MUSE_SVELTE_DIR ../../muse_svelte)"
LOCAL_IMAGE_SERVER_SCRIPT="$LOCAL_MODEL_SERVING_DIR/scripts/flux2_klein_server.sh"
LOCAL_LLM_SERVER_SCRIPT="$LOCAL_MODEL_SERVING_DIR/scripts/qwen35_35b_a3b_server.sh"
RUN_DIR="$ROOT_DIR/.run"
LEGACY_LOCAL_MODEL_PROJECT_NAME="muse_local_model_serving"

BACKEND_PID_FILE="$RUN_DIR/backend.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
BACKEND_LOG="$RUN_DIR/backend.log"
FRONTEND_LOG="$RUN_DIR/frontend.log"

mkdir -p "$RUN_DIR"

reset_db="false"
start_local_image_server_override=""
start_local_llm_server_override=""
start_voice_server_override=""
start_voice_stt_override=""
start_voice_tts_override=""
start_backend_override=""
start_frontend_override=""
start_tool_server_override=""

print_usage() {
  cat <<'EOF'
Usage:
  ./restart_dev.sh [options]

Which services start is controlled by ENABLE_* flags in .env:
  ENABLE_BACKEND=true/false     Start backend (Postgres + uvicorn). Default: true
  ENABLE_FRONTEND=true/false    Start frontend (npm dev server). Default: true
  ENABLE_IMAGE_SERVER=true/false  Start local image gen server. Default: false
  ENABLE_LLM_SERVER=true/false  Start local LLM server. Default: false
  ENABLE_VOICE_STT=true/false   Start Parakeet STT. Default: false
  ENABLE_VOICE_TTS=true/false   Start Kokoro + Chatterbox TTS. Default: false
  ENABLE_TOOL_SERVER=true/false Start tool server (SearXNG + crawl). Default: false

CLI flags override .env settings:
  --reset-db                    Reset backend DB (prompts for CONFIRM).
  --with-backend / --without-backend
  --with-frontend / --without-frontend
  --with-local-image-server / --without-local-image-server
  --with-local-llm-server / --without-local-llm-server
  --with-voice-server / --without-voice-server   (STT + TTS together)
  --with-voice-stt / --without-voice-stt
  --with-voice-tts / --without-voice-tts
  --with-tool-server / --without-tool-server
  -h, --help                    Show this help.

Voice-only example (.env):
  ENABLE_BACKEND=false
  ENABLE_FRONTEND=false
  ENABLE_VOICE_STT=true
  ENABLE_VOICE_TTS=true
  KOKORO_PROVIDER=cpu
EOF
}

is_truthy() {
  local value="${1:-}"
  value="$(echo "$value" | tr '[:upper:]' '[:lower:]')"
  [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "on" ]]
}

resolve_env_bool() {
  local env_file="$1"
  local key="$2"
  if [[ ! -f "$env_file" ]]; then
    return
  fi
  grep -E "^${key}=" "$env_file" | tail -n 1 | cut -d'=' -f2- | tr -d '[:space:]' || true
}

resolve_env_value() {
  local env_file="$1"
  local key="$2"
  if [[ ! -f "$env_file" ]]; then
    return
  fi
  grep -E "^${key}=" "$env_file" | tail -n 1 | cut -d'=' -f2- || true
}

trim_env_value() {
  local value="${1:-}"
  # Trim whitespace and optional single/double wrapping quotes.
  value="$(echo "$value" | sed 's/^ *//;s/ *$//')"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  echo "$value"
}

port_from_url() {
  local url="${1:-}"
  local port=""
  port="$(echo "$url" | sed -nE 's|^[a-zA-Z][a-zA-Z0-9+.-]*://[^/:]+:([0-9]+)(/.*)?$|\1|p')"
  if [[ -n "$port" ]]; then
    echo "$port"
    return
  fi
  if [[ "$url" =~ ^https:// ]]; then
    echo "443"
    return
  fi
  if [[ "$url" =~ ^http:// ]]; then
    echo "80"
    return
  fi
  echo ""
}

is_local_host_url() {
  local url="${1:-}"
  [[ "$url" =~ ^https?://(127\.0\.0\.1|localhost|0\.0\.0\.0)(:[0-9]+)?(/.*)?$ ]]
}

kill_from_pid_file() {
  local pid_file="$1"
  local label="$2"

  if [[ ! -f "$pid_file" ]]; then
    return
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  rm -f "$pid_file"

  if [[ -z "${pid:-}" ]]; then
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $label (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
}

kill_listeners_on_port() {
  local port="$1"
  local label="$2"

  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      echo "Killing process(es) on port $port for $label: $pids"
      kill $pids 2>/dev/null || true
      sleep 1
      kill -9 $pids 2>/dev/null || true
    fi
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
  fi
}

wait_for_db_health() {
  local max_tries=60
  local attempt=1
  local status=""

  while (( attempt <= max_tries )); do
    status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' muse-db 2>/dev/null || true)"
    if [[ "$status" == "healthy" ]]; then
      return
    fi
    sleep 1
    attempt=$((attempt + 1))
  done

  echo "Database did not become healthy in time (last status: $status)."
  exit 1
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local max_tries="${3:-40}"
  local attempt=1
  local code=""

  while (( attempt <= max_tries )); do
    code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 10 "$url" || true)"
    if [[ "${code:0:1}" == "2" || "${code:0:1}" == "3" ]]; then
      return 0
    fi
    sleep 1
    attempt=$((attempt + 1))
  done

  echo "$label did not become ready at $url (last status: ${code:-none})."
  return 1
}

pid_for_listening_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -n 1 || true
  fi
}

cleanup_legacy_local_model_project() {
  local legacy_project="$LEGACY_LOCAL_MODEL_PROJECT_NAME"
  local compose_file=""

  if [[ -z "$legacy_project" || ! -d "$LOCAL_MODEL_SERVING_DIR" ]]; then
    return
  fi

  for compose_file in \
    "$LOCAL_MODEL_SERVING_DIR/docker-compose.qwen35-35b-a3b.yml" \
    "$LOCAL_MODEL_SERVING_DIR/docker-compose.flux2.yml" \
    "$LOCAL_MODEL_SERVING_DIR/docker-compose.flux2-klein-9b-gguf.yml" \
    "$LOCAL_MODEL_SERVING_DIR/docker-compose.atlas-qwen35-35b-a3b-nvfp4.yml" \
    "$LOCAL_MODEL_SERVING_DIR/docker-compose.qwen35-35b-a3b-gptq-vllm.yml"
  do
    if [[ -f "$compose_file" ]]; then
      docker compose -p "$legacy_project" -f "$compose_file" down --remove-orphans >/dev/null 2>&1 || true
    fi
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset-db)
      reset_db="true"
      shift
      ;;
    --with-local-image-server)
      start_local_image_server_override="true"
      shift
      ;;
    --without-local-image-server)
      start_local_image_server_override="false"
      shift
      ;;
    --with-local-llm-server)
      start_local_llm_server_override="true"
      shift
      ;;
    --without-local-llm-server)
      start_local_llm_server_override="false"
      shift
      ;;
    --with-voice-server)
      start_voice_server_override="true"
      shift
      ;;
    --without-voice-server)
      start_voice_server_override="false"
      shift
      ;;
    --with-voice-stt)
      start_voice_stt_override="true"
      shift
      ;;
    --without-voice-stt)
      start_voice_stt_override="false"
      shift
      ;;
    --with-voice-tts)
      start_voice_tts_override="true"
      shift
      ;;
    --without-voice-tts)
      start_voice_tts_override="false"
      shift
      ;;
    --with-tool-server)
      start_tool_server_override="true"
      shift
      ;;
    --without-tool-server)
      start_tool_server_override="false"
      shift
      ;;
    --with-backend)
      start_backend_override="true"
      shift
      ;;
    --without-backend)
      start_backend_override="false"
      shift
      ;;
    --with-frontend)
      start_frontend_override="true"
      shift
      ;;
    --without-frontend)
      start_frontend_override="false"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Read ENABLE_* flags from local .env — these control which services start.
# CLI flags override these values.
# ---------------------------------------------------------------------------
_enable_flag() {
  local key="$1" default="$2" override_var="$3"
  # CLI override takes priority
  if [[ -n "${!override_var:-}" ]]; then
    echo "${!override_var}"
    return
  fi
  # Read from local .env
  local val=""
  val="$(resolve_env_bool "$ENV_FILE" "$key")"
  if [[ -n "${val:-}" ]]; then
    if is_truthy "$val"; then echo "true"; else echo "false"; fi
    return
  fi
  echo "$default"
}

enable_backend="$(_enable_flag ENABLE_BACKEND true start_backend_override)"
enable_frontend="$(_enable_flag ENABLE_FRONTEND true start_frontend_override)"
enable_image_server="$(_enable_flag ENABLE_IMAGE_SERVER false start_local_image_server_override)"
enable_llm_server="$(_enable_flag ENABLE_LLM_SERVER false start_local_llm_server_override)"
enable_tool_server="$(_enable_flag ENABLE_TOOL_SERVER false start_tool_server_override)"

# Voice: --with-voice-server sets both STT+TTS, individual flags override.
_voice_stt_default="false"
_voice_tts_default="false"
if [[ "${start_voice_server_override:-}" == "true" ]]; then
  _voice_stt_default="true"
  _voice_tts_default="true"
elif [[ "${start_voice_server_override:-}" == "false" ]]; then
  _voice_stt_default="false"
  _voice_tts_default="false"
else
  _voice_stt_default="$(_enable_flag ENABLE_VOICE_STT false start_voice_stt_override)"
  _voice_tts_default="$(_enable_flag ENABLE_VOICE_TTS false start_voice_tts_override)"
fi
# Individual CLI overrides still take precedence over --with-voice-server.
if [[ -n "${start_voice_stt_override:-}" ]]; then
  enable_voice_stt="${start_voice_stt_override}"
else
  enable_voice_stt="${_voice_stt_default}"
fi
if [[ -n "${start_voice_tts_override:-}" ]]; then
  enable_voice_tts="${start_voice_tts_override}"
else
  enable_voice_tts="${_voice_tts_default}"
fi

echo "Service plan:"
echo "  Backend:      ${enable_backend}"
echo "  Frontend:     ${enable_frontend}"
echo "  Image server: ${enable_image_server}"
echo "  LLM server:   ${enable_llm_server}"
echo "  Voice STT:    ${enable_voice_stt}"
echo "  Voice TTS:    ${enable_voice_tts}"
echo "  Tool server:  ${enable_tool_server}"
echo ""

# ---------------------------------------------------------------------------
# Helper: read config value from local .env first, then backend .env fallback.
# ---------------------------------------------------------------------------
resolve_config_value() {
  local key="$1"
  local val=""
  # Try local .env first
  val="$(resolve_env_value "$ENV_FILE" "$key")"
  val="$(trim_env_value "${val:-}")"
  if [[ -n "${val:-}" ]]; then
    echo "$val"
    return
  fi
  # Fallback to backend .env if backend dir exists
  if [[ -f "$BACKEND_DIR/.env" ]]; then
    val="$(resolve_env_value "$BACKEND_DIR/.env" "$key")"
    val="$(trim_env_value "${val:-}")"
  fi
  echo "${val:-}"
}

if [[ "$reset_db" == "true" ]] && [[ "$enable_backend" != "true" ]]; then
  echo "Warning: --reset-db ignored because backend is disabled."
  reset_db="false"
fi

if [[ "$reset_db" == "true" ]]; then
  if [[ ! -x "$BACKEND_DIR/reset_db.sh" ]]; then
    echo "Database reset script not found or not executable: $BACKEND_DIR/reset_db.sh"
    exit 1
  fi

  read -r -p "Type CONFIRM to continue: " reset_confirmation
  if [[ "$reset_confirmation" != "CONFIRM" ]]; then
    echo "Aborted. Confirmation did not match."
    exit 1
  fi

  echo "Starting Postgres for DB reset..."
  (
    cd "$BACKEND_DIR"
    docker compose up -d db
  )
  wait_for_db_health

  echo "Resetting backend database..."
  (
    cd "$BACKEND_DIR"
    ./reset_db.sh <<< "CONFIRM"
  )
fi

# ---------------------------------------------------------------------------
# Image server + LLM server config (skip entirely when neither is enabled)
# ---------------------------------------------------------------------------
start_local_image_server="false"
start_local_llm_server="false"
local_image_server_models_url=""
local_llm_models_url=""
local_llm_should_check_health="false"

if [[ "$enable_image_server" == "true" ]] || [[ "$enable_llm_server" == "true" ]]; then

image_gen_provider="$(resolve_env_value "$BACKEND_DIR/.env" "IMAGE_GEN_PROVIDER")"
image_gen_provider="$(trim_env_value "${image_gen_provider:-}")"
image_gen_provider_raw="$image_gen_provider"
if [[ -z "${image_gen_provider:-}" ]]; then
  image_gen_provider="xai"
fi
if [[ "$image_gen_provider" == local_* ]] && [[ "$image_gen_provider" != "local_sdcpp" ]]; then
  image_gen_provider="local_sdcpp"
fi

image_gen_local_base_url="$(resolve_env_value "$BACKEND_DIR/.env" "LOCAL_IMAGE_GEN_BASE_URL")"
image_gen_local_base_url="$(trim_env_value "${image_gen_local_base_url:-}")"
image_gen_local_model="$(resolve_env_value "$BACKEND_DIR/.env" "LOCAL_IMAGE_GEN_MODEL")"
image_gen_local_model="$(trim_env_value "${image_gen_local_model:-}")"
if [[ -z "${image_gen_local_model:-}" ]]; then
  case "$image_gen_provider_raw" in
    local_flux-2-klein-base-9b)
      image_gen_local_model="flux-2-klein-base-9b-Q5_K_M.gguf"
      ;;
    local_flux-2-klein-9b)
      image_gen_local_model="flux-2-klein-9b-Q5_K_M.gguf"
      ;;
  esac
fi

infer_local_image_server_from_model() {
  local model_value="$1"
  local normalized="$model_value"
  normalized="${normalized#/models/}"

  case "$normalized" in
    "")
      return 0
      ;;
    *.gguf)
      local_image_server_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.flux2-klein-9b-gguf.yml"
      local_image_server_service_name="flux2-klein-9b-gguf-image-server"
      if [[ "$normalized" == */* ]]; then
        local_image_server_diffusion_model="/models/$normalized"
      elif [[ "$normalized" == *base* ]]; then
        local_image_server_diffusion_model="/models/unsloth/FLUX.2-klein-base-9B-GGUF/$normalized"
      else
        local_image_server_diffusion_model="/models/unsloth/FLUX.2-klein-9B-GGUF/$normalized"
      fi
      ;;
    flux-2-klein-9b-fp8.safetensors)
      local_image_server_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.flux2.yml"
      local_image_server_service_name="flux2-image-server"
      local_image_server_diffusion_model="/models/black-forest-labs/FLUX.2-klein-9b-fp8/$normalized"
      ;;
    flux-2-klein-4b-fp8.safetensors)
      local_image_server_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.flux2.yml"
      local_image_server_service_name="flux2-image-server"
      local_image_server_diffusion_model="/models/black-forest-labs/FLUX.2-klein-4b-fp8/$normalized"
      ;;
    flux-2-klein-4b.safetensors)
      local_image_server_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.flux2.yml"
      local_image_server_service_name="flux2-image-server"
      local_image_server_diffusion_model="/models/Comfy-Org/vae-text-encorder-for-flux-klein-4b/split_files/diffusion_models/$normalized"
      ;;
    *)
      if [[ "$normalized" == */* ]]; then
        if [[ "$normalized" == *.gguf ]]; then
          local_image_server_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.flux2-klein-9b-gguf.yml"
          local_image_server_service_name="flux2-klein-9b-gguf-image-server"
        else
          local_image_server_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.flux2.yml"
          local_image_server_service_name="flux2-image-server"
        fi
        local_image_server_diffusion_model="/models/$normalized"
      fi
      ;;
  esac
}

start_local_image_server="false"
if [[ "${image_gen_provider:-}" == "local_sdcpp" ]]; then
  start_local_image_server="true"
fi

flux2_klein_profile="$(resolve_env_value "$BACKEND_DIR/.env" "FLUX2_KLEIN_PROFILE")"
flux2_klein_profile="$(trim_env_value "${flux2_klein_profile:-}")"

local_image_gen_base_url="${image_gen_local_base_url:-}"
if [[ -z "${local_image_gen_base_url:-}" ]]; then
  local_image_gen_base_url="$(resolve_env_value "$BACKEND_DIR/.env" "LOCAL_IMAGE_GEN_BASE_URL")"
  local_image_gen_base_url="$(trim_env_value "${local_image_gen_base_url:-}")"
fi
if [[ -z "${local_image_gen_base_url:-}" ]]; then
  local_image_gen_base_url="http://127.0.0.1:1234"
fi
local_image_gen_base_url="${local_image_gen_base_url%/}"
local_image_server_models_url="${local_image_gen_base_url}/v1/models"
local_image_server_listen_port="$(port_from_url "$local_image_gen_base_url")"

local_image_server_compose_file=""
local_image_server_service_name=""
local_image_server_project_name=""
local_image_server_diffusion_model=""
local_image_server_vae_model=""
local_image_server_llm_model=""

infer_local_image_server_from_model "${image_gen_local_model:-}"

# Legacy .env fallbacks/overrides.
env_toggle="$(resolve_env_bool "$BACKEND_DIR/.env" "USE_LOCAL_IMAGE_GEN_SERVER")"
if is_truthy "$env_toggle"; then
  start_local_image_server="true"
fi

legacy_compose_file="$(resolve_env_value "$BACKEND_DIR/.env" "LOCAL_IMAGE_SERVER_DOCKER_COMPOSE_FILE")"
legacy_compose_file="$(trim_env_value "${legacy_compose_file:-}")"
if [[ -n "${legacy_compose_file:-}" ]]; then
  local_image_server_compose_file="$legacy_compose_file"
fi
if [[ -n "${local_image_server_compose_file:-}" ]] && [[ "${local_image_server_compose_file}" != /* ]]; then
  local_image_server_compose_file="${LOCAL_MODEL_SERVING_DIR}/${local_image_server_compose_file}"
fi

legacy_service_name="$(resolve_env_value "$BACKEND_DIR/.env" "LOCAL_IMAGE_SERVER_DOCKER_SERVICE_NAME")"
legacy_service_name="$(trim_env_value "${legacy_service_name:-}")"
if [[ -n "${legacy_service_name:-}" ]]; then
  local_image_server_service_name="$legacy_service_name"
fi

local_image_server_project_name="$(resolve_env_value "$BACKEND_DIR/.env" "LOCAL_IMAGE_SERVER_DOCKER_PROJECT_NAME")"
local_image_server_project_name="$(trim_env_value "${local_image_server_project_name:-}")"

legacy_diffusion_model="$(resolve_env_value "$BACKEND_DIR/.env" "LOCAL_IMAGE_SERVER_DIFFUSION_MODEL")"
legacy_diffusion_model="$(trim_env_value "${legacy_diffusion_model:-}")"
if [[ -n "${legacy_diffusion_model:-}" ]]; then
  local_image_server_diffusion_model="$legacy_diffusion_model"
fi

local_image_server_vae_model="$(resolve_env_value "$BACKEND_DIR/.env" "LOCAL_IMAGE_SERVER_VAE_MODEL")"
local_image_server_vae_model="$(trim_env_value "${local_image_server_vae_model:-}")"

local_image_server_llm_model="$(resolve_env_value "$BACKEND_DIR/.env" "LOCAL_IMAGE_SERVER_LLM_MODEL")"
local_image_server_llm_model="$(trim_env_value "${local_image_server_llm_model:-}")"

default_llm_provider="$(resolve_env_value "$BACKEND_DIR/.env" "DEFAULT_LLM_PROVIDER")"
default_llm_provider="$(trim_env_value "${default_llm_provider:-}")"
default_llm_provider="$(echo "${default_llm_provider:-}" | tr '[:upper:]' '[:lower:]')"
local_llm_api_base_url="$(resolve_env_value "$BACKEND_DIR/.env" "LOCAL_LLM_API_BASE_URL")"
local_llm_api_base_url="$(trim_env_value "${local_llm_api_base_url:-}")"
if [[ -z "${local_llm_api_base_url:-}" ]]; then
  local_llm_api_base_url="http://127.0.0.1:12434/v1"
fi
local_llm_api_base_url="${local_llm_api_base_url%/}"
local_llm_models_url="${local_llm_api_base_url}/models"
local_llm_context_size="$(resolve_env_value "$BACKEND_DIR/.env" "LOCAL_LLM_CONTEXT_SIZE")"
local_llm_context_size="$(trim_env_value "${local_llm_context_size:-}")"
if [[ -z "${local_llm_context_size:-}" ]]; then
  local_llm_context_size="32768"
fi
local_llm_chat_model="$(resolve_env_value "$BACKEND_DIR/.env" "LOCAL_CHAT_MODEL")"
local_llm_chat_model="$(trim_env_value "${local_llm_chat_model:-}")"
local_llm_server_mode="$(resolve_config_value "LOCAL_LLM_SERVER_MODE")"
local_llm_server_mode="${local_llm_server_mode:-qwen35}"
local_llm_atlas_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.atlas-qwen35-35b-a3b-nvfp4.yml"
local_llm_atlas_service_name="atlas-qwen35-35b-a3b-nvfp4"
local_llm_atlas_downloader_service_name="atlas-qwen35-35b-a3b-nvfp4-downloader"
local_llm_qwen_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.qwen35-35b-a3b.yml"
local_llm_qwen_llama_service="qwen35-35b-a3b-llama"
local_llm_qwen_proxy_service="qwen35-35b-a3b-proxy"
local_llm_vllm_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.qwen35-35b-a3b-gptq-vllm.yml"
local_llm_vllm_service_name="qwen35-35b-a3b-gptq-vllm"
LOCAL_VLLM_SERVER_SCRIPT="$LOCAL_MODEL_SERVING_DIR/scripts/qwen35_35b_a3b_gptq_vllm_server.sh"

start_local_llm_server="false"
local_llm_should_check_health="false"

resolve_effective_local_llm_from_backend() {
  if [[ ! -d "$BACKEND_DIR/.venv" ]]; then
    return
  fi

  local resolved
  resolved="$(
    cd "$BACKEND_DIR" && uv run python - <<'PY'
from app.core.config import get_settings

s = get_settings()
print((s.local_llm_api_base_url or "").strip())
print((s.local_chat_model or "").strip())
print(int(getattr(s, "local_llm_context_size", 32768)))
PY
  )" || return

  local resolved_base_url resolved_chat_model resolved_context_size
  resolved_base_url="$(echo "$resolved" | sed -n '1p')"
  resolved_chat_model="$(echo "$resolved" | sed -n '2p')"
  resolved_context_size="$(echo "$resolved" | sed -n '3p')"

  if [[ -n "${resolved_base_url:-}" ]]; then
    local_llm_api_base_url="${resolved_base_url%/}"
    local_llm_models_url="${local_llm_api_base_url}/models"
  fi
  if [[ -n "${resolved_chat_model:-}" ]]; then
    local_llm_chat_model="${resolved_chat_model}"
  fi
  if [[ -n "${resolved_context_size:-}" ]]; then
    local_llm_context_size="${resolved_context_size}"
  fi

  local local_llm_port
  local_llm_port="$(port_from_url "$local_llm_api_base_url")"
  # Only auto-detect mode if not explicitly set via LOCAL_LLM_SERVER_MODE
  local explicit_mode
  explicit_mode="$(resolve_config_value "LOCAL_LLM_SERVER_MODE")"
  if [[ -n "${explicit_mode:-}" ]]; then
    local_llm_server_mode="${explicit_mode}"
  elif [[ "${local_llm_port:-}" == "18888" ]] || [[ "${local_llm_chat_model,,}" == *"nvfp4"* ]]; then
    local_llm_server_mode="atlas-nvfp4"
  elif [[ "${local_llm_chat_model,,}" == *"gptq"* ]]; then
    local_llm_server_mode="vllm"
  else
    local_llm_server_mode="qwen35"
  fi
}

# Final defaults if envs did not give explicit values.
if [[ -z "${local_image_server_compose_file:-}" ]]; then
  local_image_server_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.flux2.yml"
fi

compose_basename="$(basename "${local_image_server_compose_file}")"
if [[ -z "${local_image_server_service_name:-}" ]]; then
  if [[ "$compose_basename" == "docker-compose.flux2-klein-9b-gguf.yml" ]]; then
    local_image_server_service_name="flux2-klein-9b-gguf-image-server"
  else
    local_image_server_service_name="flux2-image-server"
  fi
fi
if [[ "$compose_basename" == "docker-compose.flux2-klein-9b-gguf.yml" ]]; then
  if [[ -z "${local_image_server_diffusion_model:-}" ]]; then
    local_image_server_diffusion_model="/models/unsloth/FLUX.2-klein-9B-GGUF/flux-2-klein-9b-Q5_K_M.gguf"
  fi
fi

start_local_image_server="$enable_image_server"

fi  # end image/LLM config block

# ---------------------------------------------------------------------------
# Stop old dev servers (only relevant ones)
# ---------------------------------------------------------------------------
if [[ "$enable_backend" == "true" ]]; then
  echo "Stopping old dev servers..."
  kill_from_pid_file "$BACKEND_PID_FILE" "backend"
  kill_listeners_on_port 8000 "backend"
fi
if [[ "$enable_frontend" == "true" ]]; then
  kill_from_pid_file "$FRONTEND_PID_FILE" "frontend"
  kill_listeners_on_port 5173 "frontend"
fi

# ---------------------------------------------------------------------------
# Backend: Postgres, venv, migrations
# ---------------------------------------------------------------------------
if [[ "$enable_backend" == "true" ]]; then
  echo "Starting Postgres..."
  (
    cd "$BACKEND_DIR"
    docker compose up -d db
  )
  wait_for_db_health

  if [[ ! -d "$BACKEND_DIR/.venv" ]]; then
    echo "Creating backend virtualenv with uv sync..."
    (
      cd "$BACKEND_DIR"
      uv sync
    )
  fi

  echo "Running database migrations..."
  (
    cd "$BACKEND_DIR"
    uv run alembic upgrade head
  )
fi

# ---------------------------------------------------------------------------
# Frontend: npm install
# ---------------------------------------------------------------------------
if [[ "$enable_frontend" == "true" ]]; then
  if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    echo "Installing frontend dependencies..."
    (
      cd "$FRONTEND_DIR"
      npm install
    )
  fi

  if [[ -d "$FRONTEND_DIR/node_modules/.bin" ]]; then
    chmod +x "$FRONTEND_DIR"/node_modules/.bin/* 2>/dev/null || true
  fi
fi

# ---------------------------------------------------------------------------
# LLM server auto-detection (only when backend is enabled)
# ---------------------------------------------------------------------------
if [[ "$enable_backend" == "true" ]]; then
  resolve_effective_local_llm_from_backend
fi
if [[ "$enable_llm_server" == "true" ]]; then
  if [[ "${default_llm_provider:-}" == "local" ]] && is_local_host_url "$local_llm_api_base_url"; then
    if wait_for_http "${local_llm_models_url}" "Local LLM server" 2; then
      start_local_llm_server="false"
      echo "Local LLM server already healthy at ${local_llm_models_url}; leaving it untouched."
    else
      start_local_llm_server="true"
      echo "Local LLM server is not healthy at ${local_llm_models_url}; starting it."
    fi
  else
    start_local_llm_server="true"
  fi
fi

if [[ "$start_local_image_server" == "true" ]]; then
  if [[ ! -x "$LOCAL_IMAGE_SERVER_SCRIPT" ]]; then
    echo "Local image server script not found or not executable: $LOCAL_IMAGE_SERVER_SCRIPT"
    exit 1
  fi

  echo "Restarting local image server..."
  (
    cd "$LOCAL_MODEL_SERVING_DIR"
    cleanup_legacy_local_model_project

    if [[ -n "${local_image_server_compose_file:-}" ]]; then
      if [[ ! -f "${local_image_server_compose_file}" ]]; then
        echo "Local image server compose file not found: ${local_image_server_compose_file}"
        exit 1
      fi
      export DOCKER_COMPOSE_FILE="${local_image_server_compose_file}"
      echo "Using DOCKER_COMPOSE_FILE=${DOCKER_COMPOSE_FILE}"
    fi
    if [[ -n "${local_image_server_service_name:-}" ]]; then
      export DOCKER_SERVICE_NAME="${local_image_server_service_name}"
      echo "Using DOCKER_SERVICE_NAME=${DOCKER_SERVICE_NAME}"
    fi
    if [[ -n "${local_image_server_project_name:-}" ]]; then
      export DOCKER_PROJECT_NAME="${local_image_server_project_name}"
      echo "Using DOCKER_PROJECT_NAME=${DOCKER_PROJECT_NAME}"
    fi
    if [[ -n "${local_image_server_diffusion_model:-}" ]]; then
      export DIFFUSION_MODEL="${local_image_server_diffusion_model}"
      echo "Using DIFFUSION_MODEL=${DIFFUSION_MODEL}"
    fi
    if [[ -n "${local_image_server_vae_model:-}" ]]; then
      export VAE_MODEL="${local_image_server_vae_model}"
      echo "Using VAE_MODEL=${VAE_MODEL}"
    fi
    if [[ -n "${local_image_server_llm_model:-}" ]]; then
      export LLM_MODEL="${local_image_server_llm_model}"
      echo "Using LLM_MODEL=${LLM_MODEL}"
    fi
    if [[ -n "${local_image_server_listen_port:-}" ]]; then
      export LISTEN_PORT="${local_image_server_listen_port}"
      echo "Using LISTEN_PORT=${LISTEN_PORT} (from LOCAL_IMAGE_GEN_BASE_URL)"
    fi
    if [[ -n "${flux2_klein_profile:-}" ]]; then
      export FLUX2_KLEIN_PROFILE="$flux2_klein_profile"
      echo "Using FLUX2_KLEIN_PROFILE=${FLUX2_KLEIN_PROFILE}"
    fi
    "$LOCAL_IMAGE_SERVER_SCRIPT" stop || true
    "$LOCAL_IMAGE_SERVER_SCRIPT" start
  )
  local_image_server_timeout="${LOCAL_IMAGE_SERVER_READY_TIMEOUT_SECONDS:-1200}"
  if ! wait_for_http "${local_image_server_models_url}" "Local image server" "$local_image_server_timeout"; then
    echo "Warning: continuing startup even though local image server is still warming up."
  fi
fi

if [[ "$start_local_llm_server" == "true" ]]; then
  echo "Restarting local LLM server (mode=${local_llm_server_mode}, base_url=${local_llm_api_base_url}, model=${local_llm_chat_model:-<unset>}, context_size=${local_llm_context_size:-<unset>})..."
  if [[ "$local_llm_server_mode" == "atlas-nvfp4" ]]; then
    if [[ ! -f "$local_llm_atlas_compose_file" ]]; then
      echo "Atlas compose file not found: $local_llm_atlas_compose_file"
      exit 1
    fi
    local_llm_listen_port="$(port_from_url "$local_llm_api_base_url")"
    (
      cd "$LOCAL_MODEL_SERVING_DIR"
      cleanup_legacy_local_model_project
      if [[ -f "$local_llm_qwen_compose_file" ]]; then
        echo "Stopping Qwen35 local LLM services to free GPU memory..."
        docker compose -f "$local_llm_qwen_compose_file" stop "$local_llm_qwen_proxy_service" "$local_llm_qwen_llama_service" >/dev/null 2>&1 || true
      fi
      echo "Prefetching Atlas model into Hugging Face cache (first run can take a while)..."
      docker compose -f "$local_llm_atlas_compose_file" --profile tools run --rm "$local_llm_atlas_downloader_service_name"
      if [[ -n "${local_llm_listen_port:-}" ]]; then
        export LISTEN_PORT="${local_llm_listen_port}"
      fi
      docker compose -f "$local_llm_atlas_compose_file" up -d "$local_llm_atlas_service_name"
    )
  elif [[ "$local_llm_server_mode" == "vllm" ]]; then
    if [[ ! -x "$LOCAL_VLLM_SERVER_SCRIPT" ]]; then
      echo "vLLM server script not found or not executable: $LOCAL_VLLM_SERVER_SCRIPT"
      exit 1
    fi
    (
      cd "$LOCAL_MODEL_SERVING_DIR"
      cleanup_legacy_local_model_project
      if [[ -f "$local_llm_qwen_compose_file" ]]; then
        echo "Stopping Qwen35 llama.cpp services to free GPU memory..."
        docker compose -f "$local_llm_qwen_compose_file" stop "$local_llm_qwen_proxy_service" "$local_llm_qwen_llama_service" >/dev/null 2>&1 || true
      fi
      export VLLM_MAX_MODEL_LEN="${local_llm_context_size}"
      "$LOCAL_VLLM_SERVER_SCRIPT" restart
    )
  else
    if [[ ! -x "$LOCAL_LLM_SERVER_SCRIPT" ]]; then
      echo "Local LLM server script not found or not executable: $LOCAL_LLM_SERVER_SCRIPT"
      exit 1
    fi
    (
      cd "$LOCAL_MODEL_SERVING_DIR"
      cleanup_legacy_local_model_project
      export CONTEXT_SIZE="${local_llm_context_size}"
      "$LOCAL_LLM_SERVER_SCRIPT" restart
    )
  fi
  local_llm_server_timeout="${LOCAL_LLM_SERVER_READY_TIMEOUT_SECONDS:-1800}"
  if ! wait_for_http "${local_llm_models_url}" "Local LLM server" "$local_llm_server_timeout"; then
    echo "Warning: continuing startup even though local LLM server is still warming up."
  fi
elif [[ "$local_llm_should_check_health" == "true" ]]; then
  if wait_for_http "${local_llm_models_url}" "Local LLM server" 2; then
    echo "Local LLM server already healthy at ${local_llm_models_url}; leaving it untouched."
  else
    echo "Warning: backend is configured to use the local LLM at ${local_llm_api_base_url}, but restart_dev did not restart it."
    echo "Use ./restart_dev.sh --with-local-llm-server if you want the local LLM stack restarted."
  fi
fi

# ---------------------------------------------------------------------------
# Voice servers (Parakeet STT + Chatterbox TTS + Kokoro TTS)
# ---------------------------------------------------------------------------

voice_stt_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.parakeet-stt.yml"
voice_tts_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.chatterbox-tts.yml"
kokoro_provider="$(resolve_config_value "KOKORO_PROVIDER")"
kokoro_provider="${kokoro_provider:-gpu}"
if [[ "$kokoro_provider" == "cpu" ]]; then
  voice_kokoro_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.kokoro-tts-cpu.yml"
else
  voice_kokoro_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.kokoro-tts.yml"
fi

parakeet_stt_url="$(resolve_config_value "PARAKEET_STT_URL")"
parakeet_stt_url="${parakeet_stt_url:-http://127.0.0.1:4124}"

chatterbox_tts_url="$(resolve_config_value "CHATTERBOX_TTS_URL")"
chatterbox_tts_url="${chatterbox_tts_url:-http://127.0.0.1:4123}"

kokoro_tts_url="$(resolve_config_value "KOKORO_TTS_URL")"
kokoro_tts_url="${kokoro_tts_url:-http://127.0.0.1:4125}"

voice_stt_health_url="${parakeet_stt_url%/}/health"
voice_tts_health_url="${chatterbox_tts_url%/}/health"
voice_kokoro_health_url="${kokoro_tts_url%/}/health"

if [[ "$enable_voice_stt" == "true" ]]; then
  if wait_for_http "$voice_stt_health_url" "Parakeet STT" 2; then
    echo "Parakeet STT already healthy at ${parakeet_stt_url}; leaving it untouched."
  else
    echo "Starting Parakeet STT..."
    (
      cd "$LOCAL_MODEL_SERVING_DIR"
      if [[ -f "$voice_stt_compose_file" ]]; then
        docker compose -f "$voice_stt_compose_file" up -d --build
      else
        echo "  Warning: STT compose file not found: $voice_stt_compose_file"
      fi
    )
    echo "  Waiting for Parakeet STT to become ready..."
    if ! wait_for_http "$voice_stt_health_url" "Parakeet STT" 120; then
      echo "  Warning: Parakeet STT did not become ready in time; continuing."
    fi
  fi
fi

if [[ "$enable_voice_tts" == "true" ]]; then
  chatterbox_already_healthy="false"
  kokoro_already_healthy="false"
  if wait_for_http "$voice_tts_health_url" "Chatterbox TTS" 2; then
    chatterbox_already_healthy="true"
    echo "Chatterbox TTS already healthy at ${chatterbox_tts_url}; leaving it untouched."
  fi
  if wait_for_http "$voice_kokoro_health_url" "Kokoro TTS" 2; then
    kokoro_already_healthy="true"
    echo "Kokoro TTS already healthy at ${kokoro_tts_url}; leaving it untouched."
  fi

  if [[ "$chatterbox_already_healthy" == "false" ]] || [[ "$kokoro_already_healthy" == "false" ]]; then
    echo "Starting TTS servers..."
    (
      cd "$LOCAL_MODEL_SERVING_DIR"
      if [[ "$chatterbox_already_healthy" == "false" ]]; then
        if [[ -f "$voice_tts_compose_file" ]]; then
          echo "  Building & starting Chatterbox TTS..."
          docker compose -f "$voice_tts_compose_file" up -d --build
        else
          echo "  Warning: Chatterbox TTS compose file not found: $voice_tts_compose_file"
        fi
      fi
      if [[ "$kokoro_already_healthy" == "false" ]]; then
        if [[ -f "$voice_kokoro_compose_file" ]]; then
          echo "  Building & starting Kokoro TTS..."
          docker compose -f "$voice_kokoro_compose_file" up -d --build
        else
          echo "  Warning: Kokoro TTS compose file not found: $voice_kokoro_compose_file"
        fi
      fi
    )
    if [[ "$chatterbox_already_healthy" == "false" ]]; then
      echo "  Waiting for Chatterbox TTS to become ready (model loading may take a while)..."
      if ! wait_for_http "$voice_tts_health_url" "Chatterbox TTS" 180; then
        echo "  Warning: Chatterbox TTS did not become ready in time; continuing."
      fi
    fi
    if [[ "$kokoro_already_healthy" == "false" ]]; then
      echo "  Waiting for Kokoro TTS to become ready..."
      if ! wait_for_http "$voice_kokoro_health_url" "Kokoro TTS" 120; then
        echo "  Warning: Kokoro TTS did not become ready in time; continuing."
      fi
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Tool server (SearXNG + crawl4ai)
# ---------------------------------------------------------------------------

tool_server_compose_file="$LOCAL_MODEL_SERVING_DIR/docker-compose.tool-server.yml"
tool_server_url="$(resolve_config_value "TOOL_SERVER_URL")"
tool_server_url="${tool_server_url:-http://127.0.0.1:4130}"
tool_server_health_url="${tool_server_url%/}/health"

if [[ "$enable_tool_server" == "true" ]]; then
  if wait_for_http "$tool_server_health_url" "Tool server" 2; then
    echo "Tool server already healthy at ${tool_server_url}; leaving it untouched."
  else
    echo "Starting tool server (SearXNG + crawl4ai)..."
    (
      cd "$LOCAL_MODEL_SERVING_DIR"
      if [[ -f "$tool_server_compose_file" ]]; then
        docker compose -f "$tool_server_compose_file" up -d --build
      else
        echo "  Warning: Tool server compose file not found: $tool_server_compose_file"
      fi
    )
    echo "  Waiting for tool server to become ready..."
    if ! wait_for_http "$tool_server_health_url" "Tool server" 120; then
      echo "  Warning: Tool server did not become ready in time; continuing."
    fi
  fi
fi

if [[ "$enable_backend" == "true" ]]; then
  echo "Starting backend..."
  : >"$BACKEND_LOG"
  setsid -f bash -lc "cd '$BACKEND_DIR' && exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload </dev/null >>'$BACKEND_LOG' 2>&1"
  wait_for_http "http://127.0.0.1:8000/docs" "Backend" || exit 1
  backend_pid="$(pid_for_listening_port 8000)"
  if [[ -n "${backend_pid:-}" ]]; then
    echo "$backend_pid" >"$BACKEND_PID_FILE"
  fi
fi

if [[ "$enable_frontend" == "true" ]]; then
  echo "Starting frontend..."
  : >"$FRONTEND_LOG"
  setsid -f bash -lc "cd '$FRONTEND_DIR' && exec npm run dev -- --host 0.0.0.0 --port 5173 </dev/null >>'$FRONTEND_LOG' 2>&1"
  wait_for_http "http://127.0.0.1:5173/" "Frontend" || exit 1
  frontend_pid="$(pid_for_listening_port 5173)"
  if [[ -n "${frontend_pid:-}" ]]; then
    echo "$frontend_pid" >"$FRONTEND_PID_FILE"
  fi
fi

echo ""
echo "Restart complete."
if [[ "$enable_backend" == "true" ]]; then
  echo "Backend:  http://127.0.0.1:8000/docs"
fi
if [[ "$enable_frontend" == "true" ]]; then
  echo "Frontend: http://127.0.0.1:5173/"
fi
if [[ "$start_local_image_server" == "true" ]]; then
  echo "Image server: ${local_image_server_models_url}"
fi
if [[ "$enable_voice_stt" == "true" ]]; then
  echo "Voice STT:       ${parakeet_stt_url}"
fi
if [[ "$enable_voice_tts" == "true" ]]; then
  echo "Voice TTS (CB):  ${chatterbox_tts_url}"
  echo "Voice TTS (KK):  ${kokoro_tts_url} (${kokoro_provider})"
fi
if [[ "$enable_tool_server" == "true" ]]; then
  echo "Tool server:     ${tool_server_url}"
fi
if [[ "$enable_backend" == "true" ]]; then
  echo "Logs:"
  echo "  $BACKEND_LOG"
  echo "  $FRONTEND_LOG"
fi
