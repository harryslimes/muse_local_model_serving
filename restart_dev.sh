#!/usr/bin/env bash
# Platform-aware dev restart.
# Delegates to nvidia/ or mac/ depending on --platform.
#
# Usage:
#   ./restart_dev.sh [--platform nvidia|mac] [flags...]
#
# Platform can also be set via MUSE_PLATFORM env var.
# All other flags are forwarded to the platform script as-is.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

wait_for_http() {
  local url="$1"
  local label="$2"
  local max_tries="${3:-120}"
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

# ---------------------------------------------------------------------------
# Parse --platform; split remaining args into infra vs model-server buckets.
# ---------------------------------------------------------------------------
PLATFORM="${MUSE_PLATFORM:-nvidia}"

# Args safe to forward to nvidia/scripts/restart_dev.sh (backend, frontend, db)
INFRA_ARGS=()
# Model server intent (only used in mac mode)
mac_enable_llm=""
mac_enable_stt=""
mac_enable_tts=""
db_mode_override=""
mac_llm_mode_override=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)
      PLATFORM="$2"; shift 2 ;;
    --platform=*)
      PLATFORM="${1#--platform=}"; shift ;;
    --db-mode)
      db_mode_override="$2"; shift 2 ;;
    --db-mode=*)
      db_mode_override="${1#--db-mode=}"; shift ;;
    --local-llm-server-mode|--llm-server-mode)
      mac_llm_mode_override="$2"; shift 2 ;;
    --local-llm-server-mode=*|--llm-server-mode=*)
      mac_llm_mode_override="${1#*=}"; shift ;;

    # Capture model-server intent — needed for mac routing.
    # These are NOT forwarded to nvidia/scripts/restart_dev.sh in mac mode.
    --with-local-llm-server)    mac_enable_llm="true";  shift ;;
    --without-local-llm-server) mac_enable_llm="false"; shift ;;
    --with-voice-server)        mac_enable_stt="true"; mac_enable_tts="true";  shift ;;
    --without-voice-server)     mac_enable_stt="false"; mac_enable_tts="false"; shift ;;
    --with-voice-stt)           mac_enable_stt="true";  shift ;;
    --without-voice-stt)        mac_enable_stt="false"; shift ;;
    --with-voice-tts)           mac_enable_tts="true";  shift ;;
    --without-voice-tts)        mac_enable_tts="false"; shift ;;
    # Image server and tool server flags aren't applicable on mac; skip silently.
    --with-local-image-server|--without-local-image-server) shift ;;
    --with-tool-server|--without-tool-server) shift ;;

    # Everything else (--reset-db, --with/without-backend/frontend, -h, etc.)
    *) INFRA_ARGS+=("$1"); shift ;;
  esac
done

print_usage() {
  cat <<'EOF'
Usage:
  ./restart_dev.sh [--platform nvidia|mac] [options]

Platform:
  --platform nvidia   DGX Spark / NVIDIA — Docker-based model servers (default)
  --platform mac      Apple Silicon — MLX-based model servers

  MUSE_PLATFORM env var sets the default platform.

Model server flags:
  --with-local-llm-server / --without-local-llm-server
  --local-llm-server-mode mlx|llama.cpp|gemma|gemma-gguf   Mac local LLM backend to launch
  --with-voice-server / --without-voice-server   (STT + TTS together)
  --with-voice-stt / --without-voice-stt
  --with-voice-tts / --without-voice-tts

Other flags are forwarded to the platform script (backend, frontend, db, etc.).
Run  nvidia/scripts/restart_dev.sh --help  for the full flag list.

Database flags:
  --db-mode docker   Always use docker compose for Postgres (default)
  --db-mode local    Use an already-running local Postgres from DATABASE_URL
  --db-mode auto     Use local Postgres when reachable, else fall back to docker

Examples:
  ./restart_dev.sh --platform nvidia --with-local-llm-server --with-voice-server
  ./restart_dev.sh --platform mac --with-voice-server
  ./restart_dev.sh --platform mac --with-local-llm-server --without-voice-server
  ./restart_dev.sh --platform mac --with-local-llm-server --local-llm-server-mode llama.cpp
  ./restart_dev.sh --platform mac --with-local-llm-server --local-llm-server-mode gemma
  ./restart_dev.sh --platform mac --db-mode local --local-llm-server-mode gemma-gguf
  ./restart_dev.sh --platform mac --db-mode local --without-voice-server
  MUSE_PLATFORM=mac ./restart_dev.sh --without-backend --without-frontend --with-voice-stt
EOF
}

for arg in "${INFRA_ARGS[@]+"${INFRA_ARGS[@]}"}"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    print_usage
    exit 0
  fi
done

# ---------------------------------------------------------------------------
# NVIDIA / DGX Spark — just delegate everything to the nvidia script.
# ---------------------------------------------------------------------------
if [[ "$PLATFORM" == "nvidia" || "$PLATFORM" == "dgx" ]]; then
  # Restore model-server flags so the nvidia script sees them.
  NVIDIA_ARGS=("${INFRA_ARGS[@]+"${INFRA_ARGS[@]}"}")
  [[ -n "$mac_enable_llm" ]]  && NVIDIA_ARGS+=("--$([ "$mac_enable_llm"  = true ] && echo with || echo without)-local-llm-server")
  [[ -n "$mac_enable_stt" ]]  && NVIDIA_ARGS+=("--$([ "$mac_enable_stt"  = true ] && echo with || echo without)-voice-stt")
  [[ -n "$mac_enable_tts" ]]  && NVIDIA_ARGS+=("--$([ "$mac_enable_tts"  = true ] && echo with || echo without)-voice-tts")
  [[ -n "$db_mode_override" ]] && NVIDIA_ARGS+=("--db-mode" "$db_mode_override")
  [[ -n "$mac_llm_mode_override" ]] && NVIDIA_ARGS+=("--local-llm-server-mode" "$mac_llm_mode_override")

  echo "Platform: nvidia"
  exec "$ROOT_DIR/nvidia/scripts/restart_dev.sh" "${NVIDIA_ARGS[@]+"${NVIDIA_ARGS[@]}"}"
fi

# ---------------------------------------------------------------------------
# Mac / Apple Silicon — run backend/frontend via nvidia script (which handles
# Postgres, migrations, npm), then start MLX model servers.
# ---------------------------------------------------------------------------
if [[ "$PLATFORM" == "mac" || "$PLATFORM" == "apple" ]]; then
  echo "Platform: mac"
  MAC_INFRA_ARGS=()
  mac_backend_plan="delegated"
  mac_frontend_plan="delegated"

  # Resolve model-server enable flags from mac .env (nvidia .env as fallback).
  _read_enable_flag() {
    local key="$1" default="$2"
    local env_files=("$ROOT_DIR/.env" "$ROOT_DIR/mac/.env")
    for f in "${env_files[@]}"; do
      if [[ -f "$f" ]]; then
        local val
        val="$(grep -E "^${key}=" "$f" 2>/dev/null | tail -n 1 | cut -d'=' -f2- | tr -d '[:space:]' || true)"
        if [[ -n "${val:-}" ]]; then
          echo "$val"; return
        fi
      fi
    done
    echo "$default"
  }

  _read_config_value() {
    local key="$1" default="$2"
    local env_files=("$ROOT_DIR/.env" "$ROOT_DIR/mac/.env")
    for f in "${env_files[@]}"; do
      if [[ -f "$f" ]]; then
        local val
        val="$(grep -E "^${key}=" "$f" 2>/dev/null | tail -n 1 | cut -d'=' -f2- || true)"
        val="$(echo "${val:-}" | sed 's/^ *//;s/ *$//')"
        val="${val%\"}" ; val="${val#\"}" ; val="${val%\'}" ; val="${val#\'}"
        if [[ -n "${val:-}" ]]; then
          echo "$val"; return
        fi
      fi
    done
    echo "$default"
  }

  _is_truthy() {
    local v="${1:-}"
    v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
    [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "on" ]]
  }

  # CLI flags override .env
  if [[ -z "$mac_enable_llm" ]]; then
    mac_enable_llm="$(_read_enable_flag ENABLE_LLM_SERVER false)"
  fi
  if [[ -z "$mac_enable_stt" ]]; then
    mac_enable_stt="$(_read_enable_flag ENABLE_VOICE_STT false)"
  fi
  if [[ -z "$mac_enable_tts" ]]; then
    mac_enable_tts="$(_read_enable_flag ENABLE_VOICE_TTS false)"
  fi

  mac_llm_mode=""
  if _is_truthy "$mac_enable_llm"; then
    mac_llm_mode="${mac_llm_mode_override:-$(_read_config_value LOCAL_LLM_SERVER_MODE "")}"
    mac_llm_mode="$(echo "${mac_llm_mode:-}" | tr '[:upper:]' '[:lower:]')"
    if [[ -z "${mac_llm_mode:-}" ]]; then
      echo "LOCAL_LLM_SERVER_MODE is not set in muse_local_model_serving/.env."
      echo "Set LOCAL_LLM_SERVER_MODE=mlx or LOCAL_LLM_SERVER_MODE=llama.cpp and rerun."
      exit 1
    fi
    case "$mac_llm_mode" in
      mlx) ;;
      llama|llama.cpp) mac_llm_mode="llama.cpp" ;;
      gemma|gemma-gguf) ;;
      *)
        echo "Unknown LOCAL_LLM_SERVER_MODE: $mac_llm_mode (use mlx, llama.cpp, or gemma)"
        exit 1
        ;;
    esac
  else
    mac_llm_mode="${mac_llm_mode_override:-$(_read_config_value LOCAL_LLM_SERVER_MODE "")}"
    mac_llm_mode="$(echo "${mac_llm_mode:-}" | tr '[:upper:]' '[:lower:]')"
    [[ "$mac_llm_mode" == "llama" ]] && mac_llm_mode="llama.cpp"
  fi

  for arg in "${INFRA_ARGS[@]+"${INFRA_ARGS[@]}"}"; do
    case "$arg" in
      --with-backend) mac_backend_plan="true" ;;
      --without-backend) mac_backend_plan="false" ;;
      --with-frontend) mac_frontend_plan="true" ;;
      --without-frontend) mac_frontend_plan="false" ;;
    esac
  done

  echo "Mac service plan:"
  echo "  Backend:      ${mac_backend_plan}"
  echo "  Frontend:     ${mac_frontend_plan}"
  echo "  DB mode:      ${db_mode_override:-${MUSE_DB_MODE:-docker}}"
  echo "  LLM server:   $mac_enable_llm"
  echo "  LLM mode:     $mac_llm_mode"
  echo "  Voice STT:    $mac_enable_stt"
  echo "  Voice TTS:    $mac_enable_tts"
  echo ""

  if [[ -n "$db_mode_override" ]]; then
    MAC_INFRA_ARGS+=(--db-mode "$db_mode_override")
  fi
  MAC_INFRA_ARGS+=("${INFRA_ARGS[@]+"${INFRA_ARGS[@]}"}")

  # Run backend/frontend via nvidia script, explicitly suppressing all model servers.
  _GEMMA_MLX_MODEL="${MLX_GEMMA_MODEL:-majentik/gemma-4-26B-A4B-it-RotorQuant-MLX-4bit}"
  if [[ "$mac_llm_mode" == "llama.cpp" ]]; then
    LOCAL_LLM_API_BASE_URL="${LOCAL_LLM_API_BASE_URL:-http://127.0.0.1:12436/v1}" \
    MUSE_BACKEND_RELOAD="${MUSE_BACKEND_RELOAD:-false}" \
    "$ROOT_DIR/nvidia/scripts/restart_dev.sh" \
      --without-local-llm-server \
      --without-local-image-server \
      --without-voice-stt \
      --without-voice-tts \
      --without-tool-server \
      "${MAC_INFRA_ARGS[@]+"${MAC_INFRA_ARGS[@]}"}"
  elif [[ "$mac_llm_mode" == "gemma" ]]; then
    LOCAL_LLM_API_BASE_URL="${LOCAL_LLM_API_BASE_URL:-http://127.0.0.1:12437/v1}" \
    LOCAL_CHAT_MODEL="${LOCAL_CHAT_MODEL:-${_GEMMA_MLX_MODEL}}" \
    LOCAL_SUMMARIZATION_MODEL="${LOCAL_SUMMARIZATION_MODEL:-${_GEMMA_MLX_MODEL}}" \
    LOCAL_LLM_PROFILE="${LOCAL_LLM_PROFILE:-${_GEMMA_MLX_MODEL}}" \
    MUSE_BACKEND_RELOAD="${MUSE_BACKEND_RELOAD:-false}" \
    "$ROOT_DIR/nvidia/scripts/restart_dev.sh" \
      --without-local-llm-server \
      --without-local-image-server \
      --without-voice-stt \
      --without-voice-tts \
      --without-tool-server \
      "${MAC_INFRA_ARGS[@]+"${MAC_INFRA_ARGS[@]}"}"
  elif [[ "$mac_llm_mode" == "gemma-gguf" ]]; then
    _GEMMA_GGUF_MODEL="gemma-4-26B-A4B-it-RotorQuant-Q4_K_M.gguf"
    LOCAL_LLM_API_BASE_URL="${LOCAL_LLM_API_BASE_URL:-http://127.0.0.1:12439/v1}" \
    LOCAL_CHAT_MODEL="${LOCAL_CHAT_MODEL:-${_GEMMA_GGUF_MODEL}}" \
    LOCAL_SUMMARIZATION_MODEL="${LOCAL_SUMMARIZATION_MODEL:-${_GEMMA_GGUF_MODEL}}" \
    LOCAL_LLM_PROFILE="${LOCAL_LLM_PROFILE:-${_GEMMA_GGUF_MODEL}}" \
    MUSE_BACKEND_RELOAD="${MUSE_BACKEND_RELOAD:-false}" \
    "$ROOT_DIR/nvidia/scripts/restart_dev.sh" \
      --without-local-llm-server \
      --without-local-image-server \
      --without-voice-stt \
      --without-voice-tts \
      --without-tool-server \
      "${MAC_INFRA_ARGS[@]+"${MAC_INFRA_ARGS[@]}"}"
  else
    LOCAL_LLM_API_BASE_URL="${LOCAL_LLM_API_BASE_URL:-http://127.0.0.1:12434/v1}" \
    LOCAL_CHAT_MODEL="${LOCAL_CHAT_MODEL:-mlx-community/Qwen3.5-4B-MLX-4bit}" \
    LOCAL_SUMMARIZATION_MODEL="${LOCAL_SUMMARIZATION_MODEL:-mlx-community/Qwen3.5-4B-MLX-4bit}" \
    LOCAL_LLM_PROFILE="${LOCAL_LLM_PROFILE:-mlx-community/Qwen3.5-4B-MLX-4bit}" \
    MUSE_BACKEND_RELOAD="${MUSE_BACKEND_RELOAD:-false}" \
    "$ROOT_DIR/nvidia/scripts/restart_dev.sh" \
      --without-local-llm-server \
      --without-local-image-server \
      --without-voice-stt \
      --without-voice-tts \
      --without-tool-server \
      "${MAC_INFRA_ARGS[@]+"${MAC_INFRA_ARGS[@]}"}"
  fi

  # Start Mac local model servers.
  MAC_SERVERS=()
  if _is_truthy "$mac_enable_llm"; then
    if [[ "$mac_llm_mode" == "llama.cpp" ]]; then
      mac_llama_model_path="$(_read_config_value LOCAL_LLAMA_MODEL_PATH "")"
      if [[ -z "${mac_llama_model_path:-}" ]]; then
        echo "LOCAL_LLAMA_MODEL_PATH is not set in muse_local_model_serving/.env."
        echo "Set it to a GGUF file path before using LOCAL_LLM_SERVER_MODE=llama.cpp."
        exit 1
      fi
      "$ROOT_DIR/mac/start.sh" stop llm gemma >/dev/null 2>&1 || true
      export MODEL_PATH="$mac_llama_model_path"
      MAC_SERVERS+=(llama)
    elif [[ "$mac_llm_mode" == "gemma" ]]; then
      "$ROOT_DIR/mac/start.sh" stop llama llm llama-gemma >/dev/null 2>&1 || true
      export MLX_GEMMA_MODEL="${_GEMMA_MLX_MODEL}"
      MAC_SERVERS+=(gemma)
    elif [[ "$mac_llm_mode" == "gemma-gguf" ]]; then
      "$ROOT_DIR/mac/start.sh" stop llama llm gemma >/dev/null 2>&1 || true
      gemma_gguf_path="$(_read_config_value GEMMA_GGUF_MODEL_PATH "")"
      if [[ -z "${GEMMA_GGUF_MODEL_PATH:-}" && -z "${gemma_gguf_path:-}" ]]; then
        _GEMMA_GGUF_FILENAME="gemma-4-26B-A4B-it-RotorQuant-Q4_K_M.gguf"
        _hf_hub="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
        _discovered="$(find "${_hf_hub}" -name "${_GEMMA_GGUF_FILENAME}" 2>/dev/null | head -1)"
        if [[ -n "${_discovered:-}" ]]; then
          echo "Auto-discovered GEMMA_GGUF_MODEL_PATH: ${_discovered}"
          gemma_gguf_path="${_discovered}"
        else
          echo "GEMMA_GGUF_MODEL_PATH is not set and ${_GEMMA_GGUF_FILENAME} was not found in ${_hf_hub}."
          echo "Set it in muse_local_model_serving/.env or export it before running."
          exit 1
        fi
      fi
      export GEMMA_GGUF_MODEL_PATH="${GEMMA_GGUF_MODEL_PATH:-${gemma_gguf_path}}"
      MAC_SERVERS+=(llama-gemma)
    else
      "$ROOT_DIR/mac/start.sh" stop llama gemma >/dev/null 2>&1 || true
      MAC_SERVERS+=(llm)
    fi
  fi
  _is_truthy "$mac_enable_stt" && MAC_SERVERS+=(stt)
  _is_truthy "$mac_enable_tts" && MAC_SERVERS+=(tts)

  if [[ ${#MAC_SERVERS[@]} -gt 0 ]]; then
    echo ""
    echo "Starting Mac local servers: ${MAC_SERVERS[*]}"
    "$ROOT_DIR/mac/start.sh" start "${MAC_SERVERS[@]}"

    if _is_truthy "$mac_enable_llm"; then
      echo "Waiting for Mac LLM server..."
      if [[ "$mac_llm_mode" == "llama.cpp" ]]; then
        wait_for_http "http://127.0.0.1:12436/v1/models" "Mac llama.cpp server" 240 || exit 1
      elif [[ "$mac_llm_mode" == "gemma" ]]; then
        wait_for_http "http://127.0.0.1:12437/v1/models" "Mac Gemma MLX server" 240 || exit 1
      elif [[ "$mac_llm_mode" == "gemma-gguf" ]]; then
        wait_for_http "http://127.0.0.1:12439/v1/models" "Mac Gemma GGUF server" 240 || exit 1
      else
        wait_for_http "http://127.0.0.1:12434/v1/models" "Mac MLX server" 240 || exit 1
      fi
    fi
    if _is_truthy "$mac_enable_stt"; then
      echo "Waiting for Mac STT server..."
      wait_for_http "http://127.0.0.1:4124/health" "Mac STT server" 120 || exit 1
    fi
    if _is_truthy "$mac_enable_tts"; then
      echo "Waiting for Mac TTS server..."
      wait_for_http "http://127.0.0.1:4123/health" "Mac TTS server" 240 || exit 1
    fi
  else
    echo ""
    echo "No Mac local model servers requested (use --with-local-llm-server / --with-voice-server to enable)."
  fi

  exit 0
fi

echo "Unknown platform: $PLATFORM (use 'nvidia' or 'mac')"
print_usage
exit 1
