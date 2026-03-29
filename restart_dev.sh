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

# ---------------------------------------------------------------------------
# Parse --platform; split remaining args into infra vs model-server buckets.
# ---------------------------------------------------------------------------
PLATFORM="${MUSE_PLATFORM:-nvidia}"

# Args safe to forward to nvidia/restart_dev.sh (backend, frontend, db)
INFRA_ARGS=()
# Model server intent (only used in mac mode)
mac_enable_llm=""
mac_enable_stt=""
mac_enable_tts=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)
      PLATFORM="$2"; shift 2 ;;
    --platform=*)
      PLATFORM="${1#--platform=}"; shift ;;

    # Capture model-server intent — needed for mac routing.
    # These are NOT forwarded to nvidia/restart_dev.sh in mac mode.
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
  --with-voice-server / --without-voice-server   (STT + TTS together)
  --with-voice-stt / --without-voice-stt
  --with-voice-tts / --without-voice-tts

Other flags are forwarded to the platform script (backend, frontend, db, etc.).
Run  nvidia/restart_dev.sh --help  for the full flag list.

Examples:
  ./restart_dev.sh --platform nvidia --with-local-llm-server --with-voice-server
  ./restart_dev.sh --platform mac --with-voice-server
  ./restart_dev.sh --platform mac --with-local-llm-server --without-voice-server
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

  echo "Platform: nvidia"
  exec "$ROOT_DIR/nvidia/restart_dev.sh" "${NVIDIA_ARGS[@]+"${NVIDIA_ARGS[@]}"}"
fi

# ---------------------------------------------------------------------------
# Mac / Apple Silicon — run backend/frontend via nvidia script (which handles
# Postgres, migrations, npm), then start MLX model servers.
# ---------------------------------------------------------------------------
if [[ "$PLATFORM" == "mac" || "$PLATFORM" == "apple" ]]; then
  echo "Platform: mac"

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

  # Run backend/frontend via nvidia script, explicitly suppressing all model servers.
  "$ROOT_DIR/nvidia/restart_dev.sh" \
    --without-local-llm-server \
    --without-local-image-server \
    --without-voice-stt \
    --without-voice-tts \
    --without-tool-server \
    "${INFRA_ARGS[@]+"${INFRA_ARGS[@]}"}"

  # Start Mac MLX model servers.
  MAC_SERVERS=()
  _is_truthy "$mac_enable_llm" && MAC_SERVERS+=(llm)
  _is_truthy "$mac_enable_stt" && MAC_SERVERS+=(stt)
  _is_truthy "$mac_enable_tts" && MAC_SERVERS+=(tts)

  if [[ ${#MAC_SERVERS[@]} -gt 0 ]]; then
    echo ""
    echo "Starting Mac MLX servers: ${MAC_SERVERS[*]}"
    "$ROOT_DIR/mac/start.sh" start "${MAC_SERVERS[@]}"
  else
    echo ""
    echo "No Mac MLX model servers requested (use --with-local-llm-server / --with-voice-server to enable)."
  fi

  exit 0
fi

echo "Unknown platform: $PLATFORM (use 'nvidia' or 'mac')"
print_usage
exit 1
