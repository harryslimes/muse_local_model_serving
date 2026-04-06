#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_MODEL_SERVING_ROOT="${LOCAL_MODEL_SERVING_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${LOCAL_MODEL_SERVING_ROOT}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

DOCKER_PROJECT_NAME="${DOCKER_PROJECT_NAME:-muse-local-model-serving}"

PARAKEET_COMPOSE_FILE="${LOCAL_MODEL_SERVING_ROOT}/docker-compose.parakeet-stt.yml"
CHATTERBOX_COMPOSE_FILE="${LOCAL_MODEL_SERVING_ROOT}/docker-compose.chatterbox-tts.yml"

PARAKEET_PORT="${PARAKEET_PORT:-4124}"
CHATTERBOX_PORT="${CHATTERBOX_PORT:-4123}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/voice_servers.sh <command>

Commands:
  setup            Build Docker images for both STT and TTS servers.
  start            Start both voice servers.
  stop             Stop both voice servers.
  restart          Restart both servers.
  status           Show process + endpoint health for both.
  logs-stt         Tail Parakeet STT logs.
  logs-tts         Tail Chatterbox TTS logs.
  print-env        Print relevant env config values.

Environment overrides (optional):
  PARAKEET_PORT (default: 4124)
  CHATTERBOX_PORT (default: 4123)
  PARAKEET_BIND_IP, CHATTERBOX_BIND_IP (default: 127.0.0.1)
  PARAKEET_MODEL (default: nvidia/parakeet-tdt-0.6b-v2)
  MODEL_ROOT_HOST (default: ./models)
  VOICES_DIR_HOST (default: ./voices)
  HF_TOKEN
EOF
}

docker_compose_stt() {
  docker compose -p "${DOCKER_PROJECT_NAME}" -f "${PARAKEET_COMPOSE_FILE}" "$@"
}

docker_compose_tts() {
  docker compose -p "${DOCKER_PROJECT_NAME}" -f "${CHATTERBOX_COMPOSE_FILE}" "$@"
}

check_stt_ready() {
  curl -fsS "http://127.0.0.1:${PARAKEET_PORT}/health" >/dev/null 2>&1
}

check_tts_ready() {
  curl -fsS "http://127.0.0.1:${CHATTERBOX_PORT}/health" >/dev/null 2>&1
}

cmd_setup() {
  echo "Building Parakeet STT image..."
  docker_compose_stt build parakeet-stt
  echo "Building Chatterbox TTS image..."
  docker_compose_tts build chatterbox-tts
  echo "Done."
}

cmd_start() {
  echo "Starting Parakeet STT..."
  docker_compose_stt up -d --build parakeet-stt
  echo "Starting Chatterbox TTS..."
  docker_compose_tts up -d --build chatterbox-tts

  echo ""
  if check_stt_ready; then
    echo "Parakeet STT: ready (http://127.0.0.1:${PARAKEET_PORT})"
  else
    echo "Parakeet STT: started, still warming up"
  fi
  if check_tts_ready; then
    echo "Chatterbox TTS: ready (http://127.0.0.1:${CHATTERBOX_PORT})"
  else
    echo "Chatterbox TTS: started, still warming up"
  fi
}

cmd_stop() {
  echo "Stopping Parakeet STT..."
  docker_compose_stt stop parakeet-stt 2>/dev/null || true
  docker_compose_stt rm -f parakeet-stt 2>/dev/null || true
  echo "Stopping Chatterbox TTS..."
  docker_compose_tts stop chatterbox-tts 2>/dev/null || true
  docker_compose_tts rm -f chatterbox-tts 2>/dev/null || true
  echo "Stopped."
}

cmd_status() {
  echo "=== Parakeet STT ==="
  if check_stt_ready; then
    echo "  Endpoint: ready (http://127.0.0.1:${PARAKEET_PORT})"
  else
    echo "  Endpoint: not ready"
  fi

  echo "=== Chatterbox TTS ==="
  if check_tts_ready; then
    echo "  Endpoint: ready (http://127.0.0.1:${CHATTERBOX_PORT})"
  else
    echo "  Endpoint: not ready"
  fi
}

cmd_print_env() {
  cat <<EOF
LOCAL_MODEL_SERVING_ROOT=${LOCAL_MODEL_SERVING_ROOT}
PARAKEET_PORT=${PARAKEET_PORT}
CHATTERBOX_PORT=${CHATTERBOX_PORT}
PARAKEET_MODEL=${PARAKEET_MODEL:-nvidia/parakeet-tdt-0.6b-v2}
MODEL_ROOT_HOST=${MODEL_ROOT_HOST:-./models}
VOICES_DIR_HOST=${VOICES_DIR_HOST:-./voices}
HF_TOKEN=${HF_TOKEN:+(set)}
EOF
}

main() {
  local command="${1:-}"
  case "${command}" in
    setup)          cmd_setup ;;
    start)          cmd_start ;;
    stop)           cmd_stop ;;
    restart)        cmd_stop; cmd_start ;;
    status)         cmd_status ;;
    logs-stt)       docker_compose_stt logs -f parakeet-stt ;;
    logs-tts)       docker_compose_tts logs -f chatterbox-tts ;;
    print-env)      cmd_print_env ;;
    -h|--help|help|"") usage ;;
    *)
      echo "Unknown command: ${command}"
      usage
      exit 1
      ;;
  esac
}

main "$@"
