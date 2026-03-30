#!/usr/bin/env bash
# Start all Mac MLX model servers.
# Requires: pip install -r requirements.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../runtime/mac"
mkdir -p "${LOG_DIR}"

# Activate venv if present
if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

LLM_PORT="${LLM_PORT:-12434}"
LLAMA_LLM_PORT="${LLAMA_LLM_PORT:-12436}"
TTS_PORT="${TTS_PORT:-4123}"
STT_PORT="${STT_PORT:-4124}"

# Env var overrides for model IDs
export MLX_LLM_MODEL="${MLX_LLM_MODEL:-mlx-community/Qwen3.5-4B-MLX-4bit}"
export CHATTERBOX_MLX_MODEL="${CHATTERBOX_MLX_MODEL:-mlx-community/chatterbox-turbo-8bit}"
export PARAKEET_MLX_MODEL="${PARAKEET_MLX_MODEL:-mlx-community/parakeet-tdt-0.6b-v3}"
export VOICES_DIR="${VOICES_DIR:-${SCRIPT_DIR}/../voices}"

usage() {
  cat <<'EOF'
Usage: ./start.sh [command]

Commands:
  start    Start all servers (default)
  stop     Stop all running servers
  status   Show server health
  logs     Tail all logs

Servers:
  llama  (llama.cpp Qwen3.5-4B GGUF) → http://127.0.0.1:12436  (set MODEL_PATH)
  llm    (Qwen3.5-4B-MLX-4bit)       → http://127.0.0.1:12434
  tts    (chatterbox-turbo-8bit)      → http://127.0.0.1:4123
  stt    (parakeet-tdt-0.6b-v3)      → http://127.0.0.1:4124
EOF
}

start_server() {
  local name="$1"
  local script="$2"
  local port="$3"
  local log="${LOG_DIR}/${name}.log"
  local pid_file="${LOG_DIR}/${name}.pid"

  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "${name}: already running (pid $(cat "${pid_file}"))"
    return
  fi

  echo "Starting ${name} on port ${port}..."
  LISTEN_PORT="${port}" python "${SCRIPT_DIR}/${script}" >"${log}" 2>&1 &
  echo $! >"${pid_file}"
  echo "${name}: started (pid $!, log: ${log})"
}

start_llama_server() {
  local name="llama"
  local port="${LLAMA_LLM_PORT}"
  local log="${LOG_DIR}/${name}.log"
  local pid_file="${LOG_DIR}/${name}.pid"

  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "${name}: already running (pid $(cat "${pid_file}"))"
    return
  fi

  if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "${name}: MODEL_PATH is not set — skipping (set MODEL_PATH to a GGUF file path)"
    return 1
  fi

  echo "Starting ${name} on port ${port}..."
  LLAMA_PORT="${port}" sh "${SCRIPT_DIR}/llama_entrypoint.sh" >"${log}" 2>&1 &
  echo $! >"${pid_file}"
  echo "${name}: started (pid $!, log: ${log})"
}

stop_server() {
  local name="$1"
  local pid_file="${LOG_DIR}/${name}.pid"

  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping ${name} (pid ${pid})..."
      kill "${pid}"
    fi
    rm -f "${pid_file}"
  else
    echo "${name}: not running"
  fi
}

check_health() {
  local name="$1"
  local port="$2"
  local path="${3:-/health}"
  local url="http://127.0.0.1:${port}${path}"
  if curl -fsS "${url}" >/dev/null 2>&1; then
    echo "${name}: ready (${url})"
  else
    echo "${name}: not ready"
  fi
}

cmd_start() {
  # Optional args: list of servers to start (llm tts stt).
  # If none given, start all three.
  local servers=("$@")
  if [[ ${#servers[@]} -eq 0 ]]; then
    servers=(llm tts stt)
  fi

  for srv in "${servers[@]}"; do
    case "$srv" in
      llama) start_llama_server ;;
      llm)   start_server "llm" "llm_server.py" "${LLM_PORT}" ;;
      tts)   start_server "tts" "tts_mlx_server.py" "${TTS_PORT}" ;;
      stt)   start_server "stt" "stt_server.py" "${STT_PORT}" ;;
      *)     echo "Unknown server: $srv (use llama, llm, tts, stt)" ;;
    esac
  done
  echo ""
  echo "Note: models load in the background. Run './start.sh status' to check readiness."
}

cmd_stop() {
  stop_server "llama"
  stop_server "llm"
  stop_server "tts"
  stop_server "stt"
}

cmd_status() {
  check_health "LLM llama.cpp (Qwen3.5-4B)" "${LLAMA_LLM_PORT}"
  check_health "LLM MLX      (Qwen3.5-4B)"  "${LLM_PORT}"
  check_health "TTS          (Chatterbox)"   "${TTS_PORT}"
  check_health "STT          (Parakeet)"     "${STT_PORT}"
}

cmd_logs() {
  tail -f "${LOG_DIR}/llama.log" "${LOG_DIR}/llm.log" "${LOG_DIR}/tts.log" "${LOG_DIR}/stt.log" 2>/dev/null
}

main() {
  local command="${1:-start}"
  case "${command}" in
    start)   shift; cmd_start "$@" ;;
    stop)    cmd_stop ;;
    restart) shift; cmd_stop; cmd_start "$@" ;;
    status)  cmd_status ;;
    logs)    cmd_logs ;;
    -h|--help|help) usage ;;
    *)
      echo "Unknown command: ${command}"
      usage
      exit 1
      ;;
  esac
}

main "$@"
