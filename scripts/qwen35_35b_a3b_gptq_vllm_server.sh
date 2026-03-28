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

RUN_ROOT="${RUN_ROOT:-${LOCAL_MODEL_SERVING_ROOT}/runtime/qwen35_35b_a3b_gptq_vllm_server}"
MODEL_ROOT="${MODEL_ROOT:-${LOCAL_MODEL_SERVING_ROOT}/models}"
HF_CACHE_HOST="${HF_CACHE_HOST:-${HOME}/.cache/huggingface}"

VLLM_IMAGE="${VLLM_IMAGE:-dgx-vllm:latest}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-Qwen3.5-35B-A3B}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

LISTEN_IP="${LISTEN_IP:-127.0.0.1}"
LISTEN_PORT="${LISTEN_PORT:-12434}"
SERVER_URL="http://${LISTEN_IP}:${LISTEN_PORT}"

DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-${LOCAL_MODEL_SERVING_ROOT}/docker-compose.qwen35-35b-a3b-gptq-vllm.yml}"
DOCKER_PROJECT_NAME="${DOCKER_PROJECT_NAME:-muse-local-model-serving}"
DOCKER_SERVICE_NAME="${DOCKER_SERVICE_NAME:-qwen35-35b-a3b-gptq-vllm}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/qwen35_35b_a3b_gptq_vllm_server.sh <command>

Commands:
  start            Start vLLM server for Qwen3.5-35B-A3B-GPTQ-Int4.
  stop             Stop and remove running container.
  restart          Restart container.
  status           Show container state + endpoint readiness.
  logs             Tail container logs.
  print-env        Print resolved env/config values.

Environment overrides (optional):
  ENV_FILE, LOCAL_MODEL_SERVING_ROOT, RUN_ROOT, MODEL_ROOT
  HF_CACHE_HOST            Host path for HuggingFace cache (default: ~/.cache/huggingface)
  VLLM_IMAGE               Docker image (default: dgx-vllm:latest)
  VLLM_MODEL               HF model ID (default: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4)
  VLLM_SERVED_MODEL_NAME   Model name in /v1/models (default: Qwen3.5-35B-A3B)
  VLLM_PORT                Internal port (default: 8000)
  VLLM_MAX_MODEL_LEN       Max context length (default: 32768)
  VLLM_GPU_MEM_UTIL        GPU memory fraction (default: 0.90)
  LISTEN_IP, LISTEN_PORT   Host bind address (default: 127.0.0.1:12434)
  HF_TOKEN                 HuggingFace token (for gated models)
  DOCKER_COMPOSE_FILE, DOCKER_PROJECT_NAME, DOCKER_SERVICE_NAME
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
}

docker_compose() {
  docker compose -p "${DOCKER_PROJECT_NAME}" -f "${DOCKER_COMPOSE_FILE}" "$@"
}

check_ready() {
  curl -fsS "${SERVER_URL}/v1/models" >/dev/null
}

start_server() {
  require_cmd docker
  if [[ ! -f "${DOCKER_COMPOSE_FILE}" ]]; then
    echo "Docker compose file not found: ${DOCKER_COMPOSE_FILE}"
    exit 1
  fi

  mkdir -p "${RUN_ROOT}" "${MODEL_ROOT}" "${HF_CACHE_HOST}"

  VLLM_IMAGE="${VLLM_IMAGE}" \
  VLLM_MODEL="${VLLM_MODEL}" \
  VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME}" \
  VLLM_PORT="${VLLM_PORT}" \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
  VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL}" \
  VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}" \
  LISTEN_IP="${LISTEN_IP}" \
  LISTEN_PORT="${LISTEN_PORT}" \
  MODEL_ROOT_HOST="${MODEL_ROOT}" \
  HF_CACHE_HOST="${HF_CACHE_HOST}" \
  HF_TOKEN="${HF_TOKEN:-}" \
    docker_compose up -d "${DOCKER_SERVICE_NAME}"

  if check_ready; then
    echo "vLLM server is ready at ${SERVER_URL}/v1"
  else
    echo "Container started, endpoint still warming up (model download + load may take a while)."
    echo "Check readiness with: ./scripts/qwen35_35b_a3b_gptq_vllm_server.sh status"
  fi
}

stop_server() {
  require_cmd docker
  if [[ -f "${DOCKER_COMPOSE_FILE}" ]]; then
    docker_compose stop "${DOCKER_SERVICE_NAME}" >/dev/null 2>&1 || true
    docker_compose rm -f "${DOCKER_SERVICE_NAME}" >/dev/null 2>&1 || true
  fi
  echo "Stopped vLLM docker service."
}

print_status() {
  require_cmd docker
  if [[ -f "${DOCKER_COMPOSE_FILE}" ]]; then
    local container_id
    container_id="$(docker_compose ps -q "${DOCKER_SERVICE_NAME}" 2>/dev/null || true)"

    if [[ -n "${container_id}" ]] && [[ "$(docker inspect -f '{{.State.Running}}' "${container_id}" 2>/dev/null || echo false)" == "true" ]]; then
      echo "vLLM: running (container ${container_id})"
    else
      echo "vLLM: stopped"
    fi
  else
    echo "Process: unknown (compose file missing: ${DOCKER_COMPOSE_FILE})"
  fi

  if check_ready; then
    echo "Endpoint: ready (${SERVER_URL}/v1/models)"
  else
    echo "Endpoint: not ready (${SERVER_URL}/v1/models)"
  fi
}

print_env() {
  cat <<EOF
SCRIPT_DIR=${SCRIPT_DIR}
LOCAL_MODEL_SERVING_ROOT=${LOCAL_MODEL_SERVING_ROOT}
ENV_FILE=${ENV_FILE}
RUN_ROOT=${RUN_ROOT}
MODEL_ROOT=${MODEL_ROOT}
HF_CACHE_HOST=${HF_CACHE_HOST}
VLLM_IMAGE=${VLLM_IMAGE}
VLLM_MODEL=${VLLM_MODEL}
VLLM_SERVED_MODEL_NAME=${VLLM_SERVED_MODEL_NAME}
VLLM_PORT=${VLLM_PORT}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL}
VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}
LISTEN_IP=${LISTEN_IP}
LISTEN_PORT=${LISTEN_PORT}
SERVER_URL=${SERVER_URL}
DOCKER_COMPOSE_FILE=${DOCKER_COMPOSE_FILE}
DOCKER_PROJECT_NAME=${DOCKER_PROJECT_NAME}
DOCKER_SERVICE_NAME=${DOCKER_SERVICE_NAME}
EOF
}

main() {
  local command="${1:-}"
  case "${command}" in
    start)
      start_server
      ;;
    stop)
      stop_server
      ;;
    restart)
      stop_server
      start_server
      ;;
    status)
      print_status
      ;;
    logs)
      require_cmd docker
      docker_compose logs -f "${DOCKER_SERVICE_NAME}"
      ;;
    print-env)
      print_env
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      echo "Unknown command: ${command}"
      usage
      exit 1
      ;;
  esac
}

main "$@"
