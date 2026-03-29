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

RUN_ROOT="${RUN_ROOT:-${LOCAL_MODEL_SERVING_ROOT}/runtime/qwen35_35b_a3b_server}"
MODEL_ROOT="${MODEL_ROOT:-${LOCAL_MODEL_SERVING_ROOT}/models}"

MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.5-35B-A3B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf}"
MODEL_PATH="${MODEL_PATH:-/models/${MODEL_REPO}/${MODEL_FILE}}"

LISTEN_IP="${LISTEN_IP:-127.0.0.1}"
LISTEN_PORT="${LISTEN_PORT:-12434}"
SERVER_URL="http://${LISTEN_IP}:${LISTEN_PORT}"

LLAMA_PORT="${LLAMA_PORT:-8080}"
LLAMA_PORT_HOST="${LLAMA_PORT_HOST:-18080}"
LLAMA_BIND_IP="${LLAMA_BIND_IP:-127.0.0.1}"
CONTEXT_SIZE="${CONTEXT_SIZE:-32768}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"
REASONING_BUDGET="${REASONING_BUDGET:--1}"
LLAMA_ENABLE_JINJA="${LLAMA_ENABLE_JINJA:-true}"
LLAMA_DISABLE_NHFR="${LLAMA_DISABLE_NHFR:-false}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-/app/chat-template/qwen3.jinja}"
LLAMA_EXTRA_FLAGS="${LLAMA_EXTRA_FLAGS:-}"

PROXY_MODEL_NAME="${PROXY_MODEL_NAME:-Qwen3.5-35B-A3B}"
FORCE_MODEL_NAME="${FORCE_MODEL_NAME:-true}"
ENABLE_THINKING_DEFAULT="${ENABLE_THINKING_DEFAULT:-true}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-600}"

DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-${LOCAL_MODEL_SERVING_ROOT}/docker-compose.qwen35-35b-a3b.yml}"
DOCKER_PROJECT_NAME="${DOCKER_PROJECT_NAME:-muse-local-model-serving}"
DOCKER_DOWNLOADER_SERVICE_NAME="${DOCKER_DOWNLOADER_SERVICE_NAME:-qwen35-35b-a3b-downloader}"
DOCKER_LLAMA_SERVICE_NAME="${DOCKER_LLAMA_SERVICE_NAME:-qwen35-35b-a3b-llama}"
DOCKER_PROXY_SERVICE_NAME="${DOCKER_PROXY_SERVICE_NAME:-qwen35-35b-a3b-proxy}"

# Newer llama-server builds only accept --reasoning-budget of -1 or 0.
if [[ "${REASONING_BUDGET}" != "-1" && "${REASONING_BUDGET}" != "0" ]]; then
  echo "Warning: unsupported REASONING_BUDGET='${REASONING_BUDGET}', forcing -1"
  REASONING_BUDGET="-1"
fi

usage() {
  cat <<'EOF'
Usage:
  ./scripts/qwen35_35b_a3b_server.sh <command>

Commands:
  setup            Build helper images (proxy + model-downloader).
  download-models  Download the configured GGUF model into MODEL_ROOT.
  start            Start llama.cpp + OpenAI-compatible proxy with Docker.
  stop             Stop and remove running containers.
  restart          Restart containers.
  status           Show container state + endpoint readiness.
  logs             Tail container logs.
  print-env        Print resolved env/config values.

Environment overrides (optional):
  ENV_FILE (default: ${LOCAL_MODEL_SERVING_ROOT}/.env)
  LOCAL_MODEL_SERVING_ROOT, RUN_ROOT, MODEL_ROOT
  MODEL_REPO, MODEL_FILE, MODEL_PATH
  LISTEN_IP, LISTEN_PORT
  LLAMA_PORT, LLAMA_PORT_HOST, LLAMA_BIND_IP, CONTEXT_SIZE, N_GPU_LAYERS
  TEMPERATURE, TOP_P, REASONING_FORMAT, REASONING_BUDGET
  LLAMA_ENABLE_JINJA, LLAMA_DISABLE_NHFR
  CHAT_TEMPLATE_FILE, LLAMA_EXTRA_FLAGS
  PROXY_MODEL_NAME, FORCE_MODEL_NAME, ENABLE_THINKING_DEFAULT
  REQUEST_TIMEOUT_SECONDS
  HF_TOKEN, UPSTREAM_API_KEY
  DOCKER_COMPOSE_FILE, DOCKER_PROJECT_NAME
  DOCKER_DOWNLOADER_SERVICE_NAME, DOCKER_LLAMA_SERVICE_NAME, DOCKER_PROXY_SERVICE_NAME
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

to_host_model_path() {
  local container_path="$1"
  if [[ "${container_path}" == /models/* ]]; then
    echo "${MODEL_ROOT}/${container_path#/models/}"
    return 0
  fi
  echo ""
  return 0
}

check_ready() {
  curl -fsS "${SERVER_URL}/v1/models" >/dev/null
}

setup_stack() {
  require_cmd docker
  if [[ ! -f "${DOCKER_COMPOSE_FILE}" ]]; then
    echo "Docker compose file not found: ${DOCKER_COMPOSE_FILE}"
    exit 1
  fi
  docker_compose build "${DOCKER_PROXY_SERVICE_NAME}" "${DOCKER_DOWNLOADER_SERVICE_NAME}"
  echo "Docker helper images built."
}

download_models() {
  require_cmd docker
  mkdir -p "${RUN_ROOT}" "${MODEL_ROOT}"
  if [[ ! -f "${DOCKER_COMPOSE_FILE}" ]]; then
    echo "Docker compose file not found: ${DOCKER_COMPOSE_FILE}"
    exit 1
  fi

  MODEL_ROOT_HOST="${MODEL_ROOT}" \
  MODEL_REPO="${MODEL_REPO}" \
  MODEL_FILE="${MODEL_FILE}" \
  HF_TOKEN="${HF_TOKEN:-}" \
    docker_compose run --rm "${DOCKER_DOWNLOADER_SERVICE_NAME}"

  local expected
  expected="$(to_host_model_path "${MODEL_PATH}")"
  if [[ -n "${expected}" && -f "${expected}" ]]; then
    echo "Model file ready at ${expected}"
  else
    echo "Model download finished, but expected path is missing: ${expected:-<unmappable MODEL_PATH>}"
    echo "Configured MODEL_PATH=${MODEL_PATH}"
    exit 1
  fi
}

start_server() {
  require_cmd docker
  if [[ ! -f "${DOCKER_COMPOSE_FILE}" ]]; then
    echo "Docker compose file not found: ${DOCKER_COMPOSE_FILE}"
    exit 1
  fi

  if [[ "${MODEL_PATH}" != /models/* ]]; then
    echo "MODEL_PATH must be inside /models when running Docker mode."
    echo "Current MODEL_PATH=${MODEL_PATH}"
    exit 1
  fi

  mkdir -p "${RUN_ROOT}" "${MODEL_ROOT}"

  local host_model_path
  host_model_path="$(to_host_model_path "${MODEL_PATH}")"
  if [[ ! -f "${host_model_path}" ]]; then
    echo "Model file missing: ${host_model_path}"
    echo "Downloading configured model (${MODEL_REPO}:${MODEL_FILE})..."
    download_models
  fi

  MODEL_ROOT_HOST="${MODEL_ROOT}" \
  RUN_ROOT_HOST="${RUN_ROOT}" \
  MODEL_PATH="${MODEL_PATH}" \
  MODEL_REPO="${MODEL_REPO}" \
  MODEL_FILE="${MODEL_FILE}" \
  LISTEN_PORT="${LISTEN_PORT}" \
  LLAMA_PORT="${LLAMA_PORT}" \
  LLAMA_PORT_HOST="${LLAMA_PORT_HOST}" \
  LLAMA_BIND_IP="${LLAMA_BIND_IP}" \
  CONTEXT_SIZE="${CONTEXT_SIZE}" \
  N_GPU_LAYERS="${N_GPU_LAYERS}" \
  TEMPERATURE="${TEMPERATURE}" \
  TOP_P="${TOP_P}" \
  REASONING_FORMAT="${REASONING_FORMAT}" \
  REASONING_BUDGET="${REASONING_BUDGET}" \
  LLAMA_ENABLE_JINJA="${LLAMA_ENABLE_JINJA}" \
  LLAMA_DISABLE_NHFR="${LLAMA_DISABLE_NHFR}" \
  CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE}" \
  LLAMA_EXTRA_FLAGS="${LLAMA_EXTRA_FLAGS}" \
  PROXY_MODEL_NAME="${PROXY_MODEL_NAME}" \
  FORCE_MODEL_NAME="${FORCE_MODEL_NAME}" \
  ENABLE_THINKING_DEFAULT="${ENABLE_THINKING_DEFAULT}" \
  REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  UPSTREAM_API_KEY="${UPSTREAM_API_KEY:-}" \
    docker_compose up -d --build "${DOCKER_LLAMA_SERVICE_NAME}" "${DOCKER_PROXY_SERVICE_NAME}"

  if check_ready; then
    echo "Qwen3.5 service is ready at ${SERVER_URL}/v1"
  else
    echo "Containers started, endpoint still warming up."
    echo "Check readiness with: ./scripts/qwen35_35b_a3b_server.sh status"
  fi
}

stop_server() {
  require_cmd docker
  if [[ -f "${DOCKER_COMPOSE_FILE}" ]]; then
    docker_compose stop "${DOCKER_PROXY_SERVICE_NAME}" "${DOCKER_LLAMA_SERVICE_NAME}" >/dev/null 2>&1 || true
    docker_compose rm -f "${DOCKER_PROXY_SERVICE_NAME}" "${DOCKER_LLAMA_SERVICE_NAME}" >/dev/null 2>&1 || true
  fi
  echo "Stopped docker services."
}

print_status() {
  require_cmd docker
  if [[ -f "${DOCKER_COMPOSE_FILE}" ]]; then
    local proxy_id llama_id
    proxy_id="$(docker_compose ps -q "${DOCKER_PROXY_SERVICE_NAME}" 2>/dev/null || true)"
    llama_id="$(docker_compose ps -q "${DOCKER_LLAMA_SERVICE_NAME}" 2>/dev/null || true)"

    if [[ -n "${proxy_id}" ]] && [[ "$(docker inspect -f '{{.State.Running}}' "${proxy_id}" 2>/dev/null || echo false)" == "true" ]]; then
      echo "Proxy: running (container ${proxy_id})"
    else
      echo "Proxy: stopped"
    fi

    if [[ -n "${llama_id}" ]] && [[ "$(docker inspect -f '{{.State.Running}}' "${llama_id}" 2>/dev/null || echo false)" == "true" ]]; then
      echo "Llama: running (container ${llama_id})"
    else
      echo "Llama: stopped"
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
MODEL_REPO=${MODEL_REPO}
MODEL_FILE=${MODEL_FILE}
MODEL_PATH=${MODEL_PATH}
LISTEN_IP=${LISTEN_IP}
LISTEN_PORT=${LISTEN_PORT}
SERVER_URL=${SERVER_URL}
LLAMA_PORT=${LLAMA_PORT}
LLAMA_PORT_HOST=${LLAMA_PORT_HOST}
LLAMA_BIND_IP=${LLAMA_BIND_IP}
CONTEXT_SIZE=${CONTEXT_SIZE}
N_GPU_LAYERS=${N_GPU_LAYERS}
TEMPERATURE=${TEMPERATURE}
TOP_P=${TOP_P}
REASONING_FORMAT=${REASONING_FORMAT}
REASONING_BUDGET=${REASONING_BUDGET}
LLAMA_ENABLE_JINJA=${LLAMA_ENABLE_JINJA}
LLAMA_DISABLE_NHFR=${LLAMA_DISABLE_NHFR}
CHAT_TEMPLATE_FILE=${CHAT_TEMPLATE_FILE}
LLAMA_EXTRA_FLAGS=${LLAMA_EXTRA_FLAGS}
PROXY_MODEL_NAME=${PROXY_MODEL_NAME}
FORCE_MODEL_NAME=${FORCE_MODEL_NAME}
ENABLE_THINKING_DEFAULT=${ENABLE_THINKING_DEFAULT}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS}
DOCKER_COMPOSE_FILE=${DOCKER_COMPOSE_FILE}
DOCKER_PROJECT_NAME=${DOCKER_PROJECT_NAME}
DOCKER_DOWNLOADER_SERVICE_NAME=${DOCKER_DOWNLOADER_SERVICE_NAME}
DOCKER_LLAMA_SERVICE_NAME=${DOCKER_LLAMA_SERVICE_NAME}
DOCKER_PROXY_SERVICE_NAME=${DOCKER_PROXY_SERVICE_NAME}
EOF
}

main() {
  local command="${1:-}"
  case "${command}" in
    setup)
      setup_stack
      ;;
    download-models)
      download_models
      ;;
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
      docker_compose logs -f "${DOCKER_LLAMA_SERVICE_NAME}" "${DOCKER_PROXY_SERVICE_NAME}"
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
