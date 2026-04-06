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

RUN_ROOT="${RUN_ROOT:-${LOCAL_MODEL_SERVING_ROOT}/runtime/flux2_klein_server}"
MODEL_ROOT="${MODEL_ROOT:-${LOCAL_MODEL_SERVING_ROOT}/models}"
SOURCE_ROOT="${SOURCE_ROOT:-${LOCAL_MODEL_SERVING_ROOT}/sources}"
SDCPP_DIR="${SDCPP_DIR:-${SOURCE_ROOT}/stable-diffusion.cpp}"
SDCPP_BUILD_DIR="${SDCPP_BUILD_DIR:-${SDCPP_DIR}/build}"
SD_SERVER_BIN="${SD_SERVER_BIN:-${SDCPP_BUILD_DIR}/bin/sd-server}"
PID_FILE="${RUN_ROOT}/sd-server.pid"
LOG_FILE="${RUN_ROOT}/sd-server.log"

LISTEN_IP="${LISTEN_IP:-127.0.0.1}"
LISTEN_PORT="${LISTEN_PORT:-1234}"
SERVER_URL="http://${LISTEN_IP}:${LISTEN_PORT}"
DOCKER_MODE="${DOCKER_MODE:-true}"
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-${LOCAL_MODEL_SERVING_ROOT}/docker-compose.flux2.yml}"
DOCKER_PROJECT_NAME="${DOCKER_PROJECT_NAME:-muse-local-model-serving}"
DOCKER_SERVICE_NAME="${DOCKER_SERVICE_NAME:-flux2-image-server}"
CONTAINER_LISTEN_IP="${CONTAINER_LISTEN_IP:-0.0.0.0}"

FLUX2_KLEIN_PROFILE="${FLUX2_KLEIN_PROFILE:-9b-fp8}"

resolve_model_defaults() {
  case "${FLUX2_KLEIN_PROFILE}" in
    4b)
      DEFAULT_DIFFUSION_MODEL="${MODEL_ROOT}/Comfy-Org/vae-text-encorder-for-flux-klein-4b/split_files/diffusion_models/flux-2-klein-4b.safetensors"
      DEFAULT_VAE_MODEL="${MODEL_ROOT}/Comfy-Org/vae-text-encorder-for-flux-klein-4b/split_files/vae/flux2-vae.safetensors"
      DEFAULT_LLM_MODEL="${MODEL_ROOT}/Comfy-Org/vae-text-encorder-for-flux-klein-4b/split_files/text_encoders/qwen_3_4b.safetensors"
      ;;
    4b-fp8)
      DEFAULT_DIFFUSION_MODEL="${MODEL_ROOT}/black-forest-labs/FLUX.2-klein-4b-fp8/flux-2-klein-4b-fp8.safetensors"
      DEFAULT_VAE_MODEL="${MODEL_ROOT}/Comfy-Org/vae-text-encorder-for-flux-klein-4b/split_files/vae/flux2-vae.safetensors"
      DEFAULT_LLM_MODEL="${MODEL_ROOT}/Comfy-Org/vae-text-encorder-for-flux-klein-4b/split_files/text_encoders/qwen_3_4b.safetensors"
      ;;
    9b-fp8)
      DEFAULT_DIFFUSION_MODEL="${MODEL_ROOT}/black-forest-labs/FLUX.2-klein-9b-fp8/flux-2-klein-9b-fp8.safetensors"
      DEFAULT_VAE_MODEL="${MODEL_ROOT}/Comfy-Org/vae-text-encorder-for-flux-klein-4b/split_files/vae/flux2-vae.safetensors"
      DEFAULT_LLM_MODEL="${MODEL_ROOT}/Comfy-Org/vae-text-encorder-for-flux-klein-9b/split_files/text_encoders/qwen_3_8b.safetensors"
      ;;
    *)
      echo "Unsupported FLUX2_KLEIN_PROFILE='${FLUX2_KLEIN_PROFILE}'"
      echo "Supported profiles: 4b, 4b-fp8, 9b-fp8"
      exit 1
      ;;
  esac
}

resolve_model_defaults

DIFFUSION_MODEL="${DIFFUSION_MODEL:-${DEFAULT_DIFFUSION_MODEL}}"
VAE_MODEL="${VAE_MODEL:-${DEFAULT_VAE_MODEL}}"
LLM_MODEL="${LLM_MODEL:-${DEFAULT_LLM_MODEL}}"

CFG_SCALE="${CFG_SCALE:-1.0}"
SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
STEPS="${STEPS:-4}"
OFFLOAD_TO_CPU="${OFFLOAD_TO_CPU:-false}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/flux2_klein_server.sh <command>

Commands:
  setup            Build Docker image (or local sd-server if DOCKER_MODE=false).
  download-models  Ensure required files for selected FLUX2_KLEIN_PROFILE.
  start            Start persistent sd-server (keeps weights loaded).
  stop             Stop running sd-server.
  restart          Restart server.
  status           Show process + endpoint health.
  logs             Tail server logs.
  print-env        Print relevant env config values.

Environment overrides (optional):
  ENV_FILE (default: ${LOCAL_MODEL_SERVING_ROOT}/.env)
  LOCAL_MODEL_SERVING_ROOT, RUN_ROOT
  MODEL_ROOT, SOURCE_ROOT, SDCPP_DIR, SD_SERVER_BIN
  FLUX2_KLEIN_PROFILE=4b|4b-fp8|9b-fp8
  DOCKER_MODE=true|false
  DOCKER_COMPOSE_FILE, DOCKER_PROJECT_NAME, DOCKER_SERVICE_NAME
  CONTAINER_LISTEN_IP (default: 0.0.0.0 for container bind)
  DIFFUSION_MODEL, VAE_MODEL, LLM_MODEL
  HF_TOKEN
  LISTEN_IP, LISTEN_PORT
  CFG_SCALE, SAMPLING_METHOD, STEPS
  OFFLOAD_TO_CPU=true|false
  SDCPP_EXTRA_FLAGS="--threads 12 ..."
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
}

is_truthy() {
  local value="${1:-}"
  value="$(echo "$value" | tr '[:upper:]' '[:lower:]')"
  [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "on" ]]
}

docker_compose() {
  docker compose -p "${DOCKER_PROJECT_NAME}" -f "${DOCKER_COMPOSE_FILE}" "$@"
}

to_container_model_path() {
  local path="${1:-}"
  if [[ -z "${path}" ]]; then
    echo ""
    return 0
  fi
  if [[ "${path}" == /models/* ]]; then
    echo "${path}"
    return 0
  fi
  if [[ "${path}" == "${MODEL_ROOT}/"* ]]; then
    echo "/models/${path#"${MODEL_ROOT}/"}"
    return 0
  fi
  echo "${path}"
}

hf_download() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$@"
    return 0
  fi
  if command -v hf >/dev/null 2>&1; then
    hf download "$@"
    return 0
  fi
  if command -v uvx >/dev/null 2>&1; then
    uvx --from huggingface_hub hf download "$@"
    return 0
  fi
  echo "Missing Hugging Face CLI. Install 'huggingface_hub' (hf) or provide 'huggingface-cli'."
  exit 1
}

is_running() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  if [[ -z "${pid}" ]]; then
    return 1
  fi
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

download_models() {
  if is_truthy "${DOCKER_MODE}"; then
    require_cmd docker
    if [[ ! -f "${DOCKER_COMPOSE_FILE}" ]]; then
      echo "Docker compose file not found: ${DOCKER_COMPOSE_FILE}"
      exit 1
    fi

    mkdir -p "${RUN_ROOT}" "${MODEL_ROOT}"
    local docker_diffusion_model docker_vae_model docker_llm_model
    docker_diffusion_model="$(to_container_model_path "${DIFFUSION_MODEL}")"
    docker_vae_model="$(to_container_model_path "${VAE_MODEL}")"
    docker_llm_model="$(to_container_model_path "${LLM_MODEL}")"

    FLUX2_KLEIN_PROFILE="${FLUX2_KLEIN_PROFILE}" \
    MODEL_ROOT_HOST="${MODEL_ROOT}" \
    RUN_ROOT_HOST="${RUN_ROOT}" \
    DIFFUSION_MODEL="${docker_diffusion_model}" \
    VAE_MODEL="${docker_vae_model}" \
    LLM_MODEL="${docker_llm_model}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    DOWNLOAD_ONLY="true" \
      docker_compose run --rm "${DOCKER_SERVICE_NAME}"

    echo "Model files for profile '${FLUX2_KLEIN_PROFILE}' ensured under ${MODEL_ROOT}"
    return 0
  fi

  mkdir -p "${MODEL_ROOT}"

  # Shared FLUX2 VAE (reused across profiles).
  hf_download \
    Comfy-Org/vae-text-encorder-for-flux-klein-4b \
    split_files/vae/flux2-vae.safetensors \
    --local-dir "${MODEL_ROOT}/Comfy-Org/vae-text-encorder-for-flux-klein-4b"

  case "${FLUX2_KLEIN_PROFILE}" in
    4b)
      hf_download \
        Comfy-Org/vae-text-encorder-for-flux-klein-4b \
        split_files/diffusion_models/flux-2-klein-4b.safetensors \
        --local-dir "${MODEL_ROOT}/Comfy-Org/vae-text-encorder-for-flux-klein-4b"
      hf_download \
        Comfy-Org/vae-text-encorder-for-flux-klein-4b \
        split_files/text_encoders/qwen_3_4b.safetensors \
        --local-dir "${MODEL_ROOT}/Comfy-Org/vae-text-encorder-for-flux-klein-4b"
      ;;
    4b-fp8)
      hf_download \
        black-forest-labs/FLUX.2-klein-4b-fp8 \
        flux-2-klein-4b-fp8.safetensors \
        --local-dir "${MODEL_ROOT}/black-forest-labs/FLUX.2-klein-4b-fp8"
      hf_download \
        Comfy-Org/vae-text-encorder-for-flux-klein-4b \
        split_files/text_encoders/qwen_3_4b.safetensors \
        --local-dir "${MODEL_ROOT}/Comfy-Org/vae-text-encorder-for-flux-klein-4b"
      ;;
    9b-fp8)
      hf_download \
        black-forest-labs/FLUX.2-klein-9b-fp8 \
        flux-2-klein-9b-fp8.safetensors \
        --local-dir "${MODEL_ROOT}/black-forest-labs/FLUX.2-klein-9b-fp8"
      hf_download \
        Comfy-Org/vae-text-encorder-for-flux-klein-9b \
        split_files/text_encoders/qwen_3_8b.safetensors \
        --local-dir "${MODEL_ROOT}/Comfy-Org/vae-text-encorder-for-flux-klein-9b"
      ;;
  esac

  echo "Model files for profile '${FLUX2_KLEIN_PROFILE}' downloaded under ${MODEL_ROOT}"
}

setup_sdcpp() {
  if is_truthy "${DOCKER_MODE}"; then
    require_cmd docker
    if [[ ! -f "${DOCKER_COMPOSE_FILE}" ]]; then
      echo "Docker compose file not found: ${DOCKER_COMPOSE_FILE}"
      exit 1
    fi
    docker_compose build "${DOCKER_SERVICE_NAME}"
    echo "Docker image built for ${DOCKER_SERVICE_NAME}"
    return 0
  fi

  require_cmd git
  require_cmd cmake

  mkdir -p "${RUN_ROOT}" "${SOURCE_ROOT}" "${MODEL_ROOT}"
  if [[ ! -d "${SDCPP_DIR}/.git" ]]; then
    git clone --recursive https://github.com/leejet/stable-diffusion.cpp.git "${SDCPP_DIR}"
  else
    git -C "${SDCPP_DIR}" pull --ff-only
    git -C "${SDCPP_DIR}" submodule update --init --recursive
  fi

  cmake -S "${SDCPP_DIR}" -B "${SDCPP_BUILD_DIR}" -DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "${SDCPP_BUILD_DIR}" --config Release -j "$(nproc)"

  if [[ ! -x "${SD_SERVER_BIN}" ]]; then
    echo "Build finished but sd-server binary not found at ${SD_SERVER_BIN}"
    exit 1
  fi

  echo "Built sd-server at ${SD_SERVER_BIN}"
}

check_ready() {
  curl -fsS "${SERVER_URL}/v1/models" >/dev/null
}

start_server() {
  if is_truthy "${DOCKER_MODE}"; then
    require_cmd docker
    if [[ ! -f "${DOCKER_COMPOSE_FILE}" ]]; then
      echo "Docker compose file not found: ${DOCKER_COMPOSE_FILE}"
      exit 1
    fi

    mkdir -p "${RUN_ROOT}" "${MODEL_ROOT}"
    local docker_diffusion_model docker_vae_model docker_llm_model
    docker_diffusion_model="$(to_container_model_path "${DIFFUSION_MODEL}")"
    docker_vae_model="$(to_container_model_path "${VAE_MODEL}")"
    docker_llm_model="$(to_container_model_path "${LLM_MODEL}")"

    FLUX2_KLEIN_PROFILE="${FLUX2_KLEIN_PROFILE}" \
    MODEL_ROOT_HOST="${MODEL_ROOT}" \
    RUN_ROOT_HOST="${RUN_ROOT}" \
    LISTEN_IP="${CONTAINER_LISTEN_IP}" \
    LISTEN_PORT="${LISTEN_PORT}" \
    CFG_SCALE="${CFG_SCALE}" \
    SAMPLING_METHOD="${SAMPLING_METHOD}" \
    STEPS="${STEPS}" \
    OFFLOAD_TO_CPU="${OFFLOAD_TO_CPU}" \
    DIFFUSION_MODEL="${docker_diffusion_model}" \
    VAE_MODEL="${docker_vae_model}" \
    LLM_MODEL="${docker_llm_model}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    SDCPP_EXTRA_FLAGS="${SDCPP_EXTRA_FLAGS:-}" \
      docker_compose up -d --build "${DOCKER_SERVICE_NAME}"

    if check_ready; then
      echo "Dockerized sd-server is running at ${SERVER_URL}"
    else
      echo "Docker container started, endpoint still warming up."
      echo "Check readiness with: ./scripts/flux2_klein_server.sh status"
    fi
    return 0
  fi

  mkdir -p "${RUN_ROOT}" "${MODEL_ROOT}" "${SOURCE_ROOT}"
  if is_running; then
    echo "sd-server is already running with PID $(cat "${PID_FILE}")"
    return 0
  fi

  if [[ ! -x "${SD_SERVER_BIN}" ]]; then
    echo "sd-server binary missing: ${SD_SERVER_BIN}"
    echo "Run: ./scripts/flux2_klein_server.sh setup"
    exit 1
  fi
  if [[ ! -f "${DIFFUSION_MODEL}" ]]; then
    echo "Diffusion model file not found: ${DIFFUSION_MODEL}"
    echo "Run: ./scripts/flux2_klein_server.sh download-models"
    exit 1
  fi
  if [[ ! -f "${VAE_MODEL}" ]]; then
    echo "VAE file not found: ${VAE_MODEL}"
    exit 1
  fi
  if [[ ! -f "${LLM_MODEL}" ]]; then
    echo "LLM encoder file not found: ${LLM_MODEL}"
    exit 1
  fi

  local cmd=(
    "${SD_SERVER_BIN}"
    --listen-ip "${LISTEN_IP}"
    --listen-port "${LISTEN_PORT}"
    --diffusion-model "${DIFFUSION_MODEL}"
    --vae "${VAE_MODEL}"
    --llm "${LLM_MODEL}"
    --cfg-scale "${CFG_SCALE}"
    --sampling-method "${SAMPLING_METHOD}"
    --steps "${STEPS}"
    --diffusion-fa
  )

  if [[ "${OFFLOAD_TO_CPU}" == "true" ]]; then
    cmd+=(--offload-to-cpu)
  fi

  if [[ -n "${SDCPP_EXTRA_FLAGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra_flags=( ${SDCPP_EXTRA_FLAGS} )
    cmd+=("${extra_flags[@]}")
  fi

  nohup "${cmd[@]}" >>"${LOG_FILE}" 2>&1 &
  local pid="$!"
  echo "${pid}" > "${PID_FILE}"

  sleep 2
  if ! is_running; then
    echo "Failed to start sd-server. Last log lines:"
    tail -n 80 "${LOG_FILE}" || true
    exit 1
  fi

  if check_ready; then
    echo "sd-server is running (PID ${pid}) at ${SERVER_URL}"
  else
    echo "sd-server process started (PID ${pid}), still warming up."
    echo "Check readiness with: ./scripts/flux2_klein_server.sh status"
  fi
}

stop_server() {
  if is_truthy "${DOCKER_MODE}"; then
    require_cmd docker
    if [[ -f "${DOCKER_COMPOSE_FILE}" ]]; then
      docker_compose stop "${DOCKER_SERVICE_NAME}" >/dev/null 2>&1 || true
      docker_compose rm -f "${DOCKER_SERVICE_NAME}" >/dev/null 2>&1 || true
    fi
    rm -f "${PID_FILE}"
    echo "Stopped docker service ${DOCKER_SERVICE_NAME}"
    return 0
  fi

  if ! is_running; then
    rm -f "${PID_FILE}"
    echo "sd-server is not running"
    return 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  kill "${pid}" >/dev/null 2>&1 || true

  for _ in $(seq 1 20); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      rm -f "${PID_FILE}"
      echo "Stopped sd-server (PID ${pid})"
      return 0
    fi
    sleep 0.5
  done

  echo "sd-server did not stop gracefully, sending SIGKILL"
  kill -9 "${pid}" >/dev/null 2>&1 || true
  rm -f "${PID_FILE}"
  echo "Stopped sd-server (PID ${pid})"
}

print_status() {
  if is_truthy "${DOCKER_MODE}"; then
    require_cmd docker
    if [[ -f "${DOCKER_COMPOSE_FILE}" ]]; then
      local container_id
      container_id="$(docker_compose ps -q "${DOCKER_SERVICE_NAME}" 2>/dev/null || true)"
      if [[ -n "${container_id}" ]] && [[ "$(docker inspect -f '{{.State.Running}}' "${container_id}" 2>/dev/null || echo false)" == "true" ]]; then
        echo "Process: running (container ${container_id})"
      else
        echo "Process: stopped"
      fi
    else
      echo "Process: unknown (compose file missing: ${DOCKER_COMPOSE_FILE})"
    fi

    if check_ready; then
      echo "Endpoint: ready (${SERVER_URL}/v1/models)"
    else
      echo "Endpoint: not ready (${SERVER_URL}/v1/models)"
    fi
    return 0
  fi

  if is_running; then
    echo "Process: running (PID $(cat "${PID_FILE}"))"
  else
    echo "Process: stopped"
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
SOURCE_ROOT=${SOURCE_ROOT}
FLUX2_KLEIN_PROFILE=${FLUX2_KLEIN_PROFILE}
DOCKER_MODE=${DOCKER_MODE}
DOCKER_COMPOSE_FILE=${DOCKER_COMPOSE_FILE}
DOCKER_PROJECT_NAME=${DOCKER_PROJECT_NAME}
DOCKER_SERVICE_NAME=${DOCKER_SERVICE_NAME}
CONTAINER_LISTEN_IP=${CONTAINER_LISTEN_IP}
SERVER_URL=${SERVER_URL}
SD_SERVER_BIN=${SD_SERVER_BIN}
DIFFUSION_MODEL=${DIFFUSION_MODEL}
VAE_MODEL=${VAE_MODEL}
LLM_MODEL=${LLM_MODEL}
CFG_SCALE=${CFG_SCALE}
SAMPLING_METHOD=${SAMPLING_METHOD}
STEPS=${STEPS}
OFFLOAD_TO_CPU=${OFFLOAD_TO_CPU}
EOF
}

main() {
  local command="${1:-}"
  case "${command}" in
    setup)
      setup_sdcpp
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
      if is_truthy "${DOCKER_MODE}"; then
        require_cmd docker
        docker_compose logs -f "${DOCKER_SERVICE_NAME}"
      else
        mkdir -p "${RUN_ROOT}"
        touch "${LOG_FILE}"
        tail -f "${LOG_FILE}"
      fi
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
