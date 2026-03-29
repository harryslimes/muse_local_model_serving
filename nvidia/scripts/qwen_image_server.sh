#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_MODEL_SERVING_ROOT="${LOCAL_MODEL_SERVING_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-${LOCAL_MODEL_SERVING_ROOT}/runtime/qwen_image_server}"
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

DIFFUSION_MODEL="${DIFFUSION_MODEL:-${MODEL_ROOT}/unsloth/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q5_K_M.gguf}"
VAE_MODEL="${VAE_MODEL:-${MODEL_ROOT}/Comfy-Org/Qwen-Image_ComfyUI/split_files/vae/qwen_image_vae.safetensors}"
LLM_MODEL="${LLM_MODEL:-${MODEL_ROOT}/Comfy-Org/Qwen-Image_ComfyUI/split_files/text_encoders/qwen_2.5_vl_7b.safetensors}"

CFG_SCALE="${CFG_SCALE:-2.5}"
SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
STEPS="${STEPS:-30}"
FLOW_SHIFT="${FLOW_SHIFT:-3}"
OFFLOAD_TO_CPU="${OFFLOAD_TO_CPU:-false}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/qwen_image_server.sh <command>

Commands:
  setup            Clone/build stable-diffusion.cpp with CUDA enabled.
  download-models  Download required Qwen 2511 files with huggingface-cli.
  start            Start persistent sd-server (keeps weights loaded).
  stop             Stop running sd-server.
  restart          Restart server.
  status           Show process + endpoint health.
  logs             Tail server logs.
  print-env        Print relevant env config values.

Environment overrides (optional):
  LOCAL_MODEL_SERVING_ROOT, RUN_ROOT
  MODEL_ROOT, SOURCE_ROOT, SDCPP_DIR, SD_SERVER_BIN
  DIFFUSION_MODEL, VAE_MODEL, LLM_MODEL
  LISTEN_IP, LISTEN_PORT
  CFG_SCALE, SAMPLING_METHOD, STEPS, FLOW_SHIFT
  OFFLOAD_TO_CPU=true|false
  SDCPP_EXTRA_FLAGS="--diffusion-fa ..."
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
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
  mkdir -p "${MODEL_ROOT}"

  hf_download \
    unsloth/Qwen-Image-Edit-2511-GGUF \
    qwen-image-edit-2511-Q5_K_M.gguf \
    --local-dir "${MODEL_ROOT}/unsloth/Qwen-Image-Edit-2511-GGUF"

  hf_download \
    Comfy-Org/Qwen-Image_ComfyUI \
    split_files/vae/qwen_image_vae.safetensors \
    --local-dir "${MODEL_ROOT}/Comfy-Org/Qwen-Image_ComfyUI"

  hf_download \
    Comfy-Org/Qwen-Image_ComfyUI \
    split_files/text_encoders/qwen_2.5_vl_7b.safetensors \
    --local-dir "${MODEL_ROOT}/Comfy-Org/Qwen-Image_ComfyUI"

  echo "Model files downloaded under ${MODEL_ROOT}"
}

setup_sdcpp() {
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
  mkdir -p "${RUN_ROOT}" "${MODEL_ROOT}" "${SOURCE_ROOT}"
  if is_running; then
    echo "sd-server is already running with PID $(cat "${PID_FILE}")"
    return 0
  fi

  if [[ ! -x "${SD_SERVER_BIN}" ]]; then
    echo "sd-server binary missing: ${SD_SERVER_BIN}"
    echo "Run: ./scripts/qwen_image_server.sh setup"
    exit 1
  fi
  if [[ ! -f "${DIFFUSION_MODEL}" ]]; then
    echo "Diffusion model file not found: ${DIFFUSION_MODEL}"
    echo "Run: ./scripts/qwen_image_server.sh download-models"
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
    --flow-shift "${FLOW_SHIFT}"
    --qwen-image-zero-cond-t
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
    tail -n 50 "${LOG_FILE}" || true
    exit 1
  fi

  if check_ready; then
    echo "sd-server is running (PID ${pid}) at ${SERVER_URL}"
  else
    echo "sd-server process started (PID ${pid}), still warming up."
    echo "Check readiness with: ./scripts/qwen_image_server.sh status"
  fi
}

stop_server() {
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
RUN_ROOT=${RUN_ROOT}
MODEL_ROOT=${MODEL_ROOT}
SOURCE_ROOT=${SOURCE_ROOT}
SERVER_URL=${SERVER_URL}
SD_SERVER_BIN=${SD_SERVER_BIN}
DIFFUSION_MODEL=${DIFFUSION_MODEL}
VAE_MODEL=${VAE_MODEL}
LLM_MODEL=${LLM_MODEL}
CFG_SCALE=${CFG_SCALE}
SAMPLING_METHOD=${SAMPLING_METHOD}
STEPS=${STEPS}
FLOW_SHIFT=${FLOW_SHIFT}
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
      mkdir -p "${RUN_ROOT}"
      touch "${LOG_FILE}"
      tail -f "${LOG_FILE}"
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
