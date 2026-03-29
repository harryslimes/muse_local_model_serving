#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_MODEL_SERVING_DIR="$ROOT_DIR"
ENV_FILE="${ENV_FILE:-$(cd "$ROOT_DIR/.." && pwd)/.env}"

# Read MUSE_BACKEND_DIR from .env; resolve relative paths.
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
BACKEND_ENV_FILE="${BACKEND_DIR}/.env"
LLM_SERVER_SCRIPT="${LOCAL_MODEL_SERVING_DIR}/scripts/qwen35_35b_a3b_server.sh"
IMAGE_SERVER_SCRIPT="${LOCAL_MODEL_SERVING_DIR}/scripts/flux2_klein_server.sh"

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

require_executable() {
  local script_path="$1"
  if [[ ! -x "$script_path" ]]; then
    echo "Missing executable script: $script_path"
    exit 1
  fi
}

infer_local_image_server_from_model() {
  local model_value="$1"
  local normalized="$model_value"
  normalized="${normalized#/models/}"

  IMAGE_SERVER_COMPOSE_FILE=""
  IMAGE_SERVER_SERVICE_NAME=""
  IMAGE_SERVER_DIFFUSION_MODEL=""

  case "$normalized" in
    "")
      return 0
      ;;
    *.gguf)
      IMAGE_SERVER_COMPOSE_FILE="${LOCAL_MODEL_SERVING_DIR}/docker-compose.flux2-klein-9b-gguf.yml"
      IMAGE_SERVER_SERVICE_NAME="flux2-klein-9b-gguf-image-server"
      if [[ "$normalized" == */* ]]; then
        IMAGE_SERVER_DIFFUSION_MODEL="/models/$normalized"
      elif [[ "$normalized" == *base* ]]; then
        IMAGE_SERVER_DIFFUSION_MODEL="/models/unsloth/FLUX.2-klein-base-9B-GGUF/$normalized"
      else
        IMAGE_SERVER_DIFFUSION_MODEL="/models/unsloth/FLUX.2-klein-9B-GGUF/$normalized"
      fi
      ;;
    flux-2-klein-9b-fp8.safetensors)
      IMAGE_SERVER_COMPOSE_FILE="${LOCAL_MODEL_SERVING_DIR}/docker-compose.flux2.yml"
      IMAGE_SERVER_SERVICE_NAME="flux2-image-server"
      IMAGE_SERVER_DIFFUSION_MODEL="/models/black-forest-labs/FLUX.2-klein-9b-fp8/$normalized"
      ;;
    flux-2-klein-4b-fp8.safetensors)
      IMAGE_SERVER_COMPOSE_FILE="${LOCAL_MODEL_SERVING_DIR}/docker-compose.flux2.yml"
      IMAGE_SERVER_SERVICE_NAME="flux2-image-server"
      IMAGE_SERVER_DIFFUSION_MODEL="/models/black-forest-labs/FLUX.2-klein-4b-fp8/$normalized"
      ;;
    flux-2-klein-4b.safetensors)
      IMAGE_SERVER_COMPOSE_FILE="${LOCAL_MODEL_SERVING_DIR}/docker-compose.flux2.yml"
      IMAGE_SERVER_SERVICE_NAME="flux2-image-server"
      IMAGE_SERVER_DIFFUSION_MODEL="/models/Comfy-Org/vae-text-encorder-for-flux-klein-4b/split_files/diffusion_models/$normalized"
      ;;
    *)
      if [[ "$normalized" == */* ]]; then
        if [[ "$normalized" == *.gguf ]]; then
          IMAGE_SERVER_COMPOSE_FILE="${LOCAL_MODEL_SERVING_DIR}/docker-compose.flux2-klein-9b-gguf.yml"
          IMAGE_SERVER_SERVICE_NAME="flux2-klein-9b-gguf-image-server"
        else
          IMAGE_SERVER_COMPOSE_FILE="${LOCAL_MODEL_SERVING_DIR}/docker-compose.flux2.yml"
          IMAGE_SERVER_SERVICE_NAME="flux2-image-server"
        fi
        IMAGE_SERVER_DIFFUSION_MODEL="/models/$normalized"
      fi
      ;;
  esac
}

if [[ ! -f "$BACKEND_ENV_FILE" ]]; then
  echo "Missing env file: $BACKEND_ENV_FILE"
  exit 1
fi

require_executable "$LLM_SERVER_SCRIPT"
require_executable "$IMAGE_SERVER_SCRIPT"

integration_test_llm_model="$(resolve_env_value "$BACKEND_ENV_FILE" "INTEGRATION_TEST_LLM_MODEL")"
integration_test_llm_model="$(trim_env_value "${integration_test_llm_model:-}")"

if [[ -z "$integration_test_llm_model" ]]; then
  echo "INTEGRATION_TEST_LLM_MODEL is not set in $BACKEND_ENV_FILE"
  exit 1
fi

image_gen_provider_raw="$(resolve_env_value "$BACKEND_ENV_FILE" "IMAGE_GEN_PROVIDER")"
image_gen_provider_raw="$(trim_env_value "${image_gen_provider_raw:-}")"
image_gen_provider="${image_gen_provider_raw:-xai}"
if [[ "$image_gen_provider" == local_* ]] && [[ "$image_gen_provider" != "local_sdcpp" ]]; then
  image_gen_provider="local_sdcpp"
fi

local_image_gen_model="$(resolve_env_value "$BACKEND_ENV_FILE" "LOCAL_IMAGE_GEN_MODEL")"
local_image_gen_model="$(trim_env_value "${local_image_gen_model:-}")"
if [[ -z "$local_image_gen_model" ]]; then
  case "$image_gen_provider_raw" in
    local_flux-2-klein-base-9b)
      local_image_gen_model="flux-2-klein-base-9b-Q5_K_M.gguf"
      ;;
    local_flux-2-klein-9b)
      local_image_gen_model="flux-2-klein-9b-Q5_K_M.gguf"
      ;;
  esac
fi

local_image_gen_base_url="$(resolve_env_value "$BACKEND_ENV_FILE" "LOCAL_IMAGE_GEN_BASE_URL")"
local_image_gen_base_url="$(trim_env_value "${local_image_gen_base_url:-}")"
if [[ -z "$local_image_gen_base_url" ]]; then
  local_image_gen_base_url="http://127.0.0.1:1234"
fi
local_image_gen_base_url="${local_image_gen_base_url%/}"
local_image_server_listen_port="$(port_from_url "$local_image_gen_base_url")"

echo "Restarting local LLM server for INTEGRATION_TEST_LLM_MODEL=${integration_test_llm_model}"
case "${integration_test_llm_model,,}" in
  qwen3.5-35b-a3b)
    (
      cd "$LOCAL_MODEL_SERVING_DIR"
      PROXY_MODEL_NAME="$integration_test_llm_model" "$LLM_SERVER_SCRIPT" restart
      PROXY_MODEL_NAME="$integration_test_llm_model" "$LLM_SERVER_SCRIPT" status
    )
    ;;
  *)
    echo "Unsupported INTEGRATION_TEST_LLM_MODEL: $integration_test_llm_model"
    echo "Currently supported: Qwen3.5-35B-A3B"
    exit 1
    ;;
esac

if [[ "$image_gen_provider" != "local_sdcpp" ]]; then
  echo "IMAGE_GEN_PROVIDER=${image_gen_provider_raw:-<unset>} does not require a local image server; skipping VLM restart."
  exit 0
fi

infer_local_image_server_from_model "${local_image_gen_model:-}"

echo "Restarting local VLM/image server for IMAGE_GEN_PROVIDER=${image_gen_provider_raw:-local_sdcpp}"
(
  cd "$LOCAL_MODEL_SERVING_DIR"
  if [[ -n "${IMAGE_SERVER_COMPOSE_FILE:-}" ]]; then
    export DOCKER_COMPOSE_FILE="$IMAGE_SERVER_COMPOSE_FILE"
  fi
  if [[ -n "${IMAGE_SERVER_SERVICE_NAME:-}" ]]; then
    export DOCKER_SERVICE_NAME="$IMAGE_SERVER_SERVICE_NAME"
  fi
  if [[ -n "${IMAGE_SERVER_DIFFUSION_MODEL:-}" ]]; then
    export DIFFUSION_MODEL="$IMAGE_SERVER_DIFFUSION_MODEL"
  fi
  if [[ -n "${local_image_server_listen_port:-}" ]]; then
    export LISTEN_PORT="$local_image_server_listen_port"
  fi

  "$IMAGE_SERVER_SCRIPT" restart
  "$IMAGE_SERVER_SCRIPT" status
)

echo "Done."
echo "LLM endpoint: http://127.0.0.1:12434/v1/models"
echo "Image endpoint: ${local_image_gen_base_url}/v1/models"
