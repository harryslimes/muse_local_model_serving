#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_PATH="${MODEL_PATH:-}"
LLAMA_PORT="${LLAMA_PORT:-12436}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
CONTEXT_SIZE="${CONTEXT_SIZE:-16384}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-}"
REASONING_FORMAT="${REASONING_FORMAT:-}"
REASONING_BUDGET="${REASONING_BUDGET:-}"
LLAMA_ENABLE_JINJA="${LLAMA_ENABLE_JINJA:-true}"
LLAMA_DISABLE_NHFR="${LLAMA_DISABLE_NHFR:-false}"
LLAMA_NO_WARMUP="${LLAMA_NO_WARMUP:-true}"
N_PARALLEL="${N_PARALLEL:-1}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-}"
CACHE_TYPE_K="${CACHE_TYPE_K:-}"
CACHE_TYPE_V="${CACHE_TYPE_V:-}"
LLAMA_EXTRA_FLAGS="${LLAMA_EXTRA_FLAGS:-}"

if [ -z "${MODEL_PATH}" ]; then
  echo "MODEL_PATH is not set." >&2
  echo "Set MODEL_PATH to the path of a GGUF file." >&2
  exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Model file not found: ${MODEL_PATH}" >&2
  exit 1
fi

HELP_TEXT="$("${LLAMA_SERVER_BIN}" --help 2>&1 || true)"

has_flag() {
  echo "${HELP_TEXT}" | grep -Eq -- "(^|[[:space:]])$1([[:space:],]|$)"
}

is_truthy() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

normalize_reasoning_budget() {
  requested="${1:-}"
  if echo "${HELP_TEXT}" | grep -Fq -- "only one of: -1 for unrestricted thinking budget, or 0"; then
    case "${requested}" in
      -1|0) echo "${requested}" ;;
      *) echo "-1" ;;
    esac
  else
    echo "${requested}"
  fi
}

set -- \
  "${LLAMA_SERVER_BIN}" \
  -m "${MODEL_PATH}" \
  -ngl "${N_GPU_LAYERS}" \
  -np "${N_PARALLEL}" \
  -c "${CONTEXT_SIZE}"

if has_flag "--temp"; then
  set -- "$@" --temp "${TEMPERATURE}"
elif has_flag "-temp"; then
  set -- "$@" -temp "${TEMPERATURE}"
fi

if has_flag "--top-p"; then
  set -- "$@" --top-p "${TOP_P}"
elif has_flag "-top_p"; then
  set -- "$@" -top_p "${TOP_P}"
fi

if has_flag "--flash-attn"; then
  if echo "${HELP_TEXT}" | grep -Fq -- "--flash-attn [on|off|auto]"; then
    set -- "$@" --flash-attn auto
  else
    set -- "$@" -fa
  fi
fi

if has_flag "--host"; then
  set -- "$@" --host "${LLAMA_HOST}"
fi
if has_flag "--port"; then
  set -- "$@" --port "${LLAMA_PORT}"
fi

if ! is_truthy "${LLAMA_DISABLE_NHFR}" && has_flag "-nhfr"; then
  set -- "$@" -nhfr
fi

if is_truthy "${LLAMA_NO_WARMUP}" && has_flag "--no-warmup"; then
  set -- "$@" --no-warmup
fi

if is_truthy "${LLAMA_ENABLE_JINJA}"; then
  if has_flag "--jinja"; then
    set -- "$@" --jinja
  fi
  if [ -n "${CHAT_TEMPLATE_FILE}" ] && has_flag "--chat-template-file"; then
    set -- "$@" --chat-template-file "${CHAT_TEMPLATE_FILE}"
  fi
  if has_flag "--reasoning"; then
    set -- "$@" --reasoning off
  fi
fi

if [ -n "${REASONING_FORMAT}" ] && has_flag "--reasoning-format"; then
  set -- "$@" --reasoning-format "${REASONING_FORMAT}"
fi

if [ -n "${REASONING_BUDGET}" ] && has_flag "--reasoning-budget"; then
  set -- "$@" --reasoning-budget "$(normalize_reasoning_budget "${REASONING_BUDGET}")"
fi

if [ -n "${SLOT_SAVE_PATH}" ] && has_flag "--slot-save-path"; then
  mkdir -p "${SLOT_SAVE_PATH}"
  set -- "$@" --slot-save-path "${SLOT_SAVE_PATH}"
fi

if [ -n "${CACHE_TYPE_K}" ] && has_flag "--cache-type-k"; then
  set -- "$@" --cache-type-k "${CACHE_TYPE_K}"
fi

if [ -n "${CACHE_TYPE_V}" ] && has_flag "--cache-type-v"; then
  set -- "$@" --cache-type-v "${CACHE_TYPE_V}"
fi

if [ -n "${LLAMA_EXTRA_FLAGS}" ]; then
  # Intentional word splitting for user-provided flags.
  # shellcheck disable=SC2086
  set -- "$@" ${LLAMA_EXTRA_FLAGS}
fi

echo "Launching llama-server:"
echo "$*"
exec "$@"
