#!/usr/bin/env sh
set -eu

MODEL_PATH="${MODEL_PATH:-/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
CONTEXT_SIZE="${CONTEXT_SIZE:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-/app/chat-template/qwen3.jinja}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"
REASONING_BUDGET="${REASONING_BUDGET:-1024}"
LLAMA_ENABLE_JINJA="${LLAMA_ENABLE_JINJA:-true}"
LLAMA_DISABLE_NHFR="${LLAMA_DISABLE_NHFR:-false}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-}"
LLAMA_EXTRA_FLAGS="${LLAMA_EXTRA_FLAGS:-}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Model file not found: ${MODEL_PATH}" >&2
  echo "Run download-models before start, or set MODEL_PATH to an existing file." >&2
  exit 1
fi

HELP_TEXT="$(llama-server --help 2>&1 || true)"

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
  llama-server \
  -m "${MODEL_PATH}" \
  -ngl "${N_GPU_LAYERS}" \
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
  set -- "$@" --host 0.0.0.0
fi
if has_flag "--port"; then
  set -- "$@" --port "${LLAMA_PORT}"
fi

if ! is_truthy "${LLAMA_DISABLE_NHFR}" && has_flag "-nhfr"; then
  set -- "$@" -nhfr
fi

if is_truthy "${LLAMA_ENABLE_JINJA}"; then
  if has_flag "--jinja"; then
    set -- "$@" --jinja
  fi
  if has_flag "--chat-template-file"; then
    set -- "$@" --chat-template-file "${CHAT_TEMPLATE_FILE}"
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

if [ -n "${LLAMA_EXTRA_FLAGS}" ]; then
  # Intentional word splitting for user-provided flags.
  # shellcheck disable=SC2086
  set -- "$@" ${LLAMA_EXTRA_FLAGS}
fi

echo "Launching llama-server:"
echo "$*"
exec "$@"
