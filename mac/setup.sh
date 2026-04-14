#!/usr/bin/env bash
# One-time setup for Mac model serving (Apple Silicon).
#
# What this does:
#   1. Checks prerequisites (python3, Xcode CLI tools, Swift)
#   2. Creates a Python venv and installs pip dependencies
#   3. Builds and installs fluidaudiocli (STT on Neural Engine)
#
# Usage:
#   cd muse_local_model_serving/mac
#   ./setup.sh
#
# After setup, start servers with:
#   ./start.sh          # or use restart_dev.sh --platform mac
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLUIDAUDIO_INSTALL_DIR="${HOME}/.local/bin"
FLUIDAUDIO_CLI="${FLUIDAUDIO_INSTALL_DIR}/fluidaudiocli"
FLUIDAUDIO_BUILD_DIR="${SCRIPT_DIR}/../.build/FluidAudio"
ROTORQUANT_BUILD_DIR="${SCRIPT_DIR}/../.build/llama-cpp-turboquant"
ROTORQUANT_BIN="${ROTORQUANT_BUILD_DIR}/build/bin/llama-server"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}!${NC} $*"; }
fail()  { echo -e "${RED}✗${NC} $*"; exit 1; }
step()  { echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Prerequisites ──────────────────────────────────────────────────

step "Checking prerequisites"

# macOS / Apple Silicon
[[ "$(uname)" == "Darwin" ]] || fail "This setup is for macOS only."
[[ "$(uname -m)" == "arm64" ]] || fail "Apple Silicon (arm64) required."
info "macOS Apple Silicon"

# python3
if command -v python3 &>/dev/null; then
  info "python3 found: $(python3 --version 2>&1)"
else
  fail "python3 not found. Install from https://www.python.org or: brew install python"
fi

# Xcode Command Line Tools
if xcode-select -p &>/dev/null; then
  info "Xcode Command Line Tools found"
else
  warn "Xcode Command Line Tools not found. Installing..."
  xcode-select --install
  echo "    Re-run this script after installation completes."
  exit 1
fi

# Swift
if command -v swift &>/dev/null; then
  SWIFT_VERSION="$(swift --version 2>&1 | head -1)"
  info "Swift found: ${SWIFT_VERSION}"
else
  fail "Swift not found. Install Xcode or Xcode Command Line Tools."
fi

# ── Python venv + dependencies ─────────────────────────────────────

step "Setting up Python environment"

VENV_DIR="${SCRIPT_DIR}/.venv"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  info "venv already exists at ${VENV_DIR}"
else
  echo "  Creating venv..."
  python3 -m venv "${VENV_DIR}"
  info "venv created"
fi

source "${VENV_DIR}/bin/activate"
info "venv activated (python: $(which python))"

echo "  Installing pip dependencies..."
pip install --upgrade pip --quiet
pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet 2>&1 | grep -v "already satisfied" || true
info "pip dependencies installed"

# ── FluidAudio CLI (Neural Engine STT) ─────────────────────────────

step "Setting up fluidaudiocli (STT on Neural Engine)"

if [[ -f "${FLUIDAUDIO_CLI}" ]]; then
  info "fluidaudiocli already installed at ${FLUIDAUDIO_CLI}"
  echo "  To rebuild, remove it and re-run: rm ${FLUIDAUDIO_CLI} && ./setup.sh"
else
  echo "  FluidAudio provides speech-to-text via Apple's Neural Engine,"
  echo "  keeping the GPU free for LLM and TTS."
  echo ""

  if [[ -d "${FLUIDAUDIO_BUILD_DIR}/.git" ]]; then
    info "FluidAudio source already cloned"
    echo "  Updating..."
    git -C "${FLUIDAUDIO_BUILD_DIR}" pull --quiet 2>/dev/null || true
  else
    echo "  Cloning FluidAudio..."
    mkdir -p "$(dirname "${FLUIDAUDIO_BUILD_DIR}")"
    git clone --quiet https://github.com/FluidInference/FluidAudio.git "${FLUIDAUDIO_BUILD_DIR}"
    info "FluidAudio cloned"
  fi

  echo "  Building fluidaudiocli (this may take a few minutes on first run)..."
  (cd "${FLUIDAUDIO_BUILD_DIR}" && swift build -c release 2>&1 | tail -5)

  # Find the built binary
  BUILT_BIN="${FLUIDAUDIO_BUILD_DIR}/.build/release/fluidaudiocli"
  if [[ ! -f "${BUILT_BIN}" ]]; then
    # Try arm64 subdirectory
    BUILT_BIN="${FLUIDAUDIO_BUILD_DIR}/.build/arm64-apple-macosx/release/fluidaudiocli"
  fi

  if [[ -f "${BUILT_BIN}" ]]; then
    mkdir -p "${FLUIDAUDIO_INSTALL_DIR}"
    cp "${BUILT_BIN}" "${FLUIDAUDIO_CLI}"
    chmod +x "${FLUIDAUDIO_CLI}"
    info "fluidaudiocli installed to ${FLUIDAUDIO_CLI}"
  else
    fail "Build succeeded but could not find fluidaudiocli binary. Check ${FLUIDAUDIO_BUILD_DIR}/.build/ for the output."
  fi
fi

# ── RotorQuant llama.cpp fork (Gemma-4 GGUF KV cache) ─────────────

step "Building RotorQuant llama.cpp fork (planar3/iso3 KV cache)"

if [[ -f "${ROTORQUANT_BIN}" ]]; then
  info "RotorQuant llama-server already built at ${ROTORQUANT_BIN}"
  echo "  To rebuild: rm -rf ${ROTORQUANT_BUILD_DIR}/build && re-run ./setup.sh"
else
  echo "  This builds johndpope/llama-cpp-turboquant (feature/planarquant-kv-cache)"
  echo "  with Metal support for Apple Silicon. May take 5-10 minutes."
  echo ""

  if ! command -v cmake &>/dev/null; then
    fail "cmake not found. Install it: brew install cmake"
  fi

  if [[ -d "${ROTORQUANT_BUILD_DIR}/.git" ]]; then
    info "RotorQuant source already cloned"
    echo "  Fetching latest changes..."
    git -C "${ROTORQUANT_BUILD_DIR}" fetch --quiet origin feature/planarquant-kv-cache 2>/dev/null || true
    git -C "${ROTORQUANT_BUILD_DIR}" checkout --quiet feature/planarquant-kv-cache 2>/dev/null || true
    git -C "${ROTORQUANT_BUILD_DIR}" pull --quiet 2>/dev/null || true
  else
    echo "  Cloning johndpope/llama-cpp-turboquant..."
    mkdir -p "$(dirname "${ROTORQUANT_BUILD_DIR}")"
    git clone --quiet --branch feature/planarquant-kv-cache \
      https://github.com/johndpope/llama-cpp-turboquant.git "${ROTORQUANT_BUILD_DIR}"
    info "RotorQuant source cloned"
  fi

  echo "  Configuring CMake (Metal + embedded library)..."
  cmake -B "${ROTORQUANT_BUILD_DIR}/build" \
    -S "${ROTORQUANT_BUILD_DIR}" \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    2>&1 | tail -5

  echo "  Building llama-server (this will take a few minutes)..."
  cmake --build "${ROTORQUANT_BUILD_DIR}/build" \
    --target llama-server \
    -j "$(sysctl -n hw.logicalcpu)" \
    2>&1 | tail -10

  if [[ -f "${ROTORQUANT_BIN}" ]]; then
    info "RotorQuant llama-server built: ${ROTORQUANT_BIN}"
  else
    fail "Build completed but binary not found at ${ROTORQUANT_BIN}. Check build output above."
  fi
fi

# ── Verify ─────────────────────────────────────────────────────────

step "Verifying setup"

# Check fluidaudiocli runs
if "${FLUIDAUDIO_CLI}" --help &>/dev/null 2>&1 || "${FLUIDAUDIO_CLI}" help &>/dev/null 2>&1; then
  info "fluidaudiocli runs OK"
else
  # Some CLIs return non-zero for --help, check if the binary is at least executable
  if [[ -x "${FLUIDAUDIO_CLI}" ]]; then
    info "fluidaudiocli is executable"
  else
    warn "fluidaudiocli may not be working correctly — check manually"
  fi
fi

# Check key Python imports
python -c "import fastapi, uvicorn, numpy; print('Python imports OK')" 2>/dev/null && info "Python dependencies OK" || warn "Some Python imports failed — check pip install output above"

echo ""
echo -e "${GREEN}${BOLD}Setup complete!${NC}"
echo ""
echo "  Start servers:  ./start.sh"
echo "  Download model: hf download majentik/gemma-4-26B-A4B-it-RotorQuant-GGUF-Q4_K_M --include '*.gguf' --local-dir ~/.cache/huggingface/hub/models--majentik--gemma-4-26B-A4B-it-RotorQuant-GGUF-Q4_K_M"
echo "  Gemma GGUF:     GEMMA_GGUF_MODEL_PATH=/path/to/gemma-4-26B-A4B-it-RotorQuant-GGUF-Q4_K_M.gguf ./start.sh start llama-gemma"
echo "  Or from repo root:  ./restart_dev.sh --platform mac --with-voice-server"
echo ""
