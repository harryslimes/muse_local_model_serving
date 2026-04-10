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
echo "  Or from repo root:  ./restart_dev.sh --platform mac --with-voice-server"
echo ""
