#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


def _print(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    model_root = Path(os.getenv("MODEL_ROOT", "/models"))
    model_repo = os.getenv("MODEL_REPO", "unsloth/Qwen3.5-35B-A3B-GGUF")
    model_file = os.getenv("MODEL_FILE", "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf")
    token = os.getenv("HF_TOKEN") or None

    if "/" not in model_repo:
        _print(f"Invalid MODEL_REPO={model_repo!r}. Expected '<org>/<repo>'.")
        return 1

    local_dir = model_root / model_repo
    local_dir.mkdir(parents=True, exist_ok=True)

    _print(f"Downloading model from {model_repo}:{model_file}")
    downloaded = hf_hub_download(
        repo_id=model_repo,
        filename=model_file,
        local_dir=str(local_dir),
        token=token,
    )
    downloaded_path = Path(downloaded)

    expected = model_root / model_repo / model_file
    final_path = expected if expected.exists() else downloaded_path
    _print(f"Model ready: {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
