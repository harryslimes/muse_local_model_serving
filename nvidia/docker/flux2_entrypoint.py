#!/usr/bin/env python3
import os
import shlex
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


PROFILE_SPECS = {
    "4b": {
        "diffusion": (
            "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            "split_files/diffusion_models/flux-2-klein-4b.safetensors",
        ),
        "vae": (
            "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            "split_files/vae/flux2-vae.safetensors",
        ),
        "llm": (
            "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            "split_files/text_encoders/qwen_3_4b.safetensors",
        ),
    },
    "4b-fp8": {
        "diffusion": (
            "black-forest-labs/FLUX.2-klein-4b-fp8",
            "flux-2-klein-4b-fp8.safetensors",
        ),
        "vae": (
            "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            "split_files/vae/flux2-vae.safetensors",
        ),
        "llm": (
            "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            "split_files/text_encoders/qwen_3_4b.safetensors",
        ),
    },
    "9b-fp8": {
        "diffusion": (
            "black-forest-labs/FLUX.2-klein-9b-fp8",
            "flux-2-klein-9b-fp8.safetensors",
        ),
        "vae": (
            "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
            "split_files/vae/flux2-vae.safetensors",
        ),
        "llm": (
            "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
            "split_files/text_encoders/qwen_3_8b.safetensors",
        ),
    },
}


def is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def eprint(message: str) -> None:
    print(message, flush=True)


def model_path_for(models_root: Path, repo_id: str, filename: str) -> Path:
    return models_root / repo_id / filename


def infer_repo_filename(path: Path, models_root: Path) -> tuple[str, str] | None:
    try:
        rel = path.relative_to(models_root)
    except ValueError:
        return None

    if len(rel.parts) < 3:
        return None

    repo_id = "/".join(rel.parts[:2])
    filename = "/".join(rel.parts[2:])
    return repo_id, filename


def download_file(models_root: Path, repo_id: str, filename: str, token: str | None) -> Path:
    local_dir = models_root / repo_id
    local_dir.mkdir(parents=True, exist_ok=True)

    eprint(f"Downloading {repo_id}:{filename}")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        token=token,
    )
    downloaded_path = Path(downloaded)

    expected_path = model_path_for(models_root, repo_id, filename)
    if expected_path.exists():
        return expected_path
    return downloaded_path


def resolve_model_path(
    label: str,
    configured_path: str | None,
    models_root: Path,
    default_repo_id: str,
    default_filename: str,
    token: str | None,
) -> Path:
    if configured_path:
        explicit = Path(configured_path)
        if explicit.exists():
            eprint(f"{label}: using existing configured path {explicit}")
            return explicit

        inferred = infer_repo_filename(explicit, models_root)
        if inferred:
            repo_id, filename = inferred
            eprint(f"{label}: configured path missing; trying HF download from {repo_id}:{filename}")
            downloaded = download_file(models_root, repo_id, filename, token)
            eprint(f"{label}: downloaded to {downloaded}")
            return downloaded

        raise FileNotFoundError(
            f"{label}: configured path does not exist and is not under MODEL_ROOT: {explicit}"
        )

    default_path = model_path_for(models_root, default_repo_id, default_filename)
    if default_path.exists():
        eprint(f"{label}: found {default_path}")
        return default_path

    downloaded = download_file(models_root, default_repo_id, default_filename, token)
    eprint(f"{label}: downloaded to {downloaded}")
    return downloaded


def main() -> int:
    profile = os.getenv("FLUX2_KLEIN_PROFILE", "9b-fp8")
    if profile not in PROFILE_SPECS:
        eprint(f"Unsupported FLUX2_KLEIN_PROFILE={profile!r}. Supported: {', '.join(sorted(PROFILE_SPECS))}")
        return 1

    token = os.getenv("HF_TOKEN") or None
    models_root = Path(os.getenv("MODEL_ROOT", "/models"))
    models_root.mkdir(parents=True, exist_ok=True)

    profile_spec = PROFILE_SPECS[profile]

    try:
        diffusion_path = resolve_model_path(
            "diffusion",
            os.getenv("DIFFUSION_MODEL"),
            models_root,
            profile_spec["diffusion"][0],
            profile_spec["diffusion"][1],
            token,
        )
        vae_path = resolve_model_path(
            "vae",
            os.getenv("VAE_MODEL"),
            models_root,
            profile_spec["vae"][0],
            profile_spec["vae"][1],
            token,
        )
        llm_path = resolve_model_path(
            "llm",
            os.getenv("LLM_MODEL"),
            models_root,
            profile_spec["llm"][0],
            profile_spec["llm"][1],
            token,
        )
    except Exception as exc:
        eprint(f"Model resolution failed: {exc}")
        return 1

    if is_truthy(os.getenv("DOWNLOAD_ONLY")):
        eprint("DOWNLOAD_ONLY=true, exiting after ensuring model files")
        return 0

    listen_ip = os.getenv("LISTEN_IP", "0.0.0.0")
    listen_port = os.getenv("LISTEN_PORT", "1234")
    cfg_scale = os.getenv("CFG_SCALE", "1.0")
    sampling_method = os.getenv("SAMPLING_METHOD", "euler")
    steps = os.getenv("STEPS", "4")
    offload_to_cpu = is_truthy(os.getenv("OFFLOAD_TO_CPU"))
    extra_flags = shlex.split(os.getenv("SDCPP_EXTRA_FLAGS", ""))

    cmd = [
        "sd-server",
        "--listen-ip",
        listen_ip,
        "--listen-port",
        str(listen_port),
        "--diffusion-model",
        str(diffusion_path),
        "--vae",
        str(vae_path),
        "--llm",
        str(llm_path),
        "--cfg-scale",
        str(cfg_scale),
        "--sampling-method",
        sampling_method,
        "--steps",
        str(steps),
        "--diffusion-fa",
    ]

    if offload_to_cpu:
        cmd.append("--offload-to-cpu")

    cmd.extend(extra_flags)

    eprint("Launching sd-server with command:")
    eprint(" ".join(shlex.quote(part) for part in cmd))

    os.execvp(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
