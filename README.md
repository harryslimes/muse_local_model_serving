# muse_local_model_serving

Shared workspace for local model serving across projects.

Layout:
- `models/` model weights
- `sources/` serving engine source trees (for example `stable-diffusion.cpp`)
- `scripts/` server lifecycle wrappers
- `runtime/` logs and transient runtime state

## FLUX.2 Klein (Docker)

`scripts/flux2_klein_server.sh` defaults to Docker mode (`DOCKER_MODE=true`), and will:
- build a CUDA-enabled `sd-server` image
- run it in Docker with GPU access
- auto-download missing model files on startup using `huggingface_hub`

Profiles:
- `FLUX2_KLEIN_PROFILE=9b-fp8` (default)
- `FLUX2_KLEIN_PROFILE=4b-fp8`
- `FLUX2_KLEIN_PROFILE=4b`

For Black Forest Labs gated repos, provide a token:
- `HF_TOKEN=...`

You can set this once in:
- `muse_local_model_serving/.env`
- Example:
  `HF_TOKEN=hf_xxx`

Examples:
- `./scripts/flux2_klein_server.sh setup`
- `./scripts/flux2_klein_server.sh start`
- `./scripts/flux2_klein_server.sh status`
- `./scripts/flux2_klein_server.sh logs`
- `./scripts/flux2_klein_server.sh stop`

To prefetch model files only:
- `FLUX2_KLEIN_PROFILE=9b-fp8 ./scripts/flux2_klein_server.sh download-models`

Container config files:
- `docker-compose.flux2.yml`
- `docker-compose.flux2-klein-9b-gguf.yml`
- `docker/Dockerfile.flux2`
- `docker/flux2_entrypoint.py`

## Qwen3.5-35B-A3B GGUF (Docker, OpenAI-Compatible)

Compose stack for:
- Repo: `unsloth/Qwen3.5-35B-A3B-GGUF`
- File: `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`

The stack includes:
- locally built CUDA `llama.cpp` server container (`docker/Dockerfile.qwen35-llama-cuda`)
- OpenAI-compatible proxy (request normalization + `enable_thinking` default)
- model-downloader helper container (`huggingface_hub`)

Compose workflow:
- `docker compose -f docker-compose.qwen35-35b-a3b.yml build qwen35-35b-a3b-llama qwen35-35b-a3b-proxy`
- `docker compose -f docker-compose.qwen35-35b-a3b.yml run --rm qwen35-35b-a3b-downloader`
- `docker compose -f docker-compose.qwen35-35b-a3b.yml up -d qwen35-35b-a3b-llama qwen35-35b-a3b-proxy`

Verify:
- `curl http://127.0.0.1:12434/healthz`
- `curl http://127.0.0.1:12434/v1/models`

Optional image override:
- set `LLAMA_IMAGE` to use a prebuilt image tag instead of the default local tag.
- example: `LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda`

Default endpoint:
- `http://127.0.0.1:12434/v1`

Scope note:
- `scripts/qwen35_35b_a3b_server.sh` remains unchanged by this compose-first setup.

Muse integration hint (OpenAI provider path):
- `DEFAULT_LLM_PROVIDER=openai`
- `OPENAI_API_BASE_URL=http://127.0.0.1:12434/v1`
- `OPENAI_API_KEY=local-dev-key`
- model name: `Qwen3.5-35B-A3B`

Container config files:
- `docker-compose.qwen35-35b-a3b.yml`
- `docker/Dockerfile.qwen35-llama-cuda`
- `docker/Dockerfile.qwen35-proxy`
- `docker/Dockerfile.qwen35-downloader`
- `docker/qwen35_openclaw_proxy.py`
- `docker/qwen35_model_downloader.py`
- `docker/qwen35_llama_entrypoint.sh`
- `docker/chat-templates/qwen3_openclaw.jinja`

## Tool Server (Docker — SearXNG + crawl4ai)

Compose stack providing web search and web crawl endpoints for companion tool calling:
- **SearXNG** — privacy-respecting meta search engine
- **crawl4ai** — headless Chromium web crawler returning markdown

Build + start:
- `docker compose -f docker-compose.tool-server.yml up -d --build`

Or via `restart_dev.sh`:
- `./restart_dev.sh --with-tool-server`
- Or set `ENABLE_TOOL_SERVER=true` in `.env`

Verify:
- `curl http://127.0.0.1:4130/health`
- `curl http://127.0.0.1:4130/tools`

Endpoints:
- `POST /tools/search` — `{"query": "..."}` → search results via SearXNG
- `POST /tools/crawl` — `{"url": "..."}` → page content as markdown

Environment overrides:
- `TOOL_SERVER_PORT` (default: `4130`)
- `SEARXNG_PORT` (default: `8888`)
- `CRAWL_MAX_CHARS` (default: `8000`)
- `SEARCH_MAX_RESULTS` (default: `5`)

Container config files:
- `docker-compose.tool-server.yml`
- `docker/Dockerfile.tool-server`
- `docker/tool_server.py`

## CosyVoice 3.0 TTS (Docker, Streaming)

Compose stack for FunAudioLLM CosyVoice 3.0:
- Repo: `FunAudioLLM/CosyVoice`
- Default model repo id: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- Default port: `4126`

Build + start:
- `docker compose -f docker-compose.cosyvoice-tts.yml build`
- `docker compose -f docker-compose.cosyvoice-tts.yml up -d cosyvoice-tts`

Verify:
- `curl http://127.0.0.1:4126/health`
- `curl http://127.0.0.1:4126/voices`

Voice registry:
- prompt audio clips live under `voices/`
- `voices/voices.json` entries may include:
  `file`, `description`, `mode`, `prompt_text`, `instruct_text`
- existing Chatterbox-style entries still work; for zero-shot mode, add `prompt_text`

Example `voices.json` entry:
```json
{
  "default": {
    "file": "her_prompt.wav",
    "description": "Default CosyVoice prompt clip",
    "mode": "zero_shot",
    "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
  }
}
```

Main endpoints:
- `POST /synthesize`
  JSON: `{"text":"Hello there","voice_id":"default"}`
  returns full raw PCM16 mono audio
- `POST /synthesize/stream`
  same JSON body, but streams raw PCM16 mono audio
- `POST /inference_zero_shot`
  multipart form: `tts_text`, `prompt_text`, `prompt_wav`
- `POST /inference_cross_lingual`
  multipart form: `tts_text`, `prompt_wav`
- `POST /inference_instruct2`
  multipart form: `tts_text`, `instruct_text`, `prompt_wav`

Streaming example:
- `curl -N -X POST http://127.0.0.1:4126/inference_zero_shot -F 'tts_text=Hello from CosyVoice 3.' -F 'prompt_text=You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。' -F 'prompt_wav=@./voices/her_prompt.wav' --output out.pcm`

Output format:
- raw PCM16
- mono
- sample rate reported via `X-Sample-Rate` response header

Container config files:
- `docker-compose.cosyvoice-tts.yml`
- `docker/Dockerfile.cosyvoice-tts`
- `docker/cosyvoice_tts_server.py`
