# FunASR API

FunASR API is a FastAPI-based inference gateway that wraps multiple FunASR speech models behind a single HTTP surface. It manages long-running model lifecycles, exposes health endpoints for each model family, and gives you a starting point for building higher level speech services such as transcription, keyword spotting, or voice activity detection.

## Table of Contents

- [Key Features](#key-features)
- [Project Layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [API Reference](#api-reference)
- [Extending with New Models](#extending-with-new-models)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Key Features

- Asynchronous FastAPI service with startup lifecycle for model warm-up.
- Centralized configuration via `config.toml` validated by Pydantic models.
- Baseline health/status endpoints for each bundled FunASR model.
- Ready for GPU execution (CUDA) with graceful CPU fallbacks.
- Structured logging powered by Loguru with daily log rotation.

## Project Layout

```text
funasr-api/
├── main.py               # FastAPI application entrypoint
├── config.toml           # Runtime configuration (port, devices, models)
├── src/
│   ├── app.py            # FunasrApp loader that initializes FunASR models
│   ├── config.py         # Pydantic config schema definitions
│   ├── log.py            # Logger setup helpers
│   └── api/
│       └── model/        # FastAPI routers for each supported model
└── pyproject.toml        # Package metadata and dependencies
```

## Prerequisites

- Python 3.11 or newer.
- CUDA-capable GPU for best performance (CPU execution is possible but slower).
- [PyTorch](https://pytorch.org/) wheels that match your CUDA runtime; `pip install` fetches defaults, but verify compatibility with your driver.
- Sufficient disk space and bandwidth to download FunASR model weights on first run.

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/HsiangNianian/funasr-api.git
   cd funasr-api
   ```

2. **Create and activate a virtual environment** (fish shell example)

   ```bash
   python -m venv .venv
   source .venv/bin/activate.fish
   ```

3. **Install dependencies**

   ```bash
   pip install -U pip
   pip install -e .
   ```

4. **Review `config.toml`** and adjust host, port, and model settings to match your hardware.

## Configuration

Runtime configuration lives in `config.toml` and is parsed into the Pydantic models declared in `src/config.py`. A minimal example that enables SenseVoice and Whisper on the first CUDA device looks like this:

```toml
[app]
host = "0.0.0.0"
port = 8000

[log]
level = "INFO"
verbose_exception = false

[model]
device = "cuda"
device_id = 0

[model.sensevoice]
enable = true
model_name = "iic/SenseVoiceSmall"
device = "cuda"
device_id = 0

[model.whisper]
enable = true
model_name = "funasr-whisper-base"
device = "cuda"
device_id = 0

[model.sensevoice.kwargs]
max_single_segment_time = 30000

[model.whisper.kwargs]
batch_size = 1
```

Notes:

- Setting `enable = false` prevents the model from loading during startup.
- `device` accepts `cuda` or `cpu`; use `device_id` to select a specific GPU.
- Arbitrary keyword arguments are passed to `funasr.AutoModel` via the nested `[model.<name>.kwargs]` tables.

## Running the Service

### Development (auto-reload)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production (single worker example)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```

When the application starts, `FunasrApp` loads the configured models once and keeps them in memory for subsequent requests. The default root path redirects to `/docs`, where you can interact with the auto-generated Swagger UI.

## API Reference

The current implementation exposes health/status checks for each registered model. Once you extend the routers, these endpoints can be evolved into full inference APIs.

| Method | Path                  | Description                        |
|--------|-----------------------|------------------------------------|
| GET    | `/campplus/status`    | CAMPPlus model load state.         |
| GET    | `/conformer/status`   | Conformer ASR model load state.    |
| GET    | `/ctpunc/status`      | CT-Punctuation model load state.   |
| GET    | `/emotion2vec/status` | Emotion2Vec model load state.      |
| GET    | `/fazh/status`        | FAZH model load state.             |
| GET    | `/fsmnkws/status`     | Keyword spotting model load state. |
| GET    | `/fsmnvad/status`     | Voice activity detector state.     |
| GET    | `/paraformer/status`  | Paraformer ASR model load state.   |
| GET    | `/qwenaudio/status`   | QwenAudio model load state.        |
| GET    | `/sensevoice/status`  | SenseVoice model load state.       |
| GET    | `/whisper/status`     | Whisper model load state.          |

Example request:

```bash
curl http://localhost:8000/sensevoice/status
```

Example response:

```json
{
  "model": "SenseVoice",
  "status": true
}
```

`status` is `true` when the model is loaded in memory and ready to serve requests.

## Extending with New Models

1. **Create a configuration block** in `config.toml` under `[model.<name>]`.
2. **Add a router** in `src/api/model/<name>.py`. Follow the existing pattern (define a FastAPI `APIRouter` with a lifespan context that loads your `funasr.AutoModel` instance via `FunasrApp`).
3. **Register the router** in `main.py` using `app.include_router`.
4. **Implement inference endpoints** (for example, POST `/transcribe`) alongside the status check to expose real functionality.

## Logging

- Logs are written to both stderr and a daily rotating file under `logs/<YYYY-MM-DD>.log` relative to the project root.
- Adjust verbosity through the `[log]` section in `config.toml`. Enabling `verbose_exception` surfaces full stack traces for debugging sessions.

## Troubleshooting

- **Model download failures**: ensure outbound internet access and that Hugging Face credentials (if required) are available in the environment.
- **CUDA initialization errors**: confirm that the installed PyTorch/torchaudio wheels match your CUDA driver version or switch to CPU mode by setting `device = "cpu"`.
- **Large memory footprint**: use smaller model variants in `config.toml` or disable unused models to reduce startup time and RAM usage.
- **Hot reload needs rebuild**: whenever you change `config.toml`, restart the service to make sure the new model settings are applied.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).
