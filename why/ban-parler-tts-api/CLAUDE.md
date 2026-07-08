# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the server

```bash
# From the project root
uvicorn api:app --host 0.0.0.0 --port 8000

# Or directly via the package entry point
python -m tts_api.app.main
```

The server takes ~1–2 minutes to start: it loads the model, runs `torch.compile`, and does 3 warmup inference passes before accepting requests.

## Environment setup

```bash
# Create conda environment (Python 3.10, PyTorch 2.6.0 + CUDA)
conda env create -f tts_api/environment.yml
conda activate infer

# Or install pip dependencies into an existing env
pip install -r tts_api/requirements.txt
```

The conda env (`environment.yml`) is the canonical dependency spec — `requirements.txt` only covers the FastAPI/serving deps and lacks `transformers`, `parler_tts`, and `sentencepiece`.

## Architecture

The API is a single-worker FastAPI server with an internal dynamic batching pipeline for GPU efficiency. The request flow is:

```
POST /tts
  └─ enqueue_request()      # chunk text, push items to asyncio.Queue
       └─ batcher_worker()  # background coroutine: collects items up to
            └─ run_batch()  # BATCH_MAX_SIZE or BATCH_TIMEOUT_MS, then
                 └─ merge_chunks()  # GPU inference → stitch WAV chunks
```

**Key design decisions:**

- **Text chunking** (`chunker.py`): Long input is split into ≤50-token chunks (configurable via `CHUNK_TOKEN_LIMIT`) using a hierarchy: sentence boundaries (`.!?।`) → clause boundaries (`,;:`) → conjunction lookahead → greedy word split. Chunks are processed independently then merged.
- **Dynamic batching** (`batcher.py`): A single background asyncio task blocks on the queue, then drains up to `BATCH_MAX_SIZE=16` items within a `BATCH_TIMEOUT_MS=30ms` window. GPU inference (`run_batch`) runs in a threadpool via `asyncio.to_thread` to avoid blocking the event loop.
- **Chunk reassembly** (`merge.py`): Audio chunks are collected into a per-request state dict keyed by UUID. When all chunks for a request complete, they are concatenated with 20ms silence between them, peak-normalized, and encoded as PCM_16 WAV.
- **Model loading** (`model.py`): Loads the base `indic-parler-tts-pretrained` from `ROOT_MODEL_PATH`, then patches weights from the fine-tuned checkpoint at `MODEL_PATH/pytorch_model.bin` using `load_state_dict(strict=False)`. Uses Flash Attention 2 for the decoder, eager attention for the text encoder.

## Configuration

All tuneable parameters live in `tts_api/config.py`. Key values:

| Parameter | Default | Effect |
|---|---|---|
| `MODEL_PATH` | `.../checkpoint-6240-epoch-19` | Fine-tuned checkpoint weights |
| `CHUNK_TOKEN_LIMIT` | 50 | Max tokens per TTS chunk |
| `BATCH_MAX_SIZE` | 16 | Max GPU batch size |
| `BATCH_TIMEOUT_MS` | 30 | Batch collection window |
| `COMPILE_MODEL` | `True` | Enables `torch.compile` (slow first start) |
| `WARMUP_RUNS` | 3 | Synthetic warmup passes at startup |

## API endpoints

- `GET /` — Interactive demo UI (served from `tts_api/static/index.html`)
- `POST /tts` — Single text → WAV bytes (`audio/wav`)
- `POST /tts/batch` — Multiple texts → JSON with base64-encoded WAVs
- `GET /health` — Model status, queue depth, batch metrics, GPU stats

## Notable quirks

- `enqueue_request` writes chunks to `chunked_text_output.txt` on every request (debug artifact, not cleaned up).
- The `/tts` endpoint also saves `generated_speech.wav` to the working directory on every request.
- The description tokenizer is hardcoded to `google/flan-t5-large` regardless of the checkpoint.
- Only `workers=1` is supported — the asyncio queue and `_models` global are not process-safe.
