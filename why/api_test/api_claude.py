import asyncio
import io
import time
import uuid
import logging
import numpy as np
import torch
import soundfile as sf

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("tts_server")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune/checkpoint-6240-epoch-19"
ROOT_MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune"
BASE_MODEL_PATH = "ai4bharat/indic-parler-tts-pretrained"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Parler-TTS degrades hard past ~40 prompt tokens — keep chunks short
CHUNK_TOKEN_LIMIT = 35
MAX_NEW_TOKENS    = 2048
COMPILE_MODE      = "reduce-overhead"   # change to "max-autotune" for max speed after longer warmup

DEFAULT_DESCRIPTION = (
    "Sazzad speech has a moderate pace with a moderate tone, offering a somewhat clear "
    "audio quality in a somewhat confined space with minimal background noise."
)

# ── GPU flags (safe, no attention surgery) ────────────────────────────────────
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True   # auto-tune conv kernels for fixed shapes

ml_models: dict = {}


# ── Text chunker ──────────────────────────────────────────────────────────────
def chunk_text_by_tokens(text: str, tokenizer, max_tokens: int) -> list[str]:
    """
    Split text into chunks ≤ max_tokens.
    Splits on Bengali/Latin sentence boundaries first, then word-level fallback.
    """
    import re
    sentences = re.split(r'(?<=[।.!?])\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        tok_len = len(tokenizer(sentence, add_special_tokens=False).input_ids)

        if current_len + tok_len > max_tokens and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0

        if tok_len > max_tokens:
            # Hard word-level split for very long sentences
            words = sentence.split()
            sub, sub_len = [], 0
            for word in words:
                wl = len(tokenizer(word, add_special_tokens=False).input_ids)
                if sub_len + wl > max_tokens and sub:
                    chunks.append(" ".join(sub))
                    sub, sub_len = [], 0
                sub.append(word)
                sub_len += wl
            if sub:
                chunks.append(" ".join(sub))
        else:
            current.append(sentence)
            current_len += tok_len

    if current:
        chunks.append(" ".join(current))

    return chunks


# ── Core inference (blocking — runs in a thread via asyncio.to_thread) ────────
def _run_inference(text: str, description: str, max_new_tokens: int) -> tuple[bytes, dict]:
    model      = ml_models["model"]
    desc_tok   = ml_models["desc_tok"]
    prompt_tok = ml_models["prompt_tok"]

    t_start = time.perf_counter()

    # Tokenize description once — reused across all chunks
    desc_inputs = desc_tok(description, return_tensors="pt").to(DEVICE)

    chunks = chunk_text_by_tokens(text, prompt_tok, CHUNK_TOKEN_LIMIT)
    log.info(f"Text split into {len(chunks)} chunk(s)")

    audio_segments = []
    sampling_rate  = model.config.sampling_rate

    for i, chunk in enumerate(chunks):
        t_chunk = time.perf_counter()

        prompt_inputs = prompt_tok(chunk, return_tensors="pt").to(DEVICE)

        with torch.inference_mode():
            # autocast to bfloat16 — safe, doesn't touch model weights or logits path
            # This is what actually drives up GPU utilization on L40S
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                generation = model.generate(
                    input_ids=desc_inputs.input_ids,
                    attention_mask=desc_inputs.attention_mask,
                    prompt_input_ids=prompt_inputs.input_ids,
                    prompt_attention_mask=prompt_inputs.attention_mask,
                    do_sample=True,
                    temperature=0.9,
                    max_new_tokens=max_new_tokens,
                )

        # Cast to float32 for audio post-processing — always safe
        audio_np = generation.cpu().float().numpy()

        # Normalize shape: (batch, channels, time) → (time,)
        if audio_np.ndim == 3:
            audio_np = audio_np[0, 0]
        elif audio_np.ndim == 2:
            audio_np = audio_np[0]
        else:
            audio_np = audio_np.squeeze()

        chunk_time = time.perf_counter() - t_chunk
        chunk_dur  = len(audio_np) / sampling_rate
        log.info(
            f"  chunk {i+1}/{len(chunks)} | "
            f'"{chunk[:40]}..." | '
            f"tokens={prompt_inputs.input_ids.shape[1]} | "
            f"audio={chunk_dur:.2f}s | "
            f"took={chunk_time:.2f}s"
        )

        audio_segments.append(audio_np)

    # Concatenate chunks with a 150 ms silence gap between sentences
    if len(audio_segments) == 1:
        full_audio = audio_segments[0]
    else:
        silence = np.zeros(int(sampling_rate * 0.15), dtype=np.float32)
        parts = []
        for j, seg in enumerate(audio_segments):
            parts.append(seg)
            if j < len(audio_segments) - 1:
                parts.append(silence)
        full_audio = np.concatenate(parts)

    # Clip to [-1, 1] to prevent WAV clipping artifacts
    full_audio = np.clip(full_audio, -1.0, 1.0)

    buf = io.BytesIO()
    sf.write(buf, full_audio, sampling_rate, format="WAV", subtype="PCM_16")
    audio_bytes = buf.getvalue()

    t_total   = time.perf_counter() - t_start
    audio_dur = len(full_audio) / sampling_rate
    meta = {
        "chunks":         len(chunks),
        "audio_duration": round(audio_dur, 2),
        "inference_time": round(t_total, 3),
        "rtf":            round(t_total / audio_dur, 3) if audio_dur > 0 else 0,
        "audio_bytes":    len(audio_bytes),
        "sampling_rate":  sampling_rate,
    }
    return audio_bytes, meta


# ── Warmup — forces torch.compile to JIT before first real request ────────────
def _warmup():
    log.info("Running warmup (compiling CUDA kernels, ~60s) ...")
    model      = ml_models["model"]
    desc_tok   = ml_models["desc_tok"]
    prompt_tok = ml_models["prompt_tok"]

    desc_inputs   = desc_tok("A clear voice.", return_tensors="pt").to(DEVICE)
    prompt_inputs = prompt_tok("হ্যালো", return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                do_sample=False,
                max_new_tokens=50,
            )
    log.info("Warmup complete.")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Loading model on {DEVICE} ...")

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # attn_implementation="eager",   # only safe option until T5+decoder both support FA2
        attn_implementation = {
            "decoder": "flash_attention_2",
            "text_encoder": "eager"
        }
    )

    ckpt_path = f"{MODEL_PATH}/pytorch_model.bin"
    log.info(f"Applying fine-tuned weights: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()

    if DEVICE == "cuda":
        log.info(f"Compiling with mode='{COMPILE_MODE}' ...")
        model = torch.compile(model, mode=COMPILE_MODE)

    ml_models["model"]      = model
    ml_models["desc_tok"]   = AutoTokenizer.from_pretrained("google/flan-t5-large")
    ml_models["prompt_tok"] = AutoTokenizer.from_pretrained(ROOT_MODEL_PATH)

    if DEVICE == "cuda":
        _warmup()
        vram_gb = torch.cuda.memory_allocated() / 1e9
        log.info(f"VRAM in use after warmup: {vram_gb:.2f} GB")

    log.info("✅ Server ready.")
    yield
    ml_models.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Parler-TTS Server", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, example="আমার সোনার বাংলা")
    description: str = Field(default=DEFAULT_DESCRIPTION)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS, ge=50, le=4096)


@app.post("/generate")
async def generate_audio(req: TTSRequest):
    request_id = str(uuid.uuid4())[:8]
    log.info(
        f"[{request_id}] POST /generate | "
        f"chars={len(req.text)} | "
        f'preview="{req.text[:60]}"'
    )

    try:
        t0 = time.perf_counter()
        audio_bytes, meta = await asyncio.to_thread(
            _run_inference, req.text, req.description, req.max_new_tokens
        )
        wall = time.perf_counter() - t0

        log.info(
            f"[{request_id}] ✅ | "
            f"chunks={meta['chunks']} | "
            f"audio={meta['audio_duration']}s | "
            f"inference={meta['inference_time']}s | "
            f"RTF={meta['rtf']} | "
            f"wall={wall:.3f}s | "
            f"bytes={meta['audio_bytes']}"
        )

        output_filename = "generated_speech.wav"

        # Standard python binary write
        with open(output_filename, "wb") as f:
            f.write(audio_bytes)

        print(f"Successfully saved audio to {output_filename}")

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=speech.wav",
                "X-Request-Id":        request_id,
                "X-Audio-Duration":    str(meta["audio_duration"]),
                "X-Inference-Time":    str(meta["inference_time"]),
                "X-RTF":               str(meta["rtf"]),
                "X-Chunks":            str(meta["chunks"]),
            },
        )

    except Exception as e:
        log.exception(f"[{request_id}] ❌ {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    info = {"status": "ok", "device": DEVICE}
    if DEVICE == "cuda":
        info["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        info["vram_reserved_gb"]  = round(torch.cuda.memory_reserved()   / 1e9, 2)
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="uvloop",      # pip install uvloop
        http="httptools",   # pip install httptools
    )