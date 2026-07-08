import asyncio
import io
import time
import uuid
import logging
import numpy as np
import torch
import soundfile as sf

from fastapi import FastAPI
from fastapi.responses import Response
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("tts")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune/checkpoint-6240-epoch-19"
ROOT_MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune"
BASE_MODEL_PATH = "ai4bharat/indic-parler-tts-pretrained"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHUNK_TOKEN_LIMIT = 35
BATCH_MAX_SIZE = 12
BATCH_TIMEOUT_MS = 30

DEFAULT_DESCRIPTION = (
    "Sazzad speech has a moderate pace and clear studio recording quality."
)

# ─────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────
ml_models = {}
queue = asyncio.Queue()

# ─────────────────────────────────────────────
# SAFE tokenizer fix (IMPORTANT)
# ─────────────────────────────────────────────
def fix_tokenizer(tok):

    # 🔥 CRITICAL: avoid eos==pad ambiguity
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "right"

    return tok

# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────
def chunk_text(text, tokenizer, limit):
    import re

    sentences = re.split(r'(?<=[।.!?])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, cur, cur_len = [], [], 0

    for s in sentences:
        l = len(tokenizer(s, add_special_tokens=False).input_ids)

        if cur_len + l > limit and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0

        if l > limit:
            words = s.split()
            tmp, tmp_len = [], 0

            for w in words:
                wl = len(tokenizer(w, add_special_tokens=False).input_ids)

                if tmp_len + wl > limit:
                    chunks.append(" ".join(tmp))
                    tmp, tmp_len = [], 0

                tmp.append(w)
                tmp_len += wl

            if tmp:
                chunks.append(" ".join(tmp))
        else:
            cur.append(s)
            cur_len += l

    if cur:
        chunks.append(" ".join(cur))

    return chunks

# ─────────────────────────────────────────────
# SAFE batch inference (FIXED MASK HANDLING)
# ─────────────────────────────────────────────
def run_batch(batch):

    model = ml_models["model"]
    desc_tok = ml_models["desc_tok"]
    prompt_tok = ml_models["prompt_tok"]

    texts = [b["chunk"] for b in batch]
    descs = [b["desc"] for b in batch]

    max_tokens = max(b["max_new_tokens"] for b in batch)

    # ─────────────────────────────────────────────
    # FORCE attention masks (FIX CRASH ROOT CAUSE)
    # ─────────────────────────────────────────────
    desc = desc_tok(
        descs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    ).to(DEVICE)

    prompt = prompt_tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    ).to(DEVICE)

    # 🔥 HARD FIX: ensure masks are valid (no None, no weird dtype)
    desc_attention_mask = desc.attention_mask.long()
    prompt_attention_mask = prompt.attention_mask.long()

    outputs = []

    log.info(f"📦 batch_size={len(batch)} prompt_len={prompt.input_ids.shape}")

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

            for i in range(len(batch)):

                # ─────────────────────────────
                # encode description safely
                # ─────────────────────────────
                enc = model.text_encoder(
                    input_ids=desc.input_ids[i:i+1],
                    attention_mask=desc_attention_mask[i:i+1],
                )

                hidden = enc.last_hidden_state

                # ─────────────────────────────
                # generate safely
                # ─────────────────────────────
                out = model.generate(
                    prompt_input_ids=prompt.input_ids[i:i+1],
                    prompt_attention_mask=prompt_attention_mask[i:i+1],

                    prompt_hidden_states=hidden,

                    do_sample=True,
                    temperature=0.5,
                    max_new_tokens=max_tokens,
                )

                audio = out.cpu().float().numpy()[0]

                if audio.ndim == 2:
                    audio = audio[0]

                outputs.append(np.clip(audio.squeeze(), -1, 1))

    return outputs

# ─────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────
async def worker():

    log.info("🚀 Worker started")

    sr = ml_models["model"].config.sampling_rate
    silence = np.zeros(int(sr * 0.15), dtype=np.float32)

    while True:

        batch = []

        first = await queue.get()
        batch.append(first)

        start = time.perf_counter()

        while len(batch) < BATCH_MAX_SIZE:

            if (time.perf_counter() - start) > (BATCH_TIMEOUT_MS / 1000):
                break

            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.01)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        log.info(f"📦 batch={len(batch)} queue={queue.qsize()}")

        outputs = await asyncio.to_thread(run_batch, batch)

        for item, audio in zip(batch, outputs):

            req = item["req"]
            idx = item["idx"]

            req["chunks"][idx] = audio
            req["done"] += 1

            if req["done"] == len(req["chunks"]):

                full = []

                for i, c in enumerate(req["chunks"]):
                    full.append(c)
                    if i < len(req["chunks"]) - 1:
                        full.append(silence)

                audio = np.concatenate(full)

                buf = io.BytesIO()
                sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")

                req["future"].set_result(buf.getvalue())

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
app = FastAPI()

class Req(BaseModel):
    text: str
    description: str = DEFAULT_DESCRIPTION
    max_new_tokens: int = 2048

# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):

    log.info("📦 loading model")

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation={
            "decoder": "flash_attention_2",
            "text_encoder": "eager"
        }
    )

    state = torch.load(
        f"{MODEL_PATH}/pytorch_model.bin",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state, strict=False)

    model.to(DEVICE).eval()

    ml_models["model"] = model

    ml_models["desc_tok"] = fix_tokenizer(
        AutoTokenizer.from_pretrained("google/flan-t5-large")
    )

    ml_models["prompt_tok"] = fix_tokenizer(
        AutoTokenizer.from_pretrained(ROOT_MODEL_PATH)
    )

    task = asyncio.create_task(worker())

    log.info("✅ ready")

    yield

    task.cancel()

app.router.lifespan_context = lifespan

# ─────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────
@app.post("/generate")
async def generate(req: Req):

    rid = str(uuid.uuid4())[:8]

    prompt_tok = ml_models["prompt_tok"]

    chunks = chunk_text(req.text, prompt_tok, CHUNK_TOKEN_LIMIT)

    future = asyncio.get_event_loop().create_future()

    state = {
        "future": future,
        "chunks": [None] * len(chunks),
        "done": 0
    }

    for i, c in enumerate(chunks):

        await queue.put({
            "req": state,
            "idx": i,
            "chunk": c,
            "desc": req.description,
            "max_new_tokens": req.max_new_tokens
        })

    audio = await future

    return Response(content=audio, media_type="audio/wav")

# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

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