import asyncio
import io
import re
import numpy as np
import torch
import soundfile as sf

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# =========================
# CONFIG
# =========================

MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune/checkpoint-6240-epoch-19"
ROOT_MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune"
BASE_MODEL_PATH = "ai4bharat/indic-parler-tts-pretrained"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_TEXT_LENGTH = 15000

DESCRIPTION = "Sazzad speaks clearly in a calm voice."

ml_models = {}

# =========================
# SPLIT TEXT
# =========================

def split_sentences(text):
    return [t.strip() for t in re.split(r'(?<=[।.!?])\s+', text) if t.strip()]

# =========================
# FIXED AUDIO GEN
# =========================

def generate_wav(text, description, max_new_tokens):

    model = ml_models["model"]
    desc_tok = ml_models["desc_tok"]
    prompt_tok = ml_models["prompt_tok"]

    desc = desc_tok(description, return_tensors="pt").to(DEVICE)
    prompt = prompt_tok(text, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():

        with torch.autocast("cuda", dtype=torch.float16):

            audio = model.generate(
                input_ids=desc.input_ids,
                attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
                do_sample=False,
                use_cache=True,
                max_new_tokens=max_new_tokens,
            )

    audio = audio.detach().cpu().float().numpy()

    # =========================
    # FIX 1: proper shape
    # =========================

    if audio.ndim == 3:
        audio = audio[0, 0]
    elif audio.ndim == 2:
        audio = audio[0]

    # =========================
    # FIX 2: normalization
    # =========================

    audio = np.nan_to_num(audio)

    max_val = np.max(np.abs(audio)) + 1e-8

    audio = audio / max_val  # IMPORTANT

    # =========================
    # encode WAV properly
    # =========================

    buf = io.BytesIO()

    sf.write(
        buf,
        audio.astype(np.float32),
        model.config.sampling_rate,
        format="WAV",
    )

    return buf.getvalue()

# =========================
# STREAMING (SAFE)
# =========================

async def stream_tts(text, description):

    sentences = split_sentences(text)

    for s in sentences:

        wav = await asyncio.to_thread(
            generate_wav,
            s,
            description,
            min(2048, max(256, len(s) * 12)),
        )

        # safe chunk streaming (WAV intact per sentence)
        chunk_size = 16384

        for i in range(0, len(wav), chunk_size):
            yield wav[i:i + chunk_size]

# =========================
# LIFESPAN
# =========================

@asynccontextmanager
async def lifespan(app):

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )

    ckpt = torch.load(
        f"{MODEL_PATH}/pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(ckpt, strict=False)

    model.to(DEVICE).eval()

    model.generation_config.use_cache = True

    if DEVICE == "cuda":
        model = torch.compile(model, mode="max-autotune")

    ml_models["model"] = model
    ml_models["desc_tok"] = AutoTokenizer.from_pretrained("google/flan-t5-large")
    ml_models["prompt_tok"] = AutoTokenizer.from_pretrained(ROOT_MODEL_PATH)

    yield

    ml_models.clear()

# =========================
# API
# =========================

app = FastAPI(lifespan=lifespan)

@app.get("/generate")
async def generate_audio(
    text: str = Query(..., max_length=MAX_TEXT_LENGTH),
    description: str = Query(default=DESCRIPTION),
):

    return StreamingResponse(
        stream_tts(text, description),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
        },
    )

# =========================
# RUN
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)