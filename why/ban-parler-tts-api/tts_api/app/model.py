import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

from tts_api.config import (
    COMPILE_MODEL,
    DEVICE,
    DTYPE,
    VOICE_CONFIGS,
)

log = logging.getLogger(__name__)

# _models[voice] = {model, desc_tok, prompt_tok, sampling_rate}
_models: dict = {}


def _fix_tokenizer(tok):
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _load_one(voice: str, cfg: dict) -> None:
    root = cfg["root_model_path"]
    ckpt_bin = Path(cfg["model_path"]) / "pytorch_model.bin"

    log.info(f"[{voice}] Loading base model from {root}")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        root,
        torch_dtype=DTYPE,
        attn_implementation={
            "decoder": "flash_attention_2",
            "text_encoder": "eager",
        },
    )

    if ckpt_bin.exists():
        log.info(f"[{voice}] Patching with checkpoint: {ckpt_bin}")
        ckpt = torch.load(ckpt_bin, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        log.info(f"[{voice}] Checkpoint loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
    else:
        log.warning(f"[{voice}] Checkpoint not found at {ckpt_bin}, using base model weights")

    model.to(DEVICE).eval()

    if COMPILE_MODEL:
        log.info(f"[{voice}] Compiling with torch.compile(mode='default')")
        model = torch.compile(model, mode="reduce-overhead")

    desc_tok = _fix_tokenizer(AutoTokenizer.from_pretrained("google/flan-t5-large"))
    prompt_tok = _fix_tokenizer(AutoTokenizer.from_pretrained(root))

    _models[voice] = {
        "model": model,
        "desc_tok": desc_tok,
        "prompt_tok": prompt_tok,
        "sampling_rate": model.config.sampling_rate,
    }
    log.info(f"[{voice}] Ready. Sampling rate: {model.config.sampling_rate} Hz")


def load_models() -> None:
    """Load all voices defined in VOICE_CONFIGS."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    for voice, cfg in VOICE_CONFIGS.items():
        _load_one(voice, cfg)


def run_batch(batch: List[dict], voice: str) -> List[np.ndarray]:
    """Batched inference for the given voice. Called in threadpool."""
    m = _models[voice]
    model, desc_tok, prompt_tok = m["model"], m["desc_tok"], m["prompt_tok"]

    texts = [b["chunk"] for b in batch]
    descs = [b["desc"] for b in batch]
    max_tokens = max(b["max_new_tokens"] for b in batch)

    desc_enc = desc_tok(
        descs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    ).to(DEVICE)

    prompt_enc = prompt_tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    ).to(DEVICE)

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.generate(
                input_ids=desc_enc.input_ids,
                attention_mask=desc_enc.attention_mask.long(),
                prompt_input_ids=prompt_enc.input_ids,
                prompt_attention_mask=prompt_enc.attention_mask.long(),
                do_sample=True,
                temperature=0.6,
                max_new_tokens=max_tokens,
                return_dict_in_generate=True,
            )

    results = []
    for i, length in enumerate(output.audios_length):
        audio = output.sequences[i, :length].cpu().float().numpy()
        audio = np.clip(audio.squeeze(), -1.0, 1.0)
        results.append(audio)
    return results


def get_sampling_rate(voice: str) -> int:
    return _models[voice]["sampling_rate"]


def get_prompt_tok(voice: str):
    return _models[voice]["prompt_tok"]


def is_loaded() -> bool:
    return len(_models) == len(VOICE_CONFIGS)
