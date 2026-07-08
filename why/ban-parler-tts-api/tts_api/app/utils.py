import logging

import torch

from tts_api.app.queue import enqueue_request
from tts_api.app.model import get_prompt_tok
from tts_api.config import (
    CHUNK_TOKEN_LIMIT,
    VOICE_CONFIGS,
    WARMUP_RUNS,
)

log = logging.getLogger(__name__)


def get_gpu_stats() -> dict:
    if not torch.cuda.is_available():
        return {}
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_used_gb": mem.used / 1e9,
            "gpu_memory_total_gb": mem.total / 1e9,
            "gpu_utilization_pct": float(util.gpu),
        }
    except Exception:
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_used_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "gpu_utilization_pct": None,
        }


async def run_warmup(n_runs: int = WARMUP_RUNS) -> None:
    """Warmup all voices to trigger torch.compile and prime GPU."""
    for voice in VOICE_CONFIGS:
        log.info(f"[{voice}] Running {n_runs} warmup inference(s)...")
        tok = get_prompt_tok(voice)
        for i in range(n_runs):
            try:
                await enqueue_request(
                    text="Hello, this is a warmup run.",
                    description=VOICE_CONFIGS[voice]["default_description"],
                    max_new_tokens=100,
                    tokenizer=tok,
                    limit=CHUNK_TOKEN_LIMIT,
                    voice=voice,
                )
                log.info(f"[{voice}] Warmup run {i + 1}/{n_runs} complete")
            except Exception as exc:
                log.warning(f"[{voice}] Warmup run {i + 1} failed: {exc}")
