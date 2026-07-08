import asyncio
import logging
import time

from tts_api.app.merge import merge_chunks
from tts_api.app.model import get_sampling_rate, run_batch
from tts_api.app.queue import get_metrics, get_queue, get_states
from tts_api.config import BATCH_MAX_SIZE, BATCH_TIMEOUT_MS

log = logging.getLogger(__name__)


async def batcher_worker(voice: str) -> None:
    """Background worker for one voice: pulls from its queue, batches, runs inference."""
    log.info(f"[{voice}] Batcher worker started")
    q = get_queue(voice)
    states = get_states()
    metrics = get_metrics()

    while True:
        batch = []

        first = await q.get()
        batch.append(first)

        deadline = time.perf_counter() + BATCH_TIMEOUT_MS / 1000.0

        while len(batch) < BATCH_MAX_SIZE:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(q.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        t0 = time.perf_counter()

        try:
            outputs = await asyncio.to_thread(run_batch, batch, voice)
        except Exception as exc:
            log.exception(f"[{voice}] run_batch failed for batch of {len(batch)}: {exc}")
            for item in batch:
                req_id = item["req_id"]
                state = states.get(req_id)
                if state and not state["future"].done():
                    state["future"].set_exception(exc)
                    states.pop(req_id, None)
            continue

        latency_ms = (time.perf_counter() - t0) * 1000.0

        metrics["batches"] += 1
        metrics["total_items"] += len(batch)
        metrics["total_latency_ms"] += latency_ms

        log.debug(f"[{voice}] Batch of {len(batch)} processed in {latency_ms:.1f} ms")

        sr = get_sampling_rate(voice)

        for item, audio in zip(batch, outputs):
            req_id = item["req_id"]
            idx = item["idx"]
            state = states.get(req_id)
            if state is None:
                continue

            state["chunks"][idx] = audio
            state["done"] += 1

            if state["done"] == state["total"]:
                try:
                    wav_bytes = merge_chunks(state["chunks"], sr)
                    state["future"].set_result(wav_bytes)
                except Exception as exc:
                    log.exception(f"[{voice}] merge_chunks failed for req {req_id}")
                    if not state["future"].done():
                        state["future"].set_exception(exc)
                finally:
                    states.pop(req_id, None)
