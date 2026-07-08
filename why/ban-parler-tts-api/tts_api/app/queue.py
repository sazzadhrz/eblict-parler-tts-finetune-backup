import asyncio
import uuid
from typing import Optional

from tts_api.app.chunker import chunk_text
from tts_api.config import VOICE_CONFIGS

_queues: dict = {}  # voice -> asyncio.Queue
_states: dict = {}  # req_id -> {future, chunks, done, total}
_metrics: dict = {"batches": 0, "total_items": 0, "total_latency_ms": 0.0}


def init_queues() -> None:
    global _queues
    _queues = {voice: asyncio.Queue() for voice in VOICE_CONFIGS}


async def enqueue_request(
    text: str,
    description: str,
    max_new_tokens: int,
    tokenizer,
    limit: int,
    voice: str,
) -> bytes:
    """Chunk text, register state, enqueue items for the given voice, await future."""
    chunks = chunk_text(text, tokenizer, limit)
    print(chunks)
    open("chunked_text_output.txt", "w", encoding="utf-8").write("\n".join(chunks))
    if not chunks:
        chunks = [text]

    req_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    _states[req_id] = {
        "future": future,
        "chunks": [None] * len(chunks),
        "done": 0,
        "total": len(chunks),
    }

    for i, chunk in enumerate(chunks):
        await _queues[voice].put(
            {
                "req_id": req_id,
                "idx": i,
                "chunk": chunk,
                "desc": description,
                "max_new_tokens": max_new_tokens,
                "voice": voice,
            }
        )

    return await future


def get_queue(voice: str) -> asyncio.Queue:
    return _queues[voice]


def get_states() -> dict:
    return _states


def get_metrics() -> dict:
    return _metrics
