import time 
import asyncio
import base64
import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware


_STATIC_DIR = Path(__file__).parent.parent / "static"

from tts_api.app.batcher import batcher_worker
from tts_api.app.model import get_prompt_tok, is_loaded, load_models
from tts_api.app.queue import enqueue_request, get_metrics, get_queue, init_queues
from tts_api.app.schema import HealthResponse, TTSBase64Response, TTSBatchRequest, TTSRequest
from tts_api.app.utils import get_gpu_stats, run_warmup
from tts_api.config import CHUNK_TOKEN_LIMIT, DEFAULT_VOICE, DEVICE, MAX_NEW_TOKENS, VOICE_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_queues()
    log.info("Queues initialized")
    load_models()
    log.info("All models loaded")
    tasks = [asyncio.create_task(batcher_worker(voice)) for voice in VOICE_CONFIGS]
    await run_warmup()
    log.info("Server ready")
    yield
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Parler TTS API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def demo_ui() -> HTMLResponse:
    return HTMLResponse(content=(_STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post(
    "/tts",
    response_class=Response,
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Synthesized speech as a WAV audio file.",
        }
    },
)
async def tts(req: TTSRequest) -> Response:
    """Synthesize speech for a single text string. Returns WAV audio."""
    st = time.time()
    description = req.description or VOICE_CONFIGS[req.voice]["default_description"]
    max_new_tokens = req.max_new_tokens or MAX_NEW_TOKENS
    log.info("Request received: %d chars, voice=%s, max_new_tokens=%d", len(req.text), req.voice, max_new_tokens)
    wav = await enqueue_request(
        text=req.text,
        description=description,
        max_new_tokens=max_new_tokens,
        tokenizer=get_prompt_tok(req.voice),
        limit=CHUNK_TOKEN_LIMIT,
        voice=req.voice,
    )
    et = time.time()
    log.info("TTS request completed in %.2f seconds", et - st)

    with open("generated_speech.wav", "wb") as f:
        f.write(wav)

    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="generated_speech.wav"'},
    )


@app.post("/tts/base64", response_model=TTSBase64Response)
async def tts_base64(req: TTSRequest) -> TTSBase64Response:
    """Synthesize speech and return base64-encoded WAV."""
    description = req.description or VOICE_CONFIGS[req.voice]["default_description"]
    max_new_tokens = req.max_new_tokens or MAX_NEW_TOKENS
    log.info("Base64 request: %d chars, voice=%s", len(req.text), req.voice)
    wav = await enqueue_request(
        text=req.text,
        description=description,
        max_new_tokens=max_new_tokens,
        tokenizer=get_prompt_tok(req.voice),
        limit=CHUNK_TOKEN_LIMIT,
        voice=req.voice,
    )
    return TTSBase64Response(
        voice=req.voice,
        audio=base64.b64encode(wav).decode(),
    )


@app.post("/tts/batch")
async def tts_batch(req: TTSBatchRequest) -> JSONResponse:
    """Synthesize speech for multiple texts concurrently. Returns base64-encoded WAVs."""
    tok = get_prompt_tok(req.voice)
    description = req.description or VOICE_CONFIGS[req.voice]["default_description"]
    max_new_tokens = req.max_new_tokens or MAX_NEW_TOKENS
    tasks = [
        enqueue_request(
            text=text,
            description=description,
            max_new_tokens=max_new_tokens,
            tokenizer=tok,
            limit=CHUNK_TOKEN_LIMIT,
            voice=req.voice,
        )
        for text in req.texts
    ]
    wavs = await asyncio.gather(*tasks)
    return JSONResponse(
        {
            "count": len(wavs),
            "voice": req.voice,
            "audios": [base64.b64encode(w).decode() for w in wavs],
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    metrics = get_metrics()
    n_batches = metrics["batches"]
    gpu_stats = get_gpu_stats()
    total_queue_depth = sum(get_queue(v).qsize() for v in VOICE_CONFIGS)
    return HealthResponse(
        status="ok",
        model_loaded=is_loaded(),
        device=DEVICE,
        queue_depth=total_queue_depth,
        batches_processed=n_batches,
        avg_batch_size=metrics["total_items"] / max(n_batches, 1),
        avg_batch_latency_ms=metrics["total_latency_ms"] / max(n_batches, 1),
        **gpu_stats,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="uvloop",
        http="httptools",
    )
