import torch
import numpy as np
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer

# -- Config --
MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune/checkpoint-6240-epoch-19"
ROOT_MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune"
BASE_MODEL_PATH = "ai4bharat/indic-parler-tts-pretrained"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global containers
model = None
tokenizer = None
description_tokenizer = None

def load_models():
    global model, tokenizer, description_tokenizer
    print(f"Loading model on {DEVICE}...")
    
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        attn_implementation="eager", 
    )
    
    ckpt_path = f"{MODEL_PATH}/pytorch_model.bin"
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()

    # Optimization
    if DEVICE == "cuda":
        model = torch.compile(model)
    
    description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained(ROOT_MODEL_PATH)
    print("Model loaded and compiled.")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def get_ui():
    # This assumes index.html is in the same folder as server.py
    return FileResponse('index.html')

@app.websocket("/ws/generate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Expecting JSON: {"text": "...", "description": "..."}
            data = await websocket.receive_text()
            payload = json.loads(data)
            text = payload.get("text", "")
            description = payload.get("description", "")

            input_ids = description_tokenizer(description, return_tensors="pt").to(DEVICE)
            prompt_input_ids = tokenizer(text, return_tensors="pt").to(DEVICE)

            # The Streamer handles the chunking of audio
            # play_steps determines how many tokens to wait for before yielding audio
            streamer = ParlerTTSStreamer(model, device=DEVICE, play_steps=20)

            generation_kwargs = dict(
                input_ids=input_ids.input_ids,
                prompt_input_ids=prompt_input_ids.input_ids,
                streamer=streamer,
                do_sample=True,
                temperature=1.0,
            )

            # Run generation in a separate thread so it doesn't block the event loop
            thread = asyncio.to_thread(model.generate, **generation_kwargs)
            asyncio.create_task(thread)

            for new_audio in streamer:
                if new_audio.shape[0] > 0:
                    # Convert to float32 and send as bytes
                    audio_bytes = new_audio.astype(np.float32).tobytes()
                    await websocket.send_bytes(audio_bytes)
            
            # Send an empty message or specific flag to indicate end of stream
            await websocket.send_text("EOS")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)