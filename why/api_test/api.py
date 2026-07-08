import torch
import io
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# -- Speed Optimization Settings --
# TF32 is critical for Ampere+ GPUs (RTX 30xx/40xx)
torch.set_float32_matmul_precision('high')

# -- Config --
MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune/checkpoint-6240-epoch-19"
ROOT_MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune"
BASE_MODEL_PATH = "ai4bharat/indic-parler-tts-pretrained"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DESCRIPTION = (
    "Sazzad speaks slightly slowly with a moderate tone, offering a somewhat clear audio quality in a somewhat confined space with minimal background noise."
)

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model on {DEVICE}...")
    
    # FIX: Set to "eager". 
    # T5 currently requires this, but torch.compile will optimize it later.
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        attn_implementation="eager", 
    )
    
    # Load your fine-tuned weights
    ckpt_path = f"{MODEL_PATH}/pytorch_model.bin"
    print(f"Applying fine-tuned weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    model.to(DEVICE).eval()

    # The Optimization Step:
    # This takes the "eager" code and turns it into highly optimized CUDA kernels.
    if DEVICE == "cuda":
        print("Compiling model (this will take ~1 minute on the first request)...")
        model = torch.compile(model)
    
    ml_models["model"] = model
    ml_models["desc_tok"] = AutoTokenizer.from_pretrained("google/flan-t5-large")
    ml_models["prompt_tok"] = AutoTokenizer.from_pretrained(ROOT_MODEL_PATH)
    
    print("Server is ready!")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/generate")
async def generate_audio(
    text: str = Query(..., example="আমার সোনার বাংলা"), 
    description: str = Query(default=DESCRIPTION)
):
    try:
        model = ml_models["model"]
        desc_tok = ml_models["desc_tok"]
        prompt_tok = ml_models["prompt_tok"]
        
        with torch.inference_mode():
            # Tokenize
            desc_inputs = desc_tok(description, return_tensors="pt").to(DEVICE)
            prompt_inputs = prompt_tok(text, return_tensors="pt").to(DEVICE)

            # Generate
            generation = model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                do_sample=True,
                temperature=1.0,
            )

            # Post-process
            audio_data = generation.cpu().float().numpy().squeeze()
            
            byte_io = io.BytesIO()
            sf.write(byte_io, audio_data, model.config.sampling_rate, format='WAV')
            byte_io.seek(0)

            return StreamingResponse(
                byte_io, 
                media_type="audio/wav",
                headers={"Content-Disposition": "inline; filename=speech.wav"}
            )

    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Single worker is recommended for GPU models to save VRAM
    uvicorn.run(app, host="0.0.0.0", port=8000)