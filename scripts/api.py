import torch
import io
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# -- Config --
MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune/checkpoint-6240-epoch-19"
ROOT_MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune"
BASE_MODEL_PATH = "ai4bharat/indic-parler-tts-pretrained"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KANAK_DESCRIPTION = (
    "Kanak speaks at a fast speed with a slightly expressive delivery. "
    "The recording is very clear and quite confined sounding."
)

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    
    ml_models["model"] = model
    ml_models["desc_tok"] = AutoTokenizer.from_pretrained("google/flan-t5-large")
    ml_models["prompt_tok"] = AutoTokenizer.from_pretrained(ROOT_MODEL_PATH)
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/generate")
async def generate_audio(
    text: str = Query(..., example="আমার সোনার বাংলা"), 
    description: str = Query(default=KANAK_DESCRIPTION)
):
    """
    Using GET makes the audio player work better in Swagger and Browsers.
    """
    try:
        model = ml_models["model"]
        
        with torch.inference_mode():
            desc_inputs = ml_models["desc_tok"](description, return_tensors="pt").to(DEVICE)
            prompt_inputs = ml_models["prompt_tok"](text, return_tensors="pt").to(DEVICE)

            generation = model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                do_sample=True,
                temperature=1.0,
            )

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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)