import torch
import soundfile as sf
import argparse
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_kanak_finetune/checkpoint-10000-epoch-9"
DEFAULT_OUTPUT     = "output.wav"

# Kanak's voice profile — tweak these to match your annotation bins
KANAK_DESCRIPTION = (
    "Kanak speaks at a fast speed with a slightly expressive delivery. "
    "The recording is very clear and quite confined sounding."
)

# ── Load ──────────────────────────────────────────────────────────────────────

ROOT_MODEL_PATH  = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_kanak_finetune"
BASE_MODEL_PATH  = "ai4bharat/indic-parler-tts-pretrained"

def load_model(model_path: str, device: str):
    print(f"Loading config from base model : {BASE_MODEL_PATH}")
    print(f"Loading weights from checkpoint: {model_path}")

    # 1. Build model architecture from the original base model config
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # 2. Load finetuned weights on top
    ckpt_path = f"{model_path}/pytorch_model.bin"
    print(f"Loading state dict from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device).eval()

    description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    prompt_tokenizer      = AutoTokenizer.from_pretrained(ROOT_MODEL_PATH)

    return model, description_tokenizer, prompt_tokenizer

# ── Inference ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate(
    text: str,
    description: str,
    model,
    description_tokenizer,
    prompt_tokenizer,
    device: str,
) -> tuple:
    # Tokenize description (voice style)
    desc_inputs = description_tokenizer(
        description,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Tokenize prompt (text to speak)
    prompt_inputs = prompt_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
    ).to(device)

    generation = model.generate(
        input_ids=desc_inputs.input_ids,
        attention_mask=desc_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask,
        do_sample=True,
        temperature=1.0,
        min_new_tokens=10,
    )

    # Move to CPU and convert to numpy
    audio = generation.cpu().float().numpy().squeeze()
    sample_rate = model.config.sampling_rate
    return audio, sample_rate

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kanak Bengali TTS Inference")
    parser.add_argument("--text",        type=str, required=True,             help="Bengali text to synthesize")
    parser.add_argument("--description", type=str, default=KANAK_DESCRIPTION, help="Voice description prompt")
    parser.add_argument("--model_path",  type=str, default=DEFAULT_MODEL_PATH,help="Path to finetuned model")
    parser.add_argument("--output",      type=str, default=DEFAULT_OUTPUT,    help="Output .wav file path")
    parser.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, desc_tok, prompt_tok = load_model(args.model_path, args.device)

    print(f"Synthesizing: {args.text}")
    audio, sr = generate(
        text=args.text,
        description=args.description,
        model=model,
        description_tokenizer=desc_tok,
        prompt_tokenizer=prompt_tok,
        device=args.device,
    )

    sf.write(args.output, audio, sr)
    print(f"Saved to: {args.output}  (sample_rate={sr})")


if __name__ == "__main__":
    main()



"""
######## Usage Examples ########

# Basic
python infer_kanak.py --text "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।"

# Custom output path
python infer_kanak.py \
    --text "উনিশ মে দুপুর তিনটায় হোটেল সুন্দরবনে আপনার একটি রিজার্ভেশন আছে।" \
    --output kanak_test.wav

# Override voice description
python infer_kanak.py \
    --text "আমার সোনার বাংলা" \
    --description "Kanak speaks slightly fast with a very expressive delivery. The recording is very clear."

# Use a different checkpoint
python infer_kanak.py \
    --text "আমার সোনার বাংলা" \
    --model_path "./output_dir_kanak_finetune/checkpoint-1000/"
"""