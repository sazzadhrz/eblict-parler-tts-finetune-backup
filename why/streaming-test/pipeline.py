#!/usr/bin/env python3
"""
Streaming Parler-TTS inference pipeline for finetuned indic-parler-tts model.

Usage:
    conda run -n infer python pipeline.py "Text to speak"
    conda run -n infer python pipeline.py "Text to speak" --out output.wav --play
"""
import argparse
import math
import os
import threading
import time

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer

MODEL_PATH      = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune/checkpoint-6240-epoch-19"
ROOT_MODEL_PATH = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune"

DEFAULT_DESCRIPTION = "Sazzad speech has a faster pace and clear studio recording quality."

CHUNK_DURATION_S = 0.5


def load_model(device="cuda"):
    """Load tokenizer and model, then overlay finetuned checkpoint weights."""
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(ROOT_MODEL_PATH)

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        ROOT_MODEL_PATH,
        torch_dtype=dtype,
    )

    ckpt_file = os.path.join(MODEL_PATH, "pytorch_model.bin")
    state_dict = torch.load(ckpt_file, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # strict=False because checkpoint has an extra T5 weight-tying key:
    #   text_encoder.encoder.embed_tokens.weight (duplicate of shared.weight)
    assert len(missing) == 0, f"Missing keys in checkpoint: {missing}"

    model = model.to(device).eval()
    sampling_rate = model.audio_encoder.config.sampling_rate  # 44100

    return model, tokenizer, sampling_rate


def _make_streamer(model, device, sampling_rate):
    hop_length = math.prod(model.audio_encoder.config.downsampling_ratios)  # 512
    frame_rate  = sampling_rate / hop_length                                 # ~86.13
    play_steps  = int(frame_rate * CHUNK_DURATION_S)                         # 43
    return ParlerTTSStreamer(model, device=device, play_steps=play_steps, timeout=30.0)


def generate_stream(text, description, model, tokenizer, device, sampling_rate):
    """Yield float32 numpy audio chunks as they are decoded."""
    desc_inputs = tokenizer(description, return_tensors="pt", padding=True).to(device)
    text_inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)

    streamer = _make_streamer(model, device, sampling_rate)

    generate_kwargs = dict(
        input_ids=text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask,
        prompt_input_ids=desc_inputs.input_ids,
        prompt_attention_mask=desc_inputs.attention_mask,
        streamer=streamer,
        do_sample=True,
        temperature=1.0,
    )

    thread = threading.Thread(target=model.generate, kwargs=generate_kwargs, daemon=True)
    thread.start()

    for chunk in streamer:
        yield chunk

    thread.join()


def save_to_wav(chunks_iter, output_path, sampling_rate):
    all_chunks = list(chunks_iter)
    audio = np.concatenate(all_chunks, axis=0)
    sf.write(output_path, audio, samplerate=sampling_rate, subtype="PCM_16")
    return len(audio) / sampling_rate


def play_and_save(chunks_iter, output_path, sampling_rate):
    try:
        import sounddevice as sd
        has_sd = True
    except ImportError:
        has_sd = False
        print("[WARNING] sounddevice not installed — falling back to save-only.")
        print("[WARNING] Install with: pip install sounddevice")

    all_chunks = []

    if has_sd:
        with sd.OutputStream(samplerate=sampling_rate, channels=1, dtype="float32") as stream:
            for chunk in chunks_iter:
                all_chunks.append(chunk)
                stream.write(chunk.reshape(-1, 1))
    else:
        all_chunks = list(chunks_iter)

    audio = np.concatenate(all_chunks, axis=0)
    sf.write(output_path, audio, samplerate=sampling_rate, subtype="PCM_16")
    return len(audio) / sampling_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Parler-TTS inference")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--desc", default=DEFAULT_DESCRIPTION, help="Voice/style description")
    parser.add_argument("--out", default="output.wav", help="Output WAV file path")
    parser.add_argument("--play", action="store_true", help="Play audio live while saving")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    args = parser.parse_args()

    print(f"Loading model on {args.device}...")
    t0 = time.time()
    model, tokenizer, sampling_rate = load_model(device=args.device)
    print(f"Model loaded in {time.time() - t0:.1f}s  |  sampling_rate={sampling_rate} Hz")

    print(f"Generating: {args.text[:80]!r}")
    t1 = time.time()

    stream = generate_stream(args.text, args.desc, model, tokenizer, args.device, sampling_rate)

    if args.play:
        duration = play_and_save(stream, args.out, sampling_rate)
    else:
        duration = save_to_wav(stream, args.out, sampling_rate)

    t2 = time.time()
    print(f"Done.  Audio: {duration:.2f}s  |  Wall time: {t2 - t1:.2f}s  |  Saved: {args.out}")
