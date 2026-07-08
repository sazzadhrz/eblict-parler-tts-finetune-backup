#!/usr/bin/env python3
"""
Gradio interface for the streaming Parler-TTS pipeline.

Run:
    conda run -n infer python app.py
"""
import time

import gradio as gr
import numpy as np
import torch

from pipeline import DEFAULT_DESCRIPTION, generate_stream, load_model

print("Loading model...")
t0 = time.time()
model, tokenizer, SAMPLING_RATE = load_model(device="cuda" if torch.cuda.is_available() else "cpu")
device = next(model.parameters()).device
print(f"Model ready in {time.time() - t0:.1f}s on {device}")


def synthesize(text, description):
    if not text.strip():
        return None, "Enter some text first."

    t0 = time.time()
    chunks = []
    for chunk in generate_stream(text, description, model, tokenizer, str(device), SAMPLING_RATE):
        chunks.append(chunk)

    audio = np.concatenate(chunks, axis=0)
    duration = len(audio) / SAMPLING_RATE
    elapsed = time.time() - t0
    return (SAMPLING_RATE, audio), f"Generated {duration:.2f}s of audio in {elapsed:.2f}s"


with gr.Blocks(title="Parler TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Parler TTS — Streaming Inference")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text",
                placeholder="Enter text to synthesize...",
                lines=4,
            )
            desc_input = gr.Textbox(
                label="Voice description",
                value=DEFAULT_DESCRIPTION,
                lines=2,
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Output",
                type="numpy",
                autoplay=True,
            )
            status = gr.Textbox(label="Status", interactive=False, lines=1)

    generate_btn.click(
        fn=synthesize,
        inputs=[text_input, desc_input],
        outputs=[audio_output, status],
    )
    clear_btn.click(
        fn=lambda: ("", DEFAULT_DESCRIPTION, None, ""),
        outputs=[text_input, desc_input, audio_output, status],
    )

    gr.Examples(
        examples=[
            ["Hello, this is a test of the streaming TTS system."],
            ["The quick brown fox jumps over the lazy dog."],
        ],
        inputs=text_input,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
