from threading import Thread
import gradio as gr
import numpy as np
import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer

# 1. Setup device and optimize precision
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f"Using device: {device} with dtype: {torch_dtype}")

# 2. Load model and tokenizer once globally
model_name = "/home/eblict/parler-tts-finetune/parler-tts/output_dir_vertex_male1_finetune"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch_dtype
).to(device)

sampling_rate = model.audio_encoder.config.sampling_rate

# 3. Define the streaming generator for Gradio
def stream_tts(text, description):
    if not text.strip() or not description.strip():
        return

    # Tokenize inputs
    inputs = tokenizer(description, return_tensors="pt").to(device)
    prompt = tokenizer(text, return_tensors="pt").to(device)

    # Initialize Parler Streamer
    play_steps = 15 
    streamer = ParlerTTSStreamer(model, device=device, play_steps=play_steps)

    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        prompt_input_ids=prompt.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_attention_mask=prompt.prompt_attention_mask if hasattr(prompt, 'prompt_attention_mask') else None,
        streamer=streamer,
        do_sample=True,
        temperature=1.0,
    )

    # Start generation in a background thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield audio chunks as they become available
    for new_audio in streamer:
        if new_audio.shape[0] == 0:
            continue
        
        # Gradio streaming audio format: (sampling_rate, numpy_array)
        # We need to make sure the audio is formatted cleanly as float32 or int16
        yield (sampling_rate, new_audio.astype(np.float32))

    thread.join()

# 4. Build the Gradio UI Layout
with gr.Blocks(title="Parler-TTS Streaming UI") as demo:
    gr.Markdown("# 🎙️ Parler-TTS Streaming Interface")
    gr.Markdown("Type your prompt and voice description below, then click **Generate** to stream audio live.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Text to Speak", 
                value="Streaming audio with Parler TTS directly into my web browser is working beautifully!",
                lines=3
            )
            input_desc = gr.Textbox(
                label="Voice Description (Style, Tone, Pace)", 
                value="Sazzad speaks in a fast paces studio quality voice.",
                lines=2
            )
            submit_btn = gr.Button("Generate Audio Stream", variant="primary")
            
        with gr.Column():
            # streaming=True sets up the HTML5 audio element to receive incoming generator chunks
            audio_output = gr.Audio(label="Live Audio Stream", streaming=True, autoplay=True)

    # Connect the button click to our streaming function
    submit_btn.click(
        fn=stream_tts,
        inputs=[input_text, input_desc],
        outputs=audio_output
    )

if __name__ == "__main__":
    # Launch the local web server
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)

