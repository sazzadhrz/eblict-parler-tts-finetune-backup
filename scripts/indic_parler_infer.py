import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the Indic-specific model
model_name = "ai4bharat/indic-parler-tts"
model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "বর্তমান রাষ্ট্রপতির মেয়াদ এখনো রয়ে গেছে। নতুন রাষ্ট্রপতি হিসেবে বেশি আলোচনা খন্দকার মোশাররফকে নিয়ে।"
description = "Aditi speaks with a deep, calm voice at a moderate pace in a clear studio setting."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().float().numpy().squeeze()

sf.write("custom_voice_output.wav", audio_arr, model.config.sampling_rate)