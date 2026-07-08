import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts-pretrained").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts-pretrained")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

prompt = "বর্তমান রাষ্ট্রপতির মেয়াদ এখনো রয়ে গেছে। নতুন রাষ্ট্রপতি হিসেবে বেশি আলোচনা খন্দকার মোশাররফকে নিয়ে।"
description = "Aditi speaks with a deep, calm voice at a moderate pace in a clear studio setting."

description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("indic_tts_out.wav", audio_arr, model.config.sampling_rate)
