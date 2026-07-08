from datasets import load_dataset, concatenate_datasets
from bnunicodenormalizer import Normalizer
import re

# 1. Load both datasets
audio_ds = load_dataset("sazzad-sit/kanak30-tts", split="train")
tag_ds = load_dataset("sazzad-sit/kanak30-tts-tagged", split="train")

# 2. REMOVE the duplicate 'text' column
tag_ds = tag_ds.remove_columns(["text"])

# 3. Merge columns side-by-side
merged_ds = concatenate_datasets([audio_ds, tag_ds], axis=1)

# 4. Correct Sentence Normalization Logic
norm = Normalizer(allow_english=True)

def normalize_sentence(sentence):
    if not sentence:
        return ""
    # Split by whitespace, normalize each word, and join back
    words = sentence.split()
    normalized_words = []
    for word in words:
        # The normalizer returns a dict; we extract 'normalized'
        res = norm(word)
        normalized_word = res['normalized'] if res['normalized'] is not None else word
        normalized_words.append(normalized_word)
    return " ".join(normalized_words)

def normalize_and_validate(example):
    # Normalize the main Bengali text
    example["text"] = normalize_sentence(example["text"])
    
    # Safety check for description (this is English, so we don't normalize it)
    if not example["text_description"] or len(str(example["text_description"])) < 2:
        example["text_description"] = "A clear Bengali voice with a natural pace."
    
    return example

# Apply the mapping
merged_ds = merged_ds.map(normalize_and_validate)

# 5. Push to Hub
merged_ds.push_to_hub("sazzad-sit/kanak30-tts-merged")

print("Done! The dataset is now normalized and merged.")