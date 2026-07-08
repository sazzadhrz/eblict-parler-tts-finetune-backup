from transformers import AutoTokenizer

# Load the Indic-specific prompt tokenizer
model_name = "ai4bharat/indic-parler-tts-pretrained"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bengali_sentences = [
    "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।",
    "যুক্তাক্ষর পরীক্ষা: বিজ্ঞান, পঙ্কজ, তৃষ্ণা।",
    "পার্লার টিটিএস বাংলা ভাষায় কথা বলতে পারে।",
    "আজকের আবহাওয়া খুব চমৎকার।"
]

with open("tokenizer_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Detailed Tokenizer Test: {model_name}\n")
    f.write("="*60 + "\n\n")

    for i, text in enumerate(bengali_sentences, 1):
        # 1. Convert text to numerical IDs
        input_ids = tokenizer.encode(text)
        
        # 2. Break down into visible tokens (subwords)
        tokens = tokenizer.tokenize(text)
        
        # 3. Convert IDs back into human-readable text
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        f.write(f"--- Sample {i} ---\n")
        f.write(f"Original: {text}\n")
        f.write(f"Tokens:   {' | '.join(tokens)}\n")
        f.write(f"IDs:      {input_ids}\n")
        f.write(f"Decoded:  {decoded_text}\n")
        
        # Validation Check
        if text.strip() == decoded_text.strip():
            f.write("Status:   ✅ Match Successful\n")
        else:
            f.write("Status:   ❌ ERROR: Mismatch detected!\n")
            
        f.write("\n")

print("Done! Open 'tokenizer_results.txt' to see the decoded text.")