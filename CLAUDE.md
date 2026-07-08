# CLAUDE.md — Parler TTS Finetune Project

This file tells Claude how this project is structured and how to restore, train, or run inference autonomously.

## Project Overview

Bengali TTS finetuning on top of `ai4bharat/indic-parler-tts-pretrained` (Indic Parler TTS).
Two finetuned speaker models:
- **vertex-female1** → speaker "Sadia"
- **vertex-male1** → speaker "Sazzad"

Base model: `ai4bharat/indic-parler-tts-pretrained`
Feature extractor: `ylacombe/dac_44khz`
Description tokenizer: `google/flan-t5-large`

---

## Directory Structure

```
parler-tts-finetune/
├── parler-tts/                          ← main training/inference code
│   ├── training/run_parler_tts_training.py   ← training entry point
│   ├── parler_tts/                      ← model library (local install)
│   ├── output_dir_vertex_female1_finetune/   ← female model weights
│   │   ├── model.safetensors
│   │   ├── config.json, tokenizer.*
│   │   └── checkpoint-6240-epoch-19/    ← resume-training checkpoint
│   └── output_dir_vertex_male1_finetune/     ← male model weights
│       ├── model.safetensors
│       ├── config.json, tokenizer.*
│       └── checkpoint-6240-epoch-19/
├── notebook_parler_tts_finetune/
│   ├── dataspeech/                      ← dataspeech pipeline (tagging + descriptions)
│   └── parler-tts/                      ← older copy of the library
├── scripts/
│   ├── api.py                           ← simple FastAPI inference server
│   └── infer_kanak.py                   ← CLI inference script
├── why/
│   └── ban-parler-tts-api/              ← production API with batching + queue
│       └── tts_api/app/main.py          ← production server entry point
├── commands_despina.txt                 ← full pipeline commands for female model
├── commands_enceladus.txt               ← full pipeline commands for male model
├── commands_kanak.txt                   ← pipeline commands for kanak model
├── ptts_environment.yml                 ← conda env for training
├── infer_environment.yml                ← conda env for inference
└── test.ipynb                           ← experiment notebook
```

---

## Conda Environments

| Env name | Purpose | Key packages |
|----------|---------|--------------|
| `ptts` | Training + dataspeech | torch, transformers, accelerate, dataspeech, torbi (special), penn |
| `infer` | Inference + API | torch, transformers, fastapi, uvicorn, soundfile |

Always activate the right env before running commands:
- Training → `conda activate ptts`
- Inference/API → `conda activate infer`

---

## Special Package: torbi

torbi was installed from a **local source** (now deleted). The compiled binary is backed up in HF.

**To restore torbi on same machine/arch (Python 3.10, Linux x86_64):**
```bash
huggingface-cli download sazzad-sit/vertex-female1-parler-tts-finetune \
  env_backup/torbi_1.1.0_ptts_env_backup.tar.gz \
  --local-dir /tmp/torbi_restore

cd /tmp && tar xzf torbi_restore/env_backup/torbi_1.1.0_ptts_env_backup.tar.gz
SITE_PKG=$(conda run -n ptts python -c "import site; print(site.getsitepackages()[0])")
cp -r /tmp/torbi_backup/torbi "$SITE_PKG/"
cp -r /tmp/torbi_backup/torbi-1.1.0.dist-info "$SITE_PKG/"
```

**On a different platform:**
```bash
conda run -n ptts pip install git+https://github.com/maxrmorrison/torbi.git@v1.1.0
```

---

## Backup Locations

| Asset | URL |
|-------|-----|
| Code + scripts + env YAMLs | https://github.com/sazzadhrz/eblict-parler-tts-finetune-backup |
| Female model | https://huggingface.co/sazzad-sit/vertex-female1-parler-tts-finetune (private) |
| Male model | https://huggingface.co/sazzad-sit/vertex-male1-parler-tts-finetune (private) |

HF token is saved at `~/.cache/huggingface/token` on this machine.

---

## Full Restore Procedure (new machine)

Run these steps in order:

```bash
# 1. Clone code
git clone https://github.com/sazzadhrz/eblict-parler-tts-finetune-backup.git parler-tts-finetune
cd parler-tts-finetune

# 2. Restore envs
conda env create -f ptts_environment.yml
conda env create -f infer_environment.yml

# 3. Restore torbi (see section above)

# 4. Download model weights
mkdir -p parler-tts/output_dir_vertex_female1_finetune
mkdir -p parler-tts/output_dir_vertex_male1_finetune

huggingface-cli download sazzad-sit/vertex-female1-parler-tts-finetune \
  --local-dir parler-tts/output_dir_vertex_female1_finetune \
  --exclude "env_backup/*"

huggingface-cli download sazzad-sit/vertex-male1-parler-tts-finetune \
  --local-dir parler-tts/output_dir_vertex_male1_finetune \
  --exclude "env_backup/*"

# 5. Install parler_tts library locally
conda activate ptts
cd parler-tts && pip install -e . && cd ..
```

---

## Running Inference

### CLI script
```bash
conda activate infer
python scripts/infer_kanak.py \
  --text "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।" \
  --model_path "parler-tts/output_dir_vertex_male1_finetune/checkpoint-6240-epoch-19" \
  --output output.wav
```

### Simple API (scripts/api.py)
```bash
conda activate infer
# Edit MODEL_PATH and ROOT_MODEL_PATH in scripts/api.py if needed
uvicorn scripts.api:app --host 0.0.0.0 --port 8000
# GET http://localhost:8000/generate?text=আমার+সোনার+বাংলা
```

### Production API (why/ban-parler-tts-api/)
```bash
conda activate infer
cd why/ban-parler-tts-api
pip install -r tts_api/requirements.txt
python -m tts_api.app.main
```
- `POST /tts` — WAV audio
- `POST /tts/base64` — base64 WAV
- `POST /tts/batch` — batch
- `GET /health` — status

---

## Training Pipeline

See `commands_despina.txt` (female) and `commands_enceladus.txt` (male) for the exact commands used for existing models. The pipeline has 4 steps:

### Step 1: Dataspeech tagging
```bash
conda activate ptts
cd notebook_parler_tts_finetune/dataspeech
python main.py "sazzad-sit/DATASET" --text_column_name "text" --audio_column_name "audio" \
  --cpu_num_workers 16 --num_workers_per_gpu_for_pitch 8 --rename_column \
  --repo_id "DATASET-tags" --penn_batch_size 256
```

### Step 2: Metadata to text
```bash
python ./scripts/metadata_to_text.py "sazzad-sit/DATASET-tags" \
  --repo_id "sazzad-sit/DATASET-tags" --cpu_num_workers 2 --avoid_pitch_computation \
  --save_bin_edges "./examples/tags_to_annotations/SPEAKER_bin_edges.json"
```

### Step 3: LLM prompt creation
```bash
python ./scripts/run_prompt_creation.py \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" \
  --per_device_eval_batch_size 8 \
  --dataset_name "sazzad-sit/DATASET-tags" \
  --output_dir "./SPEAKER-descriptions" \
  --hub_dataset_id "sazzad-sit/DATASET-descriptions" \
  --push_to_hub --is_single_speaker --speaker_name "SPEAKER_NAME" \
  --preprocessing_num_workers 2
```

### Step 4: Training
```bash
conda activate ptts
cd parler-tts
accelerate launch ./training/run_parler_tts_training.py \
  --model_name_or_path "ai4bharat/indic-parler-tts-pretrained" \
  --feature_extractor_name "ylacombe/dac_44khz" \
  --description_tokenizer_name "google/flan-t5-large" \
  --prompt_tokenizer_name "ai4bharat/indic-parler-tts-pretrained" \
  --report_to "none" --overwrite_output_dir true \
  --train_dataset_name "sazzad-sit/DATASET" \
  --train_metadata_dataset_name "sazzad-sit/DATASET-descriptions" \
  --train_split_name "train" --train_dataset_config_name "default" \
  --target_audio_column_name "audio" \
  --description_column_name "text_description" --prompt_column_name "text" \
  --max_duration_in_seconds 30 --min_duration_in_seconds 2.0 --max_text_length 400 \
  --do_train true --do_eval false --num_train_epochs 20 \
  --per_device_train_batch_size 4 --audio_encoder_per_device_batch_size 12 \
  --gradient_accumulation_steps 8 --gradient_checkpointing true \
  --learning_rate 1e-4 --adam_beta1 0.9 --adam_beta2 0.99 --weight_decay 0.01 \
  --lr_scheduler_type "cosine" --warmup_steps 1000 \
  --logging_steps 100 --save_steps 1000 \
  --freeze_text_encoder true --dtype "bfloat16" \
  --attn_implementation "eager" --seed 42 \
  --output_dir "./output_dir_SPEAKER_finetune/" \
  --temporary_save_to_disk "./SPEAKER_vectorized_dataset_tmp/" \
  --save_to_disk "./SPEAKER_vectorized_dataset/" \
  --preprocessing_num_workers 4 --dataloader_num_workers 4 \
  --group_by_length true --remove_unused_columns false
```

**To resume from checkpoint**, add:
```
--resume_from_checkpoint "./output_dir_SPEAKER_finetune/checkpoint-6240-epoch-19"
```

**attn_implementation notes:**
- Use `"eager"` if Flash Attention 2 is not available (safe default)
- Use `"flash_attention_2"` for ~2x speed on supported GPUs

---

## Loading Model for Inference (Python)

```python
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

BASE_MODEL = "ai4bharat/indic-parler-tts-pretrained"
MODEL_DIR = "parler-tts/output_dir_vertex_male1_finetune"
CKPT_DIR = f"{MODEL_DIR}/checkpoint-6240-epoch-19"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load architecture from base, then load finetuned weights
model = ParlerTTSForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
)
state_dict = torch.load(f"{CKPT_DIR}/pytorch_model.bin", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE).eval()

desc_tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
prompt_tok = AutoTokenizer.from_pretrained(MODEL_DIR)
```

Alternatively, load directly from `model.safetensors` (no base model needed):
```python
model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_DIR)
```

---

## HF Datasets (Training Data)

- `sazzad-sit/vertex-female1-tts-dataset` — raw audio (Sadia)
- `sazzad-sit/vertex-female1-tts-dataset-tags` — dataspeech tags
- `sazzad-sit/vertex-female1-tts-dataset-descriptions` — LLM descriptions
- `sazzad-sit/vertex-male1-tts-dataset` — raw audio (Sazzad)
- `sazzad-sit/vertex-male1-tts-dataset-tags`
- `sazzad-sit/vertex-male1-tts-dataset-descriptions`
- `sazzad-sit/kanak30-tts` — Kanak dataset (separate model, not in backup)

---

## Known Issues / Notes

- The original `torbi` source at `parler-tts-finetune/dataspeech/torbi/` was deleted. Only the compiled backup remains.
- `model.safetensors` can be used for inference directly. `pytorch_model.bin` inside the checkpoint is needed to resume training.
- `attn_implementation="eager"` is used for all existing models (flash_attention_2 caused issues on this machine).
- Training batch size is 4 per device with gradient accumulation of 8 (effective batch = 32).
- The model was trained for 20 epochs (~6240 steps) on the vertex datasets.
