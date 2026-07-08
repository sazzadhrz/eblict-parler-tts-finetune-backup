# EBLICT Parler TTS Finetune

Finetuned Bengali TTS models based on [ai4bharat/indic-parler-tts-pretrained](https://huggingface.co/ai4bharat/indic-parler-tts-pretrained).

Two speaker models:
- **vertex-female1** — speaker "Sadia"
- **vertex-male1** — speaker "Sazzad"

---

## Backup Locations

| Asset | Location |
|-------|----------|
| Code & scripts | https://github.com/sazzadhrz/eblict-parler-tts-finetune-backup |
| Female model weights + checkpoints | https://huggingface.co/sazzad-sit/vertex-female1-parler-tts-finetune (private) |
| Male model weights + checkpoints | https://huggingface.co/sazzad-sit/vertex-male1-parler-tts-finetune (private) |

Each HF repo contains:
```
model.safetensors                  ← inference weights
config.json, tokenizer.*           ← model config & tokenizer
checkpoint-6240-epoch-19/          ← full training state (resume training from here)
  optimizer.bin
  pytorch_model.bin
  scheduler.bin
  random_states_0.pkl
env_backup/
  ptts_environment.yml             ← ptts conda env (training)
  infer_environment.yml            ← infer conda env (inference/API)
  torbi_1.1.0_ptts_env_backup.tar.gz
```

---

## Full Restore from Scratch

### 1. Clone the code repo

```bash
git clone https://github.com/sazzadhrz/eblict-parler-tts-finetune-backup.git parler-tts-finetune
cd parler-tts-finetune
```

### 2. Restore conda environments

```bash
conda env create -f ptts_environment.yml     # training env
conda env create -f infer_environment.yml    # inference env
```

### 3. Restore the torbi package (special — was installed from local source)

torbi is a compiled package not available on PyPI in this form. Restore it manually:

```bash
# Download from HF
huggingface-cli download sazzad-sit/vertex-female1-parler-tts-finetune \
  env_backup/torbi_1.1.0_ptts_env_backup.tar.gz \
  --local-dir /tmp/torbi_restore

cd /tmp && tar xzf torbi_restore/env_backup/torbi_1.1.0_ptts_env_backup.tar.gz

# Install into the ptts env
SITE_PKG=$(conda run -n ptts python -c "import site; print(site.getsitepackages()[0])")
cp -r /tmp/torbi_backup/torbi "$SITE_PKG/"
cp -r /tmp/torbi_backup/torbi-1.1.0.dist-info "$SITE_PKG/"

# Verify
conda run -n ptts python -c "import torbi; print('torbi OK')"
```

> **Note:** The torbi backup is a compiled binary for **Python 3.10, Linux x86_64**. On a different platform, install from source instead:
> ```bash
> conda run -n ptts pip install git+https://github.com/maxrmorrison/torbi.git@v1.1.0
> ```

### 4. Download model weights

```bash
mkdir -p parler-tts/output_dir_vertex_female1_finetune
mkdir -p parler-tts/output_dir_vertex_male1_finetune

# Female model
huggingface-cli download sazzad-sit/vertex-female1-parler-tts-finetune \
  --local-dir parler-tts/output_dir_vertex_female1_finetune \
  --exclude "env_backup/*"

# Male model
huggingface-cli download sazzad-sit/vertex-male1-parler-tts-finetune \
  --local-dir parler-tts/output_dir_vertex_male1_finetune \
  --exclude "env_backup/*"
```

---

## Inference

### Simple script

```bash
conda activate infer
cd parler-tts/output_dir_vertex_male1_finetune

python scripts/infer_kanak.py \
  --text "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।" \
  --model_path "./checkpoint-6240-epoch-19" \
  --output output.wav
```

### FastAPI server (production)

The full API with batching, queue, and streaming lives in `why/ban-parler-tts-api/`:

```bash
conda activate infer
cd why/ban-parler-tts-api
pip install -r tts_api/requirements.txt
python -m tts_api.app.main
# or: uvicorn tts_api.app.main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /tts` — returns WAV audio
- `POST /tts/base64` — returns base64-encoded WAV
- `POST /tts/batch` — batch synthesis
- `GET /health` — server health + GPU stats

### Simple API (scripts/api.py)

```bash
conda activate infer
cd parler-tts/output_dir_vertex_male1_finetune
uvicorn scripts.api:app --host 0.0.0.0 --port 8000
# Then: GET http://localhost:8000/generate?text=আমার সোনার বাংলা
```

---

## Training a New Model (Full Pipeline)

All training commands are in `commands_despina.txt` (female) and `commands_enceladus.txt` (male).

### Step 1 — Dataspeech tagging

Run from inside `parler-tts/` with `ptts` env active:

```bash
conda activate ptts
cd notebook_parler_tts_finetune/dataspeech

python main.py "sazzad-sit/YOUR-tts-dataset" \
  --text_column_name "text" \
  --audio_column_name "audio" \
  --cpu_num_workers 16 \
  --num_workers_per_gpu_for_pitch 8 \
  --rename_column \
  --repo_id "YOUR-tts-dataset-tags" \
  --penn_batch_size 256
```

### Step 2 — Generate text descriptions

```bash
python ./scripts/metadata_to_text.py \
    "sazzad-sit/YOUR-tts-dataset-tags" \
    --repo_id "sazzad-sit/YOUR-tts-dataset-tags" \
    --cpu_num_workers 2 \
    --avoid_pitch_computation \
    --save_bin_edges "./examples/tags_to_annotations/YOUR_bin_edges.json"
```

### Step 3 — LLM prompt creation

```bash
python ./scripts/run_prompt_creation.py \
    --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" \
    --per_device_eval_batch_size 8 \
    --dataset_name "sazzad-sit/YOUR-tts-dataset-tags" \
    --output_dir "./YOUR-descriptions" \
    --hub_dataset_id "sazzad-sit/YOUR-tts-dataset-descriptions" \
    --push_to_hub \
    --is_single_speaker \
    --speaker_name "SPEAKER_NAME" \
    --preprocessing_num_workers 2
```

### Step 4 — Training

```bash
conda activate ptts
cd parler-tts

accelerate launch ./training/run_parler_tts_training.py \
    --model_name_or_path "ai4bharat/indic-parler-tts-pretrained" \
    --feature_extractor_name "ylacombe/dac_44khz" \
    --description_tokenizer_name "google/flan-t5-large" \
    --prompt_tokenizer_name "ai4bharat/indic-parler-tts-pretrained" \
    --report_to "none" \
    --overwrite_output_dir true \
    --train_dataset_name "sazzad-sit/YOUR-tts-dataset" \
    --train_metadata_dataset_name "sazzad-sit/YOUR-tts-dataset-descriptions" \
    --train_split_name "train" \
    --train_dataset_config_name "default" \
    --target_audio_column_name "audio" \
    --description_column_name "text_description" \
    --prompt_column_name "text" \
    --max_duration_in_seconds 30 \
    --min_duration_in_seconds 2.0 \
    --max_text_length 400 \
    --do_train true \
    --do_eval false \
    --num_train_epochs 20 \
    --per_device_train_batch_size 4 \
    --audio_encoder_per_device_batch_size 12 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --learning_rate 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 1000 \
    --logging_steps 100 \
    --save_steps 1000 \
    --freeze_text_encoder true \
    --dtype "bfloat16" \
    --attn_implementation "eager" \
    --seed 42 \
    --output_dir "./output_dir_YOUR_finetune/" \
    --temporary_save_to_disk "./YOUR_vectorized_dataset_tmp/" \
    --save_to_disk "./YOUR_vectorized_dataset/" \
    --preprocessing_num_workers 4 \
    --dataloader_num_workers 4 \
    --group_by_length true \
    --remove_unused_columns false
```

> Use `--attn_implementation "flash_attention_2"` if your GPU supports it (faster). Use `"eager"` if you get errors.

### Resume training from checkpoint

Add `--resume_from_checkpoint "./output_dir_YOUR_finetune/checkpoint-6240-epoch-19"` to the training command.

---

## Conda Environments

| Env | Purpose |
|-----|---------|
| `ptts` | Training + dataspeech pipeline |
| `infer` | Inference API server |

---

## HuggingFace Datasets (Training Data)

- `sazzad-sit/vertex-female1-tts-dataset` — raw audio
- `sazzad-sit/vertex-female1-tts-dataset-tags` — with dataspeech tags
- `sazzad-sit/vertex-female1-tts-dataset-descriptions` — with LLM descriptions
- `sazzad-sit/vertex-male1-tts-dataset` — raw audio
- `sazzad-sit/vertex-male1-tts-dataset-tags`
- `sazzad-sit/vertex-male1-tts-dataset-descriptions`
