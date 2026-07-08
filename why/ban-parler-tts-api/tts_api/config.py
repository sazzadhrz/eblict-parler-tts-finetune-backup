import torch

_BASE_DIR = "/home/eblict/parler-tts-finetune/parler-tts"

VOICE_CONFIGS = {
    "female": {
        "root_model_path":   f"{_BASE_DIR}/output_dir_vertex_female1_finetune",
        "model_path":        f"{_BASE_DIR}/output_dir_vertex_female1_finetune/checkpoint-6240-epoch-19",
        "default_description": "Sadia speech has a faster pace and clear studio recording quality.",
    },
    "male": {
        "root_model_path":   f"{_BASE_DIR}/output_dir_vertex_male1_finetune",
        "model_path":        f"{_BASE_DIR}/output_dir_vertex_male1_finetune/checkpoint-6240-epoch-19",
        "default_description": "Sazzad speech has a faster pace and clear studio recording quality.",
    },
}
DEFAULT_VOICE = "female"

DEVICE            = "cuda"
DTYPE             = torch.bfloat16

CHUNK_TOKEN_LIMIT = 50
BATCH_MAX_SIZE    = 16
BATCH_TIMEOUT_MS  = 30
SILENCE_MS        = 20
MAX_NEW_TOKENS    = 2048
COMPILE_MODEL     = True
WARMUP_RUNS       = 3

LOG_DIR = "logs"
HOST    = "0.0.0.0"
PORT    = 8000
