import io
from typing import List

import numpy as np
import soundfile as sf

from tts_api.config import SILENCE_MS


def merge_chunks(
    chunk_audios: List[np.ndarray],
    sampling_rate: int,
    silence_ms: int = SILENCE_MS,
) -> bytes:
    """Concatenate audio chunks with short silence padding, encode as PCM_16 WAV."""
    silence_samples = int(sampling_rate * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32)

    parts = []
    for i, chunk in enumerate(chunk_audios):
        parts.append(chunk.astype(np.float32))
        if i < len(chunk_audios) - 1:
            parts.append(silence)

    full = np.concatenate(parts)

    # Normalize to prevent clipping
    peak = np.abs(full).max()
    if peak > 0.95:
        full = full * (0.95 / peak)

    buf = io.BytesIO()
    sf.write(buf, full, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
