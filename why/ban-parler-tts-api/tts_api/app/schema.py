from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str
    voice: Literal["female", "male"] = "female"
    description: Optional[str] = Field(default=None, description="Speaker description. Uses server default if omitted.")
    max_new_tokens: Optional[int] = Field(default=None, ge=10, le=4096, description="Max tokens to generate. Uses server default (2048) if omitted.")


class TTSBatchRequest(BaseModel):
    texts: List[str]
    voice: Literal["female", "male"] = "female"
    description: Optional[str] = Field(default=None, description="Speaker description. Uses server default if omitted.")
    max_new_tokens: Optional[int] = Field(default=None, ge=10, le=4096, description="Max tokens to generate. Uses server default (2048) if omitted.")


class TTSBase64Response(BaseModel):
    voice: str
    audio: str  # base64-encoded WAV


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_name: str | None = None
    gpu_memory_used_gb: float | None = None
    gpu_memory_total_gb: float | None = None
    gpu_utilization_pct: float | None = None
    queue_depth: int
    batches_processed: int
    avg_batch_size: float
    avg_batch_latency_ms: float
