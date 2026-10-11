from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class DiagnosticCheck(BaseModel):
    id: str
    name: str
    category: str
    status: str  # "ok", "warning", "error"
    message: str
    details: str | None = None
    remediation_cmd: str | None = None
    checked_at: datetime


class DiagnosticsReport(BaseModel):
    checks: list[DiagnosticCheck]
    sampled_at: datetime
    cache_ttl_sec: int = 30


class BenchmarkRunRequest(BaseModel):
    user_confirmed: bool = True
    include_ollama: bool = True
    include_img2img: bool = False


class BenchmarkResult(BaseModel):
    id: str
    timestamp: datetime
    gpu_name: str
    total_time_ms: int
    prompt_to_first_img_ms: int
    steps_per_sec: float
    peak_vram_mb: int
    avg_vram_mb: int
    gpu_util_pct: float
    ollama_first_token_ms: float | None = None
    ollama_tokens_per_sec: float | None = None
    env_metadata: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
