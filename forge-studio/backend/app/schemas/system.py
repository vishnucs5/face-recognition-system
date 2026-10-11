from datetime import datetime
from pydantic import BaseModel, Field


class GpuStatus(BaseModel):
    available: bool = False
    name: str = "Unknown GPU"
    vram_used_mb: int = 0
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    utilization_pct: int = 0
    temperature_c: int | None = None


class SystemStatusResponse(BaseModel):
    comfyui_online: bool
    comfyui_url: str
    ollama_online: bool
    ollama_url: str
    ollama_model: str
    demo_mode: bool
    gpu: GpuStatus


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    app_name: str = "Forge Studio"


class ModelInfo(BaseModel):
    name: str
    filename: str
    size_mb: float = 0.0
    type: str = "checkpoint"  # checkpoint, lora, etc.


class LoraInfo(BaseModel):
    name: str
    filename: str
    size_mb: float = 0.0


class ObservedProcess(BaseModel):
    pid: int
    name: str
    vram_used_mb: int = 0


class ActiveLeaseInfo(BaseModel):
    lease_type: str
    job_id: str
    holder: str = "forge"
    acquired_at: datetime
    expires_at: datetime


class TelemetryResponse(BaseModel):
    gpu_name: str
    total_vram_mb: int
    used_vram_mb: int
    free_vram_mb: int
    utilization_pct: int
    temperature_c: int | None = None
    cpu_ram_used_mb: int
    cpu_ram_total_mb: int
    observed_processes: list[ObservedProcess] = Field(default_factory=list)
    active_lease: ActiveLeaseInfo | None = None
    state: str = "gpu_ready"  # gpu_ready, gpu_memory_pressure, gpu_unavailable, cuda_unavailable, ollama_active, comfyui_active, training_active, waiting_for_gpu
    pressure_threshold_pct: int = 85
    critical_threshold_pct: int = 95
    freshness_ms: int = 0
    is_stale: bool = False
    backend_latency_ms: float = 0.0
    sampled_at: datetime

