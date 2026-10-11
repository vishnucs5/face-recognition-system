from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class LoraInput(BaseModel):
    name: str
    strength: float = Field(default=0.8, ge=0.0, le=2.0)


class ControlNetModelInfo(BaseModel):
    name: str
    filename: str
    type: str
    size_mb: float = 0.0


class GenerationRequest(BaseModel):
    mode: Literal["txt2img", "img2img", "inpaint", "controlnet"] = "txt2img"
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000)
    model: str
    width: int = Field(default=1024, ge=256, le=1536)
    height: int = Field(default=1024, ge=256, le=1536)
    steps: int = Field(default=28, ge=1, le=80)
    cfg_scale: float = Field(default=6.5, ge=1.0, le=20.0)
    sampler: str = Field(default="euler")
    seed: int | None = Field(default=None, ge=0)
    denoise: float = Field(default=0.75, ge=0.0, le=1.0)
    loras: list[LoraInput] = Field(default_factory=list)
    source_image_id: str | None = None
    mask_image_id: str | None = None
    priority: int = Field(default=2, ge=1, le=3)
    client_request_id: str | None = None

    # ControlNet specific settings
    controlnet_type: Literal["canny", "depth", "openpose", "lineart", "soft_edge"] | None = None
    controlnet_model: str | None = None
    controlnet_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    controlnet_start_percent: float = Field(default=0.0, ge=0.0, le=1.0)
    controlnet_end_percent: float = Field(default=1.0, ge=0.0, le=1.0)
    controlnet_preprocessor: str | None = "canny"
    preset_name: str | None = None
    preset_version: int | None = None


class GenerationResponse(BaseModel):
    id: str
    mode: str
    status: str
    priority: int = 2
    estimated_vram_mb: int | None = None
    estimated_duration_sec: float | None = None
    client_request_id: str | None = None
    generation_hash: str | None = None
    retry_count: int = 0
    image_path: str | None = None
    image_url: str | None = None
    source_image_path: str | None = None
    source_image_url: str | None = None
    mask_image_path: str | None = None
    mask_image_url: str | None = None
    prompt: str
    negative_prompt: str
    model: str
    loras: list[LoraInput] = Field(default_factory=list)
    width: int
    height: int
    steps: int
    cfg_scale: float
    sampler: str
    seed: int | None = None
    denoise: float
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    error_message: str | None = None
    comfy_prompt_id: str | None = None

    controlnet_type: str | None = None
    controlnet_model: str | None = None
    controlnet_strength: float | None = None
    controlnet_start_percent: float | None = None
    controlnet_end_percent: float | None = None
    controlnet_preprocessor: str | None = None

    preset_name: str | None = None
    preset_version: int | None = None
    settings_snapshot_json: str | None = None
    gpu_telemetry_snapshot_json: str | None = None
    workflow_graph_hash: str | None = None
    asset_hashes_json: str | None = None


class GenerationProgressUpdate(BaseModel):
    job_id: str
    status: Literal["queued", "waiting_for_gpu", "running", "completed", "failed", "cancelled"]
    step: int = 0
    total_steps: int = 0
    progress_percentage: float = 0.0
    preview_url: str | None = None
    message: str | None = None
    image_url: str | None = None
