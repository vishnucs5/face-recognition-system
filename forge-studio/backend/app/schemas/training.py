from typing import Literal
from pydantic import BaseModel, Field


TrainingGoal = Literal["character", "product", "style", "object", "custom"]
MixedPrecision = Literal["bf16", "fp16", "no"]
CaptioningMode = Literal["txt", "trigger_only", "caption_first"]


class TrainingHealthResponse(BaseModel):
    engine_available: bool
    engine_name: str | None = None
    message: str
    supported_engines: list[str] = Field(default_factory=list)
    gpu_vram_gb: float = 0.0
    recommended_preset: dict = Field(default_factory=dict)


class DatasetValidationRequest(BaseModel):
    dataset_path: str = Field(min_length=1, max_length=500)


class DatasetIssue(BaseModel):
    type: Literal["corrupt", "duplicate", "unsupported_format", "missing_caption", "low_resolution", "high_aspect_ratio"]
    file: str
    message: str


class DatasetValidationResponse(BaseModel):
    is_valid: bool
    dataset_path: str
    total_images: int
    supported_formats: list[str] = Field(default_factory=list)
    missing_captions_count: int = 0
    corrupt_images_count: int = 0
    duplicates_count: int = 0
    resolution_range: str = "N/A"
    suggested_resolution: int = 1024
    issues: list[DatasetIssue] = Field(default_factory=list)
    sample_images: list[str] = Field(default_factory=list)


class CaptionPreviewRequest(BaseModel):
    dataset_path: str = Field(min_length=1, max_length=500)
    trigger_word: str | None = Field(default=None, max_length=100)
    action: Literal["inspect", "prepend_trigger", "suggest_qwen"] = "inspect"


class CaptionPreviewItem(BaseModel):
    image_file: str
    caption_file: str
    existing_caption: str
    proposed_caption: str
    changed: bool


class CaptionPreviewResponse(BaseModel):
    items: list[CaptionPreviewItem] = Field(default_factory=list)
    total_files: int = 0
    changed_files: int = 0


class CaptionUpdateEntry(BaseModel):
    caption_file: str
    content: str


class ApplyCaptionsRequest(BaseModel):
    dataset_path: str = Field(min_length=1, max_length=500)
    updates: list[CaptionUpdateEntry] = Field(default_factory=list)
    backup_existing: bool = True


class ApplyCaptionsResponse(BaseModel):
    success: bool
    applied_count: int
    backed_up_count: int
    message: str


class TrainingConfigRequest(BaseModel):
    goal: TrainingGoal = "character"
    dataset_path: str = Field(min_length=1, max_length=500)
    base_model: str = Field(default="v1-5-pruned-emaonly.safetensors", max_length=255)
    lora_name: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_\-]+$")
    trigger_word: str = Field(default="", max_length=100)
    resolution: int = Field(default=1024, ge=256, le=1536)
    network_dim: int = Field(default=16, ge=1, le=128)
    network_alpha: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=0.0001, ge=1e-6, le=0.01)
    text_encoder_lr: float = Field(default=0.00005, ge=1e-6, le=0.01)
    batch_size: int = Field(default=1, ge=1, le=4)
    repeats: int = Field(default=10, ge=1, le=100)
    epochs: int = Field(default=10, ge=1, le=100)
    max_train_steps: int = Field(default=1000, ge=10, le=10000)
    save_interval: int = Field(default=2, ge=1, le=50)
    seed: int = Field(default=42, ge=0)
    mixed_precision: MixedPrecision = "bf16"
    gradient_checkpointing: bool = True
    captioning_mode: CaptioningMode = "txt"


class TrainingJobResponse(BaseModel):
    id: str
    status: str
    goal: str
    training_engine: str
    dataset_path: str
    dataset_image_count: int
    base_model: str
    lora_name: str
    trigger_word: str
    config: dict = Field(default_factory=dict)
    current_epoch: int = 0
    total_epochs: int = 10
    current_step: int = 0
    total_steps: int = 1000
    current_loss: float | None = None
    vram_usage_mb: int | None = None
    logs: list[str] = Field(default_factory=list)
    error_message: str | None = None
    artifact_path: str | None = None
    preview_images: list[dict] = Field(default_factory=list)
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
