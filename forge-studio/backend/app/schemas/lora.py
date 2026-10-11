from pydantic import BaseModel, Field


class LoraVaultItem(BaseModel):
    id: str
    name: str
    filename: str
    file_path: str
    file_size_mb: float = 0.0
    base_model: str = "SDXL 1.0"
    trigger_words: list[str] = Field(default_factory=list)
    preview_image_url: str | None = None
    recommended_weight: float = 0.8
    training_status: str = "imported"
    notes: str | None = None
    is_available: bool = True
    created_at: str | None = None
    updated_at: str | None = None


class LoraImportRequest(BaseModel):
    file_path: str = Field(min_length=1, max_length=500, description="Absolute local path to .safetensors file")
    name: str | None = Field(default=None, max_length=255)
    trigger_words: list[str] = Field(default_factory=list)
    base_model: str = Field(default="SDXL 1.0", max_length=50)
    recommended_weight: float = Field(default=0.8, ge=0.0, le=2.0)
    notes: str | None = Field(default=None, max_length=2000)


class LoraUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    trigger_words: list[str] | None = None
    base_model: str | None = Field(default=None, max_length=50)
    recommended_weight: float | None = Field(default=None, ge=0.0, le=2.0)
    notes: str | None = Field(default=None, max_length=2000)
    preview_image_url: str | None = None
