import re
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from app.schemas.generation import LoraInput

ALLOWED_SAMPLERS = {
    "euler",
    "euler_ancestral",
    "heun",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
    "dpm++ 2m",
    "dpm++ 2m karras",
    "dpm++ sde",
}

DISALLOWED_PATTERNS = [
    (re.compile(r"<\s*script\b", re.IGNORECASE), "script tags"),
    (re.compile(r"https?://", re.IGNORECASE), "URLs (http/https)"),
    (re.compile(r"\.\./|\.\.\\"), "directory traversal (../)"),
    (re.compile(r"[;&|`$]"), "shell metacharacters (; & | ` $)"),
]


def sanitize_text(val: str, field_name: str) -> str:
    if not val:
        return val
    for pattern, name in DISALLOWED_PATTERNS:
        if pattern.search(val):
            raise ValueError(f"Field '{field_name}' contains disallowed {name}.")

    # Check for arbitrary node graph JSON injection
    val_stripped = val.strip()
    if (val_stripped.startswith("{") and val_stripped.endswith("}")) or (
        val_stripped.startswith("[") and val_stripped.endswith("]")
    ):
        if ("\"nodes\"" in val or "'nodes'" in val) and ("\"links\"" in val or "'links'" in val):
            raise ValueError(f"Field '{field_name}' contains arbitrary node graph JSON structure.")
        if "\"last_node_id\"" in val or "'last_node_id'" in val:
            raise ValueError(f"Field '{field_name}' contains arbitrary node graph JSON structure.")

    return val


class PresetBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=300)
    mode: Literal["txt2img", "img2img", "inpaint", "controlnet"] = "txt2img"
    base_model_family: str = Field(default="sd15", max_length=20)
    checkpoint_name: str | None = Field(default=None, max_length=255)
    prompt_template: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    steps: int = Field(default=28, ge=1, le=150)
    cfg_scale: float = Field(default=6.5, ge=1.0, le=30.0)
    sampler: str = Field(default="euler", max_length=50)
    scheduler: str = Field(default="karras", max_length=50)
    seed_policy: Literal["random", "fixed", "increment"] = "random"
    fixed_seed: int | None = Field(default=None, ge=0)
    denoise: float = Field(default=0.75, ge=0.0, le=1.0)
    loras: list[LoraInput] = Field(default_factory=list)
    controlnet_settings: dict[str, Any] = Field(default_factory=dict)
    vram_profile: Literal["speed", "balanced", "conservative", "extreme"] = "balanced"

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return sanitize_text(v, "name")

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        return sanitize_text(v, "description")

    @field_validator("checkpoint_name")
    @classmethod
    def validate_checkpoint_name(cls, v: str | None) -> str | None:
        if v:
            return sanitize_text(v, "checkpoint_name")
        return v

    @field_validator("prompt_template")
    @classmethod
    def validate_prompt_template(cls, v: str) -> str:
        return sanitize_text(v, "prompt_template")

    @field_validator("negative_prompt")
    @classmethod
    def validate_negative_prompt(cls, v: str) -> str:
        return sanitize_text(v, "negative_prompt")

    @field_validator("sampler")
    @classmethod
    def validate_sampler(cls, v: str) -> str:
        norm = v.strip().lower()
        if norm not in ALLOWED_SAMPLERS:
            # Allow common aliases or check
            norm_slug = norm.replace("-", "_").replace(" ", "_")
            if norm_slug not in ALLOWED_SAMPLERS:
                raise ValueError(
                    f"Sampler '{v}' is not allowed. Choose from: {sorted(list(ALLOWED_SAMPLERS))}"
                )
        return norm


class PresetCreateRequest(PresetBase):
    pass


class PresetUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=300)
    mode: Literal["txt2img", "img2img", "inpaint", "controlnet"] | None = None
    base_model_family: str | None = Field(default=None, max_length=20)
    checkpoint_name: str | None = Field(default=None, max_length=255)
    prompt_template: str | None = Field(default=None, min_length=1, max_length=4000)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    width: int | None = Field(default=None, ge=256, le=2048)
    height: int | None = Field(default=None, ge=256, le=2048)
    steps: int | None = Field(default=None, ge=1, le=150)
    cfg_scale: float | None = Field(default=None, ge=1.0, le=30.0)
    sampler: str | None = Field(default=None, max_length=50)
    scheduler: str | None = Field(default=None, max_length=50)
    seed_policy: Literal["random", "fixed", "increment"] | None = None
    fixed_seed: int | None = Field(default=None, ge=0)
    denoise: float | None = Field(default=None, ge=0.0, le=1.0)
    loras: list[LoraInput] | None = None
    controlnet_settings: dict[str, Any] | None = None
    vram_profile: Literal["speed", "balanced", "conservative", "extreme"] | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        return sanitize_text(v, "name") if v is not None else None

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        return sanitize_text(v, "description") if v is not None else None

    @field_validator("checkpoint_name")
    @classmethod
    def validate_checkpoint_name(cls, v: str | None) -> str | None:
        return sanitize_text(v, "checkpoint_name") if v is not None else None

    @field_validator("prompt_template")
    @classmethod
    def validate_prompt_template(cls, v: str | None) -> str | None:
        return sanitize_text(v, "prompt_template") if v is not None else None

    @field_validator("negative_prompt")
    @classmethod
    def validate_negative_prompt(cls, v: str | None) -> str | None:
        return sanitize_text(v, "negative_prompt") if v is not None else None

    @field_validator("sampler")
    @classmethod
    def validate_sampler(cls, v: str | None) -> str | None:
        if v is None:
            return None
        norm = v.strip().lower()
        if norm not in ALLOWED_SAMPLERS:
            norm_slug = norm.replace("-", "_").replace(" ", "_")
            if norm_slug not in ALLOWED_SAMPLERS:
                raise ValueError(
                    f"Sampler '{v}' is not allowed. Choose from: {sorted(list(ALLOWED_SAMPLERS))}"
                )
        return norm


class PresetModel(PresetBase):
    schema_version: int = 1
    id: str
    category: str = "user"  # "built-in" or "user"
    version: int = 1
    is_built_in: bool = False
    created_at: str | None = None
    updated_at: str | None = None
    dependency_status: dict[str, bool] | None = None


class PresetDiffReview(BaseModel):
    preset_id: str | None = None
    preset_name: str | None = None
    differences: dict[str, dict[str, Any]] = Field(default_factory=dict)
    preset_estimated_vram_mb: int = 0
    current_estimated_vram_mb: int = 0
    estimated_vram_diff_mb: int = 0
    summary: str = ""
