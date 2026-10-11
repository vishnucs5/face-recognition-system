from typing import Any, Literal
from pydantic import BaseModel, Field


class CopilotSettingsSuggestion(BaseModel):
    prompt: str | None = None
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    sampler: str | None = None
    denoise: float | None = None
    controlnet_type: str | None = None
    controlnet_strength: float | None = None
    suggested_loras: list[str] = Field(default_factory=list)


class CopilotRequest(BaseModel):
    task: Literal[
        "improve_prompt",
        "variations",
        "configure_generation",
        "chat",
        "explain",
        "negative_prompt",
        "inpaint_advice",
        "controlnet_advice",
    ] = "improve_prompt"
    message: str = Field(min_length=1, max_length=2000)
    current_context: dict = Field(default_factory=dict)


class CopilotResponse(BaseModel):
    reply: str
    suggested_settings: CopilotSettingsSuggestion | None = None
    variations: list[str] = Field(default_factory=list)


class CopilotActionRequest(BaseModel):
    action_type: Literal[
        "fast_generation",
        "quality",
        "vram_safe_resolution",
        "explain_slowness",
        "suggest_preset",
    ]
    current_form: dict = Field(default_factory=dict)


class CopilotActionResponse(BaseModel):
    action_type: str
    explanation: str
    suggested_changes: dict[str, Any] = Field(default_factory=dict)
    preset_id: str | None = None
    preset_name: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


