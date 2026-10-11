import json
import uuid
from datetime import datetime, timezone
from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, String, Text
from app.database import Base


class PresetRecord(Base):
    __tablename__ = "presets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(String(300), nullable=False, default="")
    category = Column(String(20), nullable=False, default="user")  # "built-in" or "user"
    version = Column(Integer, nullable=False, default=1)
    is_built_in = Column(Boolean, nullable=False, default=False)
    mode = Column(String(20), nullable=False, default="txt2img")
    base_model_family = Column(String(20), nullable=False, default="sd15")
    checkpoint_name = Column(String(255), nullable=True)
    prompt_template = Column(Text, nullable=False)
    negative_prompt = Column(Text, nullable=False, default="")
    width = Column(Integer, nullable=False, default=1024)
    height = Column(Integer, nullable=False, default=1024)
    steps = Column(Integer, nullable=False, default=28)
    cfg_scale = Column(Float, nullable=False, default=6.5)
    sampler = Column(String(50), nullable=False, default="euler")
    scheduler = Column(String(50), nullable=False, default="karras")
    seed_policy = Column(String(20), nullable=False, default="random")  # "random", "fixed", "increment"
    fixed_seed = Column(BigInteger, nullable=True)
    denoise = Column(Float, nullable=False, default=0.75)
    loras_json = Column(Text, nullable=False, default="[]")
    controlnet_settings_json = Column(Text, nullable=False, default="{}")
    vram_profile = Column(String(20), nullable=False, default="balanced")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    @property
    def loras(self) -> list:
        try:
            return json.loads(self.loras_json) if self.loras_json else []
        except Exception:
            return []

    @property
    def controlnet_settings(self) -> dict:
        try:
            return json.loads(self.controlnet_settings_json) if self.controlnet_settings_json else {}
        except Exception:
            return {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "is_built_in": self.is_built_in,
            "mode": self.mode,
            "base_model_family": self.base_model_family,
            "checkpoint_name": self.checkpoint_name,
            "prompt_template": self.prompt_template,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "seed_policy": self.seed_policy,
            "fixed_seed": self.fixed_seed,
            "denoise": self.denoise,
            "loras": self.loras,
            "loras_json": self.loras_json,
            "controlnet_settings": self.controlnet_settings,
            "controlnet_settings_json": self.controlnet_settings_json,
            "vram_profile": self.vram_profile,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
