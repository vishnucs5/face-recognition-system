import json
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from app.database import Base


class TrainingJobRecord(Base):
    __tablename__ = "training_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(20), nullable=False, default="queued")
    goal = Column(String(50), nullable=False, default="character")
    training_engine = Column(String(50), nullable=False, default="none")
    dataset_path = Column(String(500), nullable=False)
    dataset_image_count = Column(Integer, nullable=False, default=0)
    base_model = Column(String(255), nullable=False)
    lora_name = Column(String(255), nullable=False)
    trigger_word = Column(String(255), nullable=False, default="")
    config_json = Column(Text, nullable=False, default="{}")
    
    current_epoch = Column(Integer, nullable=False, default=0)
    total_epochs = Column(Integer, nullable=False, default=10)
    current_step = Column(Integer, nullable=False, default=0)
    total_steps = Column(Integer, nullable=False, default=1000)
    current_loss = Column(Float, nullable=True)
    vram_usage_mb = Column(Integer, nullable=True)
    logs = Column(Text, nullable=False, default="")
    error_message = Column(Text, nullable=True)
    artifact_path = Column(String(500), nullable=True)
    preview_images_json = Column(Text, nullable=False, default="[]")
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    @property
    def config(self) -> dict:
        try:
            return json.loads(self.config_json) if self.config_json else {}
        except Exception:
            return {}

    @property
    def preview_images(self) -> list[dict]:
        try:
            return json.loads(self.preview_images_json) if self.preview_images_json else []
        except Exception:
            return []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "goal": self.goal,
            "training_engine": self.training_engine,
            "dataset_path": self.dataset_path,
            "dataset_image_count": self.dataset_image_count,
            "base_model": self.base_model,
            "lora_name": self.lora_name,
            "trigger_word": self.trigger_word,
            "config": self.config,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_loss": self.current_loss,
            "vram_usage_mb": self.vram_usage_mb,
            "logs": self.logs.splitlines()[-200:] if self.logs else [],
            "error_message": self.error_message,
            "artifact_path": self.artifact_path,
            "preview_images": self.preview_images,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
