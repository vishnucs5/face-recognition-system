import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import Column, DateTime, Float, String, Text
from app.database import Base


class LoraVaultRecord(Base):
    __tablename__ = "lora_vault"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False, unique=True)
    file_path = Column(String(500), nullable=False)
    file_size_mb = Column(Float, nullable=False, default=0.0)
    base_model = Column(String(50), nullable=False, default="SDXL 1.0")
    trigger_words_json = Column(Text, nullable=False, default="[]")
    preview_image_url = Column(String(500), nullable=True)
    recommended_weight = Column(Float, nullable=False, default=0.8)
    training_status = Column(String(50), nullable=False, default="imported")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    @property
    def is_available(self) -> bool:
        return Path(self.file_path).exists()

    @property
    def trigger_words(self) -> list[str]:
        try:
            return json.loads(self.trigger_words_json) if self.trigger_words_json else []
        except Exception:
            return []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "filename": self.filename,
            "file_path": self.file_path,
            "file_size_mb": round(self.file_size_mb, 1),
            "base_model": self.base_model,
            "trigger_words": self.trigger_words,
            "preview_image_url": self.preview_image_url,
            "recommended_weight": self.recommended_weight,
            "training_status": self.training_status,
            "notes": self.notes,
            "is_available": self.is_available,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
