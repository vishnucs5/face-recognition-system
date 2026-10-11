import uuid
from datetime import datetime, timezone
from sqlalchemy import BigInteger, Column, DateTime, Float, Integer, String, Text
from app.database import Base


class GenerationRecord(Base):
    __tablename__ = "generations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    mode = Column(String(20), nullable=False, default="txt2img")
    status = Column(String(20), nullable=False, default="queued")  # queued, running, completed, failed, cancelled

    image_path = Column(String(500), nullable=True)
    source_image_path = Column(String(500), nullable=True)
    mask_image_path = Column(String(500), nullable=True)

    prompt = Column(Text, nullable=False)
    negative_prompt = Column(Text, nullable=False, default="")
    model = Column(String(255), nullable=False)
    loras_json = Column(Text, nullable=False, default="[]")

    width = Column(Integer, nullable=False, default=1024)
    height = Column(Integer, nullable=False, default=1024)
    steps = Column(Integer, nullable=False, default=28)
    cfg_scale = Column(Float, nullable=False, default=6.5)
    sampler = Column(String(50), nullable=False, default="euler")
    seed = Column(BigInteger, nullable=True)
    denoise = Column(Float, nullable=False, default=0.75)

    # ControlNet specific fields
    controlnet_type = Column(String(50), nullable=True)
    controlnet_model = Column(String(255), nullable=True)
    controlnet_strength = Column(Float, nullable=True, default=1.0)
    controlnet_start_percent = Column(Float, nullable=True, default=0.0)
    controlnet_end_percent = Column(Float, nullable=True, default=1.0)
    controlnet_preprocessor = Column(String(50), nullable=True)

    # Queue scheduling and telemetry fields
    priority = Column(Integer, nullable=False, default=2)  # 1=High, 2=Normal, 3=Low
    estimated_vram_mb = Column(Integer, nullable=True)
    estimated_duration_sec = Column(Float, nullable=True)
    client_request_id = Column(String(36), nullable=True)
    generation_hash = Column(String(64), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)

    # Preset and Snapshot fields
    preset_name = Column(String(100), nullable=True)
    preset_version = Column(Integer, nullable=True)
    settings_snapshot_json = Column(Text, nullable=True)
    gpu_telemetry_snapshot_json = Column(Text, nullable=True)
    workflow_graph_hash = Column(String(64), nullable=True)
    asset_hashes_json = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    comfy_prompt_id = Column(String(100), nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "mode": self.mode,
            "status": self.status,
            "priority": self.priority,
            "estimated_vram_mb": self.estimated_vram_mb,
            "estimated_duration_sec": self.estimated_duration_sec,
            "client_request_id": self.client_request_id,
            "generation_hash": self.generation_hash,
            "retry_count": self.retry_count,
            "preset_name": self.preset_name,
            "preset_version": self.preset_version,
            "settings_snapshot_json": self.settings_snapshot_json,
            "gpu_telemetry_snapshot_json": self.gpu_telemetry_snapshot_json,
            "workflow_graph_hash": self.workflow_graph_hash,
            "asset_hashes_json": self.asset_hashes_json,
            "image_path": self.image_path,
            "source_image_path": self.source_image_path,
            "mask_image_path": self.mask_image_path,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model": self.model,
            "loras_json": self.loras_json,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "sampler": self.sampler,
            "seed": self.seed,
            "denoise": self.denoise,
            "controlnet_type": self.controlnet_type,
            "controlnet_model": self.controlnet_model,
            "controlnet_strength": self.controlnet_strength,
            "controlnet_start_percent": self.controlnet_start_percent,
            "controlnet_end_percent": self.controlnet_end_percent,
            "controlnet_preprocessor": self.controlnet_preprocessor,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "comfy_prompt_id": self.comfy_prompt_id,
        }
