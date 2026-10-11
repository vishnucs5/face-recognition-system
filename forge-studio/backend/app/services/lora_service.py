import json
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.lora import LoraVaultRecord
from app.schemas.lora import LoraImportRequest, LoraUpdateRequest, LoraVaultItem


class LoraService:
    COMFY_LORAS_DIR = Path("c:/vishnu/ComfyUI/models/loras")

    @classmethod
    def read_safetensors_metadata(cls, file_path: Path) -> dict:
        """Reads safetensors header JSON metadata without external dependencies."""
        if not file_path.exists() or file_path.stat().st_size < 8:
            return {}
        try:
            with open(file_path, "rb") as f:
                header_len_bytes = f.read(8)
                if len(header_len_bytes) < 8:
                    return {}
                header_len = struct.unpack("<Q", header_len_bytes)[0]
                if header_len > 10 * 1024 * 1024:  # Safety cap at 10MB
                    return {}
                header_json_bytes = f.read(header_len)
                data = json.loads(header_json_bytes.decode("utf-8", errors="ignore"))
                return data.get("__metadata__", {})
        except Exception:
            return {}

    @classmethod
    async def sync_filesystem_loras(cls, db: AsyncSession) -> None:
        """Discovers any local LoRA files in forge or ComfyUI directories and adds them to DB."""
        scan_dirs = [settings.loras_path, cls.COMFY_LORAS_DIR]
        seen_filenames = set()

        # Query existing filenames in DB
        result = await db.execute(select(LoraVaultRecord))
        existing_records = {r.filename: r for r in result.scalars().all()}

        for sdir in scan_dirs:
            if not sdir.exists():
                continue
            for f in sdir.glob("*.*"):
                if f.suffix.lower() == ".safetensors" and f.name not in seen_filenames:
                    seen_filenames.add(f.name)
                    if f.name not in existing_records:
                        size_mb = round(f.stat().st_size / (1024 * 1024), 1)
                        metadata = cls.read_safetensors_metadata(f)
                        base_model = metadata.get("ss_base_model_version", "SDXL 1.0")
                        trigger_word = metadata.get("ss_tag_frequency", "")
                        
                        trigger_words = []
                        if isinstance(trigger_word, str) and trigger_word.strip():
                            trigger_words = [trigger_word.strip()]

                        record = LoraVaultRecord(
                            name=f.stem.replace("_", " ").title(),
                            filename=f.name,
                            file_path=str(f.resolve()),
                            file_size_mb=size_mb,
                            base_model=base_model if "sd" in base_model.lower() or "flux" in base_model.lower() else "SDXL 1.0",
                            trigger_words_json=json.dumps(trigger_words),
                            preview_image_url=None,
                            recommended_weight=0.8,
                            training_status="imported",
                            notes=f"Auto-indexed from {sdir.name}",
                        )
                        db.add(record)
                        existing_records[f.name] = record

        await db.commit()

    @classmethod
    async def list_loras(cls, db: AsyncSession) -> list[LoraVaultItem]:
        await cls.sync_filesystem_loras(db)
        result = await db.execute(select(LoraVaultRecord).order_by(LoraVaultRecord.created_at.desc()))
        records = result.scalars().all()
        return [LoraVaultItem(**r.to_dict()) for r in records]

    @classmethod
    async def get_lora(cls, db: AsyncSession, lora_id: str) -> LoraVaultItem | None:
        result = await db.execute(select(LoraVaultRecord).where(LoraVaultRecord.id == lora_id))
        rec = result.scalar_one_or_none()
        return LoraVaultItem(**rec.to_dict()) if rec else None

    @classmethod
    async def import_lora(cls, db: AsyncSession, req: LoraImportRequest) -> LoraVaultItem:
        src_path = Path(req.file_path).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"File not found: {src_path}")
        if src_path.suffix.lower() != ".safetensors":
            raise ValueError("Only .safetensors LoRA format is supported.")

        filename = src_path.name
        size_mb = round(src_path.stat().st_size / (1024 * 1024), 1)

        # Ensure copied to ComfyUI models/loras so ComfyUI can immediately execute generations
        if cls.COMFY_LORAS_DIR.exists():
            dest_comfy = cls.COMFY_LORAS_DIR / filename
            if not dest_comfy.exists() and src_path != dest_comfy:
                try:
                    shutil.copy2(src_path, dest_comfy)
                except Exception:
                    pass

        # Also place in Forge Studio's models/loras
        dest_forge = settings.loras_path / filename
        if not dest_forge.exists() and src_path != dest_forge:
            try:
                shutil.copy2(src_path, dest_forge)
            except Exception:
                pass

        # Check if record already exists
        result = await db.execute(select(LoraVaultRecord).where(LoraVaultRecord.filename == filename))
        existing = result.scalar_one_or_none()

        trigger_words = req.trigger_words
        if not trigger_words:
            meta = cls.read_safetensors_metadata(src_path)
            tag = meta.get("ss_tag_frequency", "")
            if isinstance(tag, str) and tag.strip():
                trigger_words = [tag.strip()]

        display_name = req.name or src_path.stem.replace("_", " ").title()

        if existing:
            existing.name = display_name
            existing.file_path = str(src_path)
            existing.file_size_mb = size_mb
            existing.base_model = req.base_model
            existing.trigger_words_json = json.dumps(trigger_words)
            existing.recommended_weight = req.recommended_weight
            existing.notes = req.notes or existing.notes
            existing.updated_at = datetime.now(timezone.utc)
            await db.commit()
            await db.refresh(existing)
            return LoraVaultItem(**existing.to_dict())

        new_rec = LoraVaultRecord(
            name=display_name,
            filename=filename,
            file_path=str(src_path),
            file_size_mb=size_mb,
            base_model=req.base_model,
            trigger_words_json=json.dumps(trigger_words),
            preview_image_url=None,
            recommended_weight=req.recommended_weight,
            training_status="imported",
            notes=req.notes,
        )
        db.add(new_rec)
        await db.commit()
        await db.refresh(new_rec)
        return LoraVaultItem(**new_rec.to_dict())

    @classmethod
    async def update_lora(cls, db: AsyncSession, lora_id: str, req: LoraUpdateRequest) -> LoraVaultItem:
        result = await db.execute(select(LoraVaultRecord).where(LoraVaultRecord.id == lora_id))
        rec = result.scalar_one_or_none()
        if not rec:
            raise KeyError(f"LoRA with id {lora_id} not found.")

        if req.name is not None:
            rec.name = req.name
        if req.trigger_words is not None:
            rec.trigger_words_json = json.dumps(req.trigger_words)
        if req.base_model is not None:
            rec.base_model = req.base_model
        if req.recommended_weight is not None:
            rec.recommended_weight = req.recommended_weight
        if req.notes is not None:
            rec.notes = req.notes
        if req.preview_image_url is not None:
            rec.preview_image_url = req.preview_image_url

        rec.updated_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(rec)
        return LoraVaultItem(**rec.to_dict())

    @classmethod
    async def remove_lora(cls, db: AsyncSession, lora_id: str) -> bool:
        """Removes the LoRA record from the index without deleting the physical file."""
        result = await db.execute(select(LoraVaultRecord).where(LoraVaultRecord.id == lora_id))
        rec = result.scalar_one_or_none()
        if not rec:
            return False
        await db.delete(rec)
        await db.commit()
        return True
