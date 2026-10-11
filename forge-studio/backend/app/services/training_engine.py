import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.training import TrainingJobRecord
from app.schemas.training import (
    ApplyCaptionsRequest,
    ApplyCaptionsResponse,
    CaptionPreviewItem,
    CaptionPreviewRequest,
    CaptionPreviewResponse,
    DatasetIssue,
    DatasetValidationResponse,
    TrainingConfigRequest,
    TrainingHealthResponse,
    TrainingJobResponse,
)
from app.services.gpu_monitor import GpuMonitor


class TrainingEngineService:
    SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

    @classmethod
    def detect_engine(cls) -> tuple[bool, str | None, str]:
        """Detects whether a supported training engine (kohya_ss, ai-toolkit) is installed."""
        # 1. Check PATH executables
        kohya_candidates = ["kohya_gui", "sd-scripts", "train_network.py"]
        for cand in kohya_candidates:
            if shutil.which(cand):
                return True, "kohya_ss", "kohya_ss training engine detected in system PATH."

        if shutil.which("ai-toolkit") or shutil.which("toolkit"):
            return True, "ai-toolkit", "ai-toolkit training engine detected in system PATH."

        # 2. Check local directories in c:/vishnu and configured workspace
        search_dirs = [
            Path("c:/vishnu"),
            settings.base_dir,
            settings.base_dir / "training_engines",
        ]
        
        env_cnet = os.getenv("AI_TOOLKIT_PATH")
        if env_cnet and Path(env_cnet).exists():
            return True, "ai-toolkit", f"ai-toolkit engine detected at {env_cnet}"

        env_kohya = os.getenv("KOHYA_SS_PATH")
        if env_kohya and Path(env_kohya).exists():
            return True, "kohya_ss", f"kohya_ss engine detected at {env_kohya}"

        for base in search_dirs:
            # AI-Toolkit check
            if (base / "ai-toolkit" / "run.py").exists() or (base / "ai-toolkit" / "toolkit").exists():
                return True, "ai-toolkit", f"Local ai-toolkit engine detected at {base / 'ai-toolkit'}."
            # kohya_ss / sd-scripts check
            if (base / "kohya_ss" / "train_network.py").exists() or (base / "kohya_ss" / "sd-scripts" / "train_network.py").exists():
                return True, "kohya_ss", f"Local kohya_ss engine detected at {base / 'kohya_ss'}."
            if (base / "sd-scripts" / "train_network.py").exists():
                return True, "kohya_ss", f"Local sd-scripts engine detected at {base / 'sd-scripts'}."

        # 3. Native PyTorch GPU engine check
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                return True, "forge_trainer", f"Native PyTorch LoRA Engine ready ({gpu_name})"
        except Exception:
            pass

        return (
            False,
            None,
            "Training engine unavailable — install kohya_ss or AI-Toolkit locally to enable training.",
        )

    @classmethod
    def get_health(cls) -> TrainingHealthResponse:
        available, name, msg = cls.detect_engine()
        gpu_status = GpuMonitor.get_status()
        vram_gb = round(gpu_status.vram_total_mb / 1024, 1) if gpu_status.available else 16.0

        preset = {
            "resolution": 1024,
            "batch_size": 1,
            "network_dim": 16,
            "network_alpha": 16,
            "learning_rate": 0.0001,
            "text_encoder_lr": 0.00005,
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "max_train_steps": 1000,
            "estimated_vram_gb": 12.5,
            "notes": "Optimized for NVIDIA GeForce RTX 5070 Ti (16GB VRAM)",
        }

        return TrainingHealthResponse(
            engine_available=available,
            engine_name=name,
            message=msg,
            supported_engines=["forge_trainer", "kohya_ss", "ai-toolkit"],
            gpu_vram_gb=vram_gb,
            recommended_preset=preset,
        )

    @classmethod
    def validate_dataset(cls, dataset_path_str: str) -> DatasetValidationResponse:
        dataset_path = Path(dataset_path_str).resolve()
        issues: list[DatasetIssue] = []

        if not dataset_path.exists():
            return DatasetValidationResponse(
                is_valid=False,
                dataset_path=str(dataset_path),
                total_images=0,
                issues=[
                    DatasetIssue(
                        type="unsupported_format",
                        file=str(dataset_path),
                        message=f"Directory does not exist: {dataset_path}",
                    )
                ],
            )

        if not dataset_path.is_dir():
            return DatasetValidationResponse(
                is_valid=False,
                dataset_path=str(dataset_path),
                total_images=0,
                issues=[
                    DatasetIssue(
                        type="unsupported_format",
                        file=str(dataset_path),
                        message="Specified path is a file, not a directory.",
                    )
                ],
            )

        images = []
        formats_found = set()
        seen_hashes: dict[str, str] = {}
        widths: list[int] = []
        heights: list[int] = []
        missing_captions_count = 0
        corrupt_count = 0
        duplicates_count = 0

        for f in sorted(dataset_path.glob("*.*")):
            ext = f.suffix.lower()
            if ext in cls.SUPPORTED_IMAGE_EXTS:
                formats_found.add(ext.replace(".", "").upper())
                images.append(f)

                # 1. Integrity check
                try:
                    with Image.open(f) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        if min(w, h) < 512:
                            issues.append(
                                DatasetIssue(
                                    type="low_resolution",
                                    file=f.name,
                                    message=f"Image resolution {w}x{h} is below recommended 512px minimum.",
                                )
                            )
                        aspect = max(w, h) / max(1, min(w, h))
                        if aspect > 2.5:
                            issues.append(
                                DatasetIssue(
                                    type="high_aspect_ratio",
                                    file=f.name,
                                    message=f"High aspect ratio ({round(aspect, 2)}:1) may cause crop distortion.",
                                )
                            )
                except Exception as e:
                    corrupt_count += 1
                    issues.append(
                        DatasetIssue(
                            type="corrupt",
                            file=f.name,
                            message=f"Unreadable or corrupt image file: {e}",
                        )
                    )
                    continue

                # 2. Duplicate detection via MD5
                try:
                    with open(f, "rb") as bf:
                        fhash = hashlib.md5(bf.read()).hexdigest()
                    if fhash in seen_hashes:
                        duplicates_count += 1
                        issues.append(
                            DatasetIssue(
                                type="duplicate",
                                file=f.name,
                                message=f"Identical duplicate of {seen_hashes[fhash]}.",
                            )
                        )
                    else:
                        seen_hashes[fhash] = f.name
                except Exception:
                    pass

                # 3. Caption file check
                caption_file = f.with_suffix(".txt")
                if not caption_file.exists() or not caption_file.read_text(encoding="utf-8", errors="ignore").strip():
                    missing_captions_count += 1
                    issues.append(
                        DatasetIssue(
                            type="missing_caption",
                            file=f.name,
                            message=f"Missing caption file: {caption_file.name}",
                        )
                    )

        total_images = len(images)
        is_valid = total_images >= 4 and corrupt_count == 0

        res_range = f"{min(widths)}x{min(heights)} - {max(widths)}x{max(heights)}" if widths else "N/A"
        suggested_res = 1024 if any(w >= 1024 for w in widths) else 512

        sample_images = [img.name for img in images[:8]]

        return DatasetValidationResponse(
            is_valid=is_valid,
            dataset_path=str(dataset_path),
            total_images=total_images,
            supported_formats=sorted(list(formats_found)),
            missing_captions_count=missing_captions_count,
            corrupt_images_count=corrupt_count,
            duplicates_count=duplicates_count,
            resolution_range=res_range,
            suggested_resolution=suggested_res,
            issues=issues,
            sample_images=sample_images,
        )

    @classmethod
    def preview_captions(cls, req: CaptionPreviewRequest) -> CaptionPreviewResponse:
        dataset_path = Path(req.dataset_path).resolve()
        if not dataset_path.exists() or not dataset_path.is_dir():
            return CaptionPreviewResponse(items=[], total_files=0, changed_files=0)

        items: list[CaptionPreviewItem] = []
        changed_count = 0

        for f in sorted(dataset_path.glob("*.*")):
            if f.suffix.lower() in cls.SUPPORTED_IMAGE_EXTS:
                caption_path = f.with_suffix(".txt")
                existing_caption = ""
                if caption_path.exists():
                    existing_caption = caption_path.read_text(encoding="utf-8", errors="ignore").strip()

                proposed = existing_caption
                if req.action == "prepend_trigger" and req.trigger_word:
                    tw = req.trigger_word.strip()
                    if tw and not existing_caption.lower().startswith(tw.lower()):
                        proposed = f"{tw}, {existing_caption}" if existing_caption else tw

                has_changed = proposed != existing_caption
                if has_changed:
                    changed_count += 1

                items.append(
                    CaptionPreviewItem(
                        image_file=f.name,
                        caption_file=caption_path.name,
                        existing_caption=existing_caption,
                        proposed_caption=proposed,
                        changed=has_changed,
                    )
                )

        return CaptionPreviewResponse(
            items=items,
            total_files=len(items),
            changed_files=changed_count,
        )

    @classmethod
    def apply_captions(cls, req: ApplyCaptionsRequest) -> ApplyCaptionsResponse:
        dataset_path = Path(req.dataset_path).resolve()
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dataset_path}")

        applied = 0
        backed_up = 0

        for entry in req.updates:
            caption_path = dataset_path / entry.caption_file
            # Security guard against directory traversal
            if not str(caption_path.resolve()).startswith(str(dataset_path)):
                continue

            # Backup if requested and exists
            if req.backup_existing and caption_path.exists():
                bak_path = caption_path.with_suffix(".txt.bak")
                try:
                    shutil.copy2(caption_path, bak_path)
                    backed_up += 1
                except Exception:
                    pass

            caption_path.write_text(entry.content.strip(), encoding="utf-8")
            applied += 1

        return ApplyCaptionsResponse(
            success=True,
            applied_count=applied,
            backed_up_count=backed_up,
            message=f"Successfully applied {applied} captions (backed up {backed_up}).",
        )

    @classmethod
    async def list_jobs(cls, db: AsyncSession) -> list[TrainingJobResponse]:
        result = await db.execute(select(TrainingJobRecord).order_by(TrainingJobRecord.created_at.desc()))
        records = result.scalars().all()
        return [TrainingJobResponse(**r.to_dict()) for r in records]

    @classmethod
    async def get_job(cls, db: AsyncSession, job_id: str) -> TrainingJobResponse | None:
        result = await db.execute(select(TrainingJobRecord).where(TrainingJobRecord.id == job_id))
        rec = result.scalar_one_or_none()
        return TrainingJobResponse(**rec.to_dict()) if rec else None

    @classmethod
    async def start_job(cls, db: AsyncSession, req: TrainingConfigRequest) -> TrainingJobResponse:
        # 1. Concurrency guard: Only one active training job allowed initially
        active_res = await db.execute(
            select(TrainingJobRecord).where(
                TrainingJobRecord.status.in_(["queued", "validating", "training", "saving"])
            )
        )
        if active_res.scalar_one_or_none():
            raise ValueError("A training job is already currently running. Only one active training job permitted.")

        # 2. Check engine availability
        engine_available, engine_name, msg = cls.detect_engine()
        if not engine_available:
            raise RuntimeError("Training engine unavailable — install kohya_ss or AI-Toolkit locally to enable training.")

        # 3. Validate dataset path
        ds_val = cls.validate_dataset(req.dataset_path)
        if ds_val.total_images == 0:
            raise ValueError(f"No valid images found in dataset path: {req.dataset_path}")

        # 4. Prepare training output directory
        import asyncio
        from app.services.lora_trainer import RealLoRATrainer

        job_rec = TrainingJobRecord(
            goal=req.goal,
            training_engine=engine_name or "forge_trainer",
            dataset_path=str(Path(req.dataset_path).resolve()),
            dataset_image_count=ds_val.total_images,
            base_model=req.base_model,
            lora_name=req.lora_name,
            trigger_word=req.trigger_word,
            config_json=json.dumps(req.model_dump()),
            total_epochs=req.epochs,
            total_steps=req.max_train_steps,
            status="queued",
            logs=f"[{datetime.now().strftime('%H:%M:%S')}] Job queued. Target engine: {engine_name}.\n",
        )
        db.add(job_rec)
        await db.commit()
        await db.refresh(job_rec)

        # Launch training task asynchronously
        asyncio.create_task(RealLoRATrainer.train_job(job_rec.id))

        return TrainingJobResponse(**job_rec.to_dict())

    @classmethod
    async def cancel_job(cls, db: AsyncSession, job_id: str) -> bool:
        result = await db.execute(select(TrainingJobRecord).where(TrainingJobRecord.id == job_id))
        rec = result.scalar_one_or_none()
        if not rec:
            return False

        if rec.status in ["queued", "validating", "training"]:
            rec.status = "cancelled"
            rec.completed_at = datetime.now(timezone.utc)
            rec.logs += f"\n[{datetime.now().strftime('%H:%M:%S')}] Training cancelled by user."
            await db.commit()
            return True
        return False

    @classmethod
    async def retry_job(cls, db: AsyncSession, job_id: str) -> TrainingJobResponse:
        result = await db.execute(select(TrainingJobRecord).where(TrainingJobRecord.id == job_id))
        old_job = result.scalar_one_or_none()
        if not old_job:
            raise KeyError(f"Training job {job_id} not found.")

        cfg = old_job.config
        new_job = TrainingJobRecord(
            goal=old_job.goal,
            training_engine=old_job.training_engine,
            dataset_path=old_job.dataset_path,
            dataset_image_count=old_job.dataset_image_count,
            base_model=old_job.base_model,
            lora_name=old_job.lora_name,
            trigger_word=old_job.trigger_word,
            config_json=old_job.config_json,
            total_epochs=old_job.total_epochs,
            total_steps=old_job.total_steps,
            status="queued",
            logs=f"[{datetime.now().strftime('%H:%M:%S')}] Retry created from job {job_id}.\n",
        )
        db.add(new_job)
        await db.commit()
        await db.refresh(new_job)
        return TrainingJobResponse(**new_job.to_dict())
