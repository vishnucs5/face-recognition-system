import asyncio
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.websocket import manager
from app.config import settings
from app.database import async_session_maker, get_db
from app.models.generation import GenerationRecord
from app.schemas.generation import (
    GenerationProgressUpdate,
    GenerationRequest,
    GenerationResponse,
    LoraInput,
)
from app.schemas.preset import PresetCreateRequest, PresetModel
from app.services.comfyui_client import ComfyUIClient
from app.services.demo_generator import DemoGenerator
from app.services.error_sanitizer import sanitize_error_message
from app.services.gpu_safety import gpu_safety_manager
from app.services.preset_engine import preset_engine
from app.services.queue_scheduler import queue_scheduler

router = APIRouter(prefix="/generations", tags=["Generations"])

# Active generation tasks map for cancellation
active_tasks: dict[str, asyncio.Task] = {}


def record_to_response(rec: GenerationRecord) -> GenerationResponse:
    loras = []
    try:
        raw_loras = json.loads(rec.loras_json) if rec.loras_json else []
        loras = [LoraInput(**l) for l in raw_loras]
    except Exception:
        pass

    image_url = f"/data/generated/{Path(rec.image_path).name}" if rec.image_path else None
    source_url = f"/data/uploads/{Path(rec.source_image_path).name}" if rec.source_image_path else None
    mask_url = f"/data/uploads/{Path(rec.mask_image_path).name}" if rec.mask_image_path else None

    return GenerationResponse(
        id=rec.id,
        mode=rec.mode,
        status=rec.status,
        priority=getattr(rec, "priority", 2) or 2,
        estimated_vram_mb=getattr(rec, "estimated_vram_mb", None),
        estimated_duration_sec=getattr(rec, "estimated_duration_sec", None),
        client_request_id=getattr(rec, "client_request_id", None),
        generation_hash=getattr(rec, "generation_hash", None),
        retry_count=getattr(rec, "retry_count", 0) or 0,
        image_path=rec.image_path,
        image_url=image_url,
        source_image_path=rec.source_image_path,
        source_image_url=source_url,
        mask_image_path=rec.mask_image_path,
        mask_image_url=mask_url,
        prompt=rec.prompt,
        negative_prompt=rec.negative_prompt,
        model=rec.model,
        loras=loras,
        width=rec.width,
        height=rec.height,
        steps=rec.steps,
        cfg_scale=rec.cfg_scale,
        sampler=rec.sampler,
        seed=rec.seed,
        denoise=rec.denoise,
        controlnet_type=rec.controlnet_type,
        controlnet_model=rec.controlnet_model,
        controlnet_strength=rec.controlnet_strength,
        controlnet_start_percent=rec.controlnet_start_percent,
        controlnet_end_percent=rec.controlnet_end_percent,
        controlnet_preprocessor=rec.controlnet_preprocessor,
        preset_name=getattr(rec, "preset_name", None),
        preset_version=getattr(rec, "preset_version", None),
        settings_snapshot_json=getattr(rec, "settings_snapshot_json", None),
        gpu_telemetry_snapshot_json=getattr(rec, "gpu_telemetry_snapshot_json", None),
        workflow_graph_hash=getattr(rec, "workflow_graph_hash", None),
        asset_hashes_json=getattr(rec, "asset_hashes_json", None),
        created_at=rec.created_at.isoformat() if rec.created_at else None,
        started_at=rec.started_at.isoformat() if getattr(rec, "started_at", None) else None,
        completed_at=rec.completed_at.isoformat() if rec.completed_at else None,
        duration_ms=rec.duration_ms,
        error_message=rec.error_message,
        comfy_prompt_id=rec.comfy_prompt_id,
    )


async def execute_job(
    job_id: str,
    request: GenerationRequest,
    source_img_path: str | None,
    mask_img_path: str | None = None
):
    start_time = time.time()
    comfy_client = ComfyUIClient()

    # 1. Wait for dispatch_next to select this job and acquire GPU safety lease
    try:
        while True:
            if queue_scheduler.is_paused():
                await asyncio.sleep(0.05)
                continue

            async with async_session_maker() as session:
                result = await session.execute(select(GenerationRecord).where(GenerationRecord.id == job_id))
                rec = result.scalar_one_or_none()
                if not rec or rec.status == "cancelled":
                    return

                dispatched = await queue_scheduler.dispatch_next(session)
                if dispatched and dispatched.id == job_id:
                    break

            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        async with async_session_maker() as session:
            result = await session.execute(select(GenerationRecord).where(GenerationRecord.id == job_id))
            rec = result.scalar_one_or_none()
            if rec and rec.status not in ["completed", "failed"]:
                rec.status = "cancelled"
                rec.error_message = "Generation cancelled by user"
                rec.completed_at = datetime.now(timezone.utc)
                await session.commit()
        await manager.broadcast_progress(GenerationProgressUpdate(
            job_id=job_id,
            status="cancelled",
            message="Generation cancelled"
        ))
        await gpu_safety_manager.release_lease(job_id)
        gpu_safety_manager.unregister_waiting(job_id)
        if job_id in active_tasks:
            del active_tasks[job_id]
        return

    async def on_progress(update: GenerationProgressUpdate):
        await manager.broadcast_progress(update)

    async def on_prompt_id(prompt_id: str):
        async with async_session_maker() as session:
            result = await session.execute(select(GenerationRecord).where(GenerationRecord.id == job_id))
            rec = result.scalar_one_or_none()
            if rec:
                rec.comfy_prompt_id = prompt_id
                await session.commit()

    try:
        if settings.DEMO_MODE:
            saved_path = await DemoGenerator.generate(job_id, request, on_progress)
        else:
            if not (await comfy_client.is_available()):
                raise RuntimeError(
                    f"ComfyUI is offline at {comfy_client.base_url}. Please start ComfyUI on port 8188."
                )
            checkpoints = await comfy_client.get_checkpoints()
            if not checkpoints:
                raise RuntimeError(
                    "No model checkpoints found in ComfyUI models/checkpoints directory. Please download or place a model checkpoint."
                )
            src_path = Path(source_img_path) if source_img_path else None
            mask_path = Path(mask_img_path) if mask_img_path else None
            saved_path = await comfy_client.execute_generation(
                job_id,
                request,
                src_path,
                mask_path,
                progress_callback=on_progress,
                on_prompt_id=on_prompt_id
            )

        duration_ms = int((time.time() - start_time) * 1000)

        async with async_session_maker() as session:
            result = await session.execute(select(GenerationRecord).where(GenerationRecord.id == job_id))
            rec = result.scalar_one_or_none()
            if rec:
                rec.status = "completed"
                rec.image_path = str(saved_path)
                rec.duration_ms = duration_ms
                rec.completed_at = datetime.now(timezone.utc)
                if rec.gpu_telemetry_snapshot_json:
                    try:
                        g_snap = json.loads(rec.gpu_telemetry_snapshot_json)
                    except Exception:
                        g_snap = {}
                else:
                    g_snap = {}
                g_snap["actual_duration_ms"] = duration_ms
                rec.gpu_telemetry_snapshot_json = json.dumps(g_snap)
                await session.commit()

        # Send completed progress event
        final_update = GenerationProgressUpdate(
            job_id=job_id,
            status="completed",
            step=request.steps,
            total_steps=request.steps,
            progress_percentage=100.0,
            image_url=f"/data/generated/{saved_path.name}",
            message="Generation complete"
        )
        await manager.broadcast_progress(final_update)

    except asyncio.CancelledError:
        async with async_session_maker() as session:
            result = await session.execute(select(GenerationRecord).where(GenerationRecord.id == job_id))
            rec = result.scalar_one_or_none()
            if rec:
                rec.status = "cancelled"
                rec.error_message = "Generation cancelled by user"
                rec.completed_at = datetime.now(timezone.utc)
                await session.commit()
        await manager.broadcast_progress(GenerationProgressUpdate(
            job_id=job_id,
            status="cancelled",
            message="Generation cancelled"
        ))
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        sanitized_err = sanitize_error_message(str(e))
        async with async_session_maker() as session:
            result = await session.execute(select(GenerationRecord).where(GenerationRecord.id == job_id))
            rec = result.scalar_one_or_none()
            if rec:
                rec.status = "failed"
                rec.duration_ms = duration_ms
                rec.error_message = sanitized_err
                rec.completed_at = datetime.now(timezone.utc)
                await session.commit()
        await manager.broadcast_progress(GenerationProgressUpdate(
            job_id=job_id,
            status="failed",
            message=f"Error: {sanitized_err}"
        ))
    finally:
        await gpu_safety_manager.release_lease(job_id)
        gpu_safety_manager.unregister_waiting(job_id)
        if job_id in active_tasks:
            del active_tasks[job_id]


@router.post("", response_model=GenerationResponse, status_code=status.HTTP_201_CREATED)
async def create_generation(request: GenerationRequest, db: AsyncSession = Depends(get_db)):
    # 1. Validation checks
    if len(request.loras) > 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MVP allows maximum 2 LoRAs per generation to guarantee stability."
        )

    source_path = None
    mask_path = None

    if request.mode == "img2img":
        if not request.source_image_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A source image is required for Image-to-Image mode."
            )
        candidate = settings.uploads_path / request.source_image_id
        if not candidate.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Source image '{request.source_image_id}' not found in uploads directory."
            )
        source_path = str(candidate)

    elif request.mode == "inpaint":
        if not request.source_image_id or not request.mask_image_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inpainting requires both a source image and a painted mask."
            )
        src_cand = settings.uploads_path / request.source_image_id
        if not src_cand.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Source image '{request.source_image_id}' not found in uploads directory."
            )
        mask_cand = settings.uploads_path / request.mask_image_id
        if not mask_cand.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Mask image '{request.mask_image_id}' not found in uploads directory."
            )
        source_path = str(src_cand)
        mask_path = str(mask_cand)

    elif request.mode == "controlnet":
        if not request.source_image_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A source image is required for ControlNet mode."
            )
        candidate = settings.uploads_path / request.source_image_id
        if not candidate.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Source image '{request.source_image_id}' not found in uploads directory."
            )
        source_path = str(candidate)

        # Validate ControlNet model existence
        comfy_client = ComfyUIClient()
        available_cnet = await comfy_client.get_controlnet_models()
        avail_names = [m.filename for m in available_cnet]
        if not request.controlnet_model or request.controlnet_model not in avail_names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ControlNet model unavailable — place a controlnet model (e.g. control_v11p_sd15_canny.safetensors) in ComfyUI/models/controlnet/ to enable ControlNet generation."
            )

    # 2. Check deduplication (within last 5 seconds)
    gen_hash = queue_scheduler.compute_generation_hash(
        mode=request.mode,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        model=request.model,
        width=request.width,
        height=request.height,
        steps=request.steps,
        cfg_scale=request.cfg_scale,
        sampler=request.sampler,
        seed=request.seed,
        denoise=request.denoise,
        loras_json=[l.model_dump() for l in request.loras],
        controlnet_type=request.controlnet_type,
    )
    duplicate = await queue_scheduler.find_duplicate_job(
        db=db,
        client_request_id=request.client_request_id,
        generation_hash=gen_hash,
        window_seconds=5.0,
    )
    if duplicate:
        return record_to_response(duplicate)

    # 3. Estimate resources
    est_vram, est_dur = queue_scheduler.estimate_resources(
        model=request.model,
        width=request.width,
        height=request.height,
        steps=request.steps,
        loras_count=len(request.loras),
        has_controlnet=bool(request.controlnet_type or request.controlnet_model),
    )

    # 4. Snapshots and hashes
    settings_snapshot = {
        "mode": request.mode,
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "model": request.model,
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "cfg_scale": request.cfg_scale,
        "sampler": request.sampler,
        "seed": request.seed,
        "denoise": request.denoise,
        "loras": [l.model_dump() for l in request.loras],
        "controlnet_type": request.controlnet_type,
        "controlnet_model": request.controlnet_model,
        "controlnet_strength": request.controlnet_strength,
        "controlnet_start_percent": request.controlnet_start_percent,
        "controlnet_end_percent": request.controlnet_end_percent,
        "controlnet_preprocessor": request.controlnet_preprocessor,
        "preset_name": request.preset_name,
        "preset_version": request.preset_version,
    }
    settings_snapshot_json = json.dumps(settings_snapshot)
    workflow_graph_hash = hashlib.sha256(settings_snapshot_json.encode("utf-8")).hexdigest()

    asset_hashes = {}
    if source_path and Path(source_path).exists():
        try:
            with open(source_path, "rb") as f:
                asset_hashes["source_image"] = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass
    if mask_path and Path(mask_path).exists():
        try:
            with open(mask_path, "rb") as f:
                asset_hashes["mask_image"] = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass
    asset_hashes_json = json.dumps(asset_hashes)

    gpu_telemetry = {
        "estimated_vram_mb": est_vram,
        "estimated_duration_sec": est_dur,
    }
    try:
        status_data = gpu_safety_manager.get_status()
        if status_data:
            gpu_telemetry["vram_usage"] = status_data
    except Exception:
        pass
    gpu_telemetry_snapshot_json = json.dumps(gpu_telemetry)

    # 5. Persist record to DB
    job_id = str(uuid.uuid4())
    record = GenerationRecord(
        id=job_id,
        mode=request.mode,
        status="queued",
        priority=request.priority,
        estimated_vram_mb=est_vram,
        estimated_duration_sec=est_dur,
        client_request_id=request.client_request_id,
        generation_hash=gen_hash,
        retry_count=0,
        preset_name=request.preset_name,
        preset_version=request.preset_version,
        settings_snapshot_json=settings_snapshot_json,
        gpu_telemetry_snapshot_json=gpu_telemetry_snapshot_json,
        workflow_graph_hash=workflow_graph_hash,
        asset_hashes_json=asset_hashes_json,
        source_image_path=source_path,
        mask_image_path=mask_path,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        model=request.model,
        loras_json=json.dumps([l.model_dump() for l in request.loras]),
        width=request.width,
        height=request.height,
        steps=request.steps,
        cfg_scale=request.cfg_scale,
        sampler=request.sampler,
        seed=request.seed,
        denoise=request.denoise,
        controlnet_type=request.controlnet_type,
        controlnet_model=request.controlnet_model,
        controlnet_strength=request.controlnet_strength,
        controlnet_start_percent=request.controlnet_start_percent,
        controlnet_end_percent=request.controlnet_end_percent,
        controlnet_preprocessor=request.controlnet_preprocessor,
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    # 5. Spawn background generation
    task = asyncio.create_task(execute_job(job_id, request, source_path, mask_path))
    active_tasks[job_id] = task

    return record_to_response(record)


@router.get("", response_model=list[GenerationResponse])
async def list_generations(
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
    mode: str | None = None,
    db: AsyncSession = Depends(get_db)
):
    query = select(GenerationRecord).order_by(desc(GenerationRecord.created_at))
    if search:
        query = query.where(GenerationRecord.prompt.ilike(f"%{search}%"))
    if mode and mode != "all":
        query = query.where(GenerationRecord.mode == mode)
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    records = result.scalars().all()

    # Reconcile any orphaned queued/running records (e.g. from server restart)
    orphaned = False
    for r in records:
        if r.status in ["queued", "waiting_for_gpu", "running"] and r.id not in active_tasks:
            r.status = "cancelled"
            r.error_message = "Task cancelled or interrupted by restart"
            r.completed_at = datetime.now(timezone.utc)
            orphaned = True
    if orphaned:
        await db.commit()

    return [record_to_response(r) for r in records]


@router.post("/clear-queue")
async def clear_queue(db: AsyncSession = Depends(get_db)):
    """Removes local queued and cancelled history records. Never deletes files from ComfyUI."""
    result = await db.execute(
        select(GenerationRecord).where(GenerationRecord.status.in_(["queued", "waiting_for_gpu", "cancelled"]))
    )
    records = result.scalars().all()
    count = len(records)
    for rec in records:
        if rec.id in active_tasks:
            active_tasks[rec.id].cancel()
        gpu_safety_manager.unregister_waiting(rec.id)
        await db.delete(rec)
    await db.commit()
    return {"status": "cleared", "count": count}


@router.get("/{id}", response_model=GenerationResponse)
async def get_generation(id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(GenerationRecord).where(GenerationRecord.id == id))
    rec = result.scalar_one_or_none()
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")
    return record_to_response(rec)


@router.delete("/{id}")
async def delete_generation(id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(GenerationRecord).where(GenerationRecord.id == id))
    rec = result.scalar_one_or_none()
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")
    
    # Remove file from disk if present
    if rec.image_path:
        p = Path(rec.image_path)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    await db.delete(rec)
    await db.commit()
    return {"status": "deleted", "id": id}


@router.post("/{id}/cancel")
async def cancel_generation(id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(GenerationRecord).where(GenerationRecord.id == id))
    rec = result.scalar_one_or_none()
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")

    comfy_client = ComfyUIClient()

    # If it was in the ComfyUI queue, remove it
    if rec.comfy_prompt_id:
        try:
            await comfy_client.delete_queue_item(rec.comfy_prompt_id)
        except Exception:
            pass

    # If it was the currently executing task, interrupt ComfyUI
    if rec.status == "running" or id in active_tasks:
        try:
            await comfy_client.interrupt()
        except Exception:
            pass
        await gpu_safety_manager.release_lease(id)

    gpu_safety_manager.unregister_waiting(id)

    if id in active_tasks:
        task = active_tasks[id]
        task.cancel()

    if rec.status in ["queued", "waiting_for_gpu", "running"]:
        rec.status = "cancelled"
        rec.error_message = "Generation cancelled by user"
        rec.completed_at = datetime.now(timezone.utc)
        await db.commit()

    await manager.broadcast_progress(GenerationProgressUpdate(
        job_id=id,
        status="cancelled",
        message="Generation cancelled by user"
    ))

    return {"status": "cancelled", "job_id": id}


@router.post("/upload")
async def upload_source_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    ext = Path(file.filename or "image.png").suffix or ".png"
    file_id = f"upload_{uuid.uuid4().hex[:12]}{ext}"
    dest_path = settings.uploads_path / file_id

    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    return {
        "source_image_id": file_id,
        "filename": file.filename,
        "url": f"/data/uploads/{file_id}"
    }


@router.post("/upload-mask")
async def upload_mask_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    ext = Path(file.filename or "mask.png").suffix or ".png"
    file_id = f"mask_{uuid.uuid4().hex[:12]}{ext}"
    dest_path = settings.uploads_path / file_id

    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    return {
        "mask_image_id": file_id,
        "filename": file.filename,
        "url": f"/data/uploads/{file_id}"
    }


@router.post("/{id}/copy-as-preset", response_model=PresetModel)
async def copy_generation_as_preset(id: str, db: AsyncSession = Depends(get_db)):
    """Converts a generation snapshot and parameters into a user preset."""
    result = await db.execute(select(GenerationRecord).where(GenerationRecord.id == id))
    rec = result.scalar_one_or_none()
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")

    loras_list: list[LoraInput] = []
    try:
        if rec.loras_json:
            for l in json.loads(rec.loras_json):
                loras_list.append(LoraInput(**l))
    except Exception:
        pass

    controlnet_settings = {}
    if rec.controlnet_type or rec.controlnet_model:
        controlnet_settings = {
            "controlnet_type": rec.controlnet_type,
            "controlnet_model": rec.controlnet_model,
            "controlnet_strength": rec.controlnet_strength,
            "controlnet_start_percent": rec.controlnet_start_percent,
            "controlnet_end_percent": rec.controlnet_end_percent,
            "controlnet_preprocessor": rec.controlnet_preprocessor,
        }

    family = "sdxl" if (rec.model and "xl" in rec.model.lower()) else "sd15"

    clean_prompt = re.sub(r"[;&|`$]", "", rec.prompt or "Custom Generation").strip()
    if len(clean_prompt) > 25:
        preset_name = f"Preset from {clean_prompt[:25]}..."
    else:
        preset_name = f"Preset: {clean_prompt or 'Custom'}"
    if len(preset_name) > 100:
        preset_name = preset_name[:100]

    req = PresetCreateRequest(
        name=preset_name,
        description=f"Generated from generation record {rec.id[:8]}",
        mode=rec.mode if rec.mode in ["txt2img", "img2img", "inpaint", "controlnet"] else "txt2img",
        base_model_family=family,
        checkpoint_name=rec.model,
        prompt_template=clean_prompt or "High quality photo",
        negative_prompt=re.sub(r"[;&|`$]", "", rec.negative_prompt or "").strip(),
        width=rec.width,
        height=rec.height,
        steps=rec.steps,
        cfg_scale=rec.cfg_scale,
        sampler=rec.sampler,
        scheduler="karras",
        seed_policy="random",
        fixed_seed=rec.seed,
        denoise=rec.denoise,
        loras=loras_list,
        controlnet_settings=controlnet_settings,
        vram_profile="balanced",
    )
    return await preset_engine.create_user_preset(db, req)


@router.get("/{id}/reproduce")
async def get_generation_reproduction(id: str, db: AsyncSession = Depends(get_db)):
    """Returns exact reproduction config and dependency verification for a generation."""
    result = await db.execute(select(GenerationRecord).where(GenerationRecord.id == id))
    rec = result.scalar_one_or_none()
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")

    loras_list = []
    try:
        loras_list = json.loads(rec.loras_json) if rec.loras_json else []
    except Exception:
        pass

    cnet_settings = {}
    if rec.controlnet_type or rec.controlnet_model:
        cnet_settings = {
            "controlnet_type": rec.controlnet_type,
            "controlnet_model": rec.controlnet_model,
            "controlnet_strength": rec.controlnet_strength,
            "controlnet_start_percent": rec.controlnet_start_percent,
            "controlnet_end_percent": rec.controlnet_end_percent,
            "controlnet_preprocessor": rec.controlnet_preprocessor,
        }

    family = "sdxl" if (rec.model and "xl" in rec.model.lower()) else "sd15"

    dummy_preset = PresetModel(
        schema_version=1,
        id=f"repr-{rec.id}",
        name="Reproduction Preset",
        description="",
        mode=rec.mode if rec.mode in ["txt2img", "img2img", "inpaint", "controlnet"] else "txt2img",
        base_model_family=family,
        checkpoint_name=rec.model,
        prompt_template=re.sub(r"[;&|`$]", "", rec.prompt or "prompt"),
        negative_prompt=re.sub(r"[;&|`$]", "", rec.negative_prompt or ""),
        width=rec.width,
        height=rec.height,
        steps=rec.steps,
        cfg_scale=rec.cfg_scale,
        sampler=rec.sampler,
        scheduler="karras",
        seed_policy="fixed" if rec.seed is not None else "random",
        fixed_seed=rec.seed,
        denoise=rec.denoise,
        loras=[LoraInput(**l) for l in loras_list],
        controlnet_settings=cnet_settings,
        vram_profile="balanced",
    )
    deps = await preset_engine.inspect_dependencies(dummy_preset)

    return {
        "generation_id": rec.id,
        "config": {
            "mode": rec.mode,
            "prompt": rec.prompt,
            "negative_prompt": rec.negative_prompt,
            "model": rec.model,
            "width": rec.width,
            "height": rec.height,
            "steps": rec.steps,
            "cfg_scale": rec.cfg_scale,
            "sampler": rec.sampler,
            "seed": rec.seed,
            "denoise": rec.denoise,
            "loras": loras_list,
            "controlnet": cnet_settings,
            "preset_name": rec.preset_name,
            "preset_version": rec.preset_version,
        },
        "seed": rec.seed,
        "workflow_graph_hash": rec.workflow_graph_hash,
        "asset_hashes": json.loads(rec.asset_hashes_json) if rec.asset_hashes_json else {},
        "settings_snapshot": json.loads(rec.settings_snapshot_json) if rec.settings_snapshot_json else {},
        "dependencies": deps,
        "can_reproduce": deps.get("all_satisfied", True),
    }

