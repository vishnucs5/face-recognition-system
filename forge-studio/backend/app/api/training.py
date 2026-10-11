from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.lora import LoraImportRequest, LoraVaultItem
from app.schemas.training import (
    ApplyCaptionsRequest,
    ApplyCaptionsResponse,
    CaptionPreviewRequest,
    CaptionPreviewResponse,
    DatasetValidationRequest,
    DatasetValidationResponse,
    TrainingConfigRequest,
    TrainingHealthResponse,
    TrainingJobResponse,
)
from app.services.lora_service import LoraService
from app.services.training_engine import TrainingEngineService

router = APIRouter(prefix="/training", tags=["Training"])


@router.get("/health", response_model=TrainingHealthResponse)
async def training_health():
    return TrainingEngineService.get_health()


@router.post("/validate-dataset", response_model=DatasetValidationResponse)
async def validate_dataset(req: DatasetValidationRequest):
    return TrainingEngineService.validate_dataset(req.dataset_path)


@router.post("/preview-captions", response_model=CaptionPreviewResponse)
async def preview_captions(req: CaptionPreviewRequest):
    return TrainingEngineService.preview_captions(req)


@router.post("/apply-captions", response_model=ApplyCaptionsResponse)
async def apply_captions(req: ApplyCaptionsRequest):
    try:
        return TrainingEngineService.apply_captions(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply captions: {e}")


@router.get("/jobs", response_model=list[TrainingJobResponse])
async def list_training_jobs(db: AsyncSession = Depends(get_db)):
    return await TrainingEngineService.list_jobs(db)


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await TrainingEngineService.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return job


@router.post("/start", response_model=TrainingJobResponse)
async def start_training(req: TrainingConfigRequest, db: AsyncSession = Depends(get_db)):
    try:
        return await TrainingEngineService.start_job(db, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Engine unavailable
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error starting training: {e}")


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str, db: AsyncSession = Depends(get_db)):
    success = await TrainingEngineService.cancel_job(db, job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Training job not found or cannot be cancelled.")
    return {"status": "cancelled", "job_id": job_id}


@router.post("/jobs/{job_id}/retry", response_model=TrainingJobResponse)
async def retry_training_job(job_id: str, db: AsyncSession = Depends(get_db)):
    try:
        return await TrainingEngineService.retry_job(db, job_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retry job: {e}")


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await TrainingEngineService.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return {"job_id": job.id, "logs": job.logs}


@router.get("/jobs/{job_id}/artifacts")
async def get_job_artifacts(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await TrainingEngineService.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")
    artifacts = []
    if job.artifact_path and Path(job.artifact_path).exists():
        p = Path(job.artifact_path)
        artifacts.append({
            "name": p.name,
            "path": str(p),
            "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
        })
    return {"job_id": job.id, "artifacts": artifacts}


@router.post("/jobs/{job_id}/import-vault", response_model=LoraVaultItem)
async def import_job_artifact_to_vault(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await TrainingEngineService.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if not job.artifact_path or not Path(job.artifact_path).exists():
        raise HTTPException(status_code=404, detail="No output artifact file found for this training job.")

    import_req = LoraImportRequest(
        file_path=job.artifact_path,
        name=job.lora_name,
        trigger_words=[job.trigger_word] if job.trigger_word else [],
        base_model=job.base_model,
        recommended_weight=0.8,
        notes=f"Trained in Forge Studio (Job {job.id[:8]})",
    )
    return await LoraService.import_lora(db, import_req)
