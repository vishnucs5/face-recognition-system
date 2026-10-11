from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.generation import GenerationRecord
from app.services.queue_scheduler import queue_scheduler

router = APIRouter(prefix="/queue", tags=["Queue"])


class ReorderPriorityRequest(BaseModel):
    job_id: str
    new_priority: int = Field(ge=1, le=3, description="1=High, 2=Normal, 3=Low")


class InterruptActiveRequest(BaseModel):
    job_id: str | None = None


@router.post("/pause")
async def pause_queue():
    queue_scheduler.pause()
    return {"status": "paused"}


@router.post("/resume")
async def resume_queue():
    queue_scheduler.resume()
    return {"status": "active"}


@router.get("/status")
async def get_queue_status(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(GenerationRecord.id).where(
            GenerationRecord.status.in_(["queued", "waiting_for_gpu"])
        )
    )
    queued_count = len(result.scalars().all())

    running_result = await db.execute(
        select(GenerationRecord.id).where(GenerationRecord.status == "running")
    )
    running_id = running_result.scalar_one_or_none()
    if not running_id:
        lease = await queue_scheduler.gpu_safety.get_active_lease()
        if lease and lease.lease_type in ("generation", "comfyui"):
            running_id = lease.job_id

    return {
        "paused": queue_scheduler.is_paused(),
        "queued_count": queued_count,
        "running_job_id": running_id,
    }


@router.post("/reorder")
async def reorder_priority(
    payload: ReorderPriorityRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        rec = await queue_scheduler.reorder_priority(db, payload.job_id, payload.new_priority)
        return {"status": "updated", "job_id": rec.id, "priority": rec.priority}
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Generation job '{payload.job_id}' not found."
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/cancel-all-queued")
async def cancel_all_queued(db: AsyncSession = Depends(get_db)):
    count = await queue_scheduler.cancel_all_queued(db)
    return {"cancelled_count": count}


@router.post("/interrupt-active")
async def interrupt_active(
    payload: InterruptActiveRequest | None = Body(None),
    db: AsyncSession = Depends(get_db)
):
    target_job_id = payload.job_id if payload else None
    interrupted_id = await queue_scheduler.interrupt_active(db, job_id=target_job_id)
    return {"status": "interrupted", "job_id": interrupted_id}
