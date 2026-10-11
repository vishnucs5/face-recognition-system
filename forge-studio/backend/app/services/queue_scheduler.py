import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any
from sqlalchemy import desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.generation import GenerationRecord
from app.services.comfyui_client import ComfyUIClient
from app.services.gpu_safety import GpuSafetyManager, gpu_safety_manager


class _ClassOrInstanceMethod:
    """Descriptor enabling methods to be invoked either on PriorityQueueScheduler or on an instance."""
    def __init__(self, fn):
        self.fn = fn

    def __get__(self, instance, owner):
        if instance is None:
            return lambda *args, **kwargs: self.fn(owner.get_default(), *args, **kwargs)
        return lambda *args, **kwargs: self.fn(instance, *args, **kwargs)


class PriorityQueueScheduler:
    _instance: "PriorityQueueScheduler | None" = None

    def __init__(self, safety_manager: GpuSafetyManager | None = None):
        self._paused: bool = False
        self._lock = asyncio.Lock()
        self._gpu_safety: GpuSafetyManager = safety_manager or gpu_safety_manager

    @classmethod
    def get_default(cls) -> "PriorityQueueScheduler":
        global queue_scheduler
        if queue_scheduler is None:
            queue_scheduler = PriorityQueueScheduler()
        return queue_scheduler

    @property
    def gpu_safety(self) -> GpuSafetyManager:
        return self._gpu_safety

    @gpu_safety.setter
    def gpu_safety(self, mgr: GpuSafetyManager):
        self._gpu_safety = mgr

    def _pause(self) -> None:
        self._paused = True

    def _resume(self) -> None:
        self._paused = False

    def _is_paused(self) -> bool:
        return self._paused

    pause = _ClassOrInstanceMethod(_pause)
    resume = _ClassOrInstanceMethod(_resume)
    is_paused = _ClassOrInstanceMethod(_is_paused)

    @staticmethod
    def calculate_effective_priority(base_priority: int, wait_seconds: float) -> int:
        """
        Starvation prevention / aging:
        For every 60.0s waiting, decrement effective priority by 1 (1=High, 2=Normal, 3=Low).
        Min is 1, Max is 3.
        """
        age_boost = int(wait_seconds // 60.0) if wait_seconds > 0 else 0
        effective = base_priority - age_boost
        return max(1, min(3, effective))

    @staticmethod
    def estimate_resources(
        model: str,
        width: int,
        height: int,
        steps: int = 20,
        loras_count: int = 0,
        has_controlnet: bool = False,
    ) -> tuple[int, float]:
        """
        Compute estimated VRAM (MB) and duration (sec).
        Duration: steps * (0.05 if sd15 else 0.12).
        """
        model_lower = (model or "").lower()
        model_family = "sdxl" if ("sdxl" in model_lower or "sd_xl" in model_lower) else "sd15"
        vram_mb = GpuSafetyManager.estimate_vram(
            model_family=model_family,
            width=width,
            height=height,
            steps=steps,
            loras_count=loras_count,
            has_controlnet=has_controlnet,
        )
        factor = 0.05 if model_family == "sd15" else 0.12
        duration_sec = round(steps * factor, 2)
        return vram_mb, duration_sec

    @staticmethod
    def compute_generation_hash(
        mode: str,
        prompt: str,
        negative_prompt: str | None,
        model: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        sampler: str,
        seed: int | None,
        denoise: float,
        loras_json: str | list | None,
        controlnet_type: str | None = None,
    ) -> str:
        """
        SHA256 of (mode, prompt, negative_prompt, model, width, height, steps, cfg_scale, sampler, seed, denoise, loras_json, controlnet_type)
        """
        if isinstance(loras_json, list):
            loras_str = json.dumps(loras_json, sort_keys=True)
        else:
            loras_str = str(loras_json or "[]")

        parts = [
            str(mode or "txt2img"),
            str(prompt or ""),
            str(negative_prompt or ""),
            str(model or ""),
            str(width),
            str(height),
            str(steps),
            str(cfg_scale),
            str(sampler or "euler"),
            str(seed if seed is not None else ""),
            str(denoise),
            loras_str,
            str(controlnet_type or ""),
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _find_duplicate_job(
        self,
        db: AsyncSession,
        client_request_id: str | None = None,
        generation_hash: str | None = None,
        window_seconds: float = 5.0,
    ) -> GenerationRecord | None:
        """
        Deduplication: Check if a job with same client_request_id or identical generation_hash
        was submitted within last 5 seconds and is queued, waiting_for_gpu, or running.
        """
        if not client_request_id and not generation_hash:
            return None

        query = select(GenerationRecord).where(
            GenerationRecord.status.in_(["queued", "waiting_for_gpu", "running"])
        )
        conditions = []
        if client_request_id:
            conditions.append(GenerationRecord.client_request_id == client_request_id)
        if generation_hash:
            conditions.append(GenerationRecord.generation_hash == generation_hash)

        query = query.where(or_(*conditions)).order_by(desc(GenerationRecord.created_at))
        result = await db.execute(query)
        candidates = result.scalars().all()

        now = datetime.now(timezone.utc)
        for cand in candidates:
            c_time = cand.created_at
            if c_time is not None:
                if c_time.tzinfo is None:
                    c_time = c_time.replace(tzinfo=timezone.utc)
                diff = (now - c_time).total_seconds()
                if 0 <= diff <= window_seconds:
                    return cand
        return None

    async def _dispatch_next(self, db: AsyncSession) -> GenerationRecord | None:
        """
        Atomic Dispatch:
        Locked with asyncio.Lock().
        If paused, return None.
        Find all jobs in status queued or waiting_for_gpu.
        Calculate effective priority for each. Sort by effective_priority ASC, created_at ASC.
        For best candidate: check acquire_lease.
        If lease acquired: status='running', started_at=now. Return job.
        If lease denied: status='waiting_for_gpu'. Return None.
        """
        async with self._lock:
            if self._paused:
                return None

            result = await db.execute(
                select(GenerationRecord).where(
                    GenerationRecord.status.in_(["queued", "waiting_for_gpu"])
                )
            )
            candidates = result.scalars().all()
            if not candidates:
                return None

            now = datetime.now(timezone.utc)
            scored = []
            for job in candidates:
                c_time = job.created_at
                if c_time is not None:
                    if c_time.tzinfo is None:
                        c_time = c_time.replace(tzinfo=timezone.utc)
                    wait_sec = max(0.0, (now - c_time).total_seconds())
                else:
                    wait_sec = 0.0
                eff_prio = self.calculate_effective_priority(job.priority or 2, wait_sec)
                scored.append((eff_prio, c_time or datetime.min, job))

            # Sort by effective_priority ASC, created_at ASC
            scored.sort(key=lambda item: (item[0], item[1]))
            candidate = scored[0][2]

            acquired, _msg = await self._gpu_safety.acquire_lease(candidate.id, "generation")
            if acquired:
                candidate.status = "running"
                candidate.started_at = now
                self._gpu_safety.unregister_waiting(candidate.id)
                await db.commit()
                await db.refresh(candidate)
                return candidate
            else:
                candidate.status = "waiting_for_gpu"
                self._gpu_safety.register_waiting(candidate.id)
                await db.commit()
                return None

    async def _reconcile_stale_jobs(self, db: AsyncSession) -> int:
        """
        Startup Crash Recovery:
        Find jobs with status in running, waiting_for_gpu.
        Mark as failed with error_message='Job interrupted by server restart', release any orphan leases.
        """
        result = await db.execute(
            select(GenerationRecord).where(
                GenerationRecord.status.in_(["running", "waiting_for_gpu"])
            )
        )
        stale_jobs = result.scalars().all()
        count = len(stale_jobs)
        now = datetime.now(timezone.utc)
        for job in stale_jobs:
            job.status = "failed"
            job.error_message = "Job interrupted by server restart"
            job.completed_at = now
            await self._gpu_safety.release_lease(job.id)
            self._gpu_safety.unregister_waiting(job.id)
        if count > 0:
            await db.commit()
        return count

    async def _cancel_all_queued(self, db: AsyncSession) -> int:
        """
        Decoupled Cancellation:
        Cancels ONLY queued and waiting_for_gpu jobs. Never touches active running or completed jobs!
        """
        result = await db.execute(
            select(GenerationRecord).where(
                GenerationRecord.status.in_(["queued", "waiting_for_gpu"])
            )
        )
        queued_jobs = result.scalars().all()
        count = len(queued_jobs)
        now = datetime.now(timezone.utc)
        try:
            from app.api.generations import active_tasks
        except ImportError:
            active_tasks = {}

        for job in queued_jobs:
            job.status = "cancelled"
            job.error_message = "Cancelled from queue"
            job.completed_at = now
            self._gpu_safety.unregister_waiting(job.id)
            if job.id in active_tasks:
                try:
                    active_tasks[job.id].cancel()
                except Exception:
                    pass
        if count > 0:
            await db.commit()
        return count

    async def _interrupt_active(self, db: AsyncSession, job_id: str | None = None) -> str | None:
        """
        Decoupled Cancellation:
        Sends /interrupt to ComfyUI, releases lease, sets running job status to cancelled.
        """
        try:
            comfy_client = ComfyUIClient()
            await comfy_client.interrupt()
        except Exception:
            pass

        try:
            from app.api.generations import active_tasks
        except ImportError:
            active_tasks = {}

        query = select(GenerationRecord).where(GenerationRecord.status == "running")
        if job_id:
            query = query.where(GenerationRecord.id == job_id)
        query = query.order_by(desc(GenerationRecord.started_at), desc(GenerationRecord.created_at))
        result = await db.execute(query)
        running_job = result.scalars().first()

        interrupted_id = None
        if running_job:
            interrupted_id = running_job.id
            running_job.status = "cancelled"
            running_job.error_message = "Generation interrupted by user"
            running_job.completed_at = datetime.now(timezone.utc)
            await self._gpu_safety.release_lease(running_job.id)
            if running_job.id in active_tasks:
                try:
                    active_tasks[running_job.id].cancel()
                except Exception:
                    pass
            await db.commit()
        else:
            lease = await self._gpu_safety.get_active_lease()
            if lease and (job_id is None or lease.job_id == job_id):
                interrupted_id = lease.job_id
                await self._gpu_safety.release_lease(lease.job_id)
                if lease.job_id in active_tasks:
                    try:
                        active_tasks[lease.job_id].cancel()
                    except Exception:
                        pass
        return interrupted_id

    async def _reorder_priority(self, db: AsyncSession, job_id: str, new_priority: int) -> GenerationRecord:
        if new_priority not in (1, 2, 3):
            raise ValueError("Priority must be 1 (High), 2 (Normal), or 3 (Low)")
        result = await db.execute(select(GenerationRecord).where(GenerationRecord.id == job_id))
        rec = result.scalar_one_or_none()
        if not rec:
            raise KeyError(f"Job {job_id} not found")
        rec.priority = new_priority
        await db.commit()
        await db.refresh(rec)
        return rec

    # Bindings allowing class-level or instance-level calls
    find_duplicate_job = _ClassOrInstanceMethod(_find_duplicate_job)
    dispatch_next = _ClassOrInstanceMethod(_dispatch_next)
    reconcile_stale_jobs = _ClassOrInstanceMethod(_reconcile_stale_jobs)
    cancel_all_queued = _ClassOrInstanceMethod(_cancel_all_queued)
    interrupt_active = _ClassOrInstanceMethod(_interrupt_active)
    reorder_priority = _ClassOrInstanceMethod(_reorder_priority)


queue_scheduler = PriorityQueueScheduler()
