import asyncio
from datetime import datetime, timedelta, timezone
import uuid
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update

from app.database import async_session_maker
from app.main import app
from app.models.generation import GenerationRecord
from app.schemas.generation import GenerationRequest
from app.services.gpu_safety import GpuSafetyManager, gpu_safety_manager
from app.services.queue_scheduler import PriorityQueueScheduler, queue_scheduler


@pytest.fixture(autouse=True)
def reset_queue_and_gpu():
    queue_scheduler.resume()
    gpu_safety_manager.reset_leases()
    yield
    queue_scheduler.resume()
    gpu_safety_manager.reset_leases()


# =====================================================================
# 1. Priority Calculation & Starvation Aging Tests
# =====================================================================

def test_priority_calculation_and_starvation_aging():
    """
    For every 60.0s waiting, decrement effective priority by 1.
    Min is 1, Max is 3.
    """
    calc = PriorityQueueScheduler.calculate_effective_priority

    # Base priority 3 (Low)
    assert calc(3, 0.0) == 3
    assert calc(3, 59.9) == 3
    assert calc(3, 60.0) == 2
    assert calc(3, 119.9) == 2
    assert calc(3, 120.0) == 1
    assert calc(3, 300.0) == 1  # Clamped to min 1

    # Base priority 2 (Normal)
    assert calc(2, 0.0) == 2
    assert calc(2, 59.0) == 2
    assert calc(2, 60.0) == 1
    assert calc(2, 120.0) == 1  # Clamped to min 1

    # Base priority 1 (High)
    assert calc(1, 0.0) == 1
    assert calc(1, 60.0) == 1
    assert calc(1, 180.0) == 1  # Clamped to min 1

    # Edge cases
    assert calc(3, -10.0) == 3
    assert calc(4, 0.0) == 3   # Max clamped to 3
    assert calc(0, 0.0) == 1   # Min clamped to 1


def test_resource_estimation_and_hashing():
    # SD 1.5 estimation
    vram_15, dur_15 = PriorityQueueScheduler.estimate_resources(
        model="v1-5-pruned.safetensors",
        width=512,
        height=512,
        steps=20,
    )
    assert vram_15 > 0
    assert dur_15 == 1.0  # 20 * 0.05

    # SDXL estimation
    vram_xl, dur_xl = PriorityQueueScheduler.estimate_resources(
        model="sd_xl_base_1.0.safetensors",
        width=1024,
        height=1024,
        steps=20,
    )
    assert vram_xl > vram_15
    assert dur_xl == 2.4  # 20 * 0.12

    # Generation hash determinism
    hash1 = PriorityQueueScheduler.compute_generation_hash(
        mode="txt2img",
        prompt="A cute owl",
        negative_prompt="blurry",
        model="sdxl",
        width=1024,
        height=1024,
        steps=30,
        cfg_scale=7.0,
        sampler="euler",
        seed=42,
        denoise=1.0,
        loras_json=[],
        controlnet_type=None,
    )
    hash2 = PriorityQueueScheduler.compute_generation_hash(
        mode="txt2img",
        prompt="A cute owl",
        negative_prompt="blurry",
        model="sdxl",
        width=1024,
        height=1024,
        steps=30,
        cfg_scale=7.0,
        sampler="euler",
        seed=42,
        denoise=1.0,
        loras_json=[],
        controlnet_type=None,
    )
    hash3 = PriorityQueueScheduler.compute_generation_hash(
        mode="txt2img",
        prompt="A cute owl",
        negative_prompt="blurry",
        model="sdxl",
        width=1024,
        height=1024,
        steps=30,
        cfg_scale=7.0,
        sampler="euler",
        seed=999,  # different seed
        denoise=1.0,
        loras_json=[],
        controlnet_type=None,
    )
    assert hash1 == hash2
    assert hash1 != hash3


# =====================================================================
# 2. Queue Pause and Resume Tests
# =====================================================================

@pytest.mark.asyncio
async def test_queue_pause_and_resume():
    assert queue_scheduler.is_paused() is False

    queue_scheduler.pause()
    assert queue_scheduler.is_paused() is True

    # When paused, dispatch_next returns None even if jobs exist
    async with async_session_maker() as session:
        job = GenerationRecord(
            id=str(uuid.uuid4()),
            mode="txt2img",
            status="queued",
            priority=1,
            prompt="Paused test",
            model="sd15",
            width=512,
            height=512,
            steps=10,
            cfg_scale=7.0,
            sampler="euler",
            denoise=1.0,
            created_at=datetime.now(timezone.utc),
        )
        session.add(job)
        await session.commit()

        dispatched = await queue_scheduler.dispatch_next(session)
        assert dispatched is None

        # Clean up job
        await session.delete(job)
        await session.commit()

    queue_scheduler.resume()
    assert queue_scheduler.is_paused() is False


# =====================================================================
# 3. Atomic Dispatch & Lease Acquisition Tests
# =====================================================================

@pytest.mark.asyncio
async def test_atomic_dispatch_and_priority_ordering():
    now = datetime.now(timezone.utc)
    job_ids = []

    async with async_session_maker() as session:
        # Create Job A: priority 3 (Low), created 10 seconds ago
        job_a = GenerationRecord(
            id="job-dispatch-a",
            status="queued",
            priority=3,
            prompt="Job A Low",
            model="sd15",
            created_at=now - timedelta(seconds=10),
        )
        # Create Job B: priority 2 (Normal), created 5 seconds ago
        job_b = GenerationRecord(
            id="job-dispatch-b",
            status="queued",
            priority=2,
            prompt="Job B Normal",
            model="sd15",
            created_at=now - timedelta(seconds=5),
        )
        # Create Job C: priority 1 (High), created just now
        job_c = GenerationRecord(
            id="job-dispatch-c",
            status="queued",
            priority=1,
            prompt="Job C High",
            model="sd15",
            created_at=now,
        )
        session.add_all([job_a, job_b, job_c])
        await session.commit()
        job_ids = [job_a.id, job_b.id, job_c.id]

        # 1. First dispatch: Job C has priority 1, must be dispatched first
        dispatched_1 = await queue_scheduler.dispatch_next(session)
        assert dispatched_1 is not None
        assert dispatched_1.id == "job-dispatch-c"
        assert dispatched_1.status == "running"
        assert dispatched_1.started_at is not None

        # Verify GPU lease is held by Job C
        lease = await gpu_safety_manager.get_active_lease()
        assert lease is not None
        assert lease.job_id == "job-dispatch-c"

        # 2. Second dispatch while Job C holds lease:
        # Candidate cannot acquire lease, transitions to 'waiting_for_gpu' and returns None
        dispatched_2 = await queue_scheduler.dispatch_next(session)
        assert dispatched_2 is None

        # Check that Job B became waiting_for_gpu
        res_b = await session.execute(select(GenerationRecord).where(GenerationRecord.id == "job-dispatch-b"))
        rec_b = res_b.scalar_one()
        assert rec_b.status == "waiting_for_gpu"

        # 3. Release Job C's lease
        await gpu_safety_manager.release_lease("job-dispatch-c")

        # 4. Third dispatch: Job B (priority 2) should now be dispatched over Job A (priority 3)
        dispatched_3 = await queue_scheduler.dispatch_next(session)
        assert dispatched_3 is not None
        assert dispatched_3.id == "job-dispatch-b"
        assert dispatched_3.status == "running"

        # Release Job B
        await gpu_safety_manager.release_lease("job-dispatch-b")

        # 5. Fourth dispatch: Job A (priority 3) is finally dispatched
        dispatched_4 = await queue_scheduler.dispatch_next(session)
        assert dispatched_4 is not None
        assert dispatched_4.id == "job-dispatch-a"

        # Clean up
        await gpu_safety_manager.release_lease("job-dispatch-a")
        for jid in job_ids:
            r = await session.execute(select(GenerationRecord).where(GenerationRecord.id == jid))
            rec = r.scalar_one_or_none()
            if rec:
                await session.delete(rec)
        await session.commit()


@pytest.mark.asyncio
async def test_dispatch_starvation_aging_priority_elevation():
    """
    Verify that an older low-priority job ages to effective priority 1
    and beats a freshly submitted normal-priority job.
    """
    now = datetime.now(timezone.utc)
    async with async_session_maker() as session:
        # Job A: base priority 3, submitted 130s ago -> effective priority 3 - 2 = 1
        job_a = GenerationRecord(
            id="job-starve-a",
            status="queued",
            priority=3,
            prompt="Aged Job A",
            model="sd15",
            created_at=now - timedelta(seconds=130),
        )
        # Job B: base priority 2, submitted 5s ago -> effective priority 2
        job_b = GenerationRecord(
            id="job-starve-b",
            status="queued",
            priority=2,
            prompt="Fresh Job B",
            model="sd15",
            created_at=now - timedelta(seconds=5),
        )
        session.add_all([job_a, job_b])
        await session.commit()

        # Job A has effective priority 1 <= Job B (effective 2) and older created_at
        dispatched = await queue_scheduler.dispatch_next(session)
        assert dispatched is not None
        assert dispatched.id == "job-starve-a"

        await gpu_safety_manager.release_lease("job-starve-a")
        await session.delete(job_a)
        await session.delete(job_b)
        await session.commit()


# =====================================================================
# 4. Decoupled Cancellation Tests
# =====================================================================

@pytest.mark.asyncio
async def test_cancel_all_queued_preserves_running_and_completed():
    now = datetime.now(timezone.utc)
    async with async_session_maker() as session:
        j_run = GenerationRecord(id="job-c-run", status="running", priority=2, prompt="p", model="m", created_at=now)
        j_done = GenerationRecord(id="job-c-done", status="completed", priority=2, prompt="p", model="m", created_at=now)
        j_queued = GenerationRecord(id="job-c-queue", status="queued", priority=2, prompt="p", model="m", created_at=now)
        j_waiting = GenerationRecord(id="job-c-wait", status="waiting_for_gpu", priority=2, prompt="p", model="m", created_at=now)

        session.add_all([j_run, j_done, j_queued, j_waiting])
        await session.commit()

        cancelled_count = await queue_scheduler.cancel_all_queued(session)
        assert cancelled_count == 2

        # Verify running job was untouched
        r_run = (await session.execute(select(GenerationRecord).where(GenerationRecord.id == "job-c-run"))).scalar_one()
        assert r_run.status == "running"

        # Verify completed job was untouched
        r_done = (await session.execute(select(GenerationRecord).where(GenerationRecord.id == "job-c-done"))).scalar_one()
        assert r_done.status == "completed"

        # Verify queued and waiting were cancelled
        r_q = (await session.execute(select(GenerationRecord).where(GenerationRecord.id == "job-c-queue"))).scalar_one()
        assert r_q.status == "cancelled"
        assert r_q.error_message == "Cancelled from queue"

        r_w = (await session.execute(select(GenerationRecord).where(GenerationRecord.id == "job-c-wait"))).scalar_one()
        assert r_w.status == "cancelled"
        assert r_w.error_message == "Cancelled from queue"

        # Cleanup
        for j in [j_run, j_done, j_queued, j_waiting]:
            await session.delete(j)
        await session.commit()


@pytest.mark.asyncio
async def test_interrupt_active():
    now = datetime.now(timezone.utc)
    async with async_session_maker() as session:
        # Ensure clean state by cancelling any stale running jobs from prior tests
        await session.execute(
            update(GenerationRecord).where(GenerationRecord.status == "running").values(status="cancelled")
        )
        await session.commit()

        test_job_id = f"job-to-interrupt-{uuid.uuid4().hex[:8]}"
        j_run = GenerationRecord(
            id=test_job_id,
            status="running",
            priority=2,
            prompt="Interrupt test",
            model="sd15",
            created_at=now,
            started_at=now,
        )
        session.add(j_run)
        await session.commit()

        # Acquire lease for it
        await gpu_safety_manager.acquire_lease(test_job_id, "generation")
        assert (await gpu_safety_manager.get_active_lease()) is not None

        # Interrupt active job
        interrupted_id = await queue_scheduler.interrupt_active(session)
        assert interrupted_id == test_job_id

        # Verify job is marked cancelled
        await session.refresh(j_run)
        assert j_run.status == "cancelled"
        assert "interrupted" in j_run.error_message.lower()

        # Verify lease was released
        assert (await gpu_safety_manager.get_active_lease()) is None

        # Interrupting again when nothing is running returns None
        none_res = await queue_scheduler.interrupt_active(session)
        assert none_res is None

        await session.delete(j_run)
        await session.commit()


# =====================================================================
# 5. Deduplication Tests
# =====================================================================

@pytest.mark.asyncio
async def test_duplicate_request_deduplication():
    now = datetime.now(timezone.utc)
    async with async_session_maker() as session:
        job = GenerationRecord(
            id="job-dedup-1",
            status="queued",
            priority=2,
            prompt="Dedup check",
            model="sd15",
            client_request_id="req-uuid-111",
            generation_hash="hash-fixed-111",
            created_at=now,
        )
        session.add(job)
        await session.commit()

        # 1. Match by client_request_id within 5s
        dup1 = await queue_scheduler.find_duplicate_job(session, client_request_id="req-uuid-111")
        assert dup1 is not None
        assert dup1.id == "job-dedup-1"

        # 2. Match by generation_hash within 5s
        dup2 = await queue_scheduler.find_duplicate_job(session, generation_hash="hash-fixed-111")
        assert dup2 is not None
        assert dup2.id == "job-dedup-1"

        # 3. Older than 5 seconds should NOT be returned
        job.created_at = now - timedelta(seconds=10)
        await session.commit()
        dup_expired = await queue_scheduler.find_duplicate_job(session, client_request_id="req-uuid-111", window_seconds=5.0)
        assert dup_expired is None

        # 4. Completed jobs should NOT be returned as duplicates
        job.created_at = now
        job.status = "completed"
        await session.commit()
        dup_completed = await queue_scheduler.find_duplicate_job(session, client_request_id="req-uuid-111")
        assert dup_completed is None

        await session.delete(job)
        await session.commit()


# =====================================================================
# 6. Startup Crash Recovery Reconciliation Tests
# =====================================================================

@pytest.mark.asyncio
async def test_crash_recovery_reconciliation():
    now = datetime.now(timezone.utc)
    async with async_session_maker() as session:
        j_stale_run = GenerationRecord(
            id="job-stale-run",
            status="running",
            priority=2,
            prompt="Stale run",
            model="sd15",
            created_at=now,
        )
        j_stale_wait = GenerationRecord(
            id="job-stale-wait",
            status="waiting_for_gpu",
            priority=2,
            prompt="Stale wait",
            model="sd15",
            created_at=now,
        )
        session.add_all([j_stale_run, j_stale_wait])
        await session.commit()

        # Simulate orphan lease and waiting set
        await gpu_safety_manager.acquire_lease("job-stale-run", "generation")
        gpu_safety_manager.register_waiting("job-stale-wait")

        reconciled = await queue_scheduler.reconcile_stale_jobs(session)
        assert reconciled == 2

        # Verify statuses updated to failed with restart explanation
        await session.refresh(j_stale_run)
        assert j_stale_run.status == "failed"
        assert j_stale_run.error_message == "Job interrupted by server restart"

        await session.refresh(j_stale_wait)
        assert j_stale_wait.status == "failed"
        assert j_stale_wait.error_message == "Job interrupted by server restart"

        # Verify orphan leases released
        assert (await gpu_safety_manager.get_active_lease()) is None

        await session.delete(j_stale_run)
        await session.delete(j_stale_wait)
        await session.commit()


# =====================================================================
# 7. Queue HTTP API Router Endpoints Tests
# =====================================================================

@pytest.mark.asyncio
async def test_queue_router_endpoints():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Pause queue
        res_pause = await ac.post("/api/queue/pause")
        assert res_pause.status_code == 200
        assert res_pause.json()["status"] == "paused"
        assert queue_scheduler.is_paused() is True

        # 2. Get status
        res_status = await ac.get("/api/queue/status")
        assert res_status.status_code == 200
        data = res_status.json()
        assert data["paused"] is True
        assert "queued_count" in data

        # 3. Resume queue
        res_resume = await ac.post("/api/queue/resume")
        assert res_resume.status_code == 200
        assert res_resume.json()["status"] == "active"
        assert queue_scheduler.is_paused() is False

        # 4. Create a job to test reorder
        gen_res = await ac.post("/api/generations", json={
            "mode": "txt2img",
            "prompt": "Test reorder queue prompt",
            "model": "v1-5-pruned-emaonly.safetensors",
            "priority": 3,
            "steps": 10,
        })
        assert gen_res.status_code == 201
        job_id = gen_res.json()["id"]

        # 5. Reorder priority
        reorder_res = await ac.post("/api/queue/reorder", json={
            "job_id": job_id,
            "new_priority": 1,
        })
        assert reorder_res.status_code == 200
        assert reorder_res.json()["priority"] == 1

        # Reorder with nonexistent ID -> 404
        reorder_404 = await ac.post("/api/queue/reorder", json={
            "job_id": "nonexistent-job-xyz",
            "new_priority": 1,
        })
        assert reorder_404.status_code == 404

        # 6. Cancel all queued
        canc_res = await ac.post("/api/queue/cancel-all-queued")
        assert canc_res.status_code == 200
        assert canc_res.json()["cancelled_count"] >= 0

        # 7. Interrupt active (when none active)
        int_res = await ac.post("/api/queue/interrupt-active")
        assert int_res.status_code == 200
        assert int_res.json()["status"] == "interrupted"


@pytest.mark.asyncio
async def test_generation_api_deduplication_e2e():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        req_id = f"client-req-{uuid.uuid4().hex[:8]}"
        payload = {
            "mode": "txt2img",
            "prompt": "End to end deduplication test",
            "model": "v1-5-pruned-emaonly.safetensors",
            "client_request_id": req_id,
            "steps": 10,
        }

        # First submission
        res1 = await ac.post("/api/generations", json=payload)
        assert res1.status_code == 201
        id1 = res1.json()["id"]

        # Second rapid submission with same client_request_id within 5s
        res2 = await ac.post("/api/generations", json=payload)
        assert res2.status_code == 201
        id2 = res2.json()["id"]

        # Must return the same job ID due to deduplication
        assert id1 == id2

        # Cancel the job
        await ac.post(f"/api/generations/{id1}/cancel")
