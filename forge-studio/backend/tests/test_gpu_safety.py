import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas.system import ActiveLeaseInfo, GpuStatus, ObservedProcess
from app.services.gpu_safety import GpuSafetyManager, gpu_safety_manager


@pytest.fixture(autouse=True)
def reset_manager():
    gpu_safety_manager.reset_leases()
    yield
    gpu_safety_manager.reset_leases()


# =====================================================================
# 1. VRAM Estimation Tests
# =====================================================================

def test_vram_estimation_sd15_baseline():
    """
    SD 1.5 formula:
    baseline: 2600 MB
    latents: 1.8e-6 * 512 * 512 = 0.4718592 MB
    graph overhead: 600 MB
    margin: 1.20
    (2600 + 0.4718592 + 600) * 1.20 = 3840.566 -> 3840 MB
    """
    est = GpuSafetyManager.estimate_vram("sd15", width=512, height=512, steps=20)
    assert est == 3840


def test_vram_estimation_sd15_with_loras():
    """
    SD 1.5 with 2 LoRAs:
    2 * 200 = 400 MB
    (2600 + 0.4718592 + 400 + 600) * 1.20 = 4320.566 -> 4320 MB
    """
    est = GpuSafetyManager.estimate_vram("sd_15", width=512, height=512, steps=20, loras_count=2)
    assert est == 4320


def test_vram_estimation_sd15_with_controlnet():
    """
    SD 1.5 with ControlNet:
    1600 MB overhead
    (2600 + 0.4718592 + 1600 + 600) * 1.20 = 5760.566 -> 5760 MB
    """
    est = GpuSafetyManager.estimate_vram("sd 1.5", width=512, height=512, steps=20, has_controlnet=True)
    assert est == 5760


def test_vram_estimation_sdxl_baseline():
    """
    SDXL formula:
    baseline: 6400 MB
    latents: 3.2e-6 * 1024 * 1024 = 3.3554432 MB
    graph overhead: 600 MB
    margin: 1.20
    (6400 + 3.3554432 + 600) * 1.20 = 8404.0265 -> 8404 MB
    """
    est = GpuSafetyManager.estimate_vram("sdxl", width=1024, height=1024, steps=30)
    assert est == 8404


def test_vram_estimation_sdxl_with_loras():
    """
    SDXL with 1 LoRA:
    500 MB overhead
    (6400 + 3.3554432 + 500 + 600) * 1.20 = 9004.0265 -> 9004 MB
    """
    est = GpuSafetyManager.estimate_vram("sdxl", width=1024, height=1024, steps=30, loras_count=1)
    assert est == 9004


def test_vram_estimation_sdxl_with_controlnet():
    """
    SDXL with ControlNet:
    1600 MB overhead
    (6400 + 3.3554432 + 1600 + 600) * 1.20 = 10324.0265 -> 10324 MB
    """
    est = GpuSafetyManager.estimate_vram("sdxl-turbo", width=1024, height=1024, steps=10, has_controlnet=True)
    assert est == 10324


def test_vram_estimation_sdxl_combined():
    """
    SDXL with 2 LoRAs (1000 MB) and ControlNet (1600 MB):
    (6400 + 3.3554432 + 1000 + 1600 + 600) * 1.20 = 11524.0265 -> 11524 MB
    """
    est = GpuSafetyManager.estimate_vram("sdxl", width=1024, height=1024, steps=30, loras_count=2, has_controlnet=True)
    assert est == 11524


# =====================================================================
# 2. User Policy Limit Tests
# =====================================================================

def test_policy_limits_16gb():
    total_16gb = 16384
    # balanced: 85%
    assert GpuSafetyManager.get_policy_limit_mb("balanced", total_vram_mb=total_16gb) == 13926
    # max_quality: 92%
    assert GpuSafetyManager.get_policy_limit_mb("max_quality", total_vram_mb=total_16gb) == 15073
    # max_speed: 78%
    assert GpuSafetyManager.get_policy_limit_mb("max_speed", total_vram_mb=total_16gb) == 12779
    # conservative: 65%
    assert GpuSafetyManager.get_policy_limit_mb("conservative", total_vram_mb=total_16gb) == 10649


def test_policy_limits_dynamic_gpu_monitor():
    mock_status = GpuStatus(
        available=True,
        name="RTX 4090",
        vram_total_mb=24576,
        vram_used_mb=4000,
        vram_free_mb=20576,
        utilization_pct=10,
    )
    with patch("app.services.gpu_safety.GpuMonitor.get_status", return_value=mock_status):
        assert GpuSafetyManager.get_policy_limit_mb("balanced") == int(24576 * 0.85)
        assert GpuSafetyManager.get_policy_limit_mb("max_quality") == int(24576 * 0.92)


def test_can_run_copilot_policies():
    # balanced policy requires >= 4.0 GB free (4096 MB)
    allowed, _ = GpuSafetyManager.can_run_copilot("balanced", free_vram_mb=5000)
    assert allowed is True

    denied, reason = GpuSafetyManager.can_run_copilot("balanced", free_vram_mb=3000)
    assert denied is False
    assert "4.0 GB" in reason

    # max_quality policy blocks copilot if heavy render is active
    allowed_mq, _ = GpuSafetyManager.can_run_copilot("max_quality", is_heavy_render_active=False)
    assert allowed_mq is True

    denied_mq, reason_mq = GpuSafetyManager.can_run_copilot("max_quality", is_heavy_render_active=True)
    assert denied_mq is False
    assert "blocked during heavy render" in reason_mq

    # conservative policy requires >= 4.0 GB free
    allowed_c, _ = GpuSafetyManager.can_run_copilot("conservative", free_vram_mb=4500)
    assert allowed_c is True

    denied_c, _ = GpuSafetyManager.can_run_copilot("conservative", free_vram_mb=2000)
    assert denied_c is False


def test_can_allocate():
    # SDXL baseline requires 8404 MB
    # On 8 GB GPU (8192 MB) with conservative policy (65% = 5324 MB), it should be denied
    allowed, msg = GpuSafetyManager.can_allocate("sdxl", 1024, 1024, policy="conservative", total_vram_mb=8192)
    assert allowed is False
    assert "exceeds conservative policy limit" in msg

    # On 16 GB GPU (16384 MB) with balanced policy (85% = 13926 MB), it should be allowed
    allowed_16, _ = GpuSafetyManager.can_allocate("sdxl", 1024, 1024, policy="balanced", total_vram_mb=16384)
    assert allowed_16 is True


# =====================================================================
# 3. Atomic GPU Lease Coordinator Tests
# =====================================================================

@pytest.mark.asyncio
async def test_atomic_mutual_exclusion_heavy_leases():
    manager = GpuSafetyManager()

    # Job 1 acquires heavy generation lease
    ok, msg = await manager.acquire_lease("job-gen-1", "generation")
    assert ok is True
    assert "Lease acquired" in msg

    # Job 2 cannot acquire heavy training lease
    ok2, msg2 = await manager.acquire_lease("job-train-1", "training")
    assert ok2 is False
    assert "GPU busy" in msg2
    assert "generation" in msg2

    # Job 3 cannot acquire benchmark lease
    ok3, msg3 = await manager.acquire_lease("job-bench-1", "benchmark")
    assert ok3 is False
    assert "GPU busy" in msg3

    # Job 1 can renew its own lease
    ok_renew, msg_renew = await manager.acquire_lease("job-gen-1", "generation")
    assert ok_renew is True
    assert "renewed" in msg_renew

    # Non-holder cannot release the lease
    rel_fake = await manager.release_lease("job-train-1")
    assert rel_fake is False

    # Holder releases lease
    rel_ok = await manager.release_lease("job-gen-1")
    assert rel_ok is True

    # Now Job 2 can acquire lease
    ok_train, _ = await manager.acquire_lease("job-train-1", "training")
    assert ok_train is True

    lease = await manager.get_active_lease()
    assert lease is not None
    assert lease.job_id == "job-train-1"
    assert lease.lease_type == "training"


@pytest.mark.asyncio
async def test_concurrent_lease_race():
    manager = GpuSafetyManager()

    async def try_acquire(job_id: str):
        return await manager.acquire_lease(job_id, "generation")

    # Launch 10 concurrent requests
    results = await asyncio.gather(*(try_acquire(f"job-{i}") for i in range(10)))
    successes = [r for r in results if r[0] is True]
    failures = [r for r in results if r[0] is False]

    assert len(successes) == 1
    assert len(failures) == 9


# =====================================================================
# 4. Lease TTL & Heartbeat Tests
# =====================================================================

@pytest.mark.asyncio
async def test_lease_ttl_and_heartbeat():
    manager = GpuSafetyManager()

    ok, _ = await manager.acquire_lease("job-ttl", "generation")
    assert ok is True

    lease = await manager.get_active_lease()
    assert lease is not None
    assert lease.expires_at > datetime.now(timezone.utc)

    # Heartbeat from wrong job fails
    hb_fail = await manager.heartbeat("job-wrong")
    assert hb_fail is False

    # Heartbeat from holder succeeds
    hb_ok = await manager.heartbeat("job-ttl")
    assert hb_ok is True

    # Simulate expired lease by moving expires_at into the past
    lease.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    # get_active_lease detects expiration and cleans up
    active = await manager.get_active_lease()
    assert active is None

    # New job can now acquire lease because previous expired
    ok_new, _ = await manager.acquire_lease("job-after-expiry", "training")
    assert ok_new is True


# =====================================================================
# 5. External Process Inspection Tests
# =====================================================================

def test_get_observed_processes():
    mock_csv = (
        "1001, C:\\Windows\\System32\\dwm.exe, [N/A]\n"
        "2002, C:\\AI\\llama-server.exe, 3500\n"
        "3003, C:\\Python311\\python.exe, 1200\n"
    )
    mock_res = MagicMock(returncode=0, stdout=mock_csv)

    with patch("shutil.which", return_value="nvidia-smi"):
        with patch("subprocess.run", return_value=mock_res):
            procs = GpuSafetyManager.get_observed_processes()

    assert len(procs) == 2
    p_names = [p.name for p in procs]
    assert "llama-server.exe" in p_names
    assert "python.exe" in p_names

    p_llama = next(p for p in procs if p.name == "llama-server.exe")
    assert p_llama.pid == 2002
    assert p_llama.vram_used_mb == 3500

    p_py = next(p for p in procs if p.name == "python.exe")
    assert p_py.pid == 3003
    assert p_py.vram_used_mb == 1200


# =====================================================================
# 6. Telemetry State & Endpoint Tests
# =====================================================================

def test_state_determination():
    manager = GpuSafetyManager()

    # 1. GPU unavailable
    gpu_off = GpuStatus(available=False, name="NVIDIA GPU Not Detected")
    assert manager.determine_state(gpu_off, None, []) == "gpu_unavailable"

    # 2. CUDA unavailable
    gpu_cuda_off = GpuStatus(available=False, name="CUDA Driver Unavailable")
    assert manager.determine_state(gpu_cuda_off, None, []) == "cuda_unavailable"

    # 3. GPU Ready
    gpu_ok = GpuStatus(available=True, name="RTX 4090", vram_total_mb=24576, vram_used_mb=3000, utilization_pct=15)
    assert manager.determine_state(gpu_ok, None, []) == "gpu_ready"

    # 4. Active training lease
    now = datetime.now(timezone.utc)
    lease_train = ActiveLeaseInfo(lease_type="training", job_id="t1", holder="forge", acquired_at=now, expires_at=now + timedelta(seconds=300))
    assert manager.determine_state(gpu_ok, lease_train, []) == "training_active"

    # 5. Active generation lease
    lease_gen = ActiveLeaseInfo(lease_type="generation", job_id="g1", holder="forge", acquired_at=now, expires_at=now + timedelta(seconds=300))
    assert manager.determine_state(gpu_ok, lease_gen, []) == "comfyui_active"

    # 6. Observed llama-server
    procs_ollama = [ObservedProcess(pid=10, name="llama-server.exe", vram_used_mb=4000)]
    assert manager.determine_state(gpu_ok, None, procs_ollama) == "ollama_active"

    # 7. Memory pressure (>= 85%)
    gpu_press = GpuStatus(available=True, name="RTX 4090", vram_total_mb=10000, vram_used_mb=8800, utilization_pct=50)
    assert manager.determine_state(gpu_press, None, []) == "gpu_memory_pressure"

    # 8. Waiting for GPU
    manager.register_waiting("wait-job-1")
    assert manager.determine_state(gpu_ok, None, []) == "waiting_for_gpu"
    manager.unregister_waiting("wait-job-1")


@pytest.mark.asyncio
async def test_telemetry_api_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/system/telemetry")

    assert response.status_code == 200
    data = response.json()

    # Validate structure & fields
    assert "gpu_name" in data
    assert "total_vram_mb" in data
    assert "used_vram_mb" in data
    assert "free_vram_mb" in data
    assert "utilization_pct" in data
    assert "cpu_ram_used_mb" in data
    assert "cpu_ram_total_mb" in data
    assert "observed_processes" in data
    assert isinstance(data["observed_processes"], list)
    assert "active_lease" in data
    assert "state" in data
    assert data["pressure_threshold_pct"] == 85
    assert data["critical_threshold_pct"] == 95
    assert "freshness_ms" in data
    assert "backend_latency_ms" in data
    assert "sampled_at" in data

    # State must be one of the recognized states
    valid_states = {
        "gpu_ready",
        "gpu_memory_pressure",
        "gpu_unavailable",
        "cuda_unavailable",
        "ollama_active",
        "comfyui_active",
        "training_active",
        "waiting_for_gpu",
    }
    assert data["state"] in valid_states


@pytest.mark.asyncio
async def test_telemetry_with_active_lease():
    # Acquire a training lease
    await gpu_safety_manager.acquire_lease("test-train-job", "training")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/system/telemetry")

    assert response.status_code == 200
    data = response.json()
    assert data["active_lease"] is not None
    assert data["active_lease"]["job_id"] == "test-train-job"
    assert data["active_lease"]["lease_type"] == "training"
    assert data["state"] == "training_active"

    # Release lease and verify state clears
    await gpu_safety_manager.release_lease("test-train-job")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response_cleared = await ac.get("/api/system/telemetry")

    data_cleared = response_cleared.json()
    assert data_cleared["active_lease"] is None
