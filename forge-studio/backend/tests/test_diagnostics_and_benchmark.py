import asyncio
import csv
from datetime import datetime, timezone
import io
import json
from unittest.mock import AsyncMock, patch
import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas.diagnostics import BenchmarkRunRequest, DiagnosticsReport
from app.services.benchmark_service import benchmark_service, BenchmarkService
from app.services.diagnostics_service import diagnostics_service, DiagnosticsService
from app.services.gpu_safety import gpu_safety_manager


@pytest.fixture(autouse=True)
def reset_gpu_leases():
    gpu_safety_manager.reset_leases()
    yield
    gpu_safety_manager.reset_leases()


# =====================================================================
# 1. Diagnostics Service Tests
# =====================================================================

@pytest.mark.asyncio
async def test_diagnostics_all_9_checks_structure():
    report = await diagnostics_service.run_diagnostics(force_refresh=True)
    assert isinstance(report, DiagnosticsReport)
    assert len(report.checks) == 9
    assert report.cache_ttl_sec == 30
    assert report.sampled_at is not None

    expected_ids = {
        "backend_api",
        "comfyui_endpoint",
        "ollama_connectivity",
        "checkpoint_availability",
        "controlnet_presence",
        "cuda_execution",
        "training_readiness",
        "output_permissions",
        "lora_vault_sync",
    }
    found_ids = {c.id for c in report.checks}
    assert found_ids == expected_ids

    for check in report.checks:
        assert check.name
        assert check.category
        assert check.status in ("ok", "warning", "error")
        assert check.message
        assert check.remediation_cmd is not None
        assert check.checked_at is not None


@pytest.mark.asyncio
async def test_diagnostics_caching():
    # Force refresh first
    r1 = await diagnostics_service.run_diagnostics(force_refresh=True)
    t1 = r1.sampled_at

    # Immediate call without force_refresh should return cached report
    r2 = await diagnostics_service.run_diagnostics(force_refresh=False)
    assert r2.sampled_at == t1
    assert r2 is r1

    # Force refresh should bypass cache
    await asyncio.sleep(0.01)
    r3 = await diagnostics_service.run_diagnostics(force_refresh=True)
    assert r3.sampled_at >= t1


@pytest.mark.asyncio
async def test_diagnostics_live_cuda_execution():
    check = await diagnostics_service._check_cuda_execution(datetime.now(timezone.utc))
    assert check.id == "cuda_execution"
    assert check.status in ("ok", "error")
    assert "pip install torch" in check.remediation_cmd
    # If on machine with CUDA, status is ok
    import torch
    if torch.cuda.is_available():
        assert check.status == "ok"
        assert "Device:" in check.details


@pytest.mark.asyncio
async def test_diagnostics_handles_offline_services():
    with patch("app.services.diagnostics_service.ComfyUIClient.is_available", new_callable=AsyncMock) as mock_comfy, \
         patch("app.services.diagnostics_service.OllamaClient.is_available", new_callable=AsyncMock) as mock_ollama:
        mock_comfy.return_value = False
        mock_ollama.return_value = False

        report = await diagnostics_service.run_diagnostics(force_refresh=True)
        checks_by_id = {c.id: c for c in report.checks}

        assert checks_by_id["comfyui_endpoint"].status == "error"
        assert "offline" in checks_by_id["comfyui_endpoint"].message.lower()

        assert checks_by_id["ollama_connectivity"].status == "error"
        assert "unreachable" in checks_by_id["ollama_connectivity"].message.lower()


# =====================================================================
# 2. Benchmark Service Tests
# =====================================================================

@pytest.mark.asyncio
async def test_benchmark_requires_user_confirmation():
    req = BenchmarkRunRequest(user_confirmed=False)
    with pytest.raises(HTTPException) as exc_info:
        await benchmark_service.run_benchmark(req)
    assert exc_info.value.status_code == 400
    assert "User confirmation required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_benchmark_mutual_exclusion():
    # Pre-occupy GPU with an active training lease
    acquired, _ = await gpu_safety_manager.acquire_lease("job_train_1", "training")
    assert acquired is True

    req = BenchmarkRunRequest(user_confirmed=True)
    with pytest.raises(HTTPException) as exc_info:
        await benchmark_service.run_benchmark(req)
    assert exc_info.value.status_code == 409
    assert "GPU is currently busy" in exc_info.value.detail

    # Clean up lease
    await gpu_safety_manager.release_lease("job_train_1")


@pytest.mark.asyncio
async def test_benchmark_execution_and_metrics():
    # Test execution with truthful fallback (ComfyUI offline or online)
    req = BenchmarkRunRequest(user_confirmed=True, include_ollama=False)
    result = await benchmark_service.run_benchmark(req)

    assert result.id.startswith("bench_")
    assert result.total_time_ms > 0
    assert result.prompt_to_first_img_ms >= 0
    assert result.steps_per_sec > 0
    assert result.peak_vram_mb >= 0
    assert result.gpu_name
    assert "pytorch_version" in result.env_metadata
    assert result.options["steps"] == 20
    assert result.options["seed"] == 42
    assert result.ollama_first_token_ms is None

    # Verify lease was properly released in finally block
    lease = await gpu_safety_manager.get_active_lease()
    assert lease is None


@pytest.mark.asyncio
async def test_benchmark_with_mocked_comfyui():
    with patch("app.services.benchmark_service.ComfyUIClient.is_available", new_callable=AsyncMock) as mock_avail, \
         patch("app.services.benchmark_service.ComfyUIClient.execute_generation", new_callable=AsyncMock) as mock_gen:
        mock_avail.return_value = True

        async def fake_execute(job_id, request, progress_callback=None):
            if progress_callback:
                from app.schemas.generation import GenerationProgressUpdate
                await asyncio.sleep(0.01)
                update = GenerationProgressUpdate(
                    job_id=job_id,
                    status="running",
                    step=1,
                    total_steps=request.steps,
                    progress_percentage=5.0,
                    message="Step 1/20",
                )
                res = progress_callback(update)
                if asyncio.iscoroutine(res):
                    await res
            await asyncio.sleep(0.02)
            from pathlib import Path
            return Path("mock_output.png")

        mock_gen.side_effect = fake_execute

        req = BenchmarkRunRequest(user_confirmed=True, include_ollama=False)
        result = await benchmark_service.run_benchmark(req)

        assert result.total_time_ms > 0
        assert result.prompt_to_first_img_ms > 0
        assert result.steps_per_sec > 0
        assert mock_gen.call_count == 2  # 1 warm-up (5 steps) + 1 benchmark (20 steps)


@pytest.mark.asyncio
async def test_benchmark_export_json_and_csv():
    # Export JSON
    json_str = benchmark_service.export_results(format="json")
    parsed_json = json.loads(json_str)
    assert isinstance(parsed_json, list)

    # Export CSV
    csv_str = benchmark_service.export_results(format="csv")
    assert "id,timestamp,gpu_name" in csv_str
    reader = csv.reader(io.StringIO(csv_str))
    header = next(reader)
    assert "total_time_ms" in header
    assert "steps_per_sec" in header


# =====================================================================
# 3. API Router Integration Tests
# =====================================================================

@pytest.mark.asyncio
async def test_api_diagnostics_system():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/diagnostics/system")
        assert res.status_code == 200
        data = res.json()
        assert "checks" in data
        assert len(data["checks"]) == 9
        assert data["cache_ttl_sec"] == 30

        # Refresh parameter
        res_refresh = await ac.get("/api/diagnostics/system?refresh=true")
        assert res_refresh.status_code == 200
        data_refresh = res_refresh.json()
        assert len(data_refresh["checks"]) == 9


@pytest.mark.asyncio
async def test_api_benchmark_endpoints():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Unconfirmed request rejected with 400
        res_bad = await ac.post("/api/diagnostics/benchmark/start", json={"user_confirmed": False})
        assert res_bad.status_code == 400

        # 2. Rejection when GPU lease occupied
        await gpu_safety_manager.acquire_lease("gen_test_busy", "generation")
        res_busy = await ac.post("/api/diagnostics/benchmark/start", json={"user_confirmed": True})
        assert res_busy.status_code == 409
        await gpu_safety_manager.release_lease("gen_test_busy")

        # 3. Run benchmark successfully
        res_start = await ac.post(
            "/api/diagnostics/benchmark/start",
            json={"user_confirmed": True, "include_ollama": False},
        )
        assert res_start.status_code == 200
        bench_data = res_start.json()
        assert "id" in bench_data
        assert bench_data["steps_per_sec"] > 0

        # 4. List results
        res_list = await ac.get("/api/diagnostics/benchmark/results")
        assert res_list.status_code == 200
        results = res_list.json()
        assert isinstance(results, list)
        assert any(r["id"] == bench_data["id"] for r in results)

        # 5. Export JSON
        res_export_json = await ac.get("/api/diagnostics/benchmark/export?format=json")
        assert res_export_json.status_code == 200
        assert res_export_json.headers["content-type"].startswith("application/json")
        assert "attachment; filename=\"benchmark_results.json\"" in res_export_json.headers["content-disposition"]

        # 6. Export CSV
        res_export_csv = await ac.get("/api/diagnostics/benchmark/export?format=csv")
        assert res_export_csv.status_code == 200
        assert "text/csv" in res_export_csv.headers["content-type"]
        assert "attachment; filename=\"benchmark_results.csv\"" in res_export_csv.headers["content-disposition"]
        assert "id,timestamp" in res_export_csv.text
