import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import ASGITransport, AsyncClient, Response

from app.database import async_session_maker
from app.main import app
from app.models.generation import GenerationRecord
from app.schemas.copilot import CopilotActionRequest
from app.schemas.generation import GenerationRequest
from app.services.error_sanitizer import (
    ErrorClassification,
    classify_error,
    sanitize_error_message,
)
from app.services.ollama_client import (
    DEFAULT_TIMEOUT,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_TOKENS,
    OllamaClient,
)
from app.api.generations import execute_job


# ============================================================================
# 1. Ollama Client Scheduling & Protection Tests
# ============================================================================

def test_ollama_default_timeout_configuration():
    client = OllamaClient()
    assert client.timeout == 120.0
    assert DEFAULT_TIMEOUT == 120.0


def test_ollama_context_ceiling_truncation():
    client = OllamaClient()
    assert MAX_CONTEXT_TOKENS == 8192
    assert MAX_CONTEXT_CHARS == 32768

    # Oversized prompt (e.g. 40,000 chars)
    giant_prompt = "A" * 40000
    truncated = client.truncate_prompt(giant_prompt)
    assert len(truncated) == 32768
    assert truncated == "A" * 32768

    # Normal prompt remains untouched
    normal_prompt = "A cinematic cyberpunk cityscape"
    assert client.truncate_prompt(normal_prompt) == normal_prompt


@pytest.mark.asyncio
async def test_ollama_truthful_offline_detection():
    client = OllamaClient(base_url="http://127.0.0.1:99999")
    with patch.object(client, "is_available", AsyncMock(return_value=False)):
        res = await client.generate_structured(prompt="Analyze this prompt")
        assert res["status"] == "offline"
        assert res["data"] is None
        assert "unreachable" in res["error"].lower() or "offline" in res["error"].lower()
        # Verify it did not output fake text in place of offline state
        assert res["metrics"]["eval_count"] == 0
        assert res["metrics"]["tokens_per_sec"] == 0.0


@pytest.mark.asyncio
async def test_ollama_generate_structured_online_metrics():
    client = OllamaClient()
    fake_json_content = json.dumps({"explanation": "Great prompt", "prompt": "Refined prompt"})

    fake_response = Response(
        status_code=200,
        json={
            "model": "qwen2.5-coder:32b",
            "message": {"role": "assistant", "content": fake_json_content},
            "eval_count": 150,
            "eval_duration": 3_000_000_000,  # 3 seconds -> 50 tokens/sec
        },
    )

    with patch.object(client, "is_available", AsyncMock(return_value=True)):
        with patch("httpx.AsyncClient.post", AsyncMock(return_value=fake_response)):
            res = await client.generate_structured(prompt="Optimize this prompt", system="You are an expert")
            assert res["status"] == "ok"
            assert res["data"]["explanation"] == "Great prompt"
            assert res["metrics"]["eval_count"] == 150
            assert res["metrics"]["tokens_per_sec"] == 50.0
            assert res["metrics"]["duration_ms"] >= 0


# ============================================================================
# 2. Copilot Structured Advisory Actions Tests
# ============================================================================

@pytest.fixture(autouse=True)
def mock_ollama_offline_default(monkeypatch):
    """By default, mock Ollama as offline so tests verify robust deterministic heuristics."""
    monkeypatch.setattr(OllamaClient, "is_available", AsyncMock(return_value=False))


@pytest.mark.asyncio
async def test_copilot_action_fast_generation():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "action_type": "fast_generation",
            "current_form": {
                "prompt": "An astronaut exploring Mars",
                "width": 1024,
                "height": 1024,
                "steps": 40,
                "cfg_scale": 8.0,
                "sampler": "dpmpp_2m_sde",
            },
        }
        res = await ac.post("/api/copilot/action", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["action_type"] == "fast_generation"
    assert "explanation" in data and len(data["explanation"]) > 20
    changes = data["suggested_changes"]
    assert changes["steps"] == 16
    assert changes["width"] == 768
    assert changes["height"] == 768
    assert changes["cfg_scale"] == 6.0
    assert changes["sampler"] == "euler"
    assert "prompt" in changes
    assert "astronaut" in changes["prompt"].lower()


@pytest.mark.asyncio
async def test_copilot_action_quality():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "action_type": "quality",
            "current_form": {
                "prompt": "A medieval knight in shining armor",
                "width": 768,
                "height": 768,
                "steps": 15,
            },
        }
        res = await ac.post("/api/copilot/action", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["action_type"] == "quality"
    changes = data["suggested_changes"]
    assert changes["steps"] >= 28 and changes["steps"] <= 30
    assert changes["width"] == 1024
    assert changes["height"] == 1024
    assert changes["cfg_scale"] == 6.5
    assert changes["sampler"] in ["dpmpp_2m", "euler"]
    assert "knight" in changes["prompt"].lower()
    assert any(term in changes["prompt"].lower() for term in ["8k", "detailed", "photography", "lighting"])


@pytest.mark.asyncio
async def test_copilot_action_vram_safe_resolution_sdxl_and_sd15():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # SDXL
        res_sdxl = await ac.post("/api/copilot/action", json={
            "action_type": "vram_safe_resolution",
            "current_form": {
                "model": "sd_xl_base_1.0.safetensors",
                "base_model_family": "sdxl",
            },
        })
        assert res_sdxl.status_code == 200
        sdxl_data = res_sdxl.json()
        assert sdxl_data["suggested_changes"]["width"] in [896, 1024]
        assert sdxl_data["suggested_changes"]["height"] in [896, 1024]
        assert "sdxl" in sdxl_data["explanation"].lower()

        # SD 1.5
        res_sd15 = await ac.post("/api/copilot/action", json={
            "action_type": "vram_safe_resolution",
            "current_form": {
                "model": "v1-5-pruned-emaonly.safetensors",
                "base_model_family": "sd15",
            },
        })
        assert res_sd15.status_code == 200
        sd15_data = res_sd15.json()
        assert sd15_data["suggested_changes"]["width"] in [512, 768]
        assert sd15_data["suggested_changes"]["height"] in [512, 768]
        assert "sd 1.5" in sd15_data["explanation"].lower()


@pytest.mark.asyncio
async def test_copilot_action_explain_slowness():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "action_type": "explain_slowness",
            "current_form": {
                "steps": 45,
                "width": 1536,
                "height": 1536,
                "sampler": "dpmpp_2m_sde",
                "loras": [{"name": "lora1"}, {"name": "lora2"}, {"name": "lora3"}],
            },
        }
        res = await ac.post("/api/copilot/action", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["action_type"] == "explain_slowness"
    expl = data["explanation"].lower()
    assert "step" in expl
    assert "resolution" in expl
    assert "lora" in expl
    assert "sampler" in expl
    assert data["suggested_changes"]["steps"] <= 28
    assert data["suggested_changes"]["width"] <= 1024


@pytest.mark.asyncio
async def test_copilot_action_suggest_preset_matching():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Portrait
        p_res = await ac.post("/api/copilot/action", json={
            "action_type": "suggest_preset",
            "current_form": {"prompt": "Close-up portrait photo of a smiling woman with freckles"},
        })
        assert p_res.status_code == 200
        p_data = p_res.json()
        assert p_data["preset_id"] == "builtin-portrait"
        assert p_data["preset_name"] == "Portrait"
        assert p_data["suggested_changes"]["steps"] == 28
        assert p_data["suggested_changes"]["width"] == 1024

        # 2. Cinematic landscape
        l_res = await ac.post("/api/copilot/action", json={
            "action_type": "suggest_preset",
            "current_form": {"prompt": "Epic panoramic landscape vista of misty mountains at golden hour"},
        })
        assert l_res.status_code == 200
        l_data = l_res.json()
        assert l_data["preset_id"] == "builtin-cinematic-landscape"
        assert l_data["preset_name"] == "Cinematic landscape"

        # 3. Product photography
        prod_res = await ac.post("/api/copilot/action", json={
            "action_type": "suggest_preset",
            "current_form": {"prompt": "Luxury perfume bottle studio shot with clean white background"},
        })
        assert prod_res.status_code == 200
        prod_data = prod_res.json()
        assert prod_data["preset_id"] == "builtin-product-photography"

        # 4. Fast draft
        draft_res = await ac.post("/api/copilot/action", json={
            "action_type": "suggest_preset",
            "current_form": {"prompt": "Quick sketch draft concept art of a robot"},
        })
        assert draft_res.status_code == 200
        draft_data = draft_res.json()
        assert draft_data["preset_id"] == "builtin-fast-draft"

        # 5. Inpaint repair
        inp_res = await ac.post("/api/copilot/action", json={
            "action_type": "suggest_preset",
            "current_form": {"mode": "inpaint", "prompt": "remove blemish and repair texture"},
        })
        assert inp_res.status_code == 200
        inp_data = inp_res.json()
        assert inp_data["preset_id"] == "builtin-inpainting-repair"


@pytest.mark.asyncio
async def test_copilot_strictly_review_only():
    """Verify Copilot actions do not modify DB records or create generations."""
    async with async_session_maker() as session:
        from sqlalchemy import func, select
        count_before = (await session.execute(select(func.count(GenerationRecord.id)))).scalar_one()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "action_type": "quality",
            "current_form": {"prompt": "Strict review only test"},
        }
        res = await ac.post("/api/copilot/action", json=payload)
    assert res.status_code == 200

    async with async_session_maker() as session:
        count_after = (await session.execute(select(func.count(GenerationRecord.id)))).scalar_one()

    assert count_before == count_after


@pytest.mark.asyncio
async def test_copilot_action_with_online_ollama():
    """Verify structured response incorporates Ollama generation when online."""
    fake_structured_result = {
        "status": "ok",
        "data": {
            "explanation": "Ollama custom advice for speed.",
            "prompt": "Optimized prompt by Qwen",
        },
        "metrics": {
            "duration_ms": 120,
            "eval_count": 80,
            "tokens_per_sec": 50.0,
        },
    }

    with patch.object(OllamaClient, "is_available", AsyncMock(return_value=True)):
        with patch.object(OllamaClient, "generate_structured", AsyncMock(return_value=fake_structured_result)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                res = await ac.post("/api/copilot/action", json={
                    "action_type": "fast_generation",
                    "current_form": {"prompt": "A swift cheetah"},
                })
            assert res.status_code == 200
            data = res.json()
            assert data["suggested_changes"]["prompt"] == "Optimized prompt by Qwen"
            assert data["explanation"] == "Ollama custom advice for speed."
            assert data["metrics"]["eval_count"] == 80
            assert data["metrics"]["tokens_per_sec"] == 50.0


# ============================================================================
# 3. Error Sanitizer & Classification Tests
# ============================================================================

def test_sanitize_windows_user_paths():
    raw_error_1 = "FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\pc\\AppData\\Local\\Forge\\model.safetensors'"
    sanitized_1 = sanitize_error_message(raw_error_1)
    assert "C:\\Users\\[REDACTED]\\AppData\\Local\\Forge\\model.safetensors" in sanitized_1
    assert "pc" not in sanitized_1

    raw_error_2 = "Failed to load c:/Users/alice.smith/Documents/weights.pt"
    sanitized_2 = sanitize_error_message(raw_error_2)
    assert "c:/Users/[REDACTED]/Documents/weights.pt" in sanitized_2
    assert "alice.smith" not in sanitized_2


def test_sanitize_api_keys_and_bearer_tokens():
    raw_err = "Auth failed: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret and sk-1234567890abcdef1234567890123"
    sanitized = sanitize_error_message(raw_err)
    assert "Bearer [REDACTED]" in sanitized
    assert "sk-[REDACTED]" in sanitized
    assert "1234567890abcdef" not in sanitized
    assert "eyJhbGci" not in sanitized


def test_sanitize_credentials_in_url_and_params():
    raw_err = "Connection failed to http://server?api_key=secret998877&token=myToken123456&password=password123"
    sanitized = sanitize_error_message(raw_err)
    assert "api_key=[REDACTED]" in sanitized
    assert "token=[REDACTED]" in sanitized
    assert "password=[REDACTED]" in sanitized
    assert "secret998877" not in sanitized
    assert "myToken123456" not in sanitized
    assert "password123" not in sanitized


def test_classify_error_gpu_oom_transient():
    err = "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB"
    classified = classify_error(err)
    assert classified.subsystem == "gpu"
    assert classified.category == "transient"
    assert classified.can_retry is True
    assert classified.error_code == "GPU_OOM"
    assert "resolution" in classified.remediation_guidance.lower()


def test_classify_error_comfyui_offline_transient():
    err = "RuntimeError: ComfyUI is offline at http://127.0.0.1:8188. Connection refused."
    classified = classify_error(err)
    assert classified.subsystem == "comfyui"
    assert classified.category == "transient"
    assert classified.can_retry is True
    assert classified.error_code == "COMFYUI_OFFLINE"
    assert "powershell" in classified.remediation_guidance.lower()
    assert "8188" in classified.remediation_guidance


def test_classify_error_comfyui_missing_model_permanent():
    err = "RuntimeError: No model checkpoints found in ComfyUI models/checkpoints directory."
    classified = classify_error(err)
    assert classified.subsystem == "comfyui"
    assert classified.category == "permanent"
    assert classified.can_retry is False
    assert classified.error_code == "COMFYUI_MISSING_CHECKPOINT"
    assert "checkpoint" in classified.remediation_guidance.lower()


def test_classify_error_ollama_offline_transient():
    err = "httpx.ConnectError: Ollama is unreachable at http://127.0.0.1:11434"
    classified = classify_error(err)
    assert classified.subsystem == "ollama"
    assert classified.category == "transient"
    assert classified.can_retry is True
    assert classified.error_code == "OLLAMA_OFFLINE"
    assert "ollama serve" in classified.remediation_guidance.lower()


def test_classify_error_validation_permanent():
    err = "ValueError: Invalid dimensions: width must be a multiple of 64"
    classified = classify_error(err, http_status=422)
    assert classified.subsystem == "validation"
    assert classified.category == "permanent"
    assert classified.can_retry is False
    assert classified.error_code == "VALIDATION_ERROR"


def test_classify_error_network_timeout_transient():
    err = "httpx.ReadTimeout: The read operation timed out after 120.0 seconds"
    classified = classify_error(err)
    assert classified.subsystem == "network"
    assert classified.category == "transient"
    assert classified.can_retry is True
    assert classified.error_code == "NETWORK_TIMEOUT"


# ============================================================================
# 4. Generation Error Sanitization Integration Test
# ============================================================================

@pytest.mark.asyncio
async def test_execute_job_sanitizes_error_message():
    import uuid
    from datetime import datetime, timezone
    from sqlalchemy import delete
    from app.services.comfyui_client import ComfyUIClient
    from app.services.gpu_safety import gpu_safety_manager
    from app.services.queue_scheduler import queue_scheduler

    queue_scheduler.resume()
    gpu_safety_manager.reset_leases()

    job_id = str(uuid.uuid4())
    req = GenerationRequest(
        prompt="Test error sanitization in execute_job",
        model="v1-5-pruned-emaonly.safetensors",
    )

    async with async_session_maker() as session:
        # Clear any lingering queued jobs from previous tests
        await session.execute(
            delete(GenerationRecord).where(GenerationRecord.status.in_(["queued", "waiting_for_gpu"]))
        )
        rec = GenerationRecord(
            id=job_id,
            mode="txt2img",
            status="queued",
            priority=1,
            prompt=req.prompt,
            model=req.model,
            width=512,
            height=512,
            steps=20,
            cfg_scale=7.0,
            sampler="euler",
            denoise=1.0,
            created_at=datetime.now(timezone.utc),
        )
        session.add(rec)
        await session.commit()

    # Trigger a failure containing sensitive Windows user path and API token
    sensitive_msg = (
        "Failed loading C:\\Users\\testuser\\secrets\\model.safetensors with Bearer secrettoken1234567890"
    )

    with patch.object(ComfyUIClient, "is_available", AsyncMock(side_effect=RuntimeError(sensitive_msg))):
        await asyncio.wait_for(execute_job(job_id, req, None, None), timeout=5.0)

    async with async_session_maker() as session:
        from sqlalchemy import select
        res = await session.execute(select(GenerationRecord).where(GenerationRecord.id == job_id))
        updated_rec = res.scalar_one_or_none()

    assert updated_rec is not None
    assert updated_rec.status == "failed"
    assert "testuser" not in updated_rec.error_message
    assert "C:\\Users\\[REDACTED]" in updated_rec.error_message
    assert "Bearer [REDACTED]" in updated_rec.error_message
    assert "secrettoken" not in updated_rec.error_message
