from unittest.mock import AsyncMock
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.schemas.copilot import CopilotRequest
from app.services.ollama_client import OllamaClient


@pytest.fixture(autouse=True)
def mock_ollama_offline_for_endpoint_tests(monkeypatch):
    """Ensure fast, deterministic testing of copilot endpoints without waiting for Ollama weights."""
    monkeypatch.setattr(OllamaClient, "is_available", AsyncMock(return_value=False))


@pytest.mark.asyncio
async def test_copilot_improve_prompt():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "task": "improve_prompt",
            "message": "A solitary lighthouse on a cliff",
            "current_context": {"model": "sd_xl_base_1.0.safetensors"},
        }
        res = await ac.post("/api/copilot/prompt", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert data["suggested_settings"] is not None
    assert "lighthouse" in data["suggested_settings"]["prompt"].lower()


@pytest.mark.asyncio
async def test_copilot_variations():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "task": "variations",
            "message": "A neon cybernetic cat",
        }
        res = await ac.post("/api/copilot/prompt", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["variations"]) >= 1
    assert "neon cybernetic cat" in data["variations"][0].lower()


@pytest.mark.asyncio
async def test_copilot_configure_generation():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "task": "configure_generation",
            "message": "Optimize settings for portrait photography",
        }
        res = await ac.post("/api/copilot/prompt", json=payload)
    assert res.status_code == 200
    data = res.json()
    s = data["suggested_settings"]
    assert s is not None
    assert s["steps"] >= 10 and s["steps"] <= 60
    assert s["cfg_scale"] >= 1.0 and s["cfg_scale"] <= 15.0


@pytest.mark.asyncio
async def test_copilot_negative_prompt():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "task": "negative_prompt",
            "message": "A photorealistic portrait of an astronaut",
        }
        res = await ac.post("/api/copilot/prompt", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["suggested_settings"] is not None
    assert data["suggested_settings"]["negative_prompt"] is not None
    assert len(data["suggested_settings"]["negative_prompt"]) > 5



def test_copilot_llm_json_parsing():
    client = OllamaClient()
    fake_output = (
        "Here is a refined prompt and optimal SDXL parameters:\n"
        "```json\n"
        '{\n'
        '  "prompt": "A majestic cybernetic falcon soaring over Neo-Tokyo",\n'
        '  "negative_prompt": "blurry, low quality, artifacts",\n'
        '  "width": 1280,\n'
        '  "height": 768,\n'
        '  "steps": 32,\n'
        '  "cfg_scale": 7.0,\n'
        '  "sampler": "euler"\n'
        '}\n'
        "```"
    )
    req = CopilotRequest(task="improve_prompt", message="A falcon")
    res = client._parse_llm_response(fake_output, req)
    assert res.suggested_settings is not None
    assert res.suggested_settings.prompt == "A majestic cybernetic falcon soaring over Neo-Tokyo"
    assert res.suggested_settings.width == 1280
    assert res.suggested_settings.steps == 32
    assert res.suggested_settings.sampler == "euler"


def test_copilot_llm_bullet_variations_parsing():
    client = OllamaClient()
    fake_output = (
        "Here are 3 distinct stylistic interpretations:\n"
        "- A sleek obsidian cyber falcon with gold optic filaments in mist\n"
        "- A vintage steampunk mechanical falcon with brass clockwork wings\n"
        "- An abstract polygonal geometric falcon rendered in crystal and light"
    )
    req = CopilotRequest(task="variations", message="cyber falcon")
    res = client._parse_llm_response(fake_output, req)
    assert len(res.variations) == 3
    assert "obsidian" in res.variations[0]


def test_copilot_safety_sanitization():
    client = OllamaClient()
    # Malicious injection attempt
    malicious = "A cute puppy; rm -rf /; echo hacked"
    sanitized = client._sanitize_string(malicious)
    assert sanitized == "[Unsafe content filtered]"

    # Clamping dimensions
    clamped_dim = client._clamp_dimensions(2048)
    assert clamped_dim <= 1536
    assert clamped_dim % 64 == 0
