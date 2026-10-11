import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_valid_txt2img_submission():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "mode": "txt2img",
            "prompt": "A serene mountain lake at dawn, crisp reflections, photorealistic 8k",
            "negative_prompt": "blurry, low quality, deformed",
            "model": "sd_xl_base_1.0.safetensors",
            "width": 1024,
            "height": 1024,
            "steps": 25,
            "cfg_scale": 7.0,
            "sampler": "euler",
            "seed": 123456,
            "denoise": 1.0,
            "loras": [
                {"name": "cinematic_style.safetensors", "strength": 0.8}
            ]
        }
        res = await ac.post("/api/generations", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["status"] in ["queued", "running", "completed"]
    assert data["prompt"] == payload["prompt"]
    assert data["seed"] == 123456
    assert len(data["loras"]) == 1


@pytest.mark.asyncio
async def test_empty_prompt_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "mode": "txt2img",
            "prompt": "",  # Empty prompt
            "model": "sd_xl_base_1.0.safetensors",
        }
        res = await ac.post("/api/generations", json=payload)
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_excessive_prompt_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "mode": "txt2img",
            "prompt": "x" * 4001,  # Exceeds max_length 4000
            "model": "sd_xl_base_1.0.safetensors",
        }
        res = await ac.post("/api/generations", json=payload)
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_invalid_dimensions_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "mode": "txt2img",
            "prompt": "Valid prompt",
            "model": "sd_xl_base_1.0.safetensors",
            "width": 2048,  # Exceeds 1536 VRAM boundary
            "height": 1024,
        }
        res = await ac.post("/api/generations", json=payload)
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_more_than_two_loras_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "mode": "txt2img",
            "prompt": "A test prompt",
            "model": "sd_xl_base_1.0.safetensors",
            "loras": [
                {"name": "lora1.safetensors", "strength": 0.5},
                {"name": "lora2.safetensors", "strength": 0.5},
                {"name": "lora3.safetensors", "strength": 0.5},  # 3 LoRAs -> exceeds MVP limit 2
            ]
        }
        res = await ac.post("/api/generations", json=payload)
    assert res.status_code == 400
    assert "maximum 2 LoRAs" in res.json()["detail"]


@pytest.mark.asyncio
async def test_img2img_requires_source_image():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "mode": "img2img",
            "prompt": "A test prompt for img2img",
            "model": "sd_xl_base_1.0.safetensors",
            "source_image_id": None,  # Missing source image
        }
        res = await ac.post("/api/generations", json=payload)
    assert res.status_code == 400
    assert "source image is required" in res.json()["detail"]


@pytest.mark.asyncio
async def test_inpaint_requires_source_and_mask():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "mode": "inpaint",
            "prompt": "A test prompt for inpaint",
            "model": "sd_xl_base_1.0.safetensors",
            "source_image_id": "upload_sample.png",
            "mask_image_id": None,  # Missing mask
        }
        res = await ac.post("/api/generations", json=payload)
    assert res.status_code == 400
    assert "Inpainting requires both a source image and a painted mask." in res.json()["detail"]


@pytest.mark.asyncio
async def test_controlnet_requires_source_and_valid_model():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Missing source image
        res1 = await ac.post("/api/generations", json={
            "mode": "controlnet",
            "prompt": "Test cnet prompt",
            "model": "sd_xl_base_1.0.safetensors",
            "source_image_id": None,
        })
        assert res1.status_code == 400
        assert "source image is required" in res1.json()["detail"]

        # 2. Missing/unavailable ControlNet model
        res2 = await ac.post("/api/generations", json={
            "mode": "controlnet",
            "prompt": "Test cnet prompt",
            "model": "sd_xl_base_1.0.safetensors",
            "source_image_id": "upload_nonexistent.png",
            "controlnet_model": "missing_cnet_model.safetensors",
        })
        assert res2.status_code == 400


@pytest.mark.asyncio
async def test_controlnet_types_and_models_endpoints():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res_types = await ac.get("/api/controlnet/types")
        assert res_types.status_code == 200
        types = res_types.json()
        assert len(types) == 5
        assert any(t["id"] == "canny" for t in types)

        res_models = await ac.get("/api/controlnet/models")
        assert res_models.status_code == 200
        assert isinstance(res_models.json(), list)

