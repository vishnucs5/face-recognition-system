import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.schemas.generation import GenerationRequest, LoraInput
from app.services.comfyui_client import ComfyUIClient


def test_prepare_workflow_txt2img_basic():
    client = ComfyUIClient()
    req = GenerationRequest(
        mode="txt2img",
        prompt="A serene cinematic vista",
        negative_prompt="blurry, distorted",
        model="sd_xl_base_1.0.safetensors",
        width=1152,
        height=896,
        steps=32,
        cfg_scale=8.0,
        sampler="dpmpp_2m",
        seed=999888,
    )
    workflow = client.prepare_workflow(req, "test_job_123")

    # Verify Checkpoint
    assert workflow["4"]["inputs"]["ckpt_name"] == "sd_xl_base_1.0.safetensors"
    # Verify Dimensions
    assert workflow["5"]["inputs"]["width"] == 1152
    assert workflow["5"]["inputs"]["height"] == 896
    # Verify Prompts
    assert workflow["6"]["inputs"]["text"] == "A serene cinematic vista"
    assert workflow["7"]["inputs"]["text"] == "blurry, distorted"
    # Verify Sampler
    assert workflow["3"]["inputs"]["steps"] == 32
    assert workflow["3"]["inputs"]["cfg"] == 8.0
    assert workflow["3"]["inputs"]["sampler_name"] == "dpmpp_2m"
    assert workflow["3"]["inputs"]["seed"] == 999888
    assert workflow["3"]["inputs"]["denoise"] == 1.0


def test_prepare_workflow_with_loras():
    client = ComfyUIClient()
    req = GenerationRequest(
        mode="txt2img",
        prompt="Cyberpunk android",
        model="sd_xl_base_1.0.safetensors",
        loras=[
            LoraInput(name="cyber_style.safetensors", strength=0.85),
            LoraInput(name="mech_detail.safetensors", strength=0.6),
        ]
    )
    workflow = client.prepare_workflow(req, "test_job_lora")

    # Verify LoRA node injections
    assert "lora_1" in workflow
    assert "lora_2" in workflow
    assert workflow["lora_1"]["inputs"]["lora_name"] == "cyber_style.safetensors"
    assert workflow["lora_1"]["inputs"]["strength_model"] == 0.85
    assert workflow["lora_2"]["inputs"]["lora_name"] == "mech_detail.safetensors"
    assert workflow["lora_2"]["inputs"]["strength_model"] == 0.6

    # Verify model connection chained to last LoRA
    assert workflow["3"]["inputs"]["model"] == ["lora_2", 0]
    assert workflow["6"]["inputs"]["clip"] == ["lora_2", 1]


def test_prepare_workflow_img2img():
    client = ComfyUIClient()
    req = GenerationRequest(
        mode="img2img",
        prompt="Transform into oil painting",
        model="sd_xl_base_1.0.safetensors",
        denoise=0.65,
    )
    workflow = client.prepare_workflow(req, "test_job_img2img", uploaded_source_filename="uploaded_portrait.png")

    assert workflow["10"]["inputs"]["image"] == "uploaded_portrait.png"
    assert workflow["3"]["inputs"]["denoise"] == 0.65


@pytest.mark.asyncio
async def test_cancel_generation_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create a generation first
        payload = {
            "mode": "txt2img",
            "prompt": "Test cancellation prompt",
            "model": "sd_xl_base_1.0.safetensors",
            "steps": 30,
        }
        res = await ac.post("/api/generations", json=payload)
        assert res.status_code == 201
        job_id = res.json()["id"]

        # Cancel the generation
        cancel_res = await ac.post(f"/api/generations/{job_id}/cancel")
        assert cancel_res.status_code == 200
        assert cancel_res.json()["job_id"] == job_id

        # Allow cancelled background task to settle cleanly
        import asyncio
        for _ in range(20):
            from app.api.generations import active_tasks
            if job_id not in active_tasks:
                break
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_get_generation_by_id():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 404 on nonexistent ID
        res_404 = await ac.get("/api/generations/nonexistent-uuid-1234")
        assert res_404.status_code == 404


@pytest.mark.asyncio
async def test_clear_queue_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Create a generation and cancel it
        payload = {
            "mode": "txt2img",
            "prompt": "Generation to cancel and clear",
            "model": "sd_xl_base_1.0.safetensors",
            "steps": 20,
        }
        res = await ac.post("/api/generations", json=payload)
        assert res.status_code == 201
        job_id = res.json()["id"]

        cancel_res = await ac.post(f"/api/generations/{job_id}/cancel")
        assert cancel_res.status_code == 200

        # Wait for task to settle
        import asyncio
        await asyncio.sleep(0.1)

        # 2. Call clear-queue
        clear_res = await ac.post("/api/generations/clear-queue")
        assert clear_res.status_code == 200
        count = clear_res.json()["count"]
        assert count >= 1

        # 3. Verify it is no longer returned in list
        list_res = await ac.get("/api/generations")
        assert list_res.status_code == 200
        ids = [g["id"] for g in list_res.json()]
        assert job_id not in ids

