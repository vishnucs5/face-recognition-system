from PIL import Image
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_training_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/api/training/health")
        assert res.status_code == 200
        data = res.json()
        assert "engine_available" in data
        assert "recommended_preset" in data
        assert data["recommended_preset"]["resolution"] == 1024


@pytest.mark.asyncio
async def test_validate_dataset_nonexistent():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/training/validate-dataset", json={"dataset_path": "c:/non_existent_dataset_dir"})
        assert res.status_code == 200
        data = res.json()
        assert data["is_valid"] is False
        assert data["total_images"] == 0


@pytest.mark.asyncio
async def test_validate_dataset_with_images_and_captions(tmp_path):
    # Create 4 dummy PNGs (512x512)
    for i in range(4):
        img_path = tmp_path / f"img_{i}.png"
        img = Image.new("RGB", (512, 512), color=(i * 40, i * 40, i * 40))
        img.save(img_path)
        # Create caption for first 3, leave 4th without caption
        if i < 3:
            (tmp_path / f"img_{i}.txt").write_text(f"sample artwork caption {i}", encoding="utf-8")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/training/validate-dataset", json={"dataset_path": str(tmp_path)})
        assert res.status_code == 200
        data = res.json()
        assert data["is_valid"] is True
        assert data["total_images"] == 4
        assert data["missing_captions_count"] == 1
        assert "PNG" in data["supported_formats"]


@pytest.mark.asyncio
async def test_caption_preview_and_apply(tmp_path):
    # Create 2 images and captions
    for i in range(2):
        img_path = tmp_path / f"sample_{i}.png"
        img = Image.new("RGB", (512, 512), color=(100, 100, 100))
        img.save(img_path)
        (tmp_path / f"sample_{i}.txt").write_text(f"portrait of a cyborg", encoding="utf-8")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Preview with trigger word prepend
        prev_res = await ac.post("/api/training/preview-captions", json={
            "dataset_path": str(tmp_path),
            "trigger_word": "cyborg_style",
            "action": "prepend_trigger",
        })
        assert prev_res.status_code == 200
        prev_data = prev_res.json()
        assert prev_data["total_files"] == 2
        assert prev_data["changed_files"] == 2
        assert prev_data["items"][0]["proposed_caption"] == "cyborg_style, portrait of a cyborg"

        # Apply captions with backup
        apply_payload = {
            "dataset_path": str(tmp_path),
            "updates": [
                {"caption_file": item["caption_file"], "content": item["proposed_caption"]}
                for item in prev_data["items"]
            ],
            "backup_existing": True,
        }
        apply_res = await ac.post("/api/training/apply-captions", json=apply_payload)
        assert apply_res.status_code == 200
        assert apply_res.json()["applied_count"] == 2
        assert apply_res.json()["backed_up_count"] == 2

        # Verify backup files exist
        assert (tmp_path / "sample_0.txt.bak").exists()
        assert (tmp_path / "sample_0.txt").read_text(encoding="utf-8") == "cyborg_style, portrait of a cyborg"


@pytest.mark.asyncio
async def test_training_start_rejection_when_engine_unavailable(tmp_path, monkeypatch):
    # When no engine is installed, /start must cleanly report the exact unavailable message
    from app.services.training_engine import TrainingEngineService
    monkeypatch.setattr(
        TrainingEngineService,
        "detect_engine",
        lambda: (False, None, "Training engine unavailable — install kohya_ss or AI-Toolkit locally to enable training.")
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        req = {
            "goal": "character",
            "dataset_path": str(tmp_path),
            "base_model": "v1-5-pruned-emaonly.safetensors",
            "lora_name": "test_character_lora",
            "trigger_word": "my_char",
            "resolution": 1024,
            "network_dim": 16,
            "network_alpha": 16,
            "learning_rate": 0.0001,
            "batch_size": 1,
            "epochs": 10,
            "max_train_steps": 1000,
        }
        res = await ac.post("/api/training/start", json=req)
        # Should return HTTP 400 with "Training engine unavailable"
        assert res.status_code == 400
        assert "engine unavailable" in res.json()["detail"].lower()
