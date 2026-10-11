import struct
import json
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_list_loras_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/api/loras")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_import_lora_invalid_extension(tmp_path):
    fake_file = tmp_path / "test.txt"
    fake_file.write_text("not a lora")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/loras/import", json={"file_path": str(fake_file)})
        assert res.status_code == 400
        assert "safetensors" in res.json()["detail"].lower()


@pytest.mark.asyncio
async def test_import_lora_file_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/loras/import", json={"file_path": "c:/non_existent_lora.safetensors"})
        assert res.status_code == 404


@pytest.mark.asyncio
async def test_import_and_update_lora_lifecycle(tmp_path):
    # Construct a valid minimal safetensors file
    dummy_safetensors = tmp_path / "cyber_glitch_test.safetensors"
    meta = {"ss_base_model_version": "SDXL 1.0", "ss_tag_frequency": "cyberglitch, neon"}
    header_dict = {"__metadata__": meta}
    header_json = json.dumps(header_dict).encode("utf-8")
    header_len = len(header_json)
    
    with open(dummy_safetensors, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        f.write(b"\x00" * 1024)  # 1KB dummy tensor data

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Import
        import_payload = {
            "file_path": str(dummy_safetensors),
            "name": "Cyber Glitch Aesthetic",
            "trigger_words": ["cyberglitch", "neon"],
            "base_model": "SDXL 1.0",
            "recommended_weight": 0.85,
            "notes": "Testing vault import",
        }
        res = await ac.post("/api/loras/import", json=import_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "Cyber Glitch Aesthetic"
        assert data["filename"] == "cyber_glitch_test.safetensors"
        assert data["recommended_weight"] == 0.85
        lora_id = data["id"]

        # 2. Get by ID
        get_res = await ac.get(f"/api/loras/{lora_id}")
        assert get_res.status_code == 200
        assert get_res.json()["id"] == lora_id

        # 3. Patch / Update
        patch_payload = {
            "name": "Cyber Glitch Concept V2",
            "recommended_weight": 0.7,
            "notes": "Updated notes",
        }
        patch_res = await ac.patch(f"/api/loras/{lora_id}", json=patch_payload)
        assert patch_res.status_code == 200
        assert patch_res.json()["name"] == "Cyber Glitch Concept V2"
        assert patch_res.json()["recommended_weight"] == 0.7

        # 4. Delete from index
        del_res = await ac.delete(f"/api/loras/{lora_id}")
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "removed"

        # Verify physical file still exists on disk!
        assert dummy_safetensors.exists()
