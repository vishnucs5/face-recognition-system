import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_gallery_search_and_mode_filtering():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create a txt2img generation with specific keyword
        txt_res = await ac.post("/api/generations", json={
            "mode": "txt2img",
            "prompt": "UniqueAstronaut on Mars in space suit",
            "model": "sd_xl_base_1.0.safetensors",
        })
        assert txt_res.status_code == 201
        txt_id = txt_res.json()["id"]

        # Search for keyword
        search_res = await ac.get("/api/generations?search=UniqueAstronaut")
        assert search_res.status_code == 200
        items = search_res.json()
        assert len(items) >= 1
        assert any(i["id"] == txt_id for i in items)

        # Search for non-existent keyword
        empty_search = await ac.get("/api/generations?search=NonExistentKeywordXYZ999")
        assert empty_search.status_code == 200
        assert len(empty_search.json()) == 0

        # Filter by mode
        mode_res = await ac.get("/api/generations?mode=txt2img")
        assert mode_res.status_code == 200
        assert all(i["mode"] == "txt2img" for i in mode_res.json())


@pytest.mark.asyncio
async def test_gallery_delete_record():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create a generation
        res = await ac.post("/api/generations", json={
            "mode": "txt2img",
            "prompt": "Generation to be deleted",
            "model": "sd_xl_base_1.0.safetensors",
        })
        assert res.status_code == 201
        job_id = res.json()["id"]

        # Delete it
        del_res = await ac.delete(f"/api/generations/{job_id}")
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "deleted"

        # Verify it no longer exists
        get_res = await ac.get(f"/api/generations/{job_id}")
        assert get_res.status_code == 404
