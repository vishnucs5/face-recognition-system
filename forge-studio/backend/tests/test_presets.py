import json
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.services.preset_engine import preset_engine
from app.schemas.preset import PresetModel, PresetCreateRequest


@pytest.mark.asyncio
async def test_builtin_presets_count_and_keys():
    """Verify exactly 7 factory built-in presets with immutable specs."""
    builtins = preset_engine.get_built_in_presets()
    assert len(builtins) == 7

    expected_ids = {
        "builtin-portrait",
        "builtin-cinematic-landscape",
        "builtin-product-photography",
        "builtin-fast-draft",
        "builtin-inpainting-repair",
        "builtin-controlnet-composition",
        "builtin-lora-style-test",
    }
    found_ids = {b.id for b in builtins}
    assert found_ids == expected_ids

    # Verify specs of each
    portrait = next(b for b in builtins if b.id == "builtin-portrait")
    assert portrait.width == 1024
    assert portrait.height == 1024
    assert portrait.steps == 28
    assert portrait.cfg_scale == 6.5
    assert portrait.sampler == "euler"
    assert portrait.vram_profile == "balanced"
    assert portrait.is_built_in is True
    assert portrait.category == "built-in"

    landscape = next(b for b in builtins if b.id == "builtin-cinematic-landscape")
    assert landscape.width == 1344
    assert landscape.height == 768
    assert landscape.steps == 30
    assert landscape.sampler in ["dpmpp_2m", "dpm++ 2m"]

    fast_draft = next(b for b in builtins if b.id == "builtin-fast-draft")
    assert fast_draft.width == 768
    assert fast_draft.height == 768
    assert fast_draft.steps == 16
    assert fast_draft.vram_profile == "speed"

    inpaint = next(b for b in builtins if b.id == "builtin-inpainting-repair")
    assert inpaint.denoise == 0.55
    assert inpaint.controlnet_settings.get("mask_padding") == 32

    cnet = next(b for b in builtins if b.id == "builtin-controlnet-composition")
    assert cnet.controlnet_settings.get("controlnet_strength") == 0.85

    lora_test = next(b for b in builtins if b.id == "builtin-lora-style-test")
    assert len(lora_test.loras) == 1
    assert lora_test.loras[0].strength == 0.75


@pytest.mark.asyncio
async def test_get_presets_api():
    """Verify GET /api/presets returns built-in presets."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/presets")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 7
    ids = [p["id"] for p in data]
    assert "builtin-portrait" in ids


@pytest.mark.asyncio
async def test_user_preset_crud_lifecycle():
    """Verify create, get, update, duplicate, and delete for user presets."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Create
        create_payload = {
            "name": "Cyberpunk Neon Cityscape",
            "description": "High contrast neon cityscape for nocturnal renders",
            "mode": "txt2img",
            "base_model_family": "sdxl",
            "checkpoint_name": "sd_xl_base_1.0.safetensors",
            "prompt_template": "Futuristic cyberpunk street glowing with neon signs, wet tarmac reflections, 8k",
            "negative_prompt": "blurry, low resolution, dim",
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "cfg_scale": 7.0,
            "sampler": "euler",
            "scheduler": "karras",
            "seed_policy": "random",
            "denoise": 0.8,
            "loras": [{"name": "neon_glow.safetensors", "strength": 0.7}],
            "controlnet_settings": {},
            "vram_profile": "balanced",
        }
        res_create = await ac.post("/api/presets", json=create_payload)
        assert res_create.status_code == 201
        created = res_create.json()
        preset_id = created["id"]
        assert created["name"] == create_payload["name"]
        assert created["version"] == 1
        assert created["is_built_in"] is False
        assert created["category"] == "user"

        # 2. Get with dependency inspection
        res_get = await ac.get(f"/api/presets/{preset_id}")
        assert res_get.status_code == 200
        preset_data = res_get.json()
        assert "dependency_status" in preset_data
        assert isinstance(preset_data["dependency_status"], dict)

        # 3. Update
        update_payload = {
            "name": "Cyberpunk Neon Cityscape v2",
            "cfg_scale": 8.0,
            "steps": 35,
        }
        res_update = await ac.put(f"/api/presets/{preset_id}", json=update_payload)
        assert res_update.status_code == 200
        updated = res_update.json()
        assert updated["name"] == "Cyberpunk Neon Cityscape v2"
        assert updated["cfg_scale"] == 8.0
        assert updated["steps"] == 35
        assert updated["version"] == 2

        # 4. Duplicate
        res_dup = await ac.post(f"/api/presets/{preset_id}/duplicate")
        assert res_dup.status_code == 200
        duplicated = res_dup.json()
        assert "(Copy)" in duplicated["name"]
        assert duplicated["id"] != preset_id
        assert duplicated["version"] == 1

        # 5. Delete
        res_del = await ac.delete(f"/api/presets/{preset_id}")
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "deleted"

        # Verify not found after delete
        res_after = await ac.get(f"/api/presets/{preset_id}")
        assert res_after.status_code == 404

        # Clean up duplicate
        await ac.delete(f"/api/presets/{duplicated['id']}")


@pytest.mark.asyncio
async def test_builtin_immutability():
    """Verify modifying or deleting built-in presets is strictly rejected."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Try updating built-in preset
        res_put = await ac.put("/api/presets/builtin-portrait", json={"name": "Hacked Portrait"})
        assert res_put.status_code == 400
        assert "immutable" in res_put.json()["detail"].lower()

        # Try deleting built-in preset
        res_del = await ac.delete("/api/presets/builtin-portrait")
        assert res_del.status_code == 400
        assert "immutable" in res_del.json()["detail"].lower()

        # But duplicating a built-in preset should succeed!
        res_dup = await ac.post("/api/presets/builtin-portrait/duplicate")
        assert res_dup.status_code == 200
        dup_data = res_dup.json()
        assert dup_data["name"] == "Portrait (Copy)"
        assert dup_data["is_built_in"] is False
        assert dup_data["category"] == "user"

        # Clean up
        await ac.delete(f"/api/presets/{dup_data['id']}")


@pytest.mark.asyncio
async def test_presets_diff_calculation():
    """Verify side-by-side diff between preset and current form state."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        diff_payload = {
            "preset_id": "builtin-portrait",
            "current_form": {
                "prompt": "Custom portrait of a pilot",
                "width": 768,
                "height": 768,
                "steps": 40,
                "cfg_scale": 8.0,
                "sampler": "euler",
                "model": "v1-5-pruned-emaonly.safetensors",
            },
        }
        res = await ac.post("/api/presets/diff", json=diff_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["preset_id"] == "builtin-portrait"
        assert "prompt" in data["differences"]
        assert "width" in data["differences"]
        assert data["differences"]["width"]["preset"] == 1024
        assert data["differences"]["width"]["current"] == 768
        assert "steps" in data["differences"]
        assert data["differences"]["steps"]["preset"] == 28
        assert data["differences"]["steps"]["current"] == 40
        assert "estimated_vram_diff_mb" in data
        assert "summary" in data


@pytest.mark.asyncio
async def test_export_and_import_roundtrip():
    """Verify exporting user presets to JSON and importing them back."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create user preset
        create_payload = {
            "name": "Export Test Preset",
            "description": "Preset to be exported",
            "mode": "txt2img",
            "base_model_family": "sd15",
            "checkpoint_name": "v1-5-pruned-emaonly.safetensors",
            "prompt_template": "An antique pocket watch resting on velvet",
            "negative_prompt": "blurry, dark",
            "width": 1024,
            "height": 1024,
            "steps": 25,
            "cfg_scale": 6.5,
            "sampler": "euler",
            "scheduler": "karras",
            "seed_policy": "random",
            "denoise": 0.75,
            "loras": [],
            "controlnet_settings": {},
            "vram_profile": "balanced",
        }
        res_create = await ac.post("/api/presets", json=create_payload)
        created_id = res_create.json()["id"]

        # Export
        res_export = await ac.get("/api/presets/export")
        assert res_export.status_code == 200
        export_text = res_export.text
        export_data = json.loads(export_text)
        assert export_data["schema_version"] == 1
        assert "exported_at" in export_data
        assert isinstance(export_data["presets"], list)
        assert any(p["name"] == "Export Test Preset" for p in export_data["presets"])

        # Import
        res_import = await ac.post("/api/presets/import", json=export_data)
        assert res_import.status_code == 200
        import_res = res_import.json()
        assert import_res["imported_count"] >= 1
        assert len(import_res["errors"]) == 0

        # Clean up original
        await ac.delete(f"/api/presets/{created_id}")


@pytest.mark.asyncio
async def test_import_strict_sanitization_rejections():
    """Verify import rejects script tags, URLs, shell characters, traversal, and node graph JSON."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Invalid schema_version
        bad_version_payload = {
            "schema_version": 99,
            "presets": [{"name": "Test", "prompt_template": "Hello"}],
        }
        res_bv = await ac.post("/api/presets/import", json=bad_version_payload)
        assert res_bv.json()["imported_count"] == 0
        assert any("schema_version" in err.lower() for err in res_bv.json()["errors"])

        # 2. XSS script tag injection
        script_payload = {
            "schema_version": 1,
            "presets": [
                {
                    "name": "XSS Preset <script>alert(1)</script>",
                    "prompt_template": "Harmful <script>fetch('evil')</script>",
                }
            ],
        }
        res_script = await ac.post("/api/presets/import", json=script_payload)
        assert res_script.json()["imported_count"] == 0
        assert any("script tags" in err.lower() for err in res_script.json()["errors"])

        # 3. URL injection
        url_payload = {
            "schema_version": 1,
            "presets": [
                {
                    "name": "URL Preset",
                    "prompt_template": "Download from http://malicious.org/payload.safetensors",
                }
            ],
        }
        res_url = await ac.post("/api/presets/import", json=url_payload)
        assert res_url.json()["imported_count"] == 0
        assert any("urls" in err.lower() for err in res_url.json()["errors"])

        # 4. Shell metacharacters injection (; & | ` $)
        shell_payload = {
            "schema_version": 1,
            "presets": [
                {
                    "name": "Shell Injection",
                    "prompt_template": "photo; rm -rf /; echo done",
                }
            ],
        }
        res_shell = await ac.post("/api/presets/import", json=shell_payload)
        assert res_shell.json()["imported_count"] == 0
        assert any("shell metacharacters" in err.lower() for err in res_shell.json()["errors"])

        # 5. Directory traversal
        traversal_payload = {
            "schema_version": 1,
            "presets": [
                {
                    "name": "Traversal Preset",
                    "checkpoint_name": "../../../etc/shadow",
                    "prompt_template": "a photo",
                }
            ],
        }
        res_trav = await ac.post("/api/presets/import", json=traversal_payload)
        assert res_trav.json()["imported_count"] == 0
        assert any("directory traversal" in err.lower() for err in res_trav.json()["errors"])

        # 6. Arbitrary node graph JSON structure
        graph_payload = {
            "schema_version": 1,
            "presets": [
                {
                    "name": "Graph JSON Injection",
                    "prompt_template": "{\"nodes\": [{\"id\": 1}], \"links\": [[1, 2]]}",
                }
            ],
        }
        res_graph = await ac.post("/api/presets/import", json=graph_payload)
        assert res_graph.json()["imported_count"] == 0
        assert any("node graph json" in err.lower() for err in res_graph.json()["errors"])


@pytest.mark.asyncio
async def test_generation_snapshot_and_copy_as_preset_and_reproduce():
    """Verify generation snapshots are captured, and copy-as-preset & reproduce work."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create a generation
        gen_payload = {
            "mode": "txt2img",
            "prompt": "Gourmet pastry on marble tabletop, golden light",
            "negative_prompt": "blurry, unappetizing",
            "model": "sd_xl_base_1.0.safetensors",
            "width": 1024,
            "height": 1024,
            "steps": 28,
            "cfg_scale": 6.5,
            "sampler": "euler",
            "seed": 998877,
            "denoise": 1.0,
            "preset_name": "Product photography",
            "preset_version": 1,
        }
        res_gen = await ac.post("/api/generations", json=gen_payload)
        assert res_gen.status_code == 201
        gen_data = res_gen.json()
        gen_id = gen_data["id"]

        # Check snapshot fields on GenerationResponse
        assert gen_data["preset_name"] == "Product photography"
        assert gen_data["preset_version"] == 1
        assert gen_data["settings_snapshot_json"] is not None
        assert gen_data["workflow_graph_hash"] is not None
        assert gen_data["gpu_telemetry_snapshot_json"] is not None
        assert gen_data["asset_hashes_json"] is not None

        # Verify reproduce endpoint
        res_repr = await ac.get(f"/api/generations/{gen_id}/reproduce")
        assert res_repr.status_code == 200
        repr_data = res_repr.json()
        assert repr_data["generation_id"] == gen_id
        assert repr_data["seed"] == 998877
        assert repr_data["config"]["prompt"] == gen_payload["prompt"]
        assert repr_data["workflow_graph_hash"] == gen_data["workflow_graph_hash"]
        assert "dependencies" in repr_data
        assert "can_reproduce" in repr_data

        # Verify copy-as-preset endpoint
        res_copy = await ac.post(f"/api/generations/{gen_id}/copy-as-preset")
        assert res_copy.status_code == 200
        copy_data = res_copy.json()
        preset_id = copy_data["id"]
        assert "Preset from Gourmet pastry" in copy_data["name"]
        assert copy_data["steps"] == 28
        assert copy_data["cfg_scale"] == 6.5
        assert copy_data["category"] == "user"
        assert copy_data["is_built_in"] is False

        # Clean up created preset
        await ac.delete(f"/api/presets/{preset_id}")
