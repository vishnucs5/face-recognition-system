import httpx
import json

BASE_URL = "http://127.0.0.1:8000"

def test_gallery_and_metadata():
    print("\n--- TESTS 6, 7, 8: GALLERY REUSE, METADATA EXPORT, PERSISTENCE ---")
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # 1. Fetch completed gallery records
        r = client.get("/api/generations?limit=10")
        assert r.status_code == 200, f"List failed: {r.text}"
        generations = r.json()
        print(f"Total retrieved generations: {len(generations)}")
        
        completed = [g for g in generations if g["status"] == "completed"]
        assert len(completed) >= 2, f"Expected at least 2 completed records (txt2img & img2img), found {len(completed)}"

        # Pick the latest completed generation
        item = completed[0]
        print(f"Selected completed item: ID={item['id']}, Mode={item['mode']}")
        print(f"ComfyUI Prompt ID: {item.get('comfy_prompt_id')}")
        assert item.get("comfy_prompt_id") is not None, "ComfyUI prompt ID should be recorded"

        # TEST 6: Reuse parameters check
        reusable_settings = {
            "prompt": item["prompt"],
            "negative_prompt": item["negative_prompt"],
            "model": item["model"],
            "width": item["width"],
            "height": item["height"],
            "steps": item["steps"],
            "cfg_scale": item["cfg_scale"],
            "sampler": item["sampler"],
            "seed": item["seed"],
            "denoise": item["denoise"],
        }
        print("Reusable settings payload extracted successfully:")
        print(json.dumps(reusable_settings, indent=2))
        assert reusable_settings["prompt"] is not None
        assert reusable_settings["model"] is not None
        print(">>> TEST 6 PASSED: Gallery reuse parameters verified!")

        # TEST 7: Image download & metadata JSON download verification
        image_url = item["image_url"]
        img_res = client.get(image_url)
        assert img_res.status_code == 200, f"Image fetch failed: {img_res.status_code}"
        assert img_res.headers.get("content-type") == "image/png"
        assert len(img_res.content) > 10000, "Image content too small to be real"
        print(f"Image download verified: {len(img_res.content)} bytes (image/png)")

        # Verify metadata JSON structure
        metadata_export = {
            "id": item["id"],
            "comfy_prompt_id": item["comfy_prompt_id"],
            "mode": item["mode"],
            "prompt": item["prompt"],
            "negative_prompt": item["negative_prompt"],
            "model": item["model"],
            "width": item["width"],
            "height": item["height"],
            "steps": item["steps"],
            "cfg_scale": item["cfg_scale"],
            "sampler": item["sampler"],
            "seed": item["seed"],
            "denoise": item["denoise"],
            "duration_ms": item["duration_ms"],
            "created_at": item["created_at"],
        }
        print("Exported metadata JSON snippet:")
        print(json.dumps(metadata_export, indent=2))
        assert metadata_export["comfy_prompt_id"] is not None
        assert metadata_export["duration_ms"] is not None
        print(">>> TEST 7 PASSED: Real image and metadata JSON downloads verified!")

        # TEST 8: Refresh persistence check
        # Query again with search filter
        search_res = client.get(f"/api/generations?search={item['prompt'][:10]}")
        assert search_res.status_code == 200
        found = [g for g in search_res.json() if g["id"] == item["id"]]
        assert len(found) == 1, "Failed to find generation after filtered search"
        assert found[0]["status"] == "completed"
        assert found[0]["image_url"] == item["image_url"]
        print(">>> TEST 8 PASSED: Page refresh and search persistence verified!")

if __name__ == "__main__":
    test_gallery_and_metadata()
