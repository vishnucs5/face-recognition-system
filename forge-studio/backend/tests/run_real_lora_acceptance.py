import httpx
import json
import time
from pathlib import Path
from PIL import Image, ImageChops, ImageStat

BASE = "http://127.0.0.1:8000"

def run_acceptance_test():
    print("==================================================")
    print("STEP 1: Import an actual .safetensors LoRA")
    print("==================================================")
    lora_file = Path("c:/vishnu/test_cyber_neon.safetensors")
    assert lora_file.exists(), "Source LoRA file does not exist!"

    client = httpx.Client(base_url=BASE, timeout=60.0)

    # Clean up index if already present
    existing_loras = client.get("/api/loras").json()
    for l in existing_loras:
        if l["filename"] == lora_file.name:
            client.delete(f"/api/loras/{l['id']}")

    import_resp = client.post("/api/loras/import", json={
        "file_path": str(lora_file),
        "name": "Cyber Neon Glow",
        "base_model": "SD 1.5",
        "recommended_weight": 0.85,
        "trigger_words": ["cyberneon", "luminescent glow"],
        "notes": "Verified real tensor adapter for acceptance testing"
    })
    assert import_resp.status_code == 200, f"Import failed: {import_resp.text}"
    lora_record = import_resp.json()
    lora_id = lora_record["id"]
    print(f"[OK] LoRA imported successfully: {lora_record['name']} (ID: {lora_id})")
    print(f"     Triggers: {lora_record['trigger_words']}")

    # Confirm it was mirrored to ComfyUI models/loras
    comfy_lora_path = Path("c:/vishnu/ComfyUI/models/loras") / lora_file.name
    assert comfy_lora_path.exists(), "LoRA was not copied to ComfyUI models/loras directory!"
    print(f"[OK] Confirmed mirrored to ComfyUI directory: {comfy_lora_path}")

    fixed_seed = 424242
    test_prompt = "cyberneon, luminescent glow, a sleek supercar driving on a desert road at dusk"

    print("\n==================================================")
    print("STEP 2: Generate Baseline Image (Without LoRA)")
    print("==================================================")
    baseline_payload = {
        "mode": "txt2img",
        "prompt": test_prompt,
        "negative_prompt": "blurry, low quality",
        "model": "v1-5-pruned-emaonly.safetensors",
        "width": 512,
        "height": 512,
        "steps": 12,
        "cfg_scale": 6.5,
        "sampler": "euler",
        "seed": fixed_seed,
        "loras": []
    }
    r_base = client.post("/api/generations", json=baseline_payload)
    assert r_base.status_code in [200, 201], f"Baseline submission failed: {r_base.text}"
    base_id = r_base.json()["id"]
    print(f"Submitted baseline job {base_id}. Waiting for completion...")

    base_gen = None
    for _ in range(50):
        time.sleep(2)
        status_data = client.get(f"/api/generations/{base_id}").json()
        if status_data["status"] in ["completed", "failed"]:
            base_gen = status_data
            break

    assert base_gen and base_gen["status"] == "completed", f"Baseline failed: {base_gen}"
    print(f"[OK] Baseline completed in {base_gen.get('duration_ms')}ms. Image: {base_gen['image_url']}")

    print("\n==================================================")
    print("STEP 3: Attach LoRA & Generate with Trigger Words")
    print("==================================================")
    lora_payload = {
        "mode": "txt2img",
        "prompt": test_prompt,
        "negative_prompt": "blurry, low quality",
        "model": "v1-5-pruned-emaonly.safetensors",
        "width": 512,
        "height": 512,
        "steps": 12,
        "cfg_scale": 6.5,
        "sampler": "euler",
        "seed": fixed_seed,
        "loras": [{"name": lora_file.name, "strength": 0.85}]
    }
    r_lora = client.post("/api/generations", json=lora_payload)
    assert r_lora.status_code in [200, 201], f"LoRA submission failed: {r_lora.text}"
    lora_gen_id = r_lora.json()["id"]
    print(f"Submitted LoRA job {lora_gen_id}. Waiting for completion...")

    lora_gen = None
    for _ in range(50):
        time.sleep(2)
        status_data = client.get(f"/api/generations/{lora_gen_id}").json()
        if status_data["status"] in ["completed", "failed"]:
            lora_gen = status_data
            break

    assert lora_gen and lora_gen["status"] == "completed", f"LoRA generation failed: {lora_gen}"
    print(f"[OK] LoRA generation completed in {lora_gen.get('duration_ms')}ms. Image: {lora_gen['image_url']}")

    print("\n==================================================")
    print("STEP 4: Confirm LoRA Visibly Affects Output")
    print("==================================================")
    # Download both images from backend
    img_base_bytes = client.get(base_gen["image_url"]).content
    img_lora_bytes = client.get(lora_gen["image_url"]).content

    img_base_file = Path("c:/vishnu/baseline_test.png")
    img_lora_file = Path("c:/vishnu/lora_test.png")
    img_base_file.write_bytes(img_base_bytes)
    img_lora_file.write_bytes(img_lora_bytes)

    im1 = Image.open(img_base_file).convert("RGB")
    im2 = Image.open(img_lora_file).convert("RGB")
    diff = ImageChops.difference(im1, im2)
    stat = ImageStat.Stat(diff)
    mean_diff = sum(stat.mean) / len(stat.mean)
    print(f"[OK] Output Comparison:")
    print(f"     Mean channel difference: {mean_diff:.2f}")
    assert mean_diff > 0.5, "LoRA did not visibly affect the output!"
    print(f"[OK] Confirmed: LoRA visibly modified the output image.")

    print("\n==================================================")
    print("STEP 5: Confirm Metadata Records All Fields")
    print("==================================================")
    rec = lora_gen
    print(f"Record ID: {rec['id']}")
    print(f"Model checkpoint: {rec['model']}")
    print(f"Seed: {rec['seed']}")
    print(f"Prompt: {rec['prompt']}")
    print(f"LoRAs in metadata: {rec['loras']}")
    print(f"Comfy Prompt ID: {rec['comfy_prompt_id']}")

    assert rec["model"] == "v1-5-pruned-emaonly.safetensors"
    assert rec["seed"] == fixed_seed
    assert "cyberneon" in rec["prompt"]
    assert len(rec["loras"]) == 1
    assert rec["loras"][0]["name"] == lora_file.name
    assert rec["loras"][0]["strength"] == 0.85
    assert rec["comfy_prompt_id"] is not None
    print("[OK] All required metadata fields accurately recorded!")

    print("\n==================================================")
    print("STEP 6: Persistence Check (Refresh Simulation)")
    print("==================================================")
    refreshed_list = client.get("/api/generations?limit=10").json()
    found_in_list = [g for g in refreshed_list if g["id"] == lora_gen_id]
    assert len(found_in_list) == 1, "Generation not found in history after refresh query!"
    print(f"[OK] Verified persistence: Record {lora_gen_id} persists intact in SQLite history.")

    print("\n==================================================")
    print("STEP 7: Remove LoRA from Index & Verify Disk File")
    print("==================================================")
    del_resp = client.delete(f"/api/loras/{lora_id}")
    assert del_resp.status_code == 200, f"De-indexing failed: {del_resp.text}"
    print(f"[OK] De-indexed LoRA {lora_id}")

    # Verify disk files still exist!
    assert lora_file.exists(), "Source file on disk was deleted!"
    print(f"[OK] Source file still safely exists at {lora_file}")
    assert comfy_lora_path.exists(), "File in ComfyUI directory still exists"
    print(f"[OK] ComfyUI file still exists at {comfy_lora_path}")

    print("\n==================================================")
    print("END-TO-END LORA ACCEPTANCE TEST PASSED 100%")
    print("==================================================")

if __name__ == "__main__":
    run_acceptance_test()
