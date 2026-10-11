import httpx
import json
from pathlib import Path

BASE = "http://127.0.0.1:8000"

def test_lora_vault_live():
    print("\n--- 1. Testing LoRA Vault Live Endpoints ---")
    with httpx.Client(base_url=BASE, timeout=10.0) as client:
        resp = client.get("/api/loras")
        assert resp.status_code == 200, f"Failed listing loras: {resp.text}"
        loras = resp.json()
        print(f"[OK] Found {len(loras)} indexed LoRAs")
        for lora in loras:
            print(f"  - {lora['name']} ({lora['filename']}) | weight: {lora['recommended_weight']} | is_available: {lora['is_available']}")

def test_training_health_live():
    print("\n--- 2. Testing Training Health Live Endpoint ---")
    with httpx.Client(base_url=BASE, timeout=10.0) as client:
        resp = client.get("/api/training/health")
        assert resp.status_code == 200, f"Failed health check: {resp.text}"
        health = resp.json()
        print(f"[OK] Health status: engine_available={health['engine_available']}, engine_name={health['engine_name']}")
        print(f"     GPU VRAM: {health['gpu_vram_gb']} GB")
        print(f"     Engine message: {health['message']}")
        rtx_preset = health['recommended_preset']
        print(f"     RTX 5070 Ti Safe Preset: {rtx_preset}")
        assert rtx_preset['resolution'] == 1024
        assert rtx_preset['batch_size'] == 1
        assert rtx_preset['gradient_checkpointing'] is True
        assert rtx_preset['estimated_vram_gb'] == 12.5

def test_dataset_validation_live():
    print("\n--- 3. Testing Dataset Validation on c:/vishnu/test_dataset ---")
    with httpx.Client(base_url=BASE, timeout=10.0) as client:
        payload = {"dataset_path": "c:/vishnu/test_dataset"}
        resp = client.post("/api/training/validate-dataset", json=payload)
        assert resp.status_code == 200, f"Validation failed: {resp.text}"
        val = resp.json()
        print(f"[OK] Dataset Validation Response:")
        print(f"     Total images: {val['total_images']}")
        print(f"     Missing captions count: {val['missing_captions_count']}")
        print(f"     Corrupt images count: {val['corrupt_images_count']}")
        print(f"     Duplicates count: {val['duplicates_count']}")
        print(f"     Resolution range: {val['resolution_range']}")
        print(f"     Suggested crop/res: {val['suggested_resolution']}")
        print(f"     Issues reported: {len(val['issues'])}")
        assert val['total_images'] == 4
        assert val['missing_captions_count'] == 1
        assert val['corrupt_images_count'] == 0
        assert val['duplicates_count'] == 0

def test_caption_preview_and_apply_live():
    print("\n--- 4. Testing Caption Preview & Apply with .txt.bak Backup ---")
    with httpx.Client(base_url=BASE, timeout=10.0) as client:
        # Preview
        prev_req = {
            "dataset_path": "c:/vishnu/test_dataset",
            "action": "prepend_trigger",
            "trigger_word": "cyber_hero"
        }
        resp = client.post("/api/training/preview-captions", json=prev_req)
        assert resp.status_code == 200, f"Preview failed: {resp.text}"
        prev_data = resp.json()
        print(f"[OK] Previewed {len(prev_data['items'])} captions. Example:")
        sample = prev_data['items'][0]
        print(f"     File: {sample['image_file']}")
        print(f"     Original: {sample['existing_caption']}")
        print(f"     Proposed: {sample['proposed_caption']}")
        assert sample['proposed_caption'].startswith("cyber_hero, ")

        # Apply
        updates = [
            {"caption_file": item["caption_file"], "content": item["proposed_caption"]}
            for item in prev_data["items"]
        ]
        apply_req = {
            "dataset_path": "c:/vishnu/test_dataset",
            "updates": updates,
            "backup_existing": True
        }
        resp = client.post("/api/training/apply-captions", json=apply_req)
        assert resp.status_code == 200, f"Apply failed: {resp.text}"
        app_data = resp.json()
        print(f"[OK] Applied captions: modified {app_data['applied_count']} files, backups created: {app_data['backed_up_count']}")
        assert app_data['applied_count'] == 4
        assert app_data['backed_up_count'] >= 3

    # Verify backup on disk
    bak_files = list(Path("c:/vishnu/test_dataset").glob("*.bak"))
    print(f"[OK] Verified {len(bak_files)} .bak files created on filesystem:")
    for b in bak_files:
        print(f"     - {b.name}")
    assert len(bak_files) >= 3

def test_training_start_rejection_live():
    print("\n--- 5. Testing Real Training Engine Requirement Guard ---")
    with httpx.Client(base_url=BASE, timeout=10.0) as client:
        start_req = {
            "goal": "character",
            "dataset_path": "c:/vishnu/test_dataset",
            "base_model": "v1-5-pruned-emaonly.safetensors",
            "lora_name": "cyber_hero_lora",
            "trigger_word": "cyber_hero",
            "resolution": 1024,
            "batch_size": 1,
            "epochs": 10,
            "learning_rate": 0.0001,
            "network_dim": 16,
            "network_alpha": 16,
            "mixed_precision": "bf16",
            "gradient_checkpointing": True
        }
        resp = client.post("/api/training/start", json=start_req)
        print(f"[OK] Response status: {resp.status_code}")
        print(f"     Response body: {resp.text}")
        assert resp.status_code == 400
        detail = resp.json().get("detail", "")
        assert "Training engine unavailable — install kohya_ss or AI-Toolkit locally to enable training." in detail
        print(f"[OK] Verified authentic engine check. Fake training rejected!")

def main():
    test_lora_vault_live()
    test_training_health_live()
    test_dataset_validation_live()
    test_caption_preview_and_apply_live()
    test_training_start_rejection_live()
    print("\n==========================================")
    print("ALL LIVE LORA & TRAINING TESTS PASSED 100%")
    print("==========================================")

if __name__ == "__main__":
    main()
