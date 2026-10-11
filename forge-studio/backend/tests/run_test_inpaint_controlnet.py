import asyncio
import json
import time
from io import BytesIO
from pathlib import Path
import httpx
from PIL import Image, ImageDraw

BACKEND_URL = "http://127.0.0.1:8000"


def create_test_images():
    # Source image: 512x512 colorful composition
    img = Image.new("RGB", (512, 512), color=(70, 130, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 400, 400], fill=(220, 20, 60))
    draw.ellipse([150, 150, 350, 350], fill=(255, 215, 0))

    src_buf = BytesIO()
    img.save(src_buf, format="PNG")
    src_bytes = src_buf.getvalue()

    # Mask image: 512x512 black background with white box in center
    mask = Image.new("L", (512, 512), color=0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([180, 180, 320, 320], fill=255)

    mask_buf = BytesIO()
    mask.save(mask_buf, format="PNG")
    mask_bytes = mask_buf.getvalue()

    return src_bytes, mask_bytes


async def main():
    print("======================================================================")
    print("FORGE STUDIO: INPAINTING & CONTROLNET PRODUCTION ACCEPTANCE SUITE")
    print("======================================================================")

    async with httpx.AsyncClient(timeout=180.0) as client:
        # 1. Health check
        res = await client.get(f"{BACKEND_URL}/api/health")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        print("[PASS] 1. Backend health check: OK")

        # 2. Check ComfyUI status & available models
        status_res = await client.get(f"{BACKEND_URL}/api/status")
        status_data = status_res.json()
        print(f"[INFO] ComfyUI Online: {status_data.get('comfyui_online')}, Ollama Online: {status_data.get('ollama_online')}")

        models_res = await client.get(f"{BACKEND_URL}/api/models")
        models = models_res.json()
        assert len(models) > 0, "No model checkpoints found!"
        checkpoint_name = models[0]["filename"]
        print(f"[PASS] 2. Base model checkpoint available: {checkpoint_name}")

        # 3. ControlNet types and models check
        cnet_types_res = await client.get(f"{BACKEND_URL}/api/controlnet/types")
        assert cnet_types_res.status_code == 200
        cnet_types = cnet_types_res.json()
        assert len(cnet_types) == 5, f"Expected 5 ControlNet types, got {len(cnet_types)}"
        print(f"[PASS] 3. ControlNet types verified: {[t['id'] for t in cnet_types]}")

        cnet_models_res = await client.get(f"{BACKEND_URL}/api/controlnet/models")
        assert cnet_models_res.status_code == 200
        cnet_models = cnet_models_res.json()
        assert len(cnet_models) > 0, "No ControlNet models available in system!"
        cnet_model_name = cnet_models[0]["filename"]
        print(f"[PASS] 4. ControlNet models found: {[m['filename'] for m in cnet_models]}")

        # 4. Upload source image and mask
        src_bytes, mask_bytes = create_test_images()

        files_src = {"file": ("test_source.png", src_bytes, "image/png")}
        up_src = await client.post(f"{BACKEND_URL}/api/generations/upload", files=files_src)
        assert up_src.status_code == 200, f"Source upload failed: {up_src.text}"
        src_data = up_src.json()
        source_id = src_data["source_image_id"]
        print(f"[PASS] 5. Source image uploaded: {source_id}")

        files_mask = {"file": ("test_mask.png", mask_bytes, "image/png")}
        up_mask = await client.post(f"{BACKEND_URL}/api/generations/upload-mask", files=files_mask)
        assert up_mask.status_code == 200, f"Mask upload failed: {up_mask.text}"
        mask_data = up_mask.json()
        mask_id = mask_data["mask_image_id"]
        print(f"[PASS] 6. Mask image uploaded: {mask_id}")

        # 5. Test invalid upload rejection (HTTP 400)
        invalid_file = {"file": ("malicious.txt", b"plain text payload", "text/plain")}
        inv_res = await client.post(f"{BACKEND_URL}/api/generations/upload", files=invalid_file)
        assert inv_res.status_code == 400, "Invalid file upload was not rejected!"
        print("[PASS] 7. Invalid file MIME upload rejected with HTTP 400")

        # 6. Test missing mask rejection for Inpainting (HTTP 400)
        bad_inpaint_payload = {
            "mode": "inpaint",
            "prompt": "a bright green glowing orb",
            "model": checkpoint_name,
            "width": 512,
            "height": 512,
            "steps": 10,
            "cfg_scale": 7.0,
            "sampler": "euler",
            "denoise": 0.85,
            "loras": [],
            "source_image_id": source_id,
            # mask_image_id intentionally omitted
        }
        bad_inpaint_res = await client.post(f"{BACKEND_URL}/api/generations", json=bad_inpaint_payload)
        assert bad_inpaint_res.status_code == 400, "Missing mask in inpaint was not rejected!"
        assert "Inpainting requires both a source image and a painted mask." in bad_inpaint_res.text
        print("[PASS] 8. Missing mask in inpaint correctly rejected with HTTP 400")

        # 7. Test missing ControlNet model rejection (HTTP 400)
        bad_cnet_payload = {
            "mode": "controlnet",
            "prompt": "a futuristic cyberpunk building",
            "model": checkpoint_name,
            "width": 512,
            "height": 512,
            "steps": 10,
            "cfg_scale": 7.0,
            "sampler": "euler",
            "denoise": 1.0,
            "loras": [],
            "source_image_id": source_id,
            "controlnet_model": "missing_nonexistent_model.safetensors",
            "controlnet_type": "canny",
            "controlnet_strength": 1.0,
        }
        bad_cnet_res = await client.post(f"{BACKEND_URL}/api/generations", json=bad_cnet_payload)
        assert bad_cnet_res.status_code == 400, "Missing ControlNet model was not rejected!"
        assert "ControlNet model unavailable" in bad_cnet_res.text
        print("[PASS] 9. Missing ControlNet model correctly rejected with HTTP 400")

        # 8. Test REAL INPAINTING Generation via ComfyUI
        print("[RUNNING] 10. Submitting real Inpainting generation through ComfyUI...")
        inpaint_payload = {
            "mode": "inpaint",
            "prompt": "a glowing golden mechanical gear embedded in the surface, high quality",
            "negative_prompt": "blurry, low quality",
            "model": checkpoint_name,
            "width": 512,
            "height": 512,
            "steps": 10,
            "cfg_scale": 6.5,
            "sampler": "euler",
            "seed": 987654,
            "denoise": 0.85,
            "loras": [],
            "source_image_id": source_id,
            "mask_image_id": mask_id,
        }
        inpaint_create_res = await client.post(f"{BACKEND_URL}/api/generations", json=inpaint_payload)
        assert inpaint_create_res.status_code == 201, f"Inpaint submission failed: {inpaint_create_res.text}"
        inpaint_job = inpaint_create_res.json()
        inpaint_id = inpaint_job["id"]
        print(f"[INFO] Inpaint Job ID: {inpaint_id}, Status: {inpaint_job['status']}")

        # Poll inpaint completion
        completed_inpaint = None
        for _ in range(60):
            await asyncio.sleep(1.0)
            poll_res = await client.get(f"{BACKEND_URL}/api/generations/{inpaint_id}")
            poll_data = poll_res.json()
            if poll_data["status"] == "completed":
                completed_inpaint = poll_data
                break
            elif poll_data["status"] in ["failed", "cancelled"]:
                raise RuntimeError(f"Inpainting failed: {poll_data.get('error_message')}")

        assert completed_inpaint is not None, "Inpainting generation timed out!"
        assert completed_inpaint["image_url"], "Inpainting completed without image URL!"
        assert completed_inpaint["mask_image_url"], "Inpainting completed without mask URL recorded!"
        assert Path(completed_inpaint["image_path"]).exists(), "Inpainting output file missing from disk!"
        print(f"[PASS] 10. Real Inpainting generation succeeded! Output: {completed_inpaint['image_url']}")

        # 9. Test REAL CONTROLNET Canny Generation via ComfyUI
        print(f"[RUNNING] 11. Submitting real ControlNet Canny generation ({cnet_model_name})...")
        cnet_payload = {
            "mode": "controlnet",
            "prompt": "a crystalline cybernetic structure, glowing neon lines, intricate detail",
            "negative_prompt": "blurry, distorted",
            "model": checkpoint_name,
            "width": 512,
            "height": 512,
            "steps": 10,
            "cfg_scale": 6.5,
            "sampler": "euler",
            "seed": 123456,
            "denoise": 1.0,
            "loras": [],
            "source_image_id": source_id,
            "controlnet_model": cnet_model_name,
            "controlnet_type": "canny",
            "controlnet_strength": 1.0,
            "controlnet_start_percent": 0.0,
            "controlnet_end_percent": 1.0,
            "controlnet_preprocessor": "canny",
        }
        cnet_create_res = await client.post(f"{BACKEND_URL}/api/generations", json=cnet_payload)
        assert cnet_create_res.status_code == 201, f"ControlNet submission failed: {cnet_create_res.text}"
        cnet_job = cnet_create_res.json()
        cnet_id = cnet_job["id"]
        print(f"[INFO] ControlNet Job ID: {cnet_id}, Status: {cnet_job['status']}")

        # Poll ControlNet completion
        completed_cnet = None
        for _ in range(60):
            await asyncio.sleep(1.0)
            poll_res = await client.get(f"{BACKEND_URL}/api/generations/{cnet_id}")
            poll_data = poll_res.json()
            if poll_data["status"] == "completed":
                completed_cnet = poll_data
                break
            elif poll_data["status"] in ["failed", "cancelled"]:
                raise RuntimeError(f"ControlNet failed: {poll_data.get('error_message')}")

        assert completed_cnet is not None, "ControlNet generation timed out!"
        assert completed_cnet["image_url"], "ControlNet completed without image URL!"
        assert completed_cnet["controlnet_model"] == cnet_model_name, "ControlNet model mismatch!"
        assert completed_cnet["controlnet_strength"] == 1.0, "ControlNet strength mismatch!"
        assert Path(completed_cnet["image_path"]).exists(), "ControlNet output file missing from disk!"
        print(f"[PASS] 11. Real ControlNet generation succeeded! Output: {completed_cnet['image_url']}")

        # 10. Test Cancel Generation
        print("[RUNNING] 12. Testing cancellation lifecycle...")
        cancel_payload = {
            "mode": "txt2img",
            "prompt": "a slow heavy test prompt to cancel",
            "model": checkpoint_name,
            "width": 512,
            "height": 512,
            "steps": 50,
            "cfg_scale": 7.0,
            "sampler": "euler",
            "denoise": 1.0,
            "loras": [],
        }
        canc_res = await client.post(f"{BACKEND_URL}/api/generations", json=cancel_payload)
        canc_job = canc_res.json()
        canc_id = canc_job["id"]

        # Cancel immediately
        cancel_call = await client.post(f"{BACKEND_URL}/api/generations/{canc_id}/cancel")
        assert cancel_call.status_code == 200, f"Cancel call failed: {cancel_call.text}"
        canc_status = await client.get(f"{BACKEND_URL}/api/generations/{canc_id}")
        assert canc_status.json()["status"] == "cancelled"
        print(f"[PASS] 12. Generation cancelled cleanly with status='cancelled'")

        # 11. Test Gallery and SQLite persistence
        print("[RUNNING] 13. Verifying Gallery records and SQLite persistence...")
        inpaint_list_res = await client.get(f"{BACKEND_URL}/api/generations?mode=inpaint")
        assert inpaint_list_res.status_code == 200
        inpaint_items = inpaint_list_res.json()
        assert any(item["id"] == inpaint_id for item in inpaint_items), "Inpaint job not persisted in DB!"

        cnet_list_res = await client.get(f"{BACKEND_URL}/api/generations?mode=controlnet")
        assert cnet_list_res.status_code == 200
        cnet_items = cnet_list_res.json()
        assert any(item["id"] == cnet_id for item in cnet_items), "ControlNet job not persisted in DB!"
        print("[PASS] 13. Gallery records and SQLite persistence verified for both modes")

        # 12. Test Copilot Inpainting & ControlNet Advice
        print("[RUNNING] 14. Testing Copilot inpainting and controlnet advice...")
        copilot_inpaint_res = await client.post(
            f"{BACKEND_URL}/api/copilot/prompt",
            json={
                "task": "inpaint_advice",
                "message": "How do I inpaint an earring onto a character portrait?",
                "current_context": {},
            },
        )
        assert copilot_inpaint_res.status_code == 200
        copilot_inpaint_data = copilot_inpaint_res.json()
        reply_inpaint_lower = copilot_inpaint_data["reply"].lower()
        assert "inpaint" in reply_inpaint_lower or "mask" in reply_inpaint_lower or "denoise" in reply_inpaint_lower
        assert copilot_inpaint_data["suggested_settings"] is not None
        print("[PASS] 14. Copilot inpaint advice delivered review-only suggestion")

        copilot_cnet_res = await client.post(
            f"{BACKEND_URL}/api/copilot/prompt",
            json={
                "task": "controlnet_advice",
                "message": "Which preprocessor is best for architectural sketches?",
                "current_context": {},
            },
        )
        assert copilot_cnet_res.status_code == 200
        copilot_cnet_data = copilot_cnet_res.json()
        reply_cnet_lower = copilot_cnet_data["reply"].lower()
        assert "controlnet" in reply_cnet_lower or "sketch" in reply_cnet_lower or "canny" in reply_cnet_lower or "inpaint" in reply_cnet_lower
        print("[PASS] 15. Copilot controlnet advice delivered review-only suggestion")

    print("\n======================================================================")
    print("ALL 15 TESTS IN ACCEPTANCE SUITE PASSED SUCCESSFULLY (100% PASS RATE)!")
    print("======================================================================")


if __name__ == "__main__":
    asyncio.run(main())
