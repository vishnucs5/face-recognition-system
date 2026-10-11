import httpx
import time
import json
import os

BASE_URL = "http://127.0.0.1:8000"

def test_img2img():
    print("\n--- TEST 4: REAL IMAGE-TO-IMAGE GENERATION ---")
    
    # 1. First test prevention when no source image exists
    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        reject_payload = {
            "mode": "img2img",
            "prompt": "This should fail because no source image is provided",
            "model": "v1-5-pruned-emaonly.safetensors",
            "denoise": 0.6
        }
        rej_res = client.post("/api/generations", json=reject_payload)
        print("Empty source image rejection status:", rej_res.status_code)
        assert rej_res.status_code == 400, f"Expected 400 for missing source image, got {rej_res.status_code}"
        print(">>> Source image requirement enforced successfully (400 Bad Request).")

        # 2. Upload source image to backend
        source_path = r"C:\vishnu\forge-studio\data\generated\03af4b2f-20c9-49af-9679-4432de6569df.png"
        assert os.path.exists(source_path), f"Source file does not exist: {source_path}"
        
        with open(source_path, "rb") as f:
            upload_res = client.post("/api/generations/upload", files={"file": ("source_owl.png", f, "image/png")})
        assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
        uploaded_data = upload_res.json()
        print("Upload successful:", uploaded_data)
        source_image_id = uploaded_data["source_image_id"]

        # 3. Submit real img2img job
        img2img_payload = {
            "mode": "img2img",
            "prompt": "An iridescent crystal glass sculpture of an owl, prismatic refractions, soft studio lighting",
            "negative_prompt": "blurry, low quality, deformed, artifacts",
            "model": "v1-5-pruned-emaonly.safetensors",
            "source_image_id": source_image_id,
            "width": 512,
            "height": 512,
            "steps": 15,
            "cfg_scale": 7.0,
            "sampler": "euler",
            "seed": 987654,
            "denoise": 0.55
        }
        gen_res = client.post("/api/generations", json=img2img_payload)
        assert gen_res.status_code == 201, f"Job submission failed: {gen_res.text}"
        job_data = gen_res.json()
        job_id = job_data["id"]
        print(f"Submitted img2img job: {job_id}")

        # Poll until complete
        for i in range(120):
            time.sleep(2)
            poll_res = client.get(f"/api/generations/{job_id}").json()
            status = poll_res.get("status")
            print(f"Poll {i*2+2}s: status={status}, comfy_id={poll_res.get('comfy_prompt_id')}")
            if status in ["completed", "failed", "cancelled"]:
                print("Final img2img record:", json.dumps(poll_res, indent=2))
                assert status == "completed", f"Expected completed, got {status} ({poll_res.get('error_message')})"
                assert poll_res.get("image_url") is not None, "Output image_url is missing"
                assert poll_res.get("denoise") == 0.55, f"Denoise mismatch: {poll_res.get('denoise')}"
                assert poll_res.get("source_image_path") is not None, "source_image_path missing"
                print(">>> TEST 4 PASSED: Real img2img generation completed and verified!")
                return poll_res

if __name__ == "__main__":
    test_img2img()
