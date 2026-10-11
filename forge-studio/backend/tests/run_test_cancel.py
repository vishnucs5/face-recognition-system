import httpx
import time
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_cancel_generation():
    print("\n--- TEST 2: CANCEL ACTIVE GENERATION ---")
    payload = {
        "mode": "txt2img",
        "prompt": "A test prompt for cancellation verification",
        "model": "v1-5-pruned-emaonly.safetensors",
        "width": 512,
        "height": 512,
        "steps": 25,
        "cfg_scale": 7.0,
        "sampler": "euler",
    }
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        r = client.post("/api/generations", json=payload)
        assert r.status_code == 201, f"Failed to submit: {r.text}"
        job_id = r.json()["id"]
        print(f"Submitted job: {job_id}")

        # Wait until it starts running and receives ComfyUI prompt_id
        for _ in range(20):
            time.sleep(0.5)
            status_r = client.get(f"/api/generations/{job_id}").json()
            if status_r.get("status") == "running":
                print(f"Job running with comfy_prompt_id: {status_r.get('comfy_prompt_id')}")
                break

        # Cancel the specific job
        cancel_r = client.post(f"/api/generations/{job_id}/cancel")
        assert cancel_r.status_code == 200, f"Cancel failed: {cancel_r.text}"
        print("Cancel endpoint response:", cancel_r.json())

        # Give it a moment to settle
        time.sleep(1.0)
        final_job = client.get(f"/api/generations/{job_id}").json()
        print(f"Final status: {final_job.get('status')}")
        print(f"Error message: {final_job.get('error_message')}")
        assert final_job["status"] == "cancelled", f"Expected status 'cancelled', got {final_job.get('status')}"
        print(">>> TEST 2 PASSED: Generation canceled successfully with proper status & error message.")

if __name__ == "__main__":
    test_cancel_generation()
