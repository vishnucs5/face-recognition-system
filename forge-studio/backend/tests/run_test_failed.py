import httpx
import time
import json

BASE_URL = "http://127.0.0.1:8000"

def test_failed_generation():
    print("\n--- TEST 3: FAILED GENERATION WITH USEFUL ERROR ---")
    payload = {
        "mode": "txt2img",
        "prompt": "Testing failure state with nonexistent checkpoint",
        "model": "nonexistent_model_error_test.safetensors",
        "width": 512,
        "height": 512,
        "steps": 10,
        "cfg_scale": 7.0,
        "sampler": "euler",
    }
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        r = client.post("/api/generations", json=payload)
        assert r.status_code == 201, f"Failed to submit: {r.text}"
        job_id = r.json()["id"]
        print(f"Submitted job: {job_id}")

        for _ in range(30):
            time.sleep(0.5)
            job = client.get(f"/api/generations/{job_id}").json()
            status = job.get("status")
            if status in ["failed", "completed"]:
                print(f"Terminal status: {status}")
                print(f"Error message: {job.get('error_message')}")
                assert status == "failed", f"Expected failed, got {status}"
                assert job.get("error_message") is not None, "Error message was None"
                print(">>> TEST 3 PASSED: Generation cleanly failed with useful ComfyUI error!")
                return job

if __name__ == "__main__":
    test_failed_generation()
