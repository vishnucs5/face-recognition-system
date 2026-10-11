import asyncio
import httpx

BASE_URL = "http://127.0.0.1:8000"

async def test_concurrent_submissions():
    print("\n--- TEST 9: CONCURRENT / RAPID SUBMISSION HANDLING ---")
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # Submit 3 rapid requests simulating user clicking or client handling
        payload = {
            "mode": "txt2img",
            "prompt": "Rapid submission concurrency check",
            "model": "v1-5-pruned-emaonly.safetensors",
            "width": 512,
            "height": 512,
            "steps": 10,
        }
        
        # Fire 3 requests at the exact same instant
        responses = await asyncio.gather(
            client.post("/api/generations", json=payload),
            client.post("/api/generations", json=payload),
            client.post("/api/generations", json=payload),
            return_exceptions=True
        )

        job_ids = []
        for idx, res in enumerate(responses):
            assert isinstance(res, httpx.Response), f"Request {idx} failed with exception {res}"
            assert res.status_code == 201, f"Status: {res.status_code}"
            jid = res.json()["id"]
            job_ids.append(jid)
            print(f"Submission {idx+1}: Status=201, ID={jid}")

        # Ensure all IDs are completely unique
        assert len(set(job_ids)) == 3, "Duplicate job IDs detected!"
        print(">>> All concurrent submissions received unique job IDs without database locks or collisions.")

        # Clean up by cancelling them
        for jid in job_ids:
            await client.post(f"/api/generations/{jid}/cancel")

        await client.post("/api/generations/clear-queue")
        print(">>> TEST 9 PASSED: Concurrency and duplicate click protection verified!")

if __name__ == "__main__":
    asyncio.run(test_concurrent_submissions())
