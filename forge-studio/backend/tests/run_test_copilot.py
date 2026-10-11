import httpx
import json

BASE_URL = "http://127.0.0.1:8000"

def test_copilot():
    print("\n--- TEST 5: QWEN3-CODER COPILOT VALIDATION ---")
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        # 1. Prompt expansion / improvement
        res1 = client.post("/api/copilot/prompt", json={
            "task": "improve_prompt",
            "message": "A cyberpunk street market at night",
            "current_context": {
                "prompt": "A cyberpunk street market at night",
                "model": "v1-5-pruned-emaonly.safetensors",
                "steps": 20,
                "cfg_scale": 7.0
            }
        })
        assert res1.status_code == 200, f"improve_prompt failed: {res1.text}"
        data1 = res1.json()
        print("1. Prompt expansion reply:")
        print(f"   Reply: {data1.get('reply')[:150]}...")
        assert data1.get("suggested_settings") is not None
        print(f"   Suggested expanded prompt: {data1['suggested_settings'].get('prompt')}")

        # 2. Negative prompt generation
        res2 = client.post("/api/copilot/prompt", json={
            "task": "negative_prompt",
            "message": "A cinematic landscape of mountains with auroras",
        })
        assert res2.status_code == 200, f"negative_prompt failed: {res2.text}"
        data2 = res2.json()
        print("\n2. Negative prompt suggestion:")
        neg = data2.get("suggested_settings", {}).get("negative_prompt")
        print(f"   Negative prompt: {neg}")
        assert neg is not None and len(neg) > 0

        # 3. Prompt variations
        res3 = client.post("/api/copilot/prompt", json={
            "task": "variations",
            "message": "An astronaut exploring ancient alien ruins",
        })
        assert res3.status_code == 200, f"variations failed: {res3.text}"
        data3 = res3.json()
        variations = data3.get("variations", [])
        print("\n3. Variations count:", len(variations))
        for idx, v in enumerate(variations):
            print(f"   Variation {idx+1}: {v}")
        assert len(variations) >= 2, "Expected at least 2 variations"

        # 4. Configure generation / parameter suggestions
        res4 = client.post("/api/copilot/prompt", json={
            "task": "configure_generation",
            "message": "Settings for fast portrait concept sketching",
        })
        assert res4.status_code == 200, f"configure_generation failed: {res4.text}"
        data4 = res4.json()
        params = data4.get("suggested_settings")
        print("\n4. Suggested generation parameters:")
        print(f"   Steps: {params.get('steps')}, CFG: {params.get('cfg_scale')}, Sampler: {params.get('sampler')}")
        assert params.get("steps") is not None
        assert params.get("cfg_scale") is not None

        print("\n>>> TEST 5 PASSED: Qwen3-Coder Copilot validated across all operations!")

if __name__ == "__main__":
    test_copilot()
