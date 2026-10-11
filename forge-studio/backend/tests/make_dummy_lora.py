import json
import struct

meta = {
    "ss_base_model_version": "SD 1.5",
    "ss_tag_frequency": "anime_portrait, soft_watercolor",
}
h_bytes = json.dumps({"__metadata__": meta}).encode("utf-8")
with open("c:/vishnu/test_anime_portrait.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(h_bytes)))
    f.write(h_bytes)
    f.write(b"\x00" * (1024 * 1024 * 5))  # 5MB weights
print("Successfully generated test_anime_portrait.safetensors")
