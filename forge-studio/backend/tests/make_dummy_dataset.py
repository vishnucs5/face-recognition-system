from PIL import Image
from pathlib import Path

p = Path("c:/vishnu/test_dataset")
p.mkdir(parents=True, exist_ok=True)

# Create 4 sample images
for i in range(4):
    img = Image.new("RGB", (1024, 1024), color=((i + 1) * 50, 100, 150))
    img.save(p / f"character_{i}.png")
    # write caption for first 3, leave 4th missing
    if i < 3:
        (p / f"character_{i}.txt").write_text(f"portrait of a futuristic cyber warrior {i}", encoding="utf-8")

print("Sample dataset created successfully at c:/vishnu/test_dataset")
