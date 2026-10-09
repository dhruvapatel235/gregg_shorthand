from datasets import load_dataset
from pathlib import Path
import pandas as pd
import random

print("Loading Gregg dataset...")
ds = load_dataset("grascii/gregg-preanniversary-words")["train"]

indices = list(range(len(ds)))
random.Random(42).shuffle(indices)
indices = indices[:1000]

out = Path("data/phase2_real_1000")
imgdir = out / "images"
imgdir.mkdir(parents=True, exist_ok=True)

rows = []

for i, j in enumerate(indices, start=1):
    item = ds[j]

    word = str(item["longhand"]).replace(" ", "_").replace("/", "_")
    filename = f"{i:04d}_{word}.png"

    item["image"].convert("L").save(imgdir / filename)

    rows.append({
        "id": i,
        "path": f"data/phase2_real_1000/images/{filename}",
        "word": item["longhand"],
        "grascii_normalized": item["grascii_normalized"],
    })

pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)

print(f"Saved {len(rows)} images to {out}")