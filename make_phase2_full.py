from datasets import load_dataset
from pathlib import Path
import csv

OUT = Path("data/phase2_real_full")
IMG_DIR = OUT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset("grascii/gregg-preanniversary-words")["train"]

with open(OUT / "labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "path", "word", "grascii_normalized"])

    for i, row in enumerate(ds):
        img = row["image"].convert("L")
        filename = f"{i:05d}.png"
        img.save(IMG_DIR / filename)

        w.writerow([
            i,
            str(IMG_DIR / filename),
            row["longhand"],
            row["grascii_normalized"]
        ])

print("TOTAL:", len(ds))
print("OUTPUT:", OUT)