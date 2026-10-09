from datasets import load_dataset
from pathlib import Path
import pandas as pd
import re

DATASET = "grascii/gregg-preanniversary-words"
OUTPUT_DIR = Path("data/phase2_real")
IMAGE_DIR = OUTPUT_DIR / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Gregg Preanniversary dataset...")
print(f"Dataset: {DATASET}")
print()

dataset = load_dataset(
    DATASET,
    split="train",
    streaming=True
)

rows = []

print("Extracting first 100 samples...")
print()

for index, sample in enumerate(dataset):
    if index >= 100:
        break

    image = sample["image"]
    word = str(sample["longhand"]).strip()
    grascii = str(sample.get("grascii_normalized", "")).strip()

    # Make the English word safe for use as a filename
    safe_word = re.sub(r"[^A-Za-z0-9_-]", "_", word)

    filename = f"{index+1:03d}_{safe_word}.png"
    image_path = IMAGE_DIR / filename

    # Convert to grayscale and save as PNG
    image.convert("L").save(image_path)

    rows.append({
        "id": index + 1,
        "path": str(image_path).replace("\\", "/"),
        "word": word,
        "grascii_normalized": grascii
    })

    print(f"{index+1:03d}/100  {word}  ->  {filename}")

df = pd.DataFrame(rows)

csv_path = OUTPUT_DIR / "labels.csv"
df.to_csv(csv_path, index=False)

print()
print("=" * 60)
print("PHASE 2 EXTRACTION COMPLETE")
print("=" * 60)
print(f"Images extracted : {len(rows)}")
print(f"Image folder     : {IMAGE_DIR}")
print(f"Labels CSV       : {csv_path}")
print()

if len(rows) != 100:
    print("WARNING: Fewer than 100 samples were extracted.")
else:
    print("SUCCESS: Exactly 100 samples extracted.")

print()
print("First 5 labels:")
print(df.head().to_string(index=False))
