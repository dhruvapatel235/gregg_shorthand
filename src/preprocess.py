from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "phase2_real"
IMAGE_DIR = DATA_DIR / "images"
OUTPUT_DIR = DATA_DIR / "preprocessed"

TARGET_HEIGHT = 64
TARGET_WIDTH = 256

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

labels = pd.read_csv(DATA_DIR / "labels.csv")
processed = []

for _, row in labels.iterrows():
    source = ROOT / row["path"]
    image = Image.open(source).convert("L")

    # Preserve the original aspect ratio
    image.thumbnail((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)

    # White canvas, with the image centered
    canvas = Image.new("L", (TARGET_WIDTH, TARGET_HEIGHT), 255)
    x = (TARGET_WIDTH - image.width) // 2
    y = (TARGET_HEIGHT - image.height) // 2
    canvas.paste(image, (x, y))

    output_path = OUTPUT_DIR / source.name
    canvas.save(output_path)

    processed.append({
        "path": output_path.relative_to(ROOT).as_posix(),
        "word": row["word"],
        "grascii_normalized": row["grascii_normalized"]
    })

pd.DataFrame(processed).to_csv(DATA_DIR / "preprocessed_labels.csv", index=False)

print(f"Processed images: {len(processed)}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Label CSV: {DATA_DIR / 'preprocessed_labels.csv'}")
print(f"Target size: {TARGET_WIDTH} x {TARGET_HEIGHT}")
