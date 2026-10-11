
from src.manifest import read_manifest
from PIL import Image, ImageDraw
from pathlib import Path

df, _ = read_manifest("data/phase2_real_full/manifest_open")
rows = df.sample(n=12, random_state=42)

W, H = 240, 120
grid = Image.new("RGB", (W * 3, H * 4), "white")
draw = ImageDraw.Draw(grid)

for i, (_, row) in enumerate(rows.iterrows()):
    img = Image.open(row["abs_path"]).convert("RGB")
    img.thumbnail((W - 20, H - 35))

    x = (i % 3) * W + (W - img.width) // 2
    y = (i // 3) * H + 25

    grid.paste(img, (x, y))
    draw.text(
        ((i % 3) * W + 5, (i // 3) * H + 5),
        str(row["english"]),
        fill="black",
    )

out = Path("results/handwriting_sample_grid.png")
out.parent.mkdir(parents=True, exist_ok=True)
grid.save(out)

print(f"Saved: {out.resolve()}")
