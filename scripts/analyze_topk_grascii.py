from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.manifest import read_manifest
from src.recognizer import Recognizer
from src.decode import prefix_beam_search


DATA = ROOT / "data" / "phase2_real_full" / "manifest_open"
CKPT = ROOT / "outputs" / "phase2_full_open" / "recognizer.pt"

BEAM = 50
TOP_K = [1, 5, 10, 50]


def load_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((256, 64))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = 1.0 - arr

    # [H,W] -> [1,1,H,W]
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


df, meta = read_manifest(DATA)

test_rows = df[df["split"] == "test"].reset_index(drop=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", device)
print("TEST SAMPLES:", len(test_rows))
print("Loading recognizer...")

rec = Recognizer(CKPT, device=device)

# Get vocabulary from recognizer
vocab = rec.vocab

counts = {k: 0 for k in TOP_K}
total = 0
examples = []

for i, row in test_rows.iterrows():

    try:
        target = str(row["grascii"])
        image_path = row["abs_path"]

        x = load_image(image_path).to(device)

        with torch.no_grad():
            logp = rec.model(x)

        # Model output: [T, B, C]
        # Beam search expects: [T, C]
        logp_np = logp[:, 0, :].detach().cpu().numpy()

        beams = prefix_beam_search(
            logp_np,
            vocab,
            beam=BEAM,
            k=BEAM,
            prune=-12.0
        )

        candidates = []

        for item in beams:
            if isinstance(item, tuple):
                text = item[0]
            else:
                text = item

            if text not in candidates:
                candidates.append(text)

        total += 1

        found_at = None

        for rank, candidate in enumerate(candidates, start=1):
            if candidate == target:
                found_at = rank
                break

        if found_at is not None:
            for k in TOP_K:
                if found_at <= k:
                    counts[k] += 1

        if found_at is not None and found_at > 1 and len(examples) < 30:
            examples.append(
                (target, found_at, candidates[:10])
            )

        if (i + 1) % 100 == 0:
            print(
                f"{i + 1}/{len(test_rows)} "
                f"| top1={counts[1]/total:.4f} "
                f"| top5={counts[5]/total:.4f} "
                f"| top10={counts[10]/total:.4f} "
                f"| top50={counts[50]/total:.4f}"
            )

    except Exception as e:
        print(f"ERROR {i}: {e}")


print("\n========================================")
print("TOP-K GRASCII RECALL")
print("========================================")

print("TOTAL:", total)

for k in TOP_K:
    print(
        f"TOP-{k} RECALL: "
        f"{counts[k] / total:.4f} "
        f"({counts[k]}/{total})"
    )


print("\n========================================")
print("EXAMPLES WHERE TRUE GRASCII WAS NOT TOP-1")
print("========================================")

for target, rank, candidates in examples:
    print(f"\nTRUE: {target}")
    print(f"FOUND AT RANK: {rank}")
    print("TOP CANDIDATES:", candidates)