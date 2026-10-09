import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.manifest import read_manifest
from src.dataset import load_gray, to_ink, to_canvas
from src.model import CRNN
from src.vocab import Vocab


DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_full_open/recognizer.pt"


print("Loading manifest...")

df, meta = read_manifest(DATA)

train = df[df["split"] == "train"].reset_index(drop=True)

print("Training samples:", len(train))


# ------------------------------------------------------------
# Label lengths
# ------------------------------------------------------------

lengths = train["grascii"].astype(str).map(len)

print()
print("=" * 70)
print("GRASCII LABEL LENGTHS")
print("=" * 70)

print("Minimum:", lengths.min())
print("Maximum:", lengths.max())
print("Average:", round(lengths.mean(), 2))

for n in sorted(lengths.value_counts().index):
    count = int((lengths == n).sum())
    print(f"length {n:2d}: {count:5d}")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print()
print("Loading checkpoint...")

checkpoint = torch.load(
    CKPT,
    map_location="cpu"
)

vocab = Vocab(
    checkpoint["vocab"]
)

model = CRNN(
    vocab.size,
    checkpoint["hidden"]
)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()

print("Model loaded.")


# ------------------------------------------------------------
# Check CNN output time dimension
# ------------------------------------------------------------

print()
print("=" * 70)
print("CTC TIME-DIMENSION CAPACITY")
print("=" * 70)

sample_rows = train.iloc[
    np.linspace(
        0,
        len(train) - 1,
        min(100, len(train)),
        dtype=int
    )
]

ratios = []

with torch.no_grad():

    for _, row in sample_rows.iterrows():

        gray = load_gray(row["abs_path"])
        ink = to_ink(gray)

        x = torch.from_numpy(
            to_canvas(ink)
        ).unsqueeze(0).unsqueeze(0)

        out = model(x)

        T = out.shape[0]
        label_len = len(str(row["grascii"]))

        ratios.append(
            (T, label_len)
        )


print(
    "CNN/CTC output time steps:",
    ratios[0][0]
)

max_label = max(
    label_len for _, label_len in ratios
)

print(
    "Maximum sampled label length:",
    max_label
)

print()

print(
    "CTC capacity ratio "
    "(T / label length):"
)

print(
    round(
        ratios[0][0] / max_label,
        2
    )
)


# ------------------------------------------------------------
# Find longest labels
# ------------------------------------------------------------

print()
print("=" * 70)
print("LONGEST GRASCII LABELS")
print("=" * 70)

longest = train.assign(
    label_len=lengths
).sort_values(
    "label_len",
    ascending=False
).head(20)

for _, row in longest.iterrows():

    print(
        f"{int(row['label_len']):2d}  "
        f"{str(row['grascii'])}"
        f"  {str(row['english'])}"
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)