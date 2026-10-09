import numpy as np
from PIL import Image
from pathlib import Path

from src.manifest import read_manifest
from src.recognizer import Recognizer
from src.decode import prefix_beam_search


DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_full_open/recognizer.pt"
LEXICON = "data/phase2_real_full/manifest_open/lexicon_grascii.tsv"

BEAM_SIZES = [50, 100, 200, 500]


print("Loading manifest...")

df, meta = read_manifest(DATA)
test = df[df["split"] == "test"].reset_index(drop=True)

print("Test samples:", len(test))


# ------------------------------------------------------------
# Load valid Grascii lexicon
# ------------------------------------------------------------

lexicon = set()

with open(LEXICON, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if s:
            lexicon.add(s)

print("Lexicon sequences:", len(lexicon))


# ------------------------------------------------------------
# Load images
# ------------------------------------------------------------

images = []
labels = []

for _, row in test.iterrows():

    img = np.array(
        Image.open(row["abs_path"]).convert("L")
    )

    ink = 255 - img

    images.append(
        ink.astype(np.uint8)
    )

    labels.append(str(row["grascii"]))


# ------------------------------------------------------------
# Load recognizer
# ------------------------------------------------------------

print("Loading recognizer...")

rec = Recognizer(CKPT)

print("Recognizer loaded.")

print("Running recognition...")

logprobs = rec.logprobs(images)


# ------------------------------------------------------------
# Evaluate beam sizes
# ------------------------------------------------------------

print()
print("=" * 70)
print("BEAM SIZE EXPERIMENT")
print("=" * 70)

for beam_size in BEAM_SIZES:

    correct = 0
    total = len(labels)

    for lp, true_label in zip(logprobs, labels):

        beams = prefix_beam_search(
            lp,
            rec.vocab,
            beam=beam_size,
            k=beam_size,
            prune=-12.0
        )

        chosen = None

        # First lexicon-valid candidate
        for candidate, score in beams:

            if candidate in lexicon:
                chosen = candidate
                break

        # Fallback
        if chosen is None:
            chosen = beams[0][0]

        if chosen == true_label:
            correct += 1

    accuracy = correct / total

    print(
        f"beam={beam_size:3d} | "
        f"exact={accuracy:.4f} "
        f"({correct}/{total})"
    )

print("=" * 70)