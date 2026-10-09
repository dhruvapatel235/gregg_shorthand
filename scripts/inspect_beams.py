import numpy as np
from PIL import Image

from src.manifest import read_manifest
from src.recognizer import Recognizer
from src.decode import prefix_beam_search


DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_full_open/recognizer.pt"


# ------------------------------------------------------------
# Load test data
# ------------------------------------------------------------

df, meta = read_manifest(DATA)

test = df[
    df["split"] == "test"
].reset_index(drop=True)

rec = Recognizer(CKPT)


# ------------------------------------------------------------
# Words to inspect
# ------------------------------------------------------------

TARGETS = [
    "there",
    "were",
    "only",
    "cannot",
    "deviation",
    "however",
    "care",
    "friendly",
    "lay",
]


# ------------------------------------------------------------
# Inspect beams
# ------------------------------------------------------------

for word in TARGETS:

    rows = test[
        test["english"].str.lower() == word
    ]

    if len(rows) == 0:
        print("\nNOT FOUND:", word)
        continue

    row = rows.iloc[0]

    print()
    print("=" * 70)
    print("WORD:", word)
    print("TRUE GRASCII:", row["grascii"])

    img = np.array(
        Image.open(
            row["abs_path"]
        ).convert("L")
    )

    ink = 255 - img

    images = [
        ink.astype(np.uint8)
    ]

    logprobs = rec.logprobs(images)

    beams = prefix_beam_search(
        logprobs[0],
        rec.vocab,
        beam=50,
        k=50,
        prune=-12.0
    )

    print()
    print("TOP BEAM CANDIDATES:")

    for rank, (candidate, score) in enumerate(
        beams,
        1
    ):

        marker = ""

        if candidate == str(row["grascii"]):
            marker = "  <-- TRUE GRASCII"

        print(
            f"{rank:2d}. "
            f"{candidate!r:20s} "
            f"{score:.4f}"
            f"{marker}"
        )