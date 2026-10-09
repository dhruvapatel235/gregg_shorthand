import numpy as np
from PIL import Image
from src.manifest import read_manifest
from src.recognizer import Recognizer
from src.decode import prefix_beam_search

DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_full_open/recognizer.pt"
LEXICON = "data/phase2_real_full/manifest_open/lexicon_grascii.tsv"

# ------------------------------------------------------------
# Load test data
# ------------------------------------------------------------

df, meta = read_manifest(DATA)
df = df[df["split"] == "test"].reset_index(drop=True)

rec = Recognizer(CKPT)

images = []
labels = []

for _, row in df.iterrows():
    img = np.array(Image.open(row["abs_path"]).convert("L"))
    ink = 255 - img
    images.append(ink.astype(np.uint8))
    labels.append(str(row["grascii"]))

# ------------------------------------------------------------
# Load valid Grascii sequences
# ------------------------------------------------------------

valid_grascii = set()

with open(LEXICON, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")

        if not line.strip():
            continue

        parts = line.split("\t")

        if len(parts) >= 2:
            valid_grascii.add(parts[0])

print("VALID LEXICON SEQUENCES:", len(valid_grascii))

# ------------------------------------------------------------
# Decode
# ------------------------------------------------------------

logprobs = rec.logprobs(images)

preds = []

for lp in logprobs:
    candidates = prefix_beam_search(
        lp,
        rec.vocab,
        beam=50,
        k=50,
        prune=-12.0
    )

    chosen = None

    for candidate, score in candidates:
        if candidate in valid_grascii:
            chosen = candidate
            break

    # If no beam candidate exists in the lexicon,
    # fall back to the highest-scoring beam candidate.
    if chosen is None:
        chosen = candidates[0][0]

    preds.append(chosen)

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def edit_distance(a, b):
    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        cur = [i]

        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    cur[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + (ca != cb)
                )
            )

        prev = cur

    return prev[-1]


total_edits = 0
total_chars = 0
exact = 0

for truth, pred in zip(labels, preds):
    total_edits += edit_distance(pred, truth)
    total_chars += len(truth)

    if pred == truth:
        exact += 1

print("TEST SAMPLES:", len(labels))
print("LEXICON TEST CER:", round(total_edits / total_chars, 4))
print("LEXICON TEST EXACT:", round(exact / len(labels), 4))
print("EXACT COUNT:", exact)

print("\nExamples:")

for truth, pred in list(zip(labels, preds))[:30]:
    print("TRUE:", truth, " PRED:", pred)