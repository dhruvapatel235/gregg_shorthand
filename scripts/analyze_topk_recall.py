import numpy as np
from PIL import Image

from src.manifest import read_manifest
from src.recognizer import Recognizer
from src.decode import prefix_beam_search


DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_full_open/recognizer.pt"

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

print("Loaded test samples:", len(labels))

# Use the project's exact preprocessing/model pipeline
logprobs = rec.logprobs(images)

top1 = 0
top5 = 0
top10 = 0
top50 = 0

examples = []

for i, (lp, truth) in enumerate(zip(logprobs, labels)):
    results = prefix_beam_search(
        lp,
        rec.vocab,
        beam=50,
        k=50,
        prune=-12.0
    )

    candidates = [r[0] for r in results]

    if truth in candidates[:1]:
        top1 += 1
    if truth in candidates[:5]:
        top5 += 1
    if truth in candidates[:10]:
        top10 += 1
    if truth in candidates[:50]:
        top50 += 1

    # Save a few useful examples where Top-1 is wrong
    # but the correct Grascii is somewhere in the beam.
    if truth != candidates[0] and truth in candidates[:50] and len(examples) < 30:
        examples.append(
            (truth, candidates[:10])
        )

n = len(labels)

print("\n==============================")
print("TOP-K GRASCII RECALL")
print("==============================")

print("TOTAL:", n)
print("TOP-1:", round(top1 / n, 4))
print("TOP-5:", round(top5 / n, 4))
print("TOP-10:", round(top10 / n, 4))
print("TOP-50:", round(top50 / n, 4))

print("\nPercentages:")
print("TOP-1:", round(100 * top1 / n, 2), "%")
print("TOP-5:", round(100 * top5 / n, 2), "%")
print("TOP-10:", round(100 * top10 / n, 2), "%")
print("TOP-50:", round(100 * top50 / n, 2), "%")

print("\nExamples where Top-1 is wrong but correct Grascii is in Top-50:")

for truth, candidates in examples:
    print("\nTRUE:", truth)
    print("TOP:", " | ".join(candidates))