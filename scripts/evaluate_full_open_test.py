import numpy as np
from PIL import Image
from src.manifest import read_manifest
from src.recognizer import Recognizer

DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_oversample/recognizer.pt"

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

logprobs = rec.logprobs(images)

def greedy_decode(logp, vocab):
    ids = np.argmax(logp, axis=1)
    collapsed = []
    prev = None

    for idx in ids:
        idx = int(idx)

        if idx == prev:
            continue

        prev = idx

        if idx != 0:
            collapsed.append(idx)

    return vocab.decode(collapsed)

preds = [greedy_decode(lp, rec.vocab) for lp in logprobs]

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
print("TEST CER:", round(total_edits / total_chars, 4))
print("TEST EXACT:", round(exact / len(labels), 4))
print("EXACT COUNT:", exact)

print("\nExamples:")

for truth, pred in list(zip(labels, preds))[:30]:
    print("TRUE:", truth, " PRED:", pred)