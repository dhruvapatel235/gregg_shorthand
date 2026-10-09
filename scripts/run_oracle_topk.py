import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recognizer import Recognizer
from src.decode import prefix_beam_search


DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_full_open/recognizer.pt"

BEAM = 50
K = 50
PRUNE = -12.0


def load_dictionary(path):
    d = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) < 2:
                continue

            grascii = row[0].strip()
            english = row[1].strip().lower()

            if grascii and english:
                d[grascii].append(english)

    return d


def load_test_rows():
    rows = []

    path = os.path.join(DATA, "manifest.csv")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"].lower() == "test":
                rows.append(row)

    return rows


print("Loading test data...")
rows = load_test_rows()

print("Test samples:", len(rows))

dictionary = load_dictionary(
    "data/grascii/dictionary_preanniversary.tsv"
)

print("Dictionary Grascii entries:", len(dictionary))

rec = Recognizer(CKPT)

images = []
truths = []

for row in rows:
    path = row["image_path"]

    if not os.path.isabs(path):
    	path = os.path.join(DATA, path)

    img = np.array(
        Image.open(path).convert("L")
    )

    ink = 255 - img

    images.append(ink.astype(np.uint8))
    truths.append(str(row["grascii"]).strip())


print("Running Top-50 beam search...")

logprobs = rec.logprobs(images)

topk = []

for i, lp in enumerate(logprobs):
    beams = prefix_beam_search(
        lp,
        rec.vocab,
        beam=BEAM,
        k=K,
        prune=PRUNE,
    )

    candidates = [x[0] for x in beams]

    topk.append(candidates)

    if (i + 1) % 200 == 0:
        print(f"{i + 1}/{len(rows)}")


# ---------------------------------------------------------
# ORACLE ANALYSIS
# ---------------------------------------------------------

top1 = 0
top5 = 0
top10 = 0
top50 = 0

oracle_dictionary = 0
oracle_english = 0

total = len(rows)

examples = []

for i in range(total):

    truth = truths[i]
    candidates = topk[i]

    if len(candidates) > 0 and candidates[0] == truth:
        top1 += 1

    if truth in candidates[:5]:
        top5 += 1

    if truth in candidates[:10]:
        top10 += 1

    if truth in candidates[:50]:
        top50 += 1

        # Oracle: pretend we know the correct Grascii candidate.
        # If it is in the dictionary, count the corresponding
        # English word as recoverable.
        if truth in dictionary:
            oracle_dictionary += 1
            oracle_english += 1

    elif len(examples) < 30:
        examples.append(
            (truth, candidates[:10])
        )


print()
print("=" * 70)
print("ORACLE TOP-K ANALYSIS")
print("=" * 70)

print(f"TOTAL: {total}")

print(
    f"TOP-1 RECALL:  {top1 / total:.4f} "
    f"({100 * top1 / total:.2f}%)"
)

print(
    f"TOP-5 RECALL:  {top5 / total:.4f} "
    f"({100 * top5 / total:.2f}%)"
)

print(
    f"TOP-10 RECALL: {top10 / total:.4f} "
    f"({100 * top10 / total:.2f}%)"
)

print(
    f"TOP-50 RECALL: {top50 / total:.4f} "
    f"({100 * top50 / total:.2f}%)"
)

print()
print("THEORETICAL ORACLE CEILING")
print(
    f"If correct Grascii is available in Top-50: "
    f"{top50 / total:.2%}"
)

print()
print("EXAMPLES WHERE CORRECT GRASCII IS NOT IN TOP-10")

shown = 0

for truth, candidates in examples:

    print()
    print("TRUE:", truth)
    print("TOP:", " | ".join(candidates))

    shown += 1

    if shown >= 10:
        break


output = {
    "total": total,
    "top1_recall": top1 / total,
    "top5_recall": top5 / total,
    "top10_recall": top10 / total,
    "top50_recall": top50 / total,
}

with open(
    "results/phase2_full_open/oracle_topk_results.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(output, f, indent=2)

print()
print(
    "Saved: "
    "results/phase2_full_open/oracle_topk_results.json"
)