
import json
from pathlib import Path
from collections import defaultdict

from src.manifest import read_manifest
from src.recognizer import Recognizer
from src.decode import prefix_beam_search

DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_oversample/recognizer.pt"
DICTIONARY = "data/grascii/dictionary_preanniversary.tsv"
OUT = "results/phase2_full_open/fuzzy_validation.json"

# Load validation split
df, meta = read_manifest(DATA)
val = df[df["split"] == "val"].reset_index(drop=True)

print("Validation samples:", len(val))
print("Checkpoint:", CKPT)

# Load Grascii -> English dictionary
dictionary = defaultdict(list)

with open(DICTIONARY, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            continue

        code = parts[0].strip()
        for word in parts[1].strip().lower().split("|"):
            if word and word not in dictionary[code]:
                dictionary[code].append(word)

codes = list(dictionary)
print("Dictionary codes:", len(codes))

# Levenshtein distance
def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        current = [i]

        for j, cb in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb)
            ))

        previous = current

    return previous[-1]

# Prepare recognizer
rec = Recognizer(CKPT)

baseline_correct = 0
exact_lookup_correct = 0
counts = defaultdict(lambda: {
    "correct": 0,
    "changed": 0,
    "helped": 0,
    "hurt": 0
})

for i, row in val.iterrows():
    image_path = row["abs_path"]
    true_code = str(row["grascii"]).strip()

    from PIL import Image
    import numpy as np

    image = np.array(
        Image.open(image_path).convert("L")
    )
    ink = 255 - image

    lp = rec.logprobs([ink.astype(np.uint8)])[0]
    beams = prefix_beam_search(
        lp, rec.vocab, beam=50, k=50, prune=-12.0
    )

    if not beams:
        continue

    pred = beams[0][0]
    baseline_correct += pred == true_code

    # Exact lookup: keep recognizer output if absent
    exact_code = pred if pred in dictionary else pred
    exact_lookup_correct += exact_code == true_code

    # Cache nearest dictionary distances for this prediction
    distances = [(edit_distance(pred, code), code) for code in codes]
    min_dist = min(d for d, _ in distances)
    nearest = sorted(code for d, code in distances if d == min_dist)

    for max_dist in (1, 2):
        stats = counts[max_dist]

        # Only correct when nearest dictionary code is within limit.
        # If tied, choose no correction to avoid arbitrary tie-breaking.
        corrected = pred
        if min_dist <= max_dist and len(nearest) == 1:
            corrected = nearest[0]

        is_correct = corrected == true_code
        stats["correct"] += is_correct
        stats["changed"] += corrected != pred
        stats["helped"] += pred != true_code and corrected == true_code
        stats["hurt"] += pred == true_code and corrected != true_code

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(val)}")

n = len(val)
results = {
    "validation_samples": n,
    "baseline_grascii_exact_accuracy": baseline_correct / n,
    "exact_dictionary_accuracy": exact_lookup_correct / n,
    "fuzzy": {}
}

for dist, stats in counts.items():
    results["fuzzy"][str(dist)] = {
        "accuracy": stats["correct"] / n,
        "changed": stats["changed"],
        "helped": stats["helped"],
        "hurt": stats["hurt"]
    }

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
Path(OUT).write_text(json.dumps(results, indent=2), encoding="utf-8")

print(json.dumps(results, indent=2))
print("Saved:", OUT)