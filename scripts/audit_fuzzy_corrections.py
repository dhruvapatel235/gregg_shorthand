
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from src.manifest import read_manifest
from src.recognizer import Recognizer
from src.decode import prefix_beam_search

DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_oversample/recognizer.pt"
DICTIONARY = "data/grascii/dictionary_preanniversary.tsv"
OUT = "results/phase2_full_open/fuzzy_correction_audit.json"
MAX_DIST = 2

# Load validation data only
df, meta = read_manifest(DATA)
val = df[df["split"] == "val"].reset_index(drop=True)

# Load dictionary codes
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

print("Validation samples:", len(val))
print("Dictionary codes:", len(codes))
print("Loading recognizer:", CKPT)
rec = Recognizer(CKPT)

records = []

for i, row in val.iterrows():
    true_code = str(row["grascii"]).strip()
    image = np.array(
        Image.open(row["abs_path"]).convert("L")
    )
    ink = (255 - image).astype(np.uint8)

    lp = rec.logprobs([ink])[0]
    beams = prefix_beam_search(
        lp, rec.vocab, beam=50, k=50, prune=-12.0
    )
    if not beams:
        continue

    pred = beams[0][0]

    distances = [
        (edit_distance(pred, code), code)
        for code in codes
    ]
    min_dist = min(d for d, _ in distances)
    nearest = sorted(code for d, code in distances if d == min_dist)

    corrected = pred
    if min_dist <= MAX_DIST and len(nearest) == 1:
        corrected = nearest[0]

    records.append({
        "image_path": str(row["image_path"]),
        "true_english": str(row["english"]),
        "true_grascii": true_code,
        "predicted_grascii": pred,
        "nearest_distance": min_dist,
        "nearest_codes": "|".join(nearest),
        "corrected_grascii": corrected,
        "baseline_correct": pred == true_code,
        "corrected_correct": corrected == true_code,
        "changed": corrected != pred,
        "outcome": (
            "unchanged_correct" if pred == true_code and corrected == true_code
            else "helped" if pred != true_code and corrected == true_code
            else "hurt" if pred == true_code and corrected != true_code
            else "still_wrong"
        )
    })

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(val)}")

result_df = pd.DataFrame(records)
summary = {
    "samples_evaluated": len(result_df),
    "baseline_accuracy": float(result_df["baseline_correct"].mean()),
    "corrected_accuracy": float(result_df["corrected_correct"].mean()),
    "changed": int(result_df["changed"].sum()),
    "helped": int((result_df["outcome"] == "helped").sum()),
    "hurt": int((result_df["outcome"] == "hurt").sum()),
    "still_wrong": int((result_df["outcome"] == "still_wrong").sum()),
    "outcome_counts": {
        str(k): int(v)
        for k, v in result_df["outcome"].value_counts().items()
    }
}

out = Path(OUT)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "settings": {
        "split": "val",
        "checkpoint": CKPT,
        "max_edit_distance": MAX_DIST,
        "require_unique_nearest_code": True
    },
    "summary": summary,
    "examples": records
}, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("Saved:", OUT)

print("\nExamples where fuzzy correction helped:")
print(result_df[result_df["outcome"] == "helped"][
    ["true_english", "true_grascii", "predicted_grascii",
     "corrected_grascii", "nearest_distance"]
].head(15).to_string(index=False))

print("\nExamples where predictions remained wrong:")
print(result_df[result_df["outcome"] == "still_wrong"][
    ["true_english", "true_grascii", "predicted_grascii",
     "corrected_grascii", "nearest_distance"]
].head(15).to_string(index=False))