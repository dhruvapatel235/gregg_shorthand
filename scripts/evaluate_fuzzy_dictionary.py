
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

from src.manifest import read_manifest
from src.recognizer import Recognizer

DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_oversample/recognizer.pt"
DICT_PATH = "data/grascii/dictionary_preanniversary.tsv"
OUT = "results/phase2_full_open/fuzzy_dictionary_evaluation.json"


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                cur[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb)
            ))
        prev = cur
    return prev[-1]


def greedy_decode(logp, vocab):
    ids = np.argmax(logp, axis=1)
    out = []
    prev = None
    for idx in ids:
        idx = int(idx)
        if idx != prev and idx != 0:
            out.append(idx)
        prev = idx
    return vocab.decode(out)


def load_dictionary(path):
    mapping = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) >= 2:
                code = parts[0].strip()
                word = parts[1].strip().lower()
                if code and word:
                    mapping[code].add(word)
    return {code: sorted(words) for code, words in mapping.items()}


def main():
    df, _ = read_manifest(DATA)
    df = df[df["split"] == "test"].reset_index(drop=True)

    rec = Recognizer(CKPT)
    dictionary = load_dictionary(DICT_PATH)
    dictionary_codes = list(dictionary.keys())

    images = []
    for _, row in df.iterrows():
        img = np.array(Image.open(row["abs_path"]).convert("L"))
        images.append((255 - img).astype(np.uint8))

    predictions = [
        greedy_decode(lp, rec.vocab)
        for lp in rec.logprobs(images)
    ]

    truths = [str(x).strip() for x in df["grascii"]]
    english = [str(x).strip().lower() for x in df["english"]]

    baseline_code_correct = 0
    corrected_code_correct = 0
    corrected_word_correct = 0
    dictionary_hit = 0
    candidate_recall = 0
    changed = 0
    examples = []

    for truth, pred, word in zip(truths, predictions, english):
        baseline_code_correct += int(pred == truth)

        # Find the closest dictionary code(s).
        distances = [
            (edit_distance(pred, code), code)
            for code in dictionary_codes
        ]
        min_dist = min(d for d, _ in distances)
        nearest_codes = sorted(
            code for d, code in distances if d == min_dist
        )

        # Deterministic tie-breaking for a reproducible baseline.
        corrected_code = nearest_codes[0]
        candidates = sorted(set(
            w for code in nearest_codes for w in dictionary[code]
        ))

        corrected_code_correct += int(corrected_code == truth)
        dictionary_hit += int(pred in dictionary)
        candidate_recall += int(word in candidates)
        corrected_word_correct += int(
            len(candidates) == 1 and candidates[0] == word
        )
        changed += int(corrected_code != pred)

        if len(examples) < 30 and (
            pred != truth or corrected_code != pred
        ):
            examples.append({
                "true_code": truth,
                "predicted_code": pred,
                "corrected_code": corrected_code,
                "true_english": word,
                "nearest_codes": nearest_codes,
                "english_candidates": candidates,
            })

    n = len(truths)
    results = {
        "test_samples": n,
        "checkpoint": CKPT,
        "dictionary_codes": len(dictionary_codes),
        "baseline_grascii_exact_accuracy":
            baseline_code_correct / n,
        "fuzzy_corrected_grascii_exact_accuracy":
            corrected_code_correct / n,
        "original_dictionary_exact_coverage":
            dictionary_hit / n,
        "true_english_in_nearest_code_candidates":
            candidate_recall / n,
        "unique_candidate_english_accuracy":
            corrected_word_correct / n,
        "prediction_changed_count": changed,
        "note": (
            "Exploratory nearest-edit-distance baseline. It does not "
            "use visual probabilities or sentence context. Ties are "
            "preserved for candidate recall, while code accuracy uses "
            "alphabetical tie-breaking. Candidate recall is not "
            "deployable English accuracy."
        ),
        "examples": examples,
    }

    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for key, value in results.items():
        if key != "examples" and key != "note":
            print(f"{key}: {value}")
    print("SAVED:", OUT)

    print("\nExamples:")
    for ex in examples[:15]:
        print(
            "TRUE:", ex["true_code"], ex["true_english"],
            "| PRED:", ex["predicted_code"],
            "| CORRECTED:", ex["corrected_code"],
            "| CANDIDATES:", ex["english_candidates"][:5]
        )


if __name__ == "__main__":
    main()