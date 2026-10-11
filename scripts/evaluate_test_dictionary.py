
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
OUTPUT_PATH = "results/phase2_full_open/test_dictionary_evaluation.json"


def greedy_decode(logp, vocab):
    ids = np.argmax(logp, axis=1)
    collapsed = []
    prev = None

    for idx in ids:
        idx = int(idx)
        if idx != prev and idx != 0:
            collapsed.append(idx)
        prev = idx

    return vocab.decode(collapsed)


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


def load_dictionary(path):
    mapping = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()

            if len(parts) >= 2:
                grascii = parts[0].strip()
                word = parts[1].strip().lower()
                if grascii and word:
                    mapping[grascii].append(word)

    return {
        code: sorted(set(words))
        for code, words in mapping.items()
    }


def main():
    df, _ = read_manifest(DATA)
    df = df[df["split"] == "test"].reset_index(drop=True)

    rec = Recognizer(CKPT)
    dictionary = load_dictionary(DICT_PATH)

    images = []
    truths = []
    english_truths = []

    for _, row in df.iterrows():
        img = np.array(Image.open(row["abs_path"]).convert("L"))
        images.append((255 - img).astype(np.uint8))
        truths.append(str(row["grascii"]).strip())
        english_truths.append(str(row["english"]).strip().lower())

    logprobs = rec.logprobs(images)
    predictions = [
        greedy_decode(lp, rec.vocab) for lp in logprobs
    ]

    total_edits = 0
    total_chars = 0
    exact = 0
    english_correct = 0
    dictionary_hits = 0
    dictionary_correct = 0
    examples = []

    for truth, pred, english in zip(
        truths, predictions, english_truths
    ):
        total_edits += edit_distance(pred, truth)
        total_chars += len(truth)
        exact += int(pred == truth)

        # Dictionary lookup is based on the recognized code.
        candidates = dictionary.get(pred, [])
        if candidates:
            dictionary_hits += 1
            if english in candidates:
                dictionary_correct += 1

        # For an ambiguous code, use the first sorted candidate.
        # This is deterministic, but not contextual language decoding.
        chosen_word = candidates[0] if candidates else ""

        if chosen_word == english:
            english_correct += 1

        if len(examples) < 50:
            examples.append({
                "true_grascii": truth,
                "predicted_grascii": pred,
                "true_english": english,
                "dictionary_candidates": candidates,
                "chosen_dictionary_word": chosen_word,
            })

    n = len(truths)
    results = {
        "dataset": DATA,
        "checkpoint": CKPT,
        "test_samples": n,
        "grascii_cer": total_edits / total_chars if total_chars else None,
        "grascii_exact_accuracy": exact / n if n else None,
        "grascii_exact_count": exact,
        "dictionary_coverage": dictionary_hits / n if n else None,
        "english_accuracy_first_sorted_candidate":
            english_correct / n if n else None,
        "english_correct_within_candidates":
            dictionary_correct / n if n else None,
        "note": (
            "Isolated-word test only. Candidate-list correctness is not "
            "a deployable decoder accuracy. First-sorted-candidate "
            "accuracy is a deterministic baseline, not contextual N-gram "
            "decoding. Verify dataset provenance before describing it "
            "as genuine handwritten performance."
        ),
        "examples": examples,
    }

    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("TEST SAMPLES:", n)
    print("GRASCII CER:", round(results["grascii_cer"], 4))
    print("GRASCII EXACT ACCURACY:",
          round(results["grascii_exact_accuracy"], 4))
    print("DICTIONARY COVERAGE:",
          round(results["dictionary_coverage"], 4))
    print("ENGLISH ACCURACY (first sorted candidate):",
          round(results["english_accuracy_first_sorted_candidate"], 4))
    print("ENGLISH CORRECT WITHIN CANDIDATES:",
          round(results["english_correct_within_candidates"], 4))
    print("SAVED:", OUTPUT_PATH)

    print("\nFirst 10 examples:")
    for ex in examples[:10]:
        print(
            "TRUE:", ex["true_grascii"], ex["true_english"],
            "| PRED:", ex["predicted_grascii"],
            "| CANDIDATES:", ex["dictionary_candidates"][:5]
        )


if __name__ == "__main__":
    main()