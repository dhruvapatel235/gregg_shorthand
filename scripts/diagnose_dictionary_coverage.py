
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.manifest import read_manifest
from src.recognizer import Recognizer

DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_oversample/recognizer.pt"
DICT_PATH = "data/grascii/dictionary_preanniversary.tsv"


def load_dictionary(path):
    mapping = {}
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
                    mapping.setdefault(code, set()).add(word)
    return mapping


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


def main():
    df, _ = read_manifest(DATA)
    df = df[df["split"] == "test"].reset_index(drop=True)

    rec = Recognizer(CKPT)
    dictionary = load_dictionary(DICT_PATH)

    images = []
    rows = []
    for _, row in df.iterrows():
        img = np.array(Image.open(row["abs_path"]).convert("L"))
        images.append((255 - img).astype(np.uint8))
        rows.append(row)

    predictions = [
        greedy_decode(lp, rec.vocab)
        for lp in rec.logprobs(images)
    ]

    exact = 0
    true_code_in_dictionary = 0
    predicted_code_in_dictionary = 0
    predicted_code_and_word_match = 0
    misses = Counter()
    examples = []

    for row, pred in zip(rows, predictions):
        truth = str(row["grascii"]).strip()
        word = str(row["english"]).strip().lower()

        exact += int(pred == truth)
        true_code_in_dictionary += int(truth in dictionary)
        predicted_code_in_dictionary += int(pred in dictionary)
        predicted_code_and_word_match += int(
            word in dictionary.get(pred, set())
        )

        if pred not in dictionary:
            misses[pred] += 1

        if pred != truth and len(examples) < 20:
            examples.append({
                "true_code": truth,
                "predicted_code": pred,
                "english": word,
                "true_code_in_dictionary": truth in dictionary,
                "predicted_code_in_dictionary": pred in dictionary,
            })

    n = len(rows)
    print("TEST SAMPLES:", n)
    print("GRASCII EXACT ACCURACY:", round(exact / n, 4))
    print("TRUE CODES IN DICTIONARY:", round(true_code_in_dictionary / n, 4))
    print("PREDICTED CODES IN DICTIONARY:", round(predicted_code_in_dictionary / n, 4))
    print("PREDICTED CODE CONTAINS TRUE ENGLISH:", round(predicted_code_and_word_match / n, 4))

    print("\nMost common predicted codes missing from dictionary:")
    for code, count in misses.most_common(20):
        print(repr(code), "count:", count)

    print("\nFirst 20 recognition errors:")
    for ex in examples:
        print(ex)

    output = Path("results/phase2_full_open/dictionary_diagnostic.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "test_samples": n,
        "exact_matches": exact,
        "true_codes_in_dictionary": true_code_in_dictionary,
        "predicted_codes_in_dictionary": predicted_code_in_dictionary,
        "predicted_code_contains_true_english": predicted_code_and_word_match,
        "common_missing_predictions": misses.most_common(20),
        "examples": examples,
    }, indent=2), encoding="utf-8")
    print("\nSAVED:", output)


if __name__ == "__main__":
    main()