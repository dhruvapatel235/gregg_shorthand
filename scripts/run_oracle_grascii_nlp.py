import csv
import json
import math
import os
import sys
from collections import defaultdict

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data", "phase2_real_full", "manifest_open")
DICTIONARY = os.path.join(
    ROOT, "data", "grascii", "dictionary_preanniversary.tsv"
)
NGRAM = os.path.join(ROOT, "outputs", "ngram_model.json")

OUTPUT = os.path.join(
    ROOT,
    "results",
    "phase2_full_open",
    "oracle_grascii_nlp_results.json",
)


# ---------------------------------------------------------
# LOAD TEST MANIFEST
# ---------------------------------------------------------

def load_test_rows():
    rows = []

    manifest = os.path.join(DATA, "manifest.csv")

    with open(manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"].strip().lower() == "test":
                rows.append(row)

    return rows


# ---------------------------------------------------------
# LOAD GRASCII -> ENGLISH DICTIONARY
# ---------------------------------------------------------

def load_dictionary(path):
    reverse = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) < 2:
                continue

            grascii = row[0].strip()
            english = row[1].strip().lower()

            if grascii and english:
                if english not in reverse[grascii]:
                    reverse[grascii].append(english)

    return reverse


# ---------------------------------------------------------
# LOAD N-GRAM MODEL
# ---------------------------------------------------------

def load_ngram(path):
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    unigrams = model["unigrams"]
    bigrams = model["bigrams"]
    trigrams = model["trigrams"]

    total_uni = sum(unigrams.values())
    total_bi = sum(bigrams.values())
    total_tri = sum(trigrams.values())

    vocab_size = len(unigrams)

    return (
        unigrams,
        bigrams,
        trigrams,
        total_uni,
        total_bi,
        total_tri,
        vocab_size,
    )


# ---------------------------------------------------------
# N-GRAM SCORE
# ---------------------------------------------------------

def ngram_score(
    history,
    word,
    unigrams,
    bigrams,
    trigrams,
    total_uni,
    total_bi,
    total_tri,
    vocab_size,
):
    word = word.lower()

    uni = unigrams.get(word, 0)

    if len(history) >= 1:
        bi = bigrams.get(
            f"{history[-1]}\t{word}",
            0
        )
    else:
        bi = 0

    if len(history) >= 2:
        tri = trigrams.get(
            f"{history[-2]}\t{history[-1]}\t{word}",
            0
        )
    else:
        tri = 0

    uni_p = (
        (uni + 1)
        / (total_uni + vocab_size)
    )

    bi_p = (
        (bi + 1)
        / (total_bi + vocab_size)
    )

    tri_p = (
        (tri + 1)
        / (total_tri + vocab_size)
    )

    return (
        0.1 * math.log(uni_p)
        + 0.3 * math.log(bi_p)
        + 0.6 * math.log(tri_p)
    )


# ---------------------------------------------------------
# SINGLE-WORD NLP PREDICTION
# ---------------------------------------------------------

def predict_word(
    grascii,
    reverse,
    ngram_model,
):
    candidates = reverse.get(grascii, [])

    if not candidates:
        return "[UNKNOWN]", []

    (
        unigrams,
        bigrams,
        trigrams,
        total_uni,
        total_bi,
        total_tri,
        vocab_size,
    ) = ngram_model

    # For an isolated word there is no sentence history.
    # Therefore the model effectively uses unigram probability.
    best_word = None
    best_score = float("-inf")

    for word in candidates:

        score = ngram_score(
            [],
            word,
            unigrams,
            bigrams,
            trigrams,
            total_uni,
            total_bi,
            total_tri,
            vocab_size,
        )

        if score > best_score:
            best_score = score
            best_word = word

    return best_word, candidates


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

print("=" * 70)
print("ORACLE GRASCII -> NLP EVALUATION")
print("=" * 70)

print()
print("Loading test manifest...")

rows = load_test_rows()

print("Test samples:", len(rows))

print()
print("Loading dictionary...")

reverse = load_dictionary(DICTIONARY)

print(
    "Dictionary Grascii entries:",
    len(reverse)
)

print()
print("Loading N-gram model...")

ngram_model = load_ngram(NGRAM)

print("N-gram model loaded.")

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

total = 0
evaluated = 0
correct = 0
unknown = 0

ambiguous_total = 0
ambiguous_correct = 0

examples_wrong = []
examples_ambiguous = []

for i, row in enumerate(rows):

    grascii = row["grascii"].strip()

    # The manifest's English label may have capitalization.
    true_word = row["english"].strip().lower()

    predicted, candidates = predict_word(
        grascii,
        reverse,
        ngram_model,
    )

    total += 1

    if not candidates:
        unknown += 1
        continue

    evaluated += 1

    if len(candidates) > 1:
        ambiguous_total += 1

    if predicted == true_word:
        correct += 1

        if len(candidates) > 1:
            ambiguous_correct += 1
    else:
        if len(examples_wrong) < 30:
            examples_wrong.append(
                {
                    "true_word": true_word,
                    "grascii": grascii,
                    "predicted": predicted,
                    "candidates": candidates[:20],
                }
            )

    if len(candidates) > 1 and len(examples_ambiguous) < 20:
        examples_ambiguous.append(
            {
                "true_word": true_word,
                "grascii": grascii,
                "predicted": predicted,
                "candidates": candidates[:20],
            }
        )

    if (i + 1) % 200 == 0:
        print(f"{i + 1}/{len(rows)}")


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

overall_accuracy = (
    correct / evaluated
    if evaluated
    else 0.0
)

ambiguous_accuracy = (
    ambiguous_correct / ambiguous_total
    if ambiguous_total
    else 0.0
)

print()
print("=" * 70)
print("ORACLE GRASCII -> NLP RESULTS")
print("=" * 70)

print(
    f"TOTAL TEST SAMPLES: {total}"
)

print(
    f"DICTIONARY-COVERED SAMPLES: {evaluated}"
)

print(
    f"UNKNOWN / NOT IN DICTIONARY: {unknown}"
)

print()

print(
    f"OVERALL NLP ACCURACY: "
    f"{overall_accuracy:.4f} "
    f"({100 * overall_accuracy:.2f}%)"
)

print(
    f"AMBIGUOUS GRASCII SAMPLES: "
    f"{ambiguous_total}"
)

print(
    f"AMBIGUOUS NLP ACCURACY: "
    f"{ambiguous_accuracy:.4f} "
    f"({100 * ambiguous_accuracy:.2f}%)"
)

print()
print("=" * 70)
print("WRONG PREDICTION EXAMPLES")
print("=" * 70)

for ex in examples_wrong[:15]:

    print()
    print("TRUE:", ex["true_word"])
    print("GRASCII:", ex["grascii"])
    print("PREDICTED:", ex["predicted"])
    print(
        "CANDIDATES:",
        " | ".join(ex["candidates"])
    )

print()
print("=" * 70)
print("AMBIGUOUS EXAMPLES")
print("=" * 70)

for ex in examples_ambiguous[:10]:

    print()
    print("TRUE:", ex["true_word"])
    print("GRASCII:", ex["grascii"])
    print("PREDICTED:", ex["predicted"])
    print(
        "CANDIDATES:",
        " | ".join(ex["candidates"])
    )


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

results = {
    "total_test_samples": total,
    "dictionary_covered_samples": evaluated,
    "unknown_samples": unknown,
    "overall_accuracy": overall_accuracy,
    "ambiguous_samples": ambiguous_total,
    "ambiguous_accuracy": ambiguous_accuracy,
    "examples_wrong": examples_wrong,
    "examples_ambiguous": examples_ambiguous,
}

with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        indent=2
    )

print()
print("Saved:")
print(OUTPUT)