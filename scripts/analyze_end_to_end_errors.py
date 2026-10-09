import json
from collections import Counter


RESULTS = "results/phase2_full_open/end_to_end_results.json"
DICTIONARY = "data/grascii/dictionary_preanniversary.tsv"


# ============================================================
# Load results
# ============================================================

with open(RESULTS, "r", encoding="utf-8") as f:
    results = json.load(f)

examples = results["examples"]


# ============================================================
# Load Grascii dictionary
# ============================================================

candidates = {}

with open(DICTIONARY, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")

        if len(parts) < 2:
            continue

        grascii = parts[0].strip()

        words = [
            w.strip().lower()
            for w in parts[1].split("|")
            if w.strip()
        ]

        candidates[grascii] = words


# ============================================================
# Counters
# ============================================================

total = 0

recognition_correct = 0
recognition_wrong = 0

english_correct = 0
english_wrong = 0

correct_grascii_wrong_english = 0
wrong_grascii_wrong_english = 0

unknown_predictions = 0

wrong_grascii_examples = []
correct_grascii_wrong_english_examples = []


# ============================================================
# Analyze every word
# ============================================================

for result in examples:

    true_words = result["true"]
    pred_grascii = result["grascii"]
    pred_english = result["predicted"]

    for true_word, pred_g, pred_e in zip(
        true_words,
        pred_grascii,
        pred_english
    ):

        total += 1

        # ----------------------------------------------------
        # Find true Grascii
        # ----------------------------------------------------

        true_g = None

        # Search manifest-independent way:
        # the end-to-end results do not store true Grascii,
        # so derive it from the known test labels by reading
        # the manifest below.
        #
        # This is filled later.
        #
        pass


# ============================================================
# Load manifest to obtain true Grascii
# ============================================================

from src.manifest import read_manifest

df, meta = read_manifest(
    "data/phase2_real_full/manifest_open"
)

test = df[
    df["split"] == "test"
].copy()

test["english"] = test["english"].str.lower()

true_grascii = dict(
    zip(
        test["english"],
        test["grascii"]
    )
)


# ============================================================
# Reset metrics
# ============================================================

total = 0

recognition_correct = 0
recognition_wrong = 0

english_correct = 0
english_wrong = 0

correct_grascii_wrong_english = 0
wrong_grascii_wrong_english = 0

unknown_predictions = 0

wrong_grascii_examples = []
correct_grascii_wrong_english_examples = []


# ============================================================
# Detailed analysis
# ============================================================

for result in examples:

    true_words = result["true"]
    pred_grascii = result["grascii"]
    pred_english = result["predicted"]

    for true_word, pred_g, pred_e in zip(
        true_words,
        pred_grascii,
        pred_english
    ):

        true_g = true_grascii.get(
            true_word,
            ""
        )

        total += 1

        g_correct = (
            pred_g == true_g
        )

        e_correct = (
            pred_e == true_word
        )

        if g_correct:
            recognition_correct += 1
        else:
            recognition_wrong += 1

        if e_correct:
            english_correct += 1
        else:
            english_wrong += 1

        if g_correct and not e_correct:

            correct_grascii_wrong_english += 1

            correct_grascii_wrong_english_examples.append(
                {
                    "true_word": true_word,
                    "true_grascii": true_g,
                    "pred_grascii": pred_g,
                    "pred_english": pred_e,
                    "dictionary_candidates":
                        candidates.get(pred_g, [])
                }
            )

        if not g_correct and not e_correct:

            wrong_grascii_wrong_english += 1

            wrong_grascii_examples.append(
                {
                    "true_word": true_word,
                    "true_grascii": true_g,
                    "pred_grascii": pred_g,
                    "pred_english": pred_e,
                    "dictionary_candidates":
                        candidates.get(pred_g, [])
                }
            )

        if pred_e == "[UNKNOWN]":
            unknown_predictions += 1


# ============================================================
# Print summary
# ============================================================

print()
print("=" * 70)
print("END-TO-END ERROR ANALYSIS")
print("=" * 70)

print("TOTAL WORDS:", total)

print()
print("RECOGNITION")
print(
    "Correct Grascii:",
    recognition_correct,
    f"({recognition_correct / total * 100:.2f}%)"
)

print(
    "Wrong Grascii:",
    recognition_wrong,
    f"({recognition_wrong / total * 100:.2f}%)"
)

print()
print("ENGLISH")
print(
    "Correct English:",
    english_correct,
    f"({english_correct / total * 100:.2f}%)"
)

print(
    "Wrong English:",
    english_wrong,
    f"({english_wrong / total * 100:.2f}%)"
)

print()
print("ERROR BREAKDOWN")

print(
    "Correct Grascii but wrong English:",
    correct_grascii_wrong_english,
    f"({correct_grascii_wrong_english / total * 100:.2f}%)"
)

print(
    "Wrong Grascii and wrong English:",
    wrong_grascii_wrong_english,
    f"({wrong_grascii_wrong_english / total * 100:.2f}%)"
)

print(
    "UNKNOWN English predictions:",
    unknown_predictions
)


# ============================================================
# Show examples
# ============================================================

print()
print("=" * 70)
print("CORRECT GRASCII → WRONG ENGLISH")
print("=" * 70)

for x in correct_grascii_wrong_english_examples[:30]:

    print(
        f"TRUE WORD: {x['true_word']} | "
        f"GRASCII: {x['true_grascii']} | "
        f"PRED: {x['pred_english']} | "
        f"CANDIDATES: {x['dictionary_candidates']}"
    )


print()
print("=" * 70)
print("WRONG GRASCII → WRONG ENGLISH")
print("=" * 70)

for x in wrong_grascii_examples[:30]:

    print(
        f"TRUE WORD: {x['true_word']} | "
        f"TRUE GRASCII: {x['true_grascii']} | "
        f"PRED GRASCII: {x['pred_grascii']} | "
        f"PRED ENGLISH: {x['pred_english']} | "
        f"CANDIDATES: {x['dictionary_candidates']}"
    )


# ============================================================
# Save analysis
# ============================================================

analysis = {
    "total_words": total,

    "recognition_correct": recognition_correct,
    "recognition_wrong": recognition_wrong,

    "english_correct": english_correct,
    "english_wrong": english_wrong,

    "correct_grascii_wrong_english":
        correct_grascii_wrong_english,

    "wrong_grascii_wrong_english":
        wrong_grascii_wrong_english,

    "unknown_predictions":
        unknown_predictions,

    "correct_grascii_wrong_english_examples":
        correct_grascii_wrong_english_examples,

    "wrong_grascii_wrong_english_examples":
        wrong_grascii_examples
}


with open(
    "results/phase2_full_open/end_to_end_error_analysis.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        analysis,
        f,
        indent=2
    )


print()
print("=" * 70)
print("Saved:")
print(
    "results/phase2_full_open/"
    "end_to_end_error_analysis.json"
)
print("=" * 70)