import csv
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(
    ROOT, "data", "phase2_real_full", "manifest_open"
)

DICTIONARY = os.path.join(
    ROOT, "data", "grascii", "dictionary_preanniversary.tsv"
)

NGRAM = os.path.join(
    ROOT, "outputs", "ngram_model.json"
)

OUTPUT = os.path.join(
    ROOT,
    "results",
    "phase2_full_open",
    "oracle_context_nlp_results.json"
)

MAX_RUNS = 200
MIN_RUN = 2
BEAM_SIZE = 10


def load_manifest():
    rows = []

    path = os.path.join(DATA, "manifest.csv")

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"].strip().lower() == "test":
                rows.append(row)

    return rows


def load_dictionary():
    reverse = defaultdict(list)

    with open(DICTIONARY, encoding="utf-8") as f:
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


def load_ngram():
    with open(NGRAM, encoding="utf-8") as f:
        model = json.load(f)

    return (
        model["unigrams"],
        model["bigrams"],
        model["trigrams"],
    )


def score(history, word, model):
    unigrams, bigrams, trigrams = model

    total_uni = sum(unigrams.values())
    total_bi = sum(bigrams.values())
    total_tri = sum(trigrams.values())

    vocab = len(unigrams)

    word = word.lower()

    uni = unigrams.get(word, 0)

    bi = 0
    if len(history) >= 1:
        bi = bigrams.get(
            f"{history[-1]}\t{word}", 0
        )

    tri = 0
    if len(history) >= 2:
        tri = trigrams.get(
            f"{history[-2]}\t{history[-1]}\t{word}", 0
        )

    uni_p = (uni + 1) / (total_uni + vocab)
    bi_p = (bi + 1) / (total_bi + vocab)
    tri_p = (tri + 1) / (total_tri + vocab)

    return (
        0.1 * math.log(uni_p)
        + 0.3 * math.log(bi_p)
        + 0.6 * math.log(tri_p)
    )


def decode(candidate_lists, model):
    beams = [(0.0, [])]

    for candidates in candidate_lists:

        if not candidates:
            candidates = ["[UNKNOWN]"]

        next_beams = []

        for total_score, history in beams:

            for word in candidates:

                s = score(
                    history,
                    word,
                    model
                )

                new_history = history + [word]

                next_beams.append(
                    (
                        total_score + s,
                        new_history
                    )
                )

        next_beams.sort(
            key=lambda x: x[0],
            reverse=True
        )

        beams = next_beams[:BEAM_SIZE]

    return beams[0][1] if beams else []


print("=" * 70)
print("ORACLE CONTEXTUAL NLP EVALUATION")
print("=" * 70)

print("\nLoading test manifest...")

rows = load_manifest()

print("Test samples:", len(rows))

print("\nLoading dictionary...")

reverse = load_dictionary()

print("Dictionary Grascii entries:", len(reverse))

print("\nLoading N-gram model...")

model = load_ngram()

print("N-gram model loaded.")


# ---------------------------------------------------------
# Map English word -> test row
# ---------------------------------------------------------

word_to_row = {}

for row in rows:

    word = row["english"].strip().lower()

    if word not in word_to_row:
        word_to_row[word] = row


# ---------------------------------------------------------
# Build corpus runs
# ---------------------------------------------------------

corpus_path = os.path.join(ROOT, "data", "corpus.txt")

eligible_runs = []

with open(corpus_path, encoding="utf-8") as f:

    for line in f:

        words = line.strip().lower().split()

        if len(words) < MIN_RUN:
            continue

        # Remove punctuation-like tokens
        words = [
            w.strip(".,!?;:\"'()[]{}")
            for w in words
        ]

        current = []

        for word in words:

            if word in word_to_row:
                current.append(word)

            else:
                if len(current) >= MIN_RUN:
                    eligible_runs.append(current)

                current = []

        if len(current) >= MIN_RUN:
            eligible_runs.append(current)


eligible_runs = eligible_runs[:MAX_RUNS]

print("\nRuns selected:", len(eligible_runs))


# ---------------------------------------------------------
# Decode each run using PERFECT GRASCII
# ---------------------------------------------------------

total_words = 0
correct_words = 0
correct_runs = 0

ambiguous_words = 0
ambiguous_correct = 0

examples = []

for idx, run in enumerate(eligible_runs, 1):

    candidate_lists = []
    true_words = []

    for word in run:

        row = word_to_row[word]

        grascii = row["grascii"].strip()

        candidates = reverse.get(grascii, [])

        candidate_lists.append(candidates)
        true_words.append(word)

        if len(candidates) > 1:
            ambiguous_words += 1

    predicted = decode(
        candidate_lists,
        model
    )

    run_correct = True

    for true, pred, candidates in zip(
        true_words,
        predicted,
        candidate_lists
    ):

        total_words += 1

        if true == pred:
            correct_words += 1
        else:
            run_correct = False

            if len(examples) < 30:
                examples.append(
                    {
                        "true": true,
                        "predicted": pred,
                        "candidates": candidates[:20],
                        "run": true_words,
                    }
                )

        if len(candidates) > 1:
            if true == pred:
                ambiguous_correct += 1

    if run_correct:
        correct_runs += 1

    print(
        f"RUN {idx}/{len(eligible_runs)}: "
        f"{' '.join(true_words)}"
    )


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

word_accuracy = (
    correct_words / total_words
    if total_words else 0
)

run_accuracy = (
    correct_runs / len(eligible_runs)
    if eligible_runs else 0
)

ambiguous_accuracy = (
    ambiguous_correct / ambiguous_words
    if ambiguous_words else 0
)

print()
print("=" * 70)
print("ORACLE CONTEXTUAL NLP RESULTS")
print("=" * 70)

print(
    f"Runs: {len(eligible_runs)}"
)

print(
    f"Words: {total_words}"
)

print(
    f"Word accuracy: "
    f"{word_accuracy:.4f} "
    f"({100 * word_accuracy:.2f}%)"
)

print(
    f"Run accuracy: "
    f"{run_accuracy:.4f} "
    f"({100 * run_accuracy:.2f}%)"
)

print(
    f"Ambiguous words: "
    f"{ambiguous_words}"
)

print(
    f"Ambiguous accuracy: "
    f"{ambiguous_accuracy:.4f} "
    f"({100 * ambiguous_accuracy:.2f}%)"
)


print()
print("=" * 70)
print("ERROR EXAMPLES")
print("=" * 70)

for ex in examples[:15]:

    print()
    print("RUN:", " ".join(ex["run"]))
    print("TRUE:", ex["true"])
    print("PRED:", ex["predicted"])
    print(
        "CANDIDATES:",
        " | ".join(ex["candidates"])
    )


results = {
    "runs": len(eligible_runs),
    "words": total_words,
    "word_accuracy": word_accuracy,
    "run_accuracy": run_accuracy,
    "ambiguous_words": ambiguous_words,
    "ambiguous_accuracy": ambiguous_accuracy,
    "examples": examples,
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