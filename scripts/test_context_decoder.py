import json
import math
import re
from pathlib import Path
from collections import defaultdict

LEXICON = Path("data/grascii/dictionary_preanniversary.tsv")
MODEL = Path("outputs/ngram_model.json")


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


# -----------------------------
# Load lexicon
# -----------------------------

candidates = defaultdict(list)

with open(LEXICON, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")

        if not line.strip():
            continue

        parts = line.split("\t")

        if len(parts) >= 2:
            grascii = parts[0]
            words = parts[1].strip().lower().split("|")

            for word in words:
                word = word.strip()

                if word:
                    candidates[grascii].append(word)


# -----------------------------
# Load n-gram model
# -----------------------------

data = json.loads(MODEL.read_text(encoding="utf-8"))

unigrams = data["unigrams"]
bigrams = data["bigrams"]
trigrams = data["trigrams"]

total_words = sum(unigrams.values())
vocab_size = len(unigrams)


def score_word(previous_words, word):
    """
    Score a candidate using trigram + bigram context.
    """

    prev1 = previous_words[-1] if len(previous_words) >= 1 else "<s>"
    prev2 = previous_words[-2] if len(previous_words) >= 2 else "<s>"

    tri_key = f"{prev2} {prev1} {word}"
    bi_key = f"{prev1} {word}"

    tri_count = trigrams.get(tri_key, 0)
    bi_count = bigrams.get(bi_key, 0)
    word_count = unigrams.get(word, 0)

    # Add-one smoothing
    tri_prob = (tri_count + 1) / (bigrams.get(f"{prev2} {prev1}", 0) + vocab_size)
    bi_prob = (bi_count + 1) / (unigrams.get(prev1, total_words) + vocab_size)
    unigram_prob = (word_count + 1) / (total_words + vocab_size)

    return (
        0.6 * math.log(tri_prob)
        + 0.3 * math.log(bi_prob)
        + 0.1 * math.log(unigram_prob)
    )


def decode_sentence(grascii_sequence):
    """
    grascii_sequence:
        list of recognized Grascii tokens, one per word.
    """

    output = []

    for grascii in grascii_sequence:
        choices = candidates.get(grascii, [])

        if not choices:
            output.append("[UNKNOWN]")
            continue

        scored = [
            (score_word(output, word), word)
            for word in choices
        ]

        scored.sort(reverse=True)

        output.append(scored[0][1])

    return output


# -----------------------------
# Controlled tests
# -----------------------------

tests = [
    ["'ALO", "'ANSM"],
    ["'AR"],
    ["'ED"],
    ["'OE"],
]

for test in tests:
    print("\nGRASCII:", test)
    print("CANDIDATES:")

    for g in test:
        print(" ", g, "->", candidates.get(g, []))

    print("OUTPUT:", decode_sentence(test))