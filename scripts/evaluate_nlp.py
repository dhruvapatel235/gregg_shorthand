import json
import math
import re
from pathlib import Path
from collections import defaultdict

LEXICON = Path("data/grascii/dictionary_preanniversary.tsv")
CORPUS = Path("data/corpus.txt")
MODEL = Path("outputs/ngram_model.json")


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


# ---------------------------------------------------------
# Load Grascii -> English candidates
# ---------------------------------------------------------

candidates = defaultdict(list)

for line in LEXICON.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#") or "\t" not in line:
        continue

    grascii, word = line.split("\t", 1)
    word = word.strip().lower()

    if word:
        candidates[grascii].append(word)


# ---------------------------------------------------------
# Load language model
# ---------------------------------------------------------

data = json.loads(MODEL.read_text(encoding="utf-8"))

unigrams = data["unigrams"]
bigrams = data["bigrams"]
trigrams = data["trigrams"]

total_words = sum(unigrams.values())
vocab_size = len(unigrams)


def score_word(previous, word):
    prev1 = previous[-1] if len(previous) >= 1 else "<s>"
    prev2 = previous[-2] if len(previous) >= 2 else "<s>"

    tri = trigrams.get(f"{prev2} {prev1} {word}", 0)
    bi = bigrams.get(f"{prev1} {word}", 0)

    tri_base = bigrams.get(f"{prev2} {prev1}", 0)
    bi_base = unigrams.get(prev1, total_words)

    tri_prob = (tri + 1) / (tri_base + vocab_size)
    bi_prob = (bi + 1) / (bi_base + vocab_size)

    return (
        0.7 * math.log(tri_prob)
        + 0.3 * math.log(bi_prob)
    )


def decode_sentence(words):
    output = []

    for word in words:
        choices = candidates.get(word)

        if not choices:
            output.append(None)
            continue

        scored = [
            (score_word(output, candidate), candidate)
            for candidate in choices
        ]

        output.append(max(scored)[1])

    return output


# ---------------------------------------------------------
# Evaluate corpus sentences
# ---------------------------------------------------------

total_words = 0
correct = 0
ambiguous = 0
ambiguous_correct = 0
ambiguous_total = 0

examples = []

for line in CORPUS.read_text(encoding="utf-8").splitlines():

    truth = tokenize(line)

    if not truth:
        continue

    # We can only evaluate words that have a Grascii
    # representation in the dictionary.
    grascii_for_truth = []

    reverse = {}

    for g, words in candidates.items():
        for w in words:
            reverse.setdefault(w, []).append(g)

    if any(word not in reverse for word in truth):
        continue

    # Use the first Grascii representation for evaluation.
    grascii_for_truth = [reverse[word][0] for word in truth]

    predicted = decode_sentence(grascii_for_truth)

    for t, p, g in zip(truth, predicted, grascii_for_truth):

        if p is None:
            continue

        total_words += 1

        if p == t:
            correct += 1

        if len(candidates[g]) > 1:
            ambiguous_total += 1

            if p == t:
                ambiguous_correct += 1

    if len(examples) < 20 and predicted != truth:
        examples.append((truth, predicted))


print("TOTAL EVALUATED WORDS:", total_words)
print("OVERALL ACCURACY:", round(correct / total_words, 4))

if ambiguous_total:
    print(
        "AMBIGUOUS WORD ACCURACY:",
        round(ambiguous_correct / ambiguous_total, 4)
    )

print("AMBIGUOUS WORDS:", ambiguous_total)

print("\nExamples:")

for truth, pred in examples:
    print("TRUE:", " ".join(truth))
    print("PRED:", " ".join(x or "[UNKNOWN]" for x in pred))
    print()