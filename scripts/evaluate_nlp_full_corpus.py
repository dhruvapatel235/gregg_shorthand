import json
import math
import re
import random
from pathlib import Path
from collections import defaultdict, Counter

LEXICON = Path("data/grascii/dictionary_preanniversary.tsv")
CORPUS = Path("data/corpus.txt")

TRAIN_RATIO = 0.80
SEED = 42


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


# ============================================================
# 1. LOAD GRASCII -> ENGLISH LEXICON
# ============================================================

candidates = defaultdict(list)

for line in LEXICON.read_text(encoding="utf-8").splitlines():

    if not line or line.startswith("#") or "\t" not in line:
        continue

    grascii, word = line.split("\t", 1)
    word = word.strip().lower()

    if word:
        candidates[grascii].append(word)


# Reverse mapping:
# English word -> possible Grascii representations

reverse = defaultdict(list)

for grascii, words in candidates.items():

    for word in words:
        reverse[word].append(grascii)


print("LEXICON GRASCII ENTRIES:", len(candidates))
print("LEXICON ENGLISH WORDS:", len(reverse))


# ============================================================
# 2. LOAD CORPUS
# ============================================================

sentences = []

for line in CORPUS.read_text(encoding="utf-8").splitlines():

    tokens = tokenize(line)

    if tokens:
        sentences.append(tokens)


print("TOTAL CORPUS SENTENCES:", len(sentences))


# ============================================================
# 3. REPRODUCIBLE 80/20 SPLIT
# ============================================================

random.seed(SEED)

indices = list(range(len(sentences)))
random.shuffle(indices)

split = int(len(indices) * TRAIN_RATIO)

train_indices = indices[:split]
test_indices = indices[split:]

train_sentences = [sentences[i] for i in train_indices]
test_sentences = [sentences[i] for i in test_indices]

print("TRAIN SENTENCES:", len(train_sentences))
print("TEST SENTENCES:", len(test_sentences))


# ============================================================
# 4. BUILD N-GRAM MODEL USING TRAINING DATA ONLY
# ============================================================

unigrams = Counter()
bigrams = Counter()
trigrams = Counter()

for tokens in train_sentences:

    for word in tokens:
        unigrams[word] += 1

    padded = ["<s>", "<s>"] + tokens + ["</s>"]

    for i in range(len(padded) - 1):
        bigrams[f"{padded[i]} {padded[i+1]}"] += 1

    for i in range(len(padded) - 2):
        trigrams[
            f"{padded[i]} {padded[i+1]} {padded[i+2]}"
        ] += 1


total_words = sum(unigrams.values())
vocab_size = len(unigrams)

print("N-GRAM TRAINING WORDS:", total_words)
print("N-GRAM VOCABULARY:", vocab_size)


# ============================================================
# 5. CONTEXTUAL SCORING
# ============================================================

def score_word(previous, word):

    prev1 = previous[-1] if len(previous) >= 1 else "<s>"
    prev2 = previous[-2] if len(previous) >= 2 else "<s>"

    tri = trigrams.get(
        f"{prev2} {prev1} {word}",
        0
    )

    bi = bigrams.get(
        f"{prev1} {word}",
        0
    )

    tri_base = bigrams.get(
        f"{prev2} {prev1}",
        0
    )

    bi_base = unigrams.get(
        prev1,
        total_words
    )

    tri_prob = (
        (tri + 1) /
        (tri_base + vocab_size)
    )

    bi_prob = (
        (bi + 1) /
        (bi_base + vocab_size)
    )

    return (
        0.7 * math.log(tri_prob)
        +
        0.3 * math.log(bi_prob)
    )


# ============================================================
# 6. EVALUATE ONE SENTENCE
# ============================================================

def evaluate_sentence(truth):

    predicted = []

    evaluated = 0
    correct = 0

    ambiguous_total = 0
    ambiguous_correct = 0

    for word in truth:

        # Word cannot be evaluated if it has no Grascii entry.
        if word not in reverse:
            continue

        grascii = reverse[word][0]

        choices = candidates.get(grascii, [])

        if not choices:
            continue

        scored = []

        for candidate in choices:

            score = score_word(
                predicted,
                candidate
            )

            scored.append(
                (score, candidate)
            )

        prediction = max(
            scored,
            key=lambda x: x[0]
        )[1]

        evaluated += 1

        if prediction == word:
            correct += 1

        if len(choices) > 1:

            ambiguous_total += 1

            if prediction == word:
                ambiguous_correct += 1

        predicted.append(prediction)

    return (
        evaluated,
        correct,
        ambiguous_total,
        ambiguous_correct
    )


# ============================================================
# 7. RUN FULL HELD-OUT TEST
# ============================================================

total_evaluated = 0
total_correct = 0

ambiguous_total = 0
ambiguous_correct = 0

sentences_with_evaluated_words = 0

for truth in test_sentences:

    (
        evaluated,
        correct,
        amb_total,
        amb_correct
    ) = evaluate_sentence(truth)

    if evaluated > 0:
        sentences_with_evaluated_words += 1

    total_evaluated += evaluated
    total_correct += correct

    ambiguous_total += amb_total
    ambiguous_correct += amb_correct


# ============================================================
# 8. RESULTS
# ============================================================

print("\n==========================================")
print("FULL CORPUS NLP EVALUATION")
print("==========================================")

print(
    "TOTAL CORPUS SENTENCES:",
    len(sentences)
)

print(
    "TRAIN SENTENCES:",
    len(train_sentences)
)

print(
    "HELD-OUT TEST SENTENCES:",
    len(test_sentences)
)

print(
    "TEST SENTENCES WITH EVALUATED WORDS:",
    sentences_with_evaluated_words
)

print(
    "TOTAL EVALUATED WORDS:",
    total_evaluated
)

if total_evaluated:

    accuracy = (
        total_correct /
        total_evaluated
    )

    print(
        "OVERALL NLP ACCURACY:",
        round(accuracy, 4)
    )

    print(
        "OVERALL NLP ACCURACY (%):",
        round(accuracy * 100, 2)
    )

if ambiguous_total:

    ambiguous_accuracy = (
        ambiguous_correct /
        ambiguous_total
    )

    print(
        "AMBIGUOUS WORDS:",
        ambiguous_total
    )

    print(
        "AMBIGUOUS WORD ACCURACY:",
        round(ambiguous_accuracy, 4)
    )

    print(
        "AMBIGUOUS WORD ACCURACY (%):",
        round(
            ambiguous_accuracy * 100,
            2
        )
    )

print("==========================================")