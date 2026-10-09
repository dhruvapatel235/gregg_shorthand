import re
import random
from pathlib import Path
from collections import defaultdict

LEXICON = Path("data/grascii/dictionary_preanniversary.tsv")
CORPUS = Path("data/corpus.txt")

TRAIN_RATIO = 0.80
SEED = 42


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


# ============================================================
# 1. LOAD GRASCII -> ENGLISH CANDIDATES
# ============================================================

candidates = defaultdict(list)

for line in LEXICON.read_text(encoding="utf-8").splitlines():

    if not line or line.startswith("#") or "\t" not in line:
        continue

    grascii, word = line.split("\t", 1)
    word = word.strip().lower()

    if word:
        candidates[grascii].append(word)


reverse = defaultdict(list)

for grascii, words in candidates.items():
    for word in words:
        reverse[word].append(grascii)


# ============================================================
# 2. LOAD SAME CORPUS
# ============================================================

sentences = []

for line in CORPUS.read_text(encoding="utf-8").splitlines():

    tokens = tokenize(line)

    if tokens:
        sentences.append(tokens)


# ============================================================
# 3. SAME 80/20 SPLIT AS NLP MODEL
# ============================================================

random.seed(SEED)

indices = list(range(len(sentences)))
random.shuffle(indices)

split = int(len(indices) * TRAIN_RATIO)

test_indices = indices[split:]
test_sentences = [sentences[i] for i in test_indices]


# ============================================================
# 4. BASELINE EVALUATION
# ============================================================
#
# Baseline does NOT use contextual NLP.
#
# For every Grascii form:
# choose the FIRST English candidate in the dictionary.
#
# This gives us a simple reference point against which
# the N-gram model can be compared.
# ============================================================

total_evaluated = 0
total_correct = 0

ambiguous_total = 0
ambiguous_correct = 0

sentences_with_evaluated_words = 0

examples = []


for truth in test_sentences:

    sentence_evaluated = 0

    for word in truth:

        if word not in reverse:
            continue

        grascii = reverse[word][0]
        choices = candidates.get(grascii, [])

        if not choices:
            continue

        prediction = choices[0]

        total_evaluated += 1
        sentence_evaluated += 1

        if prediction == word:
            total_correct += 1

        if len(choices) > 1:

            ambiguous_total += 1

            if prediction == word:
                ambiguous_correct += 1

    if sentence_evaluated > 0:
        sentences_with_evaluated_words += 1


# ============================================================
# 5. RESULTS
# ============================================================

print("\n==========================================")
print("BASELINE NLP EVALUATION")
print("==========================================")

print(
    "TOTAL CORPUS SENTENCES:",
    len(sentences)
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

    accuracy = total_correct / total_evaluated

    print(
        "BASELINE OVERALL ACCURACY:",
        round(accuracy, 4)
    )

    print(
        "BASELINE OVERALL ACCURACY (%):",
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
        "BASELINE AMBIGUOUS ACCURACY:",
        round(ambiguous_accuracy, 4)
    )

    print(
        "BASELINE AMBIGUOUS ACCURACY (%):",
        round(
            ambiguous_accuracy * 100,
            2
        )
    )

print("==========================================")