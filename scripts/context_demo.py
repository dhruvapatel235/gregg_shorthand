import json
import math
import re
from pathlib import Path
from collections import defaultdict

LEXICON = Path("data/grascii/dictionary_preanniversary.tsv")
MODEL = Path("outputs/ngram_model.json")

def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

# Load dictionary
candidates = defaultdict(list)

for line in LEXICON.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#") or "\t" not in line:
        continue

    grascii, word = line.split("\t", 1)
    candidates[grascii].append(word.strip().lower())

# Load language model
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

    return 0.7 * math.log(tri_prob) + 0.3 * math.log(bi_prob)

def choose(context, grascii):
    choices = candidates.get(grascii, [])

    print(f"\n{grascii} -> {choices}")

    scored = []

    for word in choices:
        score = score_word(context, word)
        scored.append((score, word))

    for score, word in sorted(scored, reverse=True):
        print(f"  {word:15s} {score:.4f}")

    if scored:
        return max(scored)[1]

    return "[UNKNOWN]"

tests = [
    (["the"], "AR"),
    (["a"], "AR"),
    (["he"], "AU"),
    (["send"], "AU"),
    (["an"], "ADE"),
    (["the"], "ATM"),
    (["to"], "ATMIZ"),
]

for context, grascii in tests:
    result = choose(context, grascii)
    print("CHOSEN:", result)