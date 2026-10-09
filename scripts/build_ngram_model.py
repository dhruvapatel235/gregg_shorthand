from pathlib import Path
from collections import Counter
import json
import re

CORPUS = Path("data/corpus.txt")
OUT = Path("outputs/ngram_model.json")

def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

unigrams = Counter()
bigrams = Counter()
trigrams = Counter()

sentences = 0

for line in CORPUS.read_text(encoding="utf-8").splitlines():
    tokens = tokenize(line)

    if not tokens:
        continue

    sentences += 1

    padded = ["<s>", "<s>"] + tokens + ["</s>"]

    for w in tokens:
        unigrams[w] += 1

    for i in range(len(padded) - 1):
        bigrams[f"{padded[i]} {padded[i+1]}"] += 1

    for i in range(len(padded) - 2):
        trigrams[
            f"{padded[i]} {padded[i+1]} {padded[i+2]}"
        ] += 1

data = {
    "sentences": sentences,
    "unigrams": dict(unigrams),
    "bigrams": dict(bigrams),
    "trigrams": dict(trigrams),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(data),
    encoding="utf-8"
)

print("sentences:", sentences)
print("unigrams:", len(unigrams))
print("bigrams:", len(bigrams))
print("trigrams:", len(trigrams))
print("saved:", OUT)