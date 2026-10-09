import re
from pathlib import Path
from collections import defaultdict

LEXICON = Path("data/grascii/dictionary_preanniversary.tsv")
CORPUS = Path("data/corpus.txt")

candidates = defaultdict(list)

for line in LEXICON.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#") or "\t" not in line:
        continue

    grascii, word = line.split("\t", 1)
    word = word.strip().lower()

    if word:
        candidates[grascii].append(word)

# Build reverse dictionary ONCE
reverse = {}

for g, words in candidates.items():
    for word in words:
        reverse.setdefault(word, []).append(g)

total_lines = 0
usable_lines = 0
skipped_lines = 0
usable_words = 0
skipped_words = 0

for line in CORPUS.read_text(encoding="utf-8").splitlines():

    truth = re.findall(r"[a-z]+(?:'[a-z]+)?", line.lower())

    if not truth:
        continue

    total_lines += 1

    missing = [word for word in truth if word not in reverse]

    if missing:
        skipped_lines += 1
        skipped_words += len(missing)
    else:
        usable_lines += 1
        usable_words += len(truth)

print("TOTAL CORPUS LINES:", total_lines)
print("FULLY USABLE LINES:", usable_lines)
print("SKIPPED LINES:", skipped_lines)
print("USABLE WORDS:", usable_words)
print("MISSING WORD OCCURRENCES:", skipped_words)

if total_lines:
    print("LINE COVERAGE:", round(usable_lines / total_lines * 100, 2), "%")