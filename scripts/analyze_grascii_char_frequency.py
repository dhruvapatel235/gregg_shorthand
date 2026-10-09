import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.manifest import read_manifest
from src.vocab import Vocab


DATA = "data/phase2_real_full/manifest_open"


print("Loading manifest...")

df, meta = read_manifest(DATA)

train = df[df["split"] == "train"].reset_index(drop=True)
val = df[df["split"] == "val"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)

print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))


# ------------------------------------------------------------
# Character frequency
# ------------------------------------------------------------

train_counter = Counter()
val_counter = Counter()
test_counter = Counter()

for s in train["grascii"]:
    train_counter.update(str(s))

for s in val["grascii"]:
    val_counter.update(str(s))

for s in test["grascii"]:
    test_counter.update(str(s))


all_chars = sorted(
    set(train_counter)
    | set(val_counter)
    | set(test_counter)
)


# ------------------------------------------------------------
# Character error rates from previous analysis
# ------------------------------------------------------------

# These are the characters identified as weak in the
# previous recognition analysis.
weak_chars = {
    "Z", "V", "(", ")", "&", "|",
    "I", "D", "H", "B"
}


print()
print("=" * 90)
print("GRASCII CHARACTER FREQUENCY")
print("=" * 90)

print(
    f"{'CHAR':<8}"
    f"{'TRAIN':>10}"
    f"{'VAL':>10}"
    f"{'TEST':>10}"
    f"{'TOTAL':>10}"
    f"{'TRAIN %':>12}"
)

print("-" * 90)

total_train_chars = sum(train_counter.values())

for char in sorted(
    all_chars,
    key=lambda c: train_counter[c]
):

    train_n = train_counter[char]
    val_n = val_counter[char]
    test_n = test_counter[char]
    total_n = train_n + val_n + test_n

    train_pct = (
        100 * train_n / total_train_chars
        if total_train_chars else 0
    )

    marker = "  <-- WEAK" if char in weak_chars else ""

    print(
        f"{char:<8}"
        f"{train_n:>10}"
        f"{val_n:>10}"
        f"{test_n:>10}"
        f"{total_n:>10}"
        f"{train_pct:>11.3f}%"
        f"{marker}"
    )


# ------------------------------------------------------------
# Word-level character coverage
# ------------------------------------------------------------

print()
print("=" * 90)
print("RARE CHARACTER TRAINING COVERAGE")
print("=" * 90)

for char in sorted(all_chars, key=lambda c: train_counter[c]):

    words = [
        str(x)
        for x in train["grascii"]
        if char in str(x)
    ]

    if not words:
        continue

    print(
        f"{repr(char):>6} : "
        f"{len(words):>5} training words | "
        f"{train_counter[char]:>5} occurrences"
    )


# ------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------

vocab = Vocab.from_labels(train["grascii"])

print()
print("=" * 90)
print("MODEL VOCABULARY")
print("=" * 90)

print("Vocabulary size:", len(vocab.chars))
print("Characters:", "".join(vocab.chars))

print()
print("=" * 90)
print("DONE")
print("=" * 90)
