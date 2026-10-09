import numpy as np
from PIL import Image
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.manifest import read_manifest
from src.recognizer import Recognizer
from src.decode import prefix_beam_search


DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_full_open/recognizer.pt"

print("Loading manifest...")

df, meta = read_manifest(DATA)
test = df[df["split"] == "test"].reset_index(drop=True)

print("Test samples:", len(test))

print("Loading recognizer...")

rec = Recognizer(CKPT)

print("Recognizer loaded.")


# ------------------------------------------------------------
# Load test images
# ------------------------------------------------------------

images = []
true_labels = []

for _, row in test.iterrows():

    img = np.array(
        Image.open(row["abs_path"]).convert("L")
    )

    ink = 255 - img

    images.append(ink.astype(np.uint8))
    true_labels.append(str(row["grascii"]))


print("Running recognition...")

logprobs = rec.logprobs(images)


# ------------------------------------------------------------
# Character-level alignment
# ------------------------------------------------------------

def edit_alignment(ref, hyp):
    """
    Returns aligned character pairs:
    (reference_char, predicted_char)

    None means insertion/deletion.
    """

    n = len(ref)
    m = len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            cost = 0 if ref[i - 1] == hyp[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    pairs = []

    i = n
    j = m

    while i > 0 or j > 0:

        if (
            i > 0
            and j > 0
            and dp[i][j]
            == dp[i - 1][j - 1]
            + (0 if ref[i - 1] == hyp[j - 1] else 1)
        ):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1

        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((ref[i - 1], "<DEL>"))
            i -= 1

        else:
            pairs.append(("<INS>", hyp[j - 1]))
            j -= 1

    pairs.reverse()

    return pairs


# ------------------------------------------------------------
# Greedy predictions
# ------------------------------------------------------------

confusions = Counter()
true_chars = Counter()
correct_chars = Counter()

word_errors = []

for idx, (lp, true) in enumerate(zip(logprobs, true_labels)):

    beams = prefix_beam_search(
        lp,
        rec.vocab,
        beam=50,
        k=50,
        prune=-12.0
    )

    pred = beams[0][0]

    if pred != true:
        word_errors.append(
            (true, pred)
        )

    alignment = edit_alignment(true, pred)

    for ref_char, pred_char in alignment:

        if ref_char != "<INS>":
            true_chars[ref_char] += 1

        if ref_char == pred_char:
            correct_chars[ref_char] += 1

        elif ref_char != "<INS>":
            confusions[(ref_char, pred_char)] += 1


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 70)
print("GRASCII CHARACTER ERROR ANALYSIS")
print("=" * 70)

print()
print("Most common character confusions:")

for (ref, pred), count in confusions.most_common(30):

    print(
        f"{ref:>6} -> {pred:<6} : {count}"
    )


print()
print("Character accuracy:")

for char, count in true_chars.most_common():

    acc = correct_chars[char] / count

    print(
        f"{char:>6} : "
        f"{acc:.3f} "
        f"({correct_chars[char]}/{count})"
    )


print()
print("Most common wrong words:")

word_counter = Counter(word_errors)

for (true, pred), count in word_counter.most_common(30):

    print(
        f"{true:<15} -> {pred:<15} : {count}"
    )


print()
print("Total wrong words:", len(word_errors))
print("=" * 70)