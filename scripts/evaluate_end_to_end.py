import numpy as np
from PIL import Image
from pathlib import Path
import re
import json
import math
from collections import defaultdict

from src.manifest import read_manifest
from src.recognizer import Recognizer
from src.decode import prefix_beam_search
from sentence_decoder import SentenceDecoder


DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_full_open/recognizer.pt"
DICTIONARY = "data/grascii/dictionary_preanniversary.tsv"
NGRAM = "outputs/ngram_model.json"

MAX_RUNS = 200
MIN_RUN = 2


# ============================================================
# Load test data
# ============================================================

print("Loading manifest...")

df, meta = read_manifest(DATA)

test = df[df["split"] == "test"].reset_index(drop=True)
test["english"] = test["english"].str.lower()

test_words = set(test["english"])

word_to_image = {}

for _, row in test.iterrows():
    word_to_image[row["english"]] = row["abs_path"]

print("Test vocabulary:", len(test_words))


# ============================================================
# Load Grascii -> English dictionary
# ============================================================

candidates = defaultdict(list)

with open(DICTIONARY, "r", encoding="utf-8") as f:

    for line in f:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")

        if len(parts) < 2:
            continue

        grascii = parts[0].strip()

        words = parts[1].strip().lower().split("|")

        for word in words:

            if word and word not in candidates[grascii]:
                candidates[grascii].append(word)

print("Dictionary Grascii keys:", len(candidates))


# ============================================================
# Load n-gram model
# ============================================================

with open(NGRAM, "r", encoding="utf-8") as f:
    ng = json.load(f)

unigrams = ng["unigrams"]
bigrams = ng["bigrams"]
trigrams = ng["trigrams"]

total_uni = sum(unigrams.values())
vocab_size = len(unigrams)

total_bi = sum(bigrams.values())
total_tri = sum(trigrams.values())


def ngram_score(history, word):

    word = word.lower()

    if len(history) >= 2:

        a = history[-2]
        b = history[-1]

        tri_key = f"{a}\t{b}\t{word}"
        bi_key = f"{b}\t{word}"

        tri = trigrams.get(tri_key, 0)
        bi = bigrams.get(bi_key, 0)

    elif len(history) == 1:

        b = history[-1]

        tri = 0
        bi = bigrams.get(f"{b}\t{word}", 0)

    else:

        tri = 0
        bi = 0

    uni = unigrams.get(word, 0)

    tri_p = (tri + 1) / (total_tri + vocab_size)
    bi_p = (bi + 1) / (total_bi + vocab_size)
    uni_p = (uni + 1) / (total_uni + vocab_size)

    return (
        0.6 * math.log(tri_p)
        + 0.3 * math.log(bi_p)
        + 0.1 * math.log(uni_p)
    )


def choose_context(history, words):

    if not words:
        return "[UNKNOWN]"

    return max(
        words,
        key=lambda w: ngram_score(history, w)
    )


# ============================================================
# Find corpus runs made entirely from test words
# ============================================================

print("Searching corpus for test-image sentence runs...")

runs = []

with open("data/corpus.txt", "r", encoding="utf-8") as f:

    for line in f:

        words = re.findall(
            r"[a-z]+(?:'[a-z]+)?",
            line.lower()
        )

        current = []

        for word in words:

            if word in test_words:

                current.append(word)

            else:

                if len(current) >= MIN_RUN:
                    runs.append(current)

                current = []

        if len(current) >= MIN_RUN:
            runs.append(current)


print("Candidate runs:", len(runs))

runs = runs[:MAX_RUNS]

print("Runs selected:", len(runs))

if not runs:

    raise RuntimeError(
        "No usable test-word runs found."
    )


# ============================================================
# Load recognizer
# ============================================================

print("Loading recognizer...")

rec = Recognizer(CKPT)
decoder = SentenceDecoder(
    NGRAM,
    beam_size=10
)

print("Recognizer loaded.")


# ============================================================
# Metrics
# ============================================================

total_words = 0

grascii_correct = 0
english_correct = 0
sentence_correct = 0

all_results = []


# ============================================================
# Process runs
# ============================================================

for run_id, words in enumerate(runs, 1):

    print()
    print("=" * 60)
    print(f"RUN {run_id}/{len(runs)}")
    print("TRUE:", " ".join(words))

    # --------------------------------------------------------
    # Load all images in this run
    # --------------------------------------------------------

    images = []
    labels = []

    for word in words:

        image_path = word_to_image[word]

        img = np.array(
            Image.open(image_path).convert("L")
        )

        ink = 255 - img

        images.append(
            ink.astype(np.uint8)
        )

        labels.append(word)


    # --------------------------------------------------------
    # Recognizer
    # --------------------------------------------------------

    logprobs = rec.logprobs(images)

    predicted_grascii = []
    predicted_candidates = []

    for lp in logprobs:

        beams = prefix_beam_search(
            lp,
            rec.vocab,
            beam=50,
            k=50,
            prune=-12.0
        )

        chosen = None

        # Lexicon-constrained decoding
        for candidate, score in beams:

            if candidate in candidates:

                chosen = candidate
                break

        # Fallback
        if chosen is None:

            chosen = beams[0][0]

        predicted_grascii.append(chosen)

        predicted_candidates.append(
            candidates.get(chosen, [])
        )


    # --------------------------------------------------------
    # Context-aware English decoding
    # --------------------------------------------------------

    predicted_english = decoder.decode(
        predicted_candidates
    )


    print(
        "GRASCII:",
        " ".join(predicted_grascii)
    )

    print(
        "PRED:",
        " ".join(predicted_english)
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    run_correct = True

    for true_word, pred_g, pred_e in zip(
        words,
        predicted_grascii,
        predicted_english
    ):

        row = test[
            test["english"] == true_word
        ]

        if len(row) > 0:

            true_g = str(
                row.iloc[0]["grascii"]
            )

        else:

            true_g = ""

        if pred_g == true_g:
            grascii_correct += 1

        if pred_e == true_word:

            english_correct += 1

        else:

            run_correct = False

        total_words += 1


    if run_correct:
        sentence_correct += 1


    all_results.append(
        {
            "true": words,
            "grascii": predicted_grascii,
            "predicted": predicted_english
        }
    )


# ============================================================
# Final results
# ============================================================

grascii_acc = (
    grascii_correct / total_words
)

english_acc = (
    english_correct / total_words
)

sentence_acc = (
    sentence_correct / len(runs)
)


print()
print("=" * 60)
print("END-TO-END RESULTS")
print("=" * 60)

print("Runs:", len(runs))
print("Words:", total_words)

print(
    f"Grascii exact accuracy: "
    f"{grascii_acc:.4f}"
)

print(
    f"English word accuracy:  "
    f"{english_acc:.4f}"
)

print(
    f"Sentence/run accuracy:  "
    f"{sentence_acc:.4f}"
)

print("=" * 60)


# ============================================================
# Save JSON
# ============================================================

out_dir = Path(
    "results/phase2_full_open"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    out_dir / "end_to_end_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "runs": len(runs),
            "words": total_words,
            "grascii_exact_accuracy": grascii_acc,
            "english_word_accuracy": english_acc,
            "sentence_accuracy": sentence_acc,
            "examples": all_results
        },
        f,
        indent=2
    )


# ============================================================
# Save summary
# ============================================================

with open(
    out_dir / "version3_results.txt",
    "a",
    encoding="utf-8"
) as f:

    f.write("\n\n")
    f.write(
        "Version 3 - End-to-End Gregg to English\n"
    )

    f.write(
        f"- Sentence/phrase runs: {len(runs)}\n"
    )

    f.write(
        f"- Words evaluated: {total_words}\n"
    )

    f.write(
        f"- Grascii exact accuracy: "
        f"{grascii_acc:.4f} "
        f"({grascii_acc * 100:.2f}%)\n"
    )

    f.write(
        f"- English word accuracy: "
        f"{english_acc:.4f} "
        f"({english_acc * 100:.2f}%)\n"
    )

    f.write(
        f"- Sentence/run accuracy: "
        f"{sentence_acc:.4f} "
        f"({sentence_acc * 100:.2f}%)\n"
    )

    f.write(
        "- Note: synthetic sentence reconstruction "
        "using held-out test word images; document "
        "segmentation is not included.\n"
    )


print()
print("Saved:")
print(
    "results/phase2_full_open/end_to_end_results.json"
)
print(
    "results/phase2_full_open/version3_results.txt"
)