import numpy as np
from PIL import Image
from collections import defaultdict
import json
import re

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

df, meta = read_manifest(DATA)
test = df[df["split"] == "test"].reset_index(drop=True)
test["english"] = test["english"].str.lower()

test_words = set(test["english"])

word_to_image = {}

for _, row in test.iterrows():
    word_to_image[row["english"]] = row["abs_path"]


# ============================================================
# True English -> true Grascii
# ============================================================

true_grascii = {}

for _, row in test.iterrows():
    true_grascii[row["english"]] = str(row["grascii"])


# ============================================================
# Grascii -> English dictionary
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


# ============================================================
# Find synthetic corpus runs
# ============================================================

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


runs = runs[:MAX_RUNS]

print("Runs selected:", len(runs))


# ============================================================
# Load models
# ============================================================

rec = Recognizer(CKPT)

decoder = SentenceDecoder(
    NGRAM,
    beam_size=10
)


# ============================================================
# Metrics
# ============================================================

stats = {
    "top1": {"correct": 0, "total": 0},
    "dictionary": {"correct": 0, "total": 0},
    "ngram": {"correct": 0, "total": 0},
}

sentence_stats = {
    "top1": 0,
    "dictionary": 0,
    "ngram": 0,
}

examples = []


# ============================================================
# Process runs
# ============================================================

for run_id, words in enumerate(runs, 1):

    print(
        f"\nRUN {run_id}/{len(runs)}:",
        " ".join(words)
    )

    images = []
    truths = []

    for word in words:

        img = np.array(
            Image.open(
                word_to_image[word]
            ).convert("L")
        )

        ink = 255 - img

        images.append(
            ink.astype(np.uint8)
        )

        truths.append(word)


    # --------------------------------------------------------
    # Recognizer
    # --------------------------------------------------------

    logprobs = rec.logprobs(images)

    top1_words = []
    dictionary_words = []
    ngram_candidate_lists = []

    for lp in logprobs:

        beams = prefix_beam_search(
            lp,
            rec.vocab,
            beam=50,
            k=50,
            prune=-12.0
        )

        # ----------------------------------------------------
        # A: pure recognizer Top-1
        # ----------------------------------------------------

        top1_g = beams[0][0]

        # ----------------------------------------------------
        # B: dictionary constrained
        # ----------------------------------------------------

        dictionary_g = None

        for candidate_g, score in beams:

            if candidate_g in candidates:
                dictionary_g = candidate_g
                break

        if dictionary_g is None:
            dictionary_g = top1_g

        
        # English candidates for N-gram decoder, preserving CTC scores
        english_scores = {}

        for candidate_g, score in beams:
            for word in candidates.get(candidate_g, []):
                # Keep the best visual score if multiple Grascii
                # candidates map to the same English word.
                if word not in english_scores or score > english_scores[word]:
                    english_scores[word] = float(score)

        # Fallback if no dictionary words match any beam
        if not english_scores:
            fallback_words = candidates.get(dictionary_g, ["[UNKNOWN]"])
            english_scores = {word: 0.0 for word in fallback_words}

        english_candidates = [
            {"word": word, "visual_score": score}
            for word, score in english_scores.items()
        ]

        top1_words.append(top1_g)
        dictionary_words.append(dictionary_g)
        ngram_candidate_lists.append(english_candidates)


    # --------------------------------------------------------
    # C: N-gram contextual decoding
    # --------------------------------------------------------

    ngram_words = decoder.decode(
        ngram_candidate_lists
    )

    if len(ngram_words) != len(truths):
        raise ValueError(
            f"Prediction length mismatch in run {run_id}: "
            f"truths={len(truths)}, "
            f"candidates={len(ngram_candidate_lists)}, "
            f"predictions={len(ngram_words)}"
        ) 


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    run_correct = {
        "top1": True,
        "dictionary": True,
        "ngram": True,
    }

    for i, true_word in enumerate(truths):

        true_g = true_grascii[true_word]

        # Top-1 Grascii
        if top1_words[i] == true_g:
            stats["top1"]["correct"] += 1
        else:
            run_correct["top1"] = False

        # Dictionary Grascii
        if dictionary_words[i] == true_g:
            stats["dictionary"]["correct"] += 1
        else:
            run_correct["dictionary"] = False

        # N-gram English
        if (
            i < len(ngram_words)
            and ngram_words[i] == true_word
        ):
            stats["ngram"]["correct"] += 1
        else:
            run_correct["ngram"] = False

        for key in stats:
            stats[key]["total"] += 1


    for key in sentence_stats:

        if run_correct[key]:
            sentence_stats[key] += 1


    # --------------------------------------------------------
    # Save useful examples
    # --------------------------------------------------------

    
    examples.append(
        {
            "run_id": run_id,
            "true": truths,
            "true_grascii": [
                true_grascii[w] for w in truths
            ],
            "top1_grascii": top1_words,
            "dictionary_grascii": dictionary_words,
            "ngram_english": ngram_words,
        }
    )


# ============================================================
# Final results
# ============================================================

print("\n")
print("=" * 70)
print("DECODER ABLATION RESULTS")
print("=" * 70)

for key in ["top1", "dictionary", "ngram"]:

    acc = (
        stats[key]["correct"]
        / stats[key]["total"]
    )

    print(
        f"{key:12s}: "
        f"{acc:.4f} "
        f"({acc * 100:.2f}%)"
    )


print("\nSENTENCE/RUN ACCURACY")

for key in ["top1", "dictionary", "ngram"]:

    acc = (
        sentence_stats[key]
        / len(runs)
    )

    print(
        f"{key:12s}: "
        f"{acc:.4f} "
        f"({acc * 100:.2f}%)"
    )


print("\n")
print("=" * 70)
print("EXAMPLES")
print("=" * 70)

for ex in examples[:10]:

    print("\nTRUE:")
    print(" ".join(ex["true"]))

    print("TOP-1 GRASCII:")
    print(" ".join(ex["top1_grascii"]))

    print("DICTIONARY GRASCII:")
    print(" ".join(ex["dictionary_grascii"]))

    print("N-GRAM ENGLISH:")
    print(" ".join(ex["ngram_english"]))


# ============================================================
# Save results
# ============================================================

output = {
    "runs": len(runs),
    "words": stats["top1"]["total"],
    "word_accuracy": {},
    "sentence_accuracy": {},
    "examples": examples,
}

for key in ["top1", "dictionary", "ngram"]:

    output["word_accuracy"][key] = (
        stats[key]["correct"]
        / stats[key]["total"]
    )

    output["sentence_accuracy"][key] = (
        sentence_stats[key]
        / len(runs)
    )


with open(
    "results/phase2_full_open/"
    "decoder_ablation_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )


print("\nSaved:")
print(
    "results/phase2_full_open/"
    "decoder_ablation_results.json"
)