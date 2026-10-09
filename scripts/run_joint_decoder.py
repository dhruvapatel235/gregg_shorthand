import numpy as np
from PIL import Image
from collections import defaultdict
import json
import math

from src.manifest import read_manifest
from src.recognizer import Recognizer
from src.decode import prefix_beam_search


DATA = "data/phase2_real_full/manifest_open"
CKPT = "outputs/phase2_full_open/recognizer.pt"
DICTIONARY = "data/grascii/dictionary_preanniversary.tsv"
NGRAM = "outputs/ngram_model.json"

MAX_RUNS = 2000
MIN_RUN = 2
SEED = 42

# Try several visual/context balances.
WEIGHTS = [
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
]


# ============================================================
# Load data
# ============================================================

df, meta = read_manifest(DATA)

test = df[df["split"] == "test"].reset_index(drop=True)
test["english"] = test["english"].str.lower()

test_words = set(test["english"])

word_to_image = {}
true_grascii = {}

for _, row in test.iterrows():
    word = row["english"]
    word_to_image[word] = row["abs_path"]
    true_grascii[word] = str(row["grascii"])


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

        for word in parts[1].strip().lower().split("|"):

            if word and word not in candidates[grascii]:
                candidates[grascii].append(word)


# ============================================================
# Load N-gram model
# ============================================================

with open(NGRAM, "r", encoding="utf-8") as f:
    ng = json.load(f)

unigrams = ng["unigrams"]
bigrams = ng["bigrams"]
trigrams = ng["trigrams"]

total_uni = sum(unigrams.values())
total_bi = sum(bigrams.values())
total_tri = sum(trigrams.values())
vocab_size = len(unigrams)


def ngram_score(history, word):

    word = word.lower()

    uni = unigrams.get(word, 0)

    bi = 0
    tri = 0

    if len(history) >= 1:
        bi = bigrams.get(
            f"{history[-1]}\t{word}",
            0
        )

    if len(history) >= 2:
        tri = trigrams.get(
            f"{history[-2]}\t{history[-1]}\t{word}",
            0
        )

    uni_p = (uni + 1) / (
        total_uni + vocab_size
    )

    bi_p = (bi + 1) / (
        total_bi + vocab_size
    )

    tri_p = (tri + 1) / (
        total_tri + vocab_size
    )

    return (
        0.1 * math.log(uni_p)
        + 0.3 * math.log(bi_p)
        + 0.6 * math.log(tri_p)
    )

# ============================================================
# Load the exact evaluation runs from the oracle experiment
# ============================================================

ORACLE_RUNS_PATH = (
    "results/phase2_full_open/"
    "oracle_context_nlp_results_2000_random_seed42.json"
)

with open(ORACLE_RUNS_PATH, "r", encoding="utf-8") as f:
    oracle_results = json.load(f)

runs = oracle_results["selected_runs"]

# Normalize runs into lists of lowercase English words.
normalized_runs = []

for run in runs:
    if isinstance(run, dict):
        # Handle dictionary-form runs if present.
        for key in ("words", "run", "tokens"):
            if key in run:
                run = run[key]
                break

    normalized_runs.append(
        [str(word).lower() for word in run]
    )

runs = normalized_runs

print("Evaluation source:", ORACLE_RUNS_PATH)
print("Runs selected:", len(runs))
print("Words selected:", sum(len(run) for run in runs))
print("Using the oracle experiment's exact selected runs.")




# ============================================================
# Recognizer
# ============================================================

rec = Recognizer(CKPT)


# ============================================================
# Joint decoding
# ============================================================

def decode_joint(candidate_data, alpha, beta):

    # candidate_data:
    #
    # [
    #   [(english_word, ctc_score), ...],
    #   ...
    # ]

    beams = [(0.0, [])]

    for position_candidates in candidate_data:

        next_beams = []

        for previous_score, history in beams:

            for word, ctc_score in position_candidates:

                context_score = ngram_score(
                    history,
                    word
                )

                score = (
                    previous_score
                    + alpha * ctc_score
                    + beta * context_score
                )

                next_beams.append(
                    (
                        score,
                        history + [word]
                    )
                )

        next_beams.sort(
            key=lambda x: x[0],
            reverse=True
        )

        beams = next_beams[:10]

    if not beams:
        return []

    return beams[0][1]


# ============================================================
# Results
# ============================================================

# Raw recognizer baseline: top-1 Grascii -> dictionary -> English
raw_results = {
    "correct": 0,
    "total": 0,
    "runs_correct": 0
}

results = {}

for alpha, beta in WEIGHTS:

    key = f"{alpha:.1f}_{beta:.1f}"

    results[key] = {
        "correct": 0,
        "total": 0,
        "runs_correct": 0
    }

# ============================================================
# Process runs
# ============================================================

for run_id, words in enumerate(runs, 1):

    print(
        f"\nRUN {run_id}/{len(runs)}:",
        " ".join(words)
    )

    images = []

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

    logprobs = rec.logprobs(images)


    candidate_data = []
    raw_predictions = []

    for lp in logprobs:

        beams = prefix_beam_search(
            lp,
            rec.vocab,
            beam=50,
            k=50,
            prune=-12.0
        )

        # Raw top-1 Grascii prediction, before context.
        raw_grascii = beams[0][0] if beams else ""
        raw_english_candidates = candidates.get(raw_grascii, [])

        raw_prediction = (
            raw_english_candidates[0]
            if raw_english_candidates
            else "[UNKNOWN]"
        )

        raw_predictions.append(raw_prediction)

        # Build dictionary candidates for joint decoding.
        word_candidates = []

        for grascii, ctc_score in beams:

            for word in candidates.get(grascii, []):

                word_candidates.append((word, ctc_score))

        # Keep the best CTC score for each English word.
        best_by_word = {}

        for word, score in word_candidates:

            if (
                word not in best_by_word
                or score > best_by_word[word]
            ):
                best_by_word[word] = score

        word_candidates = list(best_by_word.items())

        if not word_candidates:
            word_candidates = [("[UNKNOWN]", -20.0)]

        candidate_data.append(word_candidates)

    if len(raw_predictions) != len(words):
        raise ValueError(
            f"Run {run_id}: expected {len(words)} raw predictions, "
            f"got {len(raw_predictions)}"
        )

    
    # Evaluate raw top-1 baseline for this run.
    raw_run_correct = True

    for true_word, pred_word in zip(words, raw_predictions):
        raw_results["total"] += 1

        if pred_word == true_word:
            raw_results["correct"] += 1
        else:
            raw_run_correct = False

    if raw_run_correct:
        raw_results["runs_correct"] += 1


    # --------------------------------------------------------
    # Evaluate every weight setting
    # --------------------------------------------------------

    for alpha, beta in WEIGHTS:

        key = f"{alpha:.1f}_{beta:.1f}"

        prediction = decode_joint(
            candidate_data,
            alpha,
            beta
        )

        if len(prediction) != len(words):
            raise ValueError(
                f"Run {run_id}: expected {len(words)} predictions, "
                f"got {len(prediction)}"
            )

        run_correct = True

        for true_word, pred_word in zip(
            words,
            prediction
        ):

            results[key]["total"] += 1

            if pred_word == true_word:

                results[key]["correct"] += 1

            else:

                run_correct = False

        if run_correct:
            results[key]["runs_correct"] += 1


# ============================================================
# Print results
# ============================================================

# ============================================================
# Print results
# ============================================================

print("\n")
print("=" * 70)
print("RAW TOP-1 RECOGNIZER BASELINE")
print("=" * 70)

print("Runs:", len(runs))
print("Words:", raw_results["total"])

if raw_results["total"]:
    raw_word_acc = raw_results["correct"] / raw_results["total"]
    print(f"Word accuracy: {raw_word_acc:.4f} ({raw_word_acc * 100:.2f}%)")

if runs:
    raw_run_acc = raw_results["runs_correct"] / len(runs)
    print(f"Run accuracy: {raw_run_acc:.4f} ({raw_run_acc * 100:.2f}%)")

print("\n" + "=" * 70)
print("JOINT CTC + N-GRAM RESULTS")
print("=" * 70)

total_words = sum(
    results[k]["total"]
    for k in results
)

print("Runs:", len(runs))
print("Words:", total_words)

for alpha, beta in WEIGHTS:

    key = f"{alpha:.1f}_{beta:.1f}"

    word_acc = (
        results[key]["correct"]
        / results[key]["total"]
    )

    run_acc = (
        results[key]["runs_correct"]
        / len(runs)
    )

    print(
        f"\nCTC={alpha:.1f}, "
        f"NGRAM={beta:.1f}"
    )

    print(
        f"Word accuracy: "
        f"{word_acc:.4f} "
        f"({word_acc * 100:.2f}%)"
    )

    print(
        f"Run accuracy: "
        f"{run_acc:.4f} "
        f"({run_acc * 100:.2f}%)"
    )



# ============================================================
# Save results and selected evaluation runs
# ============================================================

out = {
    "seed": SEED,
    "max_runs": MAX_RUNS,
    "min_run": MIN_RUN,
    "total_eligible_runs": len(runs),
    "selected_runs": runs,
    "runs": len(runs),
    "words": sum(len(run) for run in runs),
    "raw_baseline": {
    "correct_words": raw_results["correct"],
    "total_words": raw_results["total"],
    "correct_runs": raw_results["runs_correct"],
    "total_runs": len(runs),
    "word_accuracy": (
        raw_results["correct"] / raw_results["total"]
        if raw_results["total"] else 0.0
    ),
    "run_accuracy": (
        raw_results["runs_correct"] / len(runs)
        if runs else 0.0
    )
    },
    "weights": {}
}

for alpha, beta in WEIGHTS:
    key = f"{alpha:.1f}_{beta:.1f}"

    total = results[key]["total"]

    out["weights"][key] = {
        "ctc_weight": alpha,
        "ngram_weight": beta,
        "correct_words": results[key]["correct"],
        "total_words": total,
        "correct_runs": results[key]["runs_correct"],
        "total_runs": len(runs),
        "word_accuracy": (
            results[key]["correct"] / total
            if total else 0.0
        ),
        "run_accuracy": (
            results[key]["runs_correct"] / len(runs)
            if runs else 0.0
        )
    }

output_path = (
    "results/phase2_full_open/"
    "joint_decoder_results_matched_oracle_seed42.json"
)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("\nSaved:", output_path)