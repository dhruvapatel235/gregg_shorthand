"""
Evaluation on sentence-level test data. Reports, SEPARATELY:
  1. recognizer quality in its own target representation (Grascii or English letters): CER / exact
  2. candidate quality: top-1 / top-k accuracy (upper bound for any re-ranker)
  3. English output: WER, word accuracy, ambiguity-resolution accuracy (by ambiguity type),
     sentence exact match for Recognition-only | Dictionary | N-gram | Transformer (+ LM-weight ablation)
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUT
from src.candidates import ambiguity_type, build_candidates
from src.dataset import load_word_image
from src.lexicon import Lexicon
from src.lm_ngram import NGramLM
from src.recognizer import Recognizer
from src.resolver import resolve
from src.utils import cer, wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--target", choices=["grascii", "english"], default="grascii")
    ap.add_argument("--ckpt")
    ap.add_argument("--sentences")
    ap.add_argument("--ngram-corpus")
    ap.add_argument("--noise", type=float, default=0.6, help="degradation strength on test images (0 = none)")
    ap.add_argument("--lm-model", default="distilgpt2")
    ap.add_argument("--beta", type=float, default=0.5, help="LM weight")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max-sentences", type=int, default=0)
    ap.add_argument("--ablation", action="store_true", help="also run the Transformer with several LM weights")
    ap.add_argument("--no-transformer", action="store_true")
    a = ap.parse_args()
    root = Path(a.data_dir)
    run = OUT / a.target
    rows = [json.loads(l) for l in open(a.sentences or root / "test_sentences.jsonl", encoding="utf-8")]
    if a.max_sentences:
        rows = rows[:a.max_sentences]
    rec = Recognizer(a.ckpt or run / "recognizer.pt")
    if rec.target != a.target:
        sys.exit(f"Checkpoint was trained with --target {rec.target}; pass --target {rec.target}")
    lex = Lexicon.load(root, a.target, rec.vocab)
    corpus = Path(a.ngram_corpus or root / "corpus_train.txt").read_text(encoding="utf-8").splitlines()
    lex.set_freq(Counter(w for s in corpus for w in s.split()))
    ngram = NGramLM(corpus)
    key_of = "grascii" if a.target == "grascii" else "words"

    print(f"Target: {a.target} | lexicon keys: {len(lex)} | recognising "
          f"{sum(len(r['words']) for r in rows)} words in {len(rows)} sentences ...")
    data = []
    for r in rows:
        inks = [load_word_image(p, s, a.noise) for p, s in zip(r["images"], r["seeds"])]
        data.append([build_candidates(lp, lex, rec.vocab) for lp in rec.logprobs(inks)])

    refs = [r["words"] for r in rows]
    flat_ref = [w for s in refs for w in s]
    flat_key = [k for r in rows for k in r[key_of]]
    flat_c = [c for cl in data for c in cl]
    amb = [ambiguity_type(c) for c in flat_c]
    n_same, n_rec = amb.count("same_outline"), amb.count("recognition")

    def unk_free(c):  # Recognition-only: no language knowledge at all
        if a.target == "english":
            return c["greedy"]
        ws = lex.words_for(c["greedy"])
        return ws[0] if ws else "<unk>"

    top1 = float(np.mean([c["cands"][0][0] == w for c, w in zip(flat_c, flat_ref)]))
    topk = float(np.mean([w in [x for x, _ in c["cands"]] for c, w in zip(flat_c, flat_ref)]))
    tgt_cer = cer(flat_key, [c["greedy"] for c in flat_c])
    tgt_exact = float(np.mean([c["greedy"] == k for c, k in zip(flat_c, flat_key)]))
    print(f"\n[1] Recognizer ({a.target}) CER {tgt_cer:.3f} | exact {tgt_exact:.3f}")
    print(f"[2] Candidates: top-1 {top1:.3f} | top-k (k<=12) {topk:.3f}   <- upper bound for any re-ranker")
    print(f"    Ambiguous words: {n_same} same-outline (e.g. homophones) + {n_rec} recognition-uncertain "
          f"of {len(flat_ref)}\n[3] English output:")

    systems = {
        "Recognition-only": lambda cl, i: [unk_free(c) for c in cl],
        "Dictionary": lambda cl, i: resolve([c["cands"] for c in cl], None),
        "N-gram LM": lambda cl, i: resolve([c["cands"] for c in cl], ngram, beta=a.beta, beam=a.beam),
    }
    if not a.no_transformer:
        from src.lm_transformer import TransformerLM
        print("    loading Transformer LM:", a.lm_model)
        tlm = TransformerLM(a.lm_model)
        systems["Transformer LM"] = lambda cl, i: resolve([c["cands"] for c in cl], tlm, beta=a.beta, beam=a.beam)
        if a.ablation:
            for b in (0.0, 0.25, 1.0):
                systems[f"  ablation: Transformer beta={b}"] = (
                    lambda cl, i, b=b: resolve([c["cands"] for c in cl], tlm, beta=b, beam=a.beam))

    results, outputs = {}, {}
    for name, fn in systems.items():
        hyps = [fn(cl, i) for i, cl in enumerate(data)]
        outputs[name] = hyps
        flat_hyp = [w for s in hyps for w in s]
        corr = np.array([h == r for h, r in zip(flat_hyp, flat_ref)])
        def acc(kind):
            m = np.array([x == kind for x in amb]) if kind else np.array([x is not None for x in amb])
            return float(corr[m].mean()) if m.any() else float("nan")
        results[name] = {"WER": wer(refs, hyps), "Word acc": float(corr.mean()),
                         "Ambiguity-resolution acc (all)": acc(None),
                         "  same-outline": acc("same_outline"), "  recognition-uncertain": acc("recognition"),
                         "Sentence exact match": float(np.mean([h == r for h, r in zip(hyps, refs)]))}
        r = results[name]
        print(f"{name:34s} WER {r['WER']:.3f} | word acc {r['Word acc']:.3f} | ambig. acc {r['Ambiguity-resolution acc (all)']:.3f} "
              f"(same-outline {r['  same-outline']:.3f}, recog. {r['  recognition-uncertain']:.3f}) | sentence exact {r['Sentence exact match']:.3f}")

    run.mkdir(parents=True, exist_ok=True)
    results["_recognizer_and_candidates"] = {
        "target": a.target, "target_CER": tgt_cer, "target_exact": tgt_exact, "top1": top1, "topk": topk,
        "ambiguous_same_outline": n_same, "ambiguous_recognition": n_rec, "n_words": len(flat_ref),
        "noise": a.noise, "beta": a.beta}
    json.dump(results, open(run / "results.json", "w"), indent=2)
    main_sys = "Transformer LM" if "Transformer LM" in outputs else "N-gram LM"
    with open(run / "error_analysis.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["sentence_id", "position", "truth", f"truth_{a.target}", f"recognized_{a.target}",
                     "recognition_only", "dictionary", "ngram", main_sys, "top5_candidates",
                     "ambiguity_type", "context_helped"])
        n = 0
        for i, (s, cl) in enumerate(zip(refs, data)):
            for j, (w, c) in enumerate(zip(s, cl)):
                d, m = outputs["Dictionary"][i][j], outputs[main_sys][i][j]
                if d != w or m != w:
                    wr.writerow([i, j, w, flat_key[n], c["greedy"], outputs["Recognition-only"][i][j], d,
                                 outputs["N-gram LM"][i][j], m, " ".join(x for x, _ in c["cands"][:5]),
                                 amb[n] or "", "yes" if (d != w and m == w) else
                                 "no (made worse)" if (d == w and m != w) else "no"])
                n += 1
    print("\nSaved:", run / "results.json", "and", run / "error_analysis.csv")


if __name__ == "__main__":
    main()
