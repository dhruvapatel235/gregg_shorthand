"""Gregg shorthand document (image or PDF) -> English text (+ the recognized Grascii when --target grascii)."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, OUT
from src.lexicon import Lexicon
from src.pipeline import GreggTranscriber
from src.recognizer import Recognizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="scanned page (png/jpg) or PDF written in Gregg shorthand")
    ap.add_argument("--data-dir", default=str(DATA / "gregg"), help="folder made by prepare_manifest.py")
    ap.add_argument("--target", choices=["grascii", "english"], default="grascii")
    ap.add_argument("--ckpt")
    ap.add_argument("--lm", choices=["transformer", "ngram", "none"], default="transformer")
    ap.add_argument("--lm-model", default="distilgpt2")
    ap.add_argument("--ngram-corpus", help="required for --lm ngram; also used to order same-outline words by frequency")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--word-gap", type=float, default=0.9)
    ap.add_argument("--out", help="default: outputs/<target>/transcription.txt")
    a = ap.parse_args()
    out = Path(a.out or OUT / a.target / "transcription.txt")

    rec = Recognizer(a.ckpt or OUT / a.target / "recognizer.pt")
    lex = Lexicon.load(a.data_dir, rec.target, rec.vocab)
    corpus = None
    if a.ngram_corpus:
        corpus = Path(a.ngram_corpus).read_text(encoding="utf-8").splitlines()
        lex.set_freq(Counter(w for s in corpus for w in s.split()))
    lm = None
    if a.lm == "transformer":
        from src.lm_transformer import TransformerLM
        lm = TransformerLM(a.lm_model)
    elif a.lm == "ngram":
        from src.lm_ngram import NGramLM
        if not corpus:
            sys.exit("--ngram-corpus is required with --lm ngram")
        lm = NGramLM(corpus)

    tr = GreggTranscriber(rec, lex, lm, beta=a.beta)
    text, target_text, details = tr.transcribe_document(a.input, word_gap=a.word_gap, debug_dir=out.parent)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".details.json").write_text(json.dumps(details, indent=1), encoding="utf-8")
    saved = f"{out}  (+ .details.json with candidates, + segmentation_page*.png)"
    if rec.target == "grascii":
        g = out.with_suffix(".grascii.txt")
        g.write_text(target_text, encoding="utf-8")
        saved += f"\nRecognized Grascii (stage 1 output): {g}"
    print("\n--- English transcription ---\n" + text)
    print("\nSaved:", saved)


if __name__ == "__main__":
    main()
