"""
Builds sentence-level TEST data from an English text corpus: every sentence is assembled from
word images of the dataset (one random image per word). Sentences are cut at words that have no
image/Grascii label (e.g. brief forms missing from the dictionary), keeping runs of >= --min-len words.
Writes into the data folder:
  test_sentences.jsonl   {"words": [...], "grascii": [...], "images": [...], "seeds": [...]}
  corpus_train.txt       the corpus WITHOUT the sentences that produced test items (for the N-gram LM)
and outputs/demo_page.png (fake document page from 5 test sentences, for transcribe.py).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUT
from src.dataset import load_word_image
from src.manifest import read_manifest
from src.utils import normalize_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-sentences", type=int, default=300)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.data_dir)
    df, meta = read_manifest(root)
    d = df if meta["split_mode"] == "closed" else df[df["split"] == a.split]
    d = d[d["grascii"] != ""]
    imgs, w2g = {}, {}
    for p, w, g in zip(d["abs_path"], d["english"], d["grascii"]):
        imgs.setdefault(w, []).append(p)
        w2g.setdefault(w, g)

    sents = [normalize_text(s) for s in Path(a.corpus).read_text(encoding="utf-8").splitlines()]
    sents = list(dict.fromkeys(s for s in sents if s))
    items = []  # (source sentence id, words)
    for i, s in enumerate(sents):
        run = []
        for w in s.split() + [None]:
            if w is not None and w in imgs:
                run.append(w)
                continue
            for k in range(0, len(run), a.max_len):
                chunk = run[k:k + a.max_len]
                if len(chunk) >= a.min_len:
                    items.append((i, chunk))
            run = []
    print(f"{len(items)} usable sentence segments from {len(sents)} corpus sentences "
          f"(dataset covers {len(imgs)} words)")
    if not items:
        sys.exit("No corpus sentence has a long enough run of words covered by the dataset. "
                 "Add brief forms (data/grascii/brief_forms_extra.tsv + images) or lower --min-len.")
    rng = np.random.RandomState(a.seed)
    rng.shuffle(items)
    items = items[:a.max_sentences]
    used_src = {i for i, _ in items}

    rows = []
    for _, ws in items:
        rows.append({"words": ws, "grascii": [w2g[w] for w in ws],
                     "images": [imgs[w][rng.randint(len(imgs[w]))] for w in ws],
                     "seeds": [int(rng.randint(1_000_000)) for _ in ws]})
    with open(root / "test_sentences.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (root / "corpus_train.txt").write_text(
        "\n".join(s for i, s in enumerate(sents) if i not in used_src), encoding="utf-8")
    print(f"{len(rows)} test sentences -> {root/'test_sentences.jsonl'}; "
          f"{len(sents)-len(used_src)} corpus sentences kept for N-gram training")

    OUT.mkdir(exist_ok=True)
    lines = []
    for r in rows[:5]:
        words = [load_word_image(p, s, 0.4) for p, s in zip(r["images"], r["seeds"])]
        h, parts = 70, []
        for w in words:
            sc = h / w.shape[0] * 0.85
            w = cv2.resize(w, (max(4, int(w.shape[1] * sc)), int(w.shape[0] * sc)))
            c = np.zeros((h, w.shape[1]), np.uint8)
            c[(h - w.shape[0]) // 2:(h - w.shape[0]) // 2 + w.shape[0]] = w
            parts += [c, np.zeros((h, 45), np.uint8)]
        lines.append(np.hstack(parts))
    Wd = max(l.shape[1] for l in lines) + 60
    page = np.zeros((len(lines) * 110 + 40, Wd), np.uint8)
    for i, l in enumerate(lines):
        page[30 + i * 110:30 + i * 110 + 70, 30:30 + l.shape[1]] = l
    cv2.imwrite(str(OUT / "demo_page.png"), 255 - page)
    print("Demo page ->", OUT / "demo_page.png", "\nGround truth of demo page:")
    for r in rows[:5]:
        print("  ", " ".join(r["words"]))


if __name__ == "__main__":
    main()
