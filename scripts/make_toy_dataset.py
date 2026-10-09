"""
Creates a TOY stand-in for Gregg word images so the WHOLE pipeline (both branches) can be tested
without the real dataset. Words come from the real Grascii dictionary; every Grascii character
gets an invented stroke shape and a word is one connected pen stroke drawn from its GRASCII string.
=> words with the same Grascii (their/there, to/too, hear/here ...) get IDENTICAL images, exactly
   like same-outline words in real shorthand. Confusable characters (S/Z, T/D, K/G, M/N, P/B, F/V,
   R/L, A/E, O/U) get near-identical shapes to create recognition ambiguity.
THIS IS NOT GREGG SHORTHAND - use prepare_manifest.py for the real images.
Also writes a toy English corpus (data/toy/corpus.txt) made of template sentences in which the
homophones can only be told apart by context.
"""
import argparse
import itertools
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, GRASCII_DICT
from src import grascii as G
from src.manifest import write_dataset

GROUPS = ["SZ", "TD", "KG", "MN", "PB", "FV", "RL", "AE", "OU"]

SUBJ = ["the man", "the woman", "the doctor", "the teacher", "the brother", "the sister",
        "the children", "the people", "they", "he", "she", "you"]
TEMPLATES = [
    "{S} will meet the {doctor|teacher|brother|sister|woman|man} {here|there}",
    "{S} will hear the {story|man|woman|teacher} {here|there}",
    "{S} will leave their {coat|hat|book|pen|paper|money} {here|there}",
    "{S} will go to the {market|station|bank|office|town|city|bridge|river} {too|today|tomorrow}",
    "{S} will read the {red|new|old|long} {book|letter|story|paper}",
    "{S} will write the letter {today|tomorrow|soon}",
    "{S} will let the {children|people|woman|man} {go|stay|come|leave}",
    "{S} will cut the {meat|bread|piece of bread}",
    "the {street|house|town|city} will stay {quiet|dark|cold} during the night",
    "the {ship|boat} will sail {near|toward|over} the {bridge|river|station}",
    "the {children|people} will keep peace {here|there|always}",
    "the hare will leave the hole {soon|now|again}",
    "the hare will gnaw the {bread|paper}",
    "{S} will know the {man|woman|teacher|doctor}",
    "{S} will leave the {town|city|house} {soon|today|tomorrow}",
    "{S} will live {near|inside|outside} the {town|city|house}",
    "the {hair|hat|coat} will look {clean|fresh|dark|bright}",
    "{S} will take the {plane|boat|car} to the {city|town}",
    "the plain will look {cold|dark|quiet}",
    "the pair of {children|people} will {come|go|stay} {here|there}",
    "{S} will stay the whole week",
    "the weak man will stay {here|there}",
    "{S} will have the {small|large|new|old} {house|farm|field} {soon|today}",
]


def expand(t):
    groups = re.findall(r"\{([^{}]*)\}", t)
    pools = [SUBJ if g == "S" else g.split("|") for g in groups]
    parts = re.split(r"\{[^{}]*\}", t)
    for combo in itertools.product(*pools):
        out = parts[0]
        for c, p in zip(combo, parts[1:]):
            out += c + p
        yield out


def make_shapes(chars, seed=7):
    rng = np.random.RandomState(seed)
    shapes = {}
    for g in GROUPS:
        base = rng.rand(5, 2)
        for ch in g:
            shapes[ch] = np.clip(base + rng.normal(0, 0.06, base.shape), 0, 1)
    for ch in sorted(chars):
        if ch not in shapes:
            shapes[ch] = rng.rand(5, 2)
    return shapes


def catmull(pts, n=8):
    p = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(p) - 2):
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p[i]) + (-p[i - 1] + p[i + 1]) * t +
                              (2 * p[i - 1] - 5 * p[i] + 4 * p[i + 1] - p[i + 2]) * t ** 2 +
                              (-p[i - 1] + 3 * p[i] - 3 * p[i + 1] + p[i + 2]) * t ** 3))
    out.append(p[-2])
    return np.array(out)


def render(grascii, shapes):
    gw, gh = 26, 40
    path, x = [], 4
    for ch in grascii:
        pts = catmull(shapes[ch] * [gw, gh] + [x, 6])
        if path:
            pts = np.vstack([path[-1][-1:], pts])
        path.append(pts)
        x += gw - 4
    pts = np.vstack(path)
    img = np.zeros((gh + 14, x + gw + 6), np.uint8)
    cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)], False, 255, 2, cv2.LINE_AA)
    return 255 - img  # dark ink on white paper, like a scan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "toy"))
    ap.add_argument("--max-sentences", type=int, default=700)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    by_word, _, _ = G.load_dictionary(GRASCII_DICT)

    sents = list(dict.fromkeys(s for t in TEMPLATES for s in expand(t)))
    sents = [s for s in sents if all(w in by_word for w in s.split())]
    rng = np.random.RandomState(0)
    rng.shuffle(sents)
    sents = sents[:a.max_sentences]
    (out / "corpus.txt").write_text("\n".join(sents), encoding="utf-8")
    vocab = sorted({w for s in sents for w in s.split()})
    shapes = make_shapes({c for w in vocab for c in by_word[w][0]})
    rows = []
    for i, w in enumerate(vocab):
        g = by_word[w][0]
        rel = out / "images" / f"{i:05d}_{w}.png"
        ok, buf = cv2.imencode(".png", render(g, shapes))
        buf.tofile(str(rel))
        rows.append({"image_path": str(rel), "english": w, "grascii": g, "source_id": "toy"})
    df, meta, _ = write_dataset(pd.DataFrame(rows), out, mode="closed", dictionary=GRASCII_DICT,
                                lexicon_scope="manifest")
    shared = df.groupby("grascii")["english"].nunique()
    print(f"Toy dataset: {len(df)} words, {len(sents)} sentences, "
          f"{int((shared > 1).sum())} Grascii strings shared by several words -> {out}")


if __name__ == "__main__":
    main()
