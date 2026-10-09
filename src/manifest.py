"""
Dataset manifest (one row per image):
  image_path, english, grascii, split, source_id, writer_id, notes
Splits are deterministic (hash of the grouping key), so every run gives identical splits.
split_mode (stored in manifest.meta.json):
  closed : every image is used for training; validation/test = the same images under synthetic
           degradation. Right for one-image-per-word dictionaries where the lexicon is fixed.
           (Optimistic: say so in your report.)
  open   : images are split by group (word, or source_id) -> test groups never seen in training.
"""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from src import grascii as G
from src.utils import valid_english

COLUMNS = ["image_path", "english", "grascii", "split", "source_id", "writer_id", "notes"]


def _bucket(key, seed):
    return int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_split(df, mode, group_by="english", seed=0):
    if mode == "closed":
        df["split"] = "all"
        return df
    if group_by not in df or (df[group_by].astype(str) == "").all():
        raise SystemExit(f"--group-by {group_by}: column is empty in the data")
    b = df[group_by].astype(str).map(lambda k: _bucket(k, seed))
    df["split"] = ["train" if x < 0.8 else "val" if x < 0.9 else "test" for x in b]
    return df


def write_dataset(df, out_dir, mode="closed", group_by="english", seed=0, dictionary=None,
                  extra_dictionary=None, lexicon_scope="dictionary"):
    """df needs columns image_path (any path) + english; optional grascii/source_id/writer_id/notes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["english"] = df["english"].astype(str).str.strip().str.lower()
    n0 = len(df)
    df = df[df["english"].map(valid_english)].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"None of the {n0} images has a usable English label (letters, ' and - only). "
                         "File names must start with the word (about.png, about_2.png) or use --csv.")
    for c in ("grascii", "source_id", "writer_id", "notes"):
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    df["grascii"] = df["grascii"].map(G.normalize)

    by_word, by_g, skipped = {}, {}, 0
    if dictionary and Path(dictionary).exists():
        by_word, by_g, skipped = G.load_dictionary(dictionary, extra_dictionary)
        fill = df["grascii"] == ""
        df.loc[fill, "grascii"] = [by_word.get(w, [""])[0] for w in df.loc[fill, "english"]]
    n_missing = int((df["grascii"] == "").sum())

    def rel(p):
        try:
            return os.path.relpath(Path(p).resolve(), out.resolve())
        except ValueError:  # different drive on Windows
            return str(Path(p).resolve())

    df["image_path"] = df["image_path"].map(rel)
    df = assign_split(df, mode, group_by, seed)[COLUMNS]
    df.to_csv(out / "manifest.csv", index=False)

    # Lexicons: what the system is allowed to output
    en = {w for w in df["english"]}
    key2words = {}
    for w, g in zip(df["english"], df["grascii"]):
        if g:
            key2words.setdefault(g, [])
            if w not in key2words[g]:
                key2words[g].append(w)
    if lexicon_scope == "dictionary" and by_g:
        en |= set(by_word)
        for g, ws in by_g.items():
            for w in ws:
                key2words.setdefault(g, [])
                if w not in key2words[g]:
                    key2words[g].append(w)
    (out / "lexicon_english.txt").write_text("\n".join(sorted(en)), encoding="utf-8")
    with open(out / "lexicon_grascii.tsv", "w", encoding="utf-8") as f:
        for g in sorted(key2words):
            f.write(g + "\t" + "|".join(sorted(key2words[g])) + "\n")
    meta = {"split_mode": mode, "group_by": group_by, "seed": seed, "n_rows": len(df),
            "n_without_grascii": n_missing, "lexicon_scope": lexicon_scope,
            "grascii_dictionary": str(dictionary) if dictionary else None}
    (out / "manifest.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    spec = {"spec_version": 1,
            "normalisation": "strip all whitespace; upper-case",
            "alphabet": sorted({c for g in df["grascii"] for c in g}),
            "tokenisation": "one CTC symbol per character",
            "source": "Grascii project dictionary (https://github.com/grascii/grascii, MIT)",
            "note": "Verify these conventions against the dataset you actually use."}
    (out / "grascii_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return df, meta, skipped


def read_manifest(path):
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.csv"
    df = pd.read_csv(path, keep_default_na=False)
    meta = json.loads((path.parent / "manifest.meta.json").read_text(encoding="utf-8"))
    df["abs_path"] = [str((path.parent / p).resolve()) if not Path(p).is_absolute() else p
                      for p in df["image_path"]]
    return df, meta


def rows_for_split(df, meta, split, target):
    """Rows of a split that have a label for the chosen target ('english' | 'grascii')."""
    d = df if meta["split_mode"] == "closed" else df[df["split"] == split]
    return d[d[target].astype(str) != ""].reset_index(drop=True)
