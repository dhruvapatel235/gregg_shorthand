"""
Data audit (run this BEFORE training). Checks files, labels, duplicates, split leakage, alphabet
coverage and ambiguity; writes outputs/audit_<name>.json and outputs/audit_<name>_samples.png.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUT
from src.dataset import load_gray
from src.manifest import read_manifest
from src.utils import valid_english


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--samples", type=int, default=24)
    a = ap.parse_args()
    df, meta = read_manifest(a.data_dir)
    name = Path(a.data_dir).name
    rep = {"rows": len(df), "split_mode": meta["split_mode"], "problems": []}
    prob = rep["problems"].append

    # --- files -------------------------------------------------------------------------------
    bad, shapes, light, md5 = [], [], 0, defaultdict(list)
    for i, p in enumerate(df["abs_path"]):
        try:
            g = load_gray(p)
            shapes.append(g.shape)
            light += g.mean() > 127
            md5[hashlib.md5(g.tobytes()).hexdigest()].append(i)
        except Exception:
            bad.append(p)
    rep["unreadable_files"] = len(bad)
    if bad:
        prob(f"{len(bad)} unreadable images, e.g. {bad[:3]}")
    if shapes:
        hs, ws = zip(*shapes)
        rep["image_height"] = [int(np.min(hs)), int(np.median(hs)), int(np.max(hs))]
        rep["image_width"] = [int(np.min(ws)), int(np.median(ws)), int(np.max(ws))]
        rep["light_background_fraction"] = round(light / len(shapes), 3)
        if 0.05 < rep["light_background_fraction"] < 0.95:
            prob("mixed polarity (some dark, some light backgrounds) - inverted automatically per image")
    dups = [v for v in md5.values() if len(v) > 1]
    rep["duplicate_image_groups"] = len(dups)
    rep["duplicate_groups_with_different_labels"] = sum(len({df.english[i] for i in v}) > 1 for v in dups)

    # --- labels --------------------------------------------------------------------------------
    rep["english_invalid"] = int((~df["english"].map(valid_english)).sum())
    rep["distinct_words"] = int(df["english"].nunique())
    rep["with_grascii"] = int((df["grascii"] != "").sum())
    rep["without_grascii"] = int((df["grascii"] == "").sum())
    if rep["without_grascii"]:
        prob(f"{rep['without_grascii']} images have no Grascii label (skipped for --target grascii); "
             f"examples: {list(df[df['grascii'] == '']['english'][:8])}")
    g_alpha = Counter(c for g in df["grascii"] for c in g)
    rep["grascii_alphabet"] = dict(sorted(g_alpha.items(), key=lambda x: -x[1]))
    rep["english_alphabet"] = dict(sorted(Counter(c for w in df["english"] for c in w).items()))
    lens = df.loc[df["grascii"] != "", "grascii"].str.len()
    if len(lens):
        rep["grascii_length_min_median_max"] = [int(lens.min()), int(lens.median()), int(lens.max())]
        if lens.max() >= 60:
            prob("Grascii label longer than CTC time steps (64) allows")
    w2g = df[df["grascii"] != ""].groupby("english")["grascii"].nunique()
    rep["words_with_conflicting_grascii"] = int((w2g > 1).sum())
    g2w = df[df["grascii"] != ""].groupby("grascii")["english"].nunique()
    rep["grascii_shared_by_several_words"] = int((g2w > 1).sum())          # same outline, different word
    rep["words_affected_by_shared_grascii"] = int(df[df["grascii"].isin(g2w[g2w > 1].index)]["english"].nunique())
    if rep["grascii_shared_by_several_words"]:
        ex = g2w[g2w > 1].index[:5]
        rep["shared_grascii_examples"] = {g: sorted(df[df["grascii"] == g]["english"].unique()) for g in ex}

    # --- splits / leakage ----------------------------------------------------------------------
    rep["split_counts"] = df["split"].value_counts().to_dict()
    if meta["split_mode"] == "open":
        key = meta["group_by"]
        sets = {s: set(df[df["split"] == s][key]) for s in ("train", "val", "test")}
        leak = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
        rep["split_group_overlap"] = len(leak)
        if leak:
            prob(f"LEAKAGE: {len(leak)} {key} values appear in more than one split")
        tr_chars = {c for g in df[df["split"] == "train"]["grascii"] for c in g}
        unseen = {c for g in df[df["split"] != "train"]["grascii"] for c in g} - tr_chars
        if unseen:
            prob(f"Grascii characters never seen in training: {sorted(unseen)}")
    else:
        prob("closed split: val/test are degraded copies of training images (optimistic by design)")
    rep["problems"] = rep["problems"] or ["none found"]

    OUT.mkdir(exist_ok=True)
    (OUT / f"audit_{name}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    # --- sample sheet --------------------------------------------------------------------------
    rng = np.random.RandomState(0)
    idx = rng.choice(len(df), min(a.samples, len(df)), replace=False)
    tiles = []
    for i in idx:
        try:
            g = load_gray(df.abs_path[i])
        except Exception:
            continue
        g = cv2.resize(g, (240, 80), interpolation=cv2.INTER_AREA)
        t = np.full((110, 240), 255, np.uint8)
        t[:80] = g
        cv2.putText(t, f"{df.english[i]} | {df.grascii[i]}"[:34], (3, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        tiles.append(t)
    if tiles:
        cols = 4
        while len(tiles) % cols:
            tiles.append(np.full((110, 240), 255, np.uint8))
        sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
        cv2.imwrite(str(OUT / f"audit_{name}_samples.png"), sheet)

    print(json.dumps({k: v for k, v in rep.items() if k not in ("grascii_alphabet", "english_alphabet")}, indent=2, default=str))
    print(f"\nSaved {OUT / ('audit_' + name + '.json')} and the sample sheet audit_{name}_samples.png - LOOK at the sheet.")


if __name__ == "__main__":
    main()
