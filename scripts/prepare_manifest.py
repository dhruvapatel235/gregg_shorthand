"""
Builds data/<name>/ from REAL Gregg word images:
    manifest.csv, manifest.meta.json, lexicon_english.txt, lexicon_grascii.tsv, grascii_spec.json

Describe your images with ONE of:
  --images DIR : file names start with the English word (about.png, about_2.png, cat-3.jpg);
                 leading '00012_' and trailing _digits / -digits are ignored
  --csv FILE   : CSV with columns  path,word  and optionally  grascii,source_id,writer_id,notes

Grascii labels are attached automatically from the bundled Grascii dictionary (matched by the
English word) unless your CSV already has a 'grascii' column. Words missing from the dictionary
get an empty Grascii label: they are skipped for --target grascii (see audit_data.py).

--mode closed  : every image is trained on; val/test = same images with degradation (one image per
                 word, fixed lexicon). Optimistic - state this in your report.
--mode open    : split by --group-by (english | source_id) so test groups are never seen in training.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, GRASCII_DICT
from src.manifest import write_dataset

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--csv")
    ap.add_argument("--out", default=str(DATA / "gregg"))
    ap.add_argument("--mode", choices=["closed", "open"], default="closed")
    ap.add_argument("--group-by", choices=["english", "source_id"], default="english")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grascii-dict", default=str(GRASCII_DICT))
    ap.add_argument("--extra-dict", default=str(DATA / "grascii" / "brief_forms_extra.tsv"))
    ap.add_argument("--lexicon-scope", choices=["dictionary", "manifest"], default="dictionary",
                    help="dictionary: the system may output ANY dictionary word; manifest: only words that have images")
    a = ap.parse_args()

    if a.csv:
        df = pd.read_csv(a.csv, keep_default_na=False)
        df = df.rename(columns={"path": "image_path", "word": "english"})
        df["image_path"] = [str(Path(p)) for p in df["image_path"]]
    elif a.images:
        rows = []
        for p in sorted(Path(a.images).rglob("*")):
            if p.suffix.lower() in EXT:
                w = re.sub(r"^\d+[_\-\s]+", "", re.sub(r"[_\-\s]*\d+$", "", p.stem)).lower()
                rows.append({"image_path": str(p), "english": w, "source_id": p.parent.name})
        df = pd.DataFrame(rows)
    else:
        sys.exit("Give --images DIR or --csv FILE")

    df, meta, skipped = write_dataset(df, a.out, a.mode, a.group_by, a.seed, a.grascii_dict,
                                      a.extra_dict, a.lexicon_scope)
    print(f"{len(df)} images, {df['english'].nunique()} distinct words -> {a.out}")
    print(f"Grascii attached for {len(df) - meta['n_without_grascii']} images; "
          f"{meta['n_without_grascii']} have no Grascii (not usable with --target grascii)")
    print("split counts:", df["split"].value_counts().to_dict(), f"(mode={a.mode})")
    print("Next: python scripts/audit_data.py --data-dir", a.out)


if __name__ == "__main__":
    main()
