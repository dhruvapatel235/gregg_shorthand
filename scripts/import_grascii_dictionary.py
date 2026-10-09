"""
(Re)builds data/grascii/dictionary_<edition>.tsv from the Grascii project's dictionary.
A copy of the pre-anniversary dictionary is ALREADY bundled, so you normally do not need this.

  pip install grascii
  python scripts/import_grascii_dictionary.py --edition preanniversary
  python scripts/import_grascii_dictionary.py --edition anniversary
  python scripts/import_grascii_dictionary.py --source path/to/folder_with_letter_files
"""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", default="preanniversary", choices=["preanniversary", "anniversary"])
    ap.add_argument("--source", help="folder with the per-letter dictionary files (default: installed grascii package)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.source:
        src = Path(a.source)
    else:
        spec = importlib.util.find_spec("grascii")
        if spec is None:
            sys.exit("Install the package first:  pip install grascii   (or pass --source)")
        src = Path(list(spec.submodule_search_locations)[0]) / "dictionary" / a.edition
    rows = []
    for f in sorted(src.iterdir()):
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    g, _, w = line.partition(" ")
                    rows.append((g, w.strip().lower()))
    out = Path(a.out or DATA / "grascii" / f"dictionary_{a.edition}.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Grascii dictionary ({a.edition}) from https://github.com/grascii/grascii (MIT). Format: GRASCII<TAB>word\n")
        for g, w in rows:
            f.write(f"{g}\t{w}\n")
    print(f"{len(rows)} entries -> {out}")


if __name__ == "__main__":
    main()
