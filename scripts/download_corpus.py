"""Optional: builds a large English text corpus (NLTK Brown, ~57k sentences) for the N-gram model
and for building sentence-level test data. Needs internet once."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA
from src.utils import normalize_text


def main():
    import nltk
    nltk.download("brown", quiet=True)
    from nltk.corpus import brown
    lines = []
    for sent in brown.sents():
        t = normalize_text(" ".join(sent))
        if 4 <= len(t.split()) <= 20:
            lines.append(t)
    (DATA / "corpus.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(lines)} sentences to {DATA/'corpus.txt'}")


if __name__ == "__main__":
    main()
