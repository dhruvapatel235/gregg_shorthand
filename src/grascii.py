"""
Grascii support.

Grascii is an ASCII encoding of Gregg shorthand (project: https://github.com/grascii/grascii, MIT).
Examples from the bundled pre-anniversary dictionary:   'ABT = habit,   TH) = their / there,   TU = to / too.
Characters seen in that dictionary:  A-Z  '  ^  ~  _  &  (  )  |  ,  .  \\
This code treats a Grascii string as a plain character sequence (CTC target); the alphabet is
read from the training labels, not hard-coded. Conventions are written to grascii_spec.json by
prepare_manifest.py so they are documented with every dataset.
"""
import re
from collections import defaultdict
from pathlib import Path

from src.utils import valid_english


def normalize(g):
    """Versioned normalisation rule (spec_version 1): remove whitespace, upper-case."""
    return re.sub(r"\s+", "", str(g)).upper() if g is not None else ""


def load_dictionary(path, extra=None):
    """
    Dictionary file: one entry per line, 'GRASCII<TAB>word'  ('#' = comment).
    Returns (by_word, by_grascii, skipped). Multi-word entries (phrases) are skipped.
    Several words with the same Grascii (homophones / same outline) are kept: they are exactly
    the ambiguity the language model has to resolve.
    """
    by_word, by_grascii, skipped = defaultdict(list), defaultdict(list), 0
    for p in [path] + ([extra] if extra else []):
        if not p or not Path(p).exists():
            continue
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            g, _, w = line.partition("\t") if "\t" in line else line.partition(" ")
            g, w = normalize(g), w.strip().lower()
            if not g or not valid_english(w):
                skipped += 1
                continue
            if g not in by_word[w]:
                by_word[w].append(g)
            if w not in by_grascii[g]:
                by_grascii[g].append(w)
    return dict(by_word), dict(by_grascii), skipped
