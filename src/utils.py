import re

from rapidfuzz.distance import Levenshtein

BLANK = 0
ENGLISH_RE = re.compile(r"^[a-z'\-]+$")


def valid_english(w):
    return bool(ENGLISH_RE.match(w))


def greedy_decode(logp, vocab):
    """logp: (T, C) numpy log-probabilities -> string (collapse repeats, drop blanks)."""
    best = logp.argmax(axis=1)
    out, prev = [], BLANK
    for b in best:
        if b != prev and b != BLANK:
            out.append(int(b))
        prev = b
    return vocab.decode(out)


def normalize_text(text):
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"[^a-z'\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b):
    return Levenshtein.distance(a, b)


def cer(refs, hyps):
    """Character error rate over lists of strings."""
    errs = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    return errs / max(1, sum(len(r) for r in refs))


def wer(ref_sents, hyp_sents):
    """Word error rate over lists of word-lists."""
    errs = sum(Levenshtein.distance(r, h) for r, h in zip(ref_sents, hyp_sents))
    return errs / max(1, sum(len(r) for r in ref_sents))
