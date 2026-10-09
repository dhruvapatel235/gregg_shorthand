from pathlib import Path

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein


class Lexicon:
    """
    key -> English words.
      Branch A (target english): key = the word itself            e.g. 'there' -> ['there']
      Branch B (target grascii): key = Grascii string; several words may share it (same outline)
                                                                  e.g. 'TH)'   -> ['their','there']
    """

    def __init__(self, key2words, vocab=None):
        self.map = {k: list(dict.fromkeys(ws)) for k, ws in key2words.items()
                    if k and (vocab is None or vocab.valid(k))}
        self.keys = sorted(self.map)
        self.set = set(self.keys)

    @classmethod
    def load(cls, data_dir, target, vocab=None):
        d = Path(data_dir)
        if target == "english":
            words = [w.strip() for w in (d / "lexicon_english.txt").read_text(encoding="utf-8").splitlines() if w.strip()]
            return cls({w: [w] for w in words}, vocab)
        m = {}
        for line in (d / "lexicon_grascii.tsv").read_text(encoding="utf-8").splitlines():
            if line.strip():
                k, _, ws = line.partition("\t")
                m[k] = ws.split("|")
        return cls(m, vocab)

    def set_freq(self, freq):
        """Order the words of each key by corpus frequency (most frequent first)."""
        for k, ws in self.map.items():
            if len(ws) > 1:
                ws.sort(key=lambda w: (-freq.get(w, 0), w))

    def words_for(self, key):
        return self.map.get(key, [])

    def fallback(self):
        return self.map[self.keys[0]][0] if self.keys else "the"

    def __contains__(self, k):
        return k in self.set

    def __len__(self):
        return len(self.keys)

    def neighbors(self, s, k=15):
        """Closest lexicon keys to string s by edit distance."""
        if not s:
            return []
        cutoff = 1 if len(s) <= 2 else 2 if len(s) <= 4 else 3
        res = process.extract(s, self.keys, scorer=Levenshtein.distance, limit=k, score_cutoff=cutoff)
        return [(w, d) for w, d, _ in res]
