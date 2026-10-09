"""Target alphabet. Built from TRAINING labels only (never hard-coded), index 0 = CTC blank."""


class Vocab:
    def __init__(self, chars):
        self.chars = list(chars)
        self.c2i = {c: i + 1 for i, c in enumerate(self.chars)}
        self.i2c = {i: c for c, i in self.c2i.items()}

    @property
    def size(self):
        return len(self.chars) + 1

    @classmethod
    def from_labels(cls, labels):
        return cls(sorted({c for s in labels for c in s}))

    def valid(self, s):
        return len(s) > 0 and all(c in self.c2i for c in s)

    def encode(self, s):
        return [self.c2i[c] for c in s]

    def decode(self, idx):
        return "".join(self.i2c[i] for i in idx if i in self.i2c)

    def __repr__(self):
        return f"Vocab({''.join(self.chars)})"
