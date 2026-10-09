import math
from collections import Counter


class NGramLM:
    """Interpolated trigram language model (classical contextual baseline)."""

    def __init__(self, sentences, l3=0.5, l2=0.3, l1=0.2):
        self.c1, self.c2, self.c3 = Counter(), Counter(), Counter()
        self.h1, self.h2 = Counter(), Counter()  # history counts
        self.l3, self.l2, self.l1 = l3, l2, l1
        for s in sentences:
            w = ["<s>", "<s>"] + s.split()
            for i in range(2, len(w)):
                self.c1[w[i]] += 1
                self.c2[(w[i - 1], w[i])] += 1
                self.c3[(w[i - 2], w[i - 1], w[i])] += 1
                self.h1[w[i - 1]] += 1
                self.h2[(w[i - 2], w[i - 1])] += 1
        self.N = sum(self.c1.values())
        self.V = len(self.c1) + 1

    def prob(self, u, v, w):
        p1 = (self.c1[w] + 0.1) / (self.N + 0.1 * self.V)
        p2 = self.c2[(v, w)] / self.h1[v] if self.h1[v] else 0.0
        p3 = self.c3[(u, v, w)] / self.h2[(u, v)] if self.h2[(u, v)] else 0.0
        return self.l3 * p3 + self.l2 * p2 + self.l1 * p1

    def score_candidates(self, prefix, cands):
        ctx = ["<s>", "<s>"] + list(prefix)
        u, v = ctx[-2], ctx[-1]
        return [math.log(self.prob(u, v, c)) for c in cands]
