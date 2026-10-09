from collections import defaultdict

import numpy as np

NEG = -1e30


def prefix_beam_search(logp, vocab, beam=10, k=5, prune=-12.0):
    """CTC prefix beam search. logp: (T, C) log-probs. Returns top-k [(string, log_prob)]."""
    T, C = logp.shape
    beams = {(): (0.0, NEG)}  # prefix -> (log p ending in blank, log p ending in non-blank)
    for t in range(T):
        nb = defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beams.items():
            ptot = np.logaddexp(pb, pnb)
            for c in range(C):
                p = logp[t, c]
                if p < prune:
                    continue
                if c == 0:
                    nb[prefix][0] = np.logaddexp(nb[prefix][0], ptot + p)
                    continue
                last = prefix[-1] if prefix else None
                newp = prefix + (c,)
                if c == last:
                    nb[newp][1] = np.logaddexp(nb[newp][1], pb + p)
                    nb[prefix][1] = np.logaddexp(nb[prefix][1], pnb + p)
                else:
                    nb[newp][1] = np.logaddexp(nb[newp][1], ptot + p)
        top = sorted(nb.items(), key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1]))[:beam]
        beams = {p: (v[0], v[1]) for p, v in top}
    res = [(vocab.decode(p), float(np.logaddexp(*v))) for p, v in beams.items()]
    res.sort(key=lambda x: -x[1])
    return res[:k]
