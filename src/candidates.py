import numpy as np
import torch
import torch.nn.functional as F

from src.decode import prefix_beam_search
from src.utils import greedy_decode


def ctc_nll(logp, keys, vocab):
    """Negative CTC log-likelihood of each candidate key (target string) given (T, C) log-probs."""
    T, C = logp.shape
    n = len(keys)
    lp = torch.from_numpy(logp).unsqueeze(1).expand(T, n, C)
    tg = [torch.tensor(vocab.encode(c), dtype=torch.long) for c in keys]
    targets = torch.cat(tg)
    tl = torch.tensor([len(t) for t in tg], dtype=torch.long)
    il = torch.full((n,), T, dtype=torch.long)
    loss = F.ctc_loss(lp, targets, il, tl, blank=0, reduction="none", zero_infinity=False)
    loss = torch.nan_to_num(loss, nan=1e4, posinf=1e4).clamp(max=1e4)
    return loss.numpy()


def build_candidates(logp, lexicon, vocab, beam=10, k_beam=5, k_lex=15, max_keys=10, max_cands=12):
    """
    Returns dict:
      greedy : recognition-only output in the TARGET representation (English letters or Grascii)
      keys   : [(lexicon key, log posterior)]          best first
      cands  : [(English word, log posterior)]         best first; words sharing one key (same
               outline / homophones) get the SAME posterior - only the language model can
               separate them.
    Keys = lexicon entries close to the greedy / beam outputs, re-scored by exact CTC likelihood.
    """
    greedy = greedy_decode(logp, vocab)
    seeds = [greedy] + [s for s, _ in prefix_beam_search(logp, vocab, beam=beam, k=k_beam)]
    pool, seen = [], set()
    for s in seeds:
        if s in lexicon and s not in seen:
            seen.add(s)
            pool.append(s)
        for k, _ in lexicon.neighbors(s, k=k_lex):
            if k not in seen:
                seen.add(k)
                pool.append(k)
    if not pool:
        return {"greedy": greedy, "keys": [], "cands": [(lexicon.fallback(), 0.0)]}
    nll = ctc_nll(logp, pool, vocab)
    order = np.argsort(nll)[:max_keys]
    scores = -nll[order]
    post = scores - np.logaddexp.reduce(scores)
    keys = [(pool[i], float(p)) for i, p in zip(order, post)]
    cands = []
    for k, p in keys:
        for w in lexicon.words_for(k):
            cands.append((w, p))
    return {"greedy": greedy, "keys": keys, "cands": cands[:max_cands]}


def ambiguity_type(c, min_post=0.1):
    """None | 'same_outline' (top candidates share one key) | 'recognition' (different keys compete)."""
    strong = [(w, p) for w, p in c["cands"] if np.exp(p) >= min_post]
    if len(strong) < 2:
        return None
    same = sum(1 for w, p in strong if c["keys"] and p == c["keys"][0][1])
    return "same_outline" if same >= 2 and len(strong) == same else "recognition"
