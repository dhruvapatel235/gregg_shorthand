def resolve(cand_lists, lm=None, alpha=1.0, beta=0.5, beam=5, left_context=()):
    """
    Pick one word per position by beam search over the whole sentence.
    score = sum_i [ alpha * log P_recognizer(word_i) + beta * log P_LM(word_i | previous words) ]
    cand_lists : list (one per word position) of [(word, rec_log_posterior), ...]
    lm         : object with score_candidates(prefix_words, candidate_words) or None
                 (None -> pure recognizer ranking = dictionary baseline)
    """
    ctx = list(left_context)
    beams = [(0.0, [])]
    for cl in cand_lists:
        words = [w for w, _ in cl]
        rec = [p for _, p in cl]
        new = []
        for score, out in beams:
            lms = lm.score_candidates(ctx + out, words) if (lm is not None and beta > 0) else [0.0] * len(words)
            for w, r, l in zip(words, rec, lms):
                new.append((score + alpha * r + beta * l, out + [w]))
        new.sort(key=lambda x: -x[0])
        beams = new[:beam]
    return beams[0][1] if beams else []
