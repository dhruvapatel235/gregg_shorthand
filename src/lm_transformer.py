import torch
import torch.nn.functional as F


class TransformerLM:
    """
    Causal Transformer language model (GPT-2 family via HuggingFace) used to score
    P(candidate | preceding words). Default 'distilgpt2' (small, CPU-friendly); use 'gpt2'
    or larger for better context modelling. Downloaded automatically on first use.
    """

    def __init__(self, model_name="distilgpt2", device=None, max_ctx_words=48):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.bos = self.tok.bos_token_id
        self.pad = self.tok.eos_token_id
        self.max_ctx = max_ctx_words

    @torch.no_grad()
    def score_candidates(self, prefix, cands):
        prefix = list(prefix)[-self.max_ctx:]
        ctx_text = " ".join(prefix)
        ctx_text = ctx_text[:1].upper() + ctx_text[1:]
        ctx_ids = [self.bos] + (self.tok.encode(ctx_text) if ctx_text else [])
        if prefix:
            cand_ids = [self.tok.encode(" " + c) for c in cands]
        else:  # sentence start: capitalised, no leading space
            cand_ids = [self.tok.encode(c.capitalize()) for c in cands]
        seqs = [ctx_ids + c for c in cand_ids]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
        logp = F.log_softmax(logits.float(), dim=-1).cpu()
        scores = []
        for i, c in enumerate(cand_ids):
            s = 0.0
            for j, t in enumerate(c):
                s += logp[i, len(ctx_ids) + j - 1, t].item()
            scores.append(s)
        return scores
