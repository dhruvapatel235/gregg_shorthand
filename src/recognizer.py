import numpy as np
import torch

from src.dataset import to_canvas
from src.model import CRNN
from src.vocab import Vocab


class Recognizer:
    """Loads a trained checkpoint (weights + alphabet + target type) and returns CTC log-probs."""

    def __init__(self, ckpt_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.vocab = Vocab(ckpt["vocab"])
        self.target = ckpt["target"]
        self.model = CRNN(self.vocab.size, ckpt.get("hidden", 128)).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def logprobs(self, inks, bs=64):
        """inks: list of uint8 ink-on-dark arrays -> list of (T, C) numpy log-prob matrices."""
        out = []
        for i in range(0, len(inks), bs):
            x = torch.from_numpy(np.stack([to_canvas(a) for a in inks[i:i + bs]])).unsqueeze(1)
            lp = self.model(x.to(self.device)).cpu().numpy()  # (T, B, C)
            out.extend(lp[:, j, :] for j in range(lp.shape[1]))
        return out
