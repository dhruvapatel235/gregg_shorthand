from pathlib import Path

from config import OUT
from src.candidates import build_candidates
from src.resolver import resolve
from src.segment import draw_boxes, load_pages, segment_page


class GreggTranscriber:
    """image -> recognizer (English letters OR Grascii) -> lexicon candidates -> LM -> English text."""

    def __init__(self, recognizer, lexicon, lm=None, alpha=1.0, beta=0.5, beam=5):
        self.rec, self.lex, self.lm = recognizer, lexicon, lm
        self.alpha, self.beta, self.beam = alpha, beta, beam

    def candidates(self, inks):
        return [build_candidates(lp, self.lex, self.rec.vocab) for lp in self.rec.logprobs(inks)]

    def transcribe_words(self, inks, left_context=()):
        info = self.candidates(inks)
        words = resolve([c["cands"] for c in info], self.lm, self.alpha, self.beta,
                        self.beam, left_context)
        return words, info

    def transcribe_document(self, path, word_gap=0.9, debug_dir=None):
        """Returns (english_text, recognized_target_text, details)."""
        debug_dir = Path(debug_dir or OUT)
        debug_dir.mkdir(parents=True, exist_ok=True)
        text_lines, target_lines, details, ctx = [], [], [], []
        for pi, gray in enumerate(load_pages(path)):
            lines, boxes = segment_page(gray, word_gap=word_gap)
            draw_boxes(gray, boxes, debug_dir / f"segmentation_page{pi + 1}.png")
            for line in lines:
                words, info = self.transcribe_words(line, left_context=ctx[-40:])
                ctx += words
                text_lines.append(" ".join(words))
                target_lines.append(" ".join(i["greedy"] for i in info))
                details.append([{"chosen": w, "recognized_" + self.rec.target: i["greedy"],
                                 "candidates": [c for c, _ in i["cands"][:5]]}
                                for w, i in zip(words, info)])
        return "\n".join(text_lines), "\n".join(target_lines), details
