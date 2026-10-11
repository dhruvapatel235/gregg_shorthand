
import json
import math


class SentenceDecoder:
    def __init__(
        self,
        ngram_path,
        beam_size=10,
        visual_weight=0.5,
        language_weight=1.0,
    ):
        with open(ngram_path, "r", encoding="utf-8") as f:
            model = json.load(f)

        self.unigrams = model["unigrams"]
        self.bigrams = model["bigrams"]
        self.trigrams = model["trigrams"]

        self.total_uni = sum(self.unigrams.values())
        self.total_bi = sum(self.bigrams.values())
        self.total_tri = sum(self.trigrams.values())
        self.vocab_size = max(len(self.unigrams), 1)

        self.beam_size = beam_size
        self.visual_weight = visual_weight
        self.language_weight = language_weight

    def score(self, history, word):
        word = word.lower()

        uni = self.unigrams.get(word, 0)
        bi = (
            self.bigrams.get(f"{history[-1]}\t{word}", 0)
            if history else 0
        )
        tri = (
            self.trigrams.get(
                f"{history[-2]}\t{history[-1]}\t{word}", 0
            )
            if len(history) >= 2 else 0
        )

        uni_p = (uni + 1) / (self.total_uni + self.vocab_size)
        bi_p = (bi + 1) / (self.total_bi + self.vocab_size)
        tri_p = (tri + 1) / (self.total_tri + self.vocab_size)

        return (
            0.2 * math.log(uni_p)
            + 0.4 * math.log(bi_p)
            + 0.4 * math.log(tri_p)
        )

    def decode(self, candidate_lists):
        # Each beam item: (total score, words)
        beams = [(0.0, [])]

        for candidates in candidate_lists:
            if not candidates:
                candidates = [{"word": "[UNKNOWN]", "visual_score": 0.0}]

            # Backward compatibility with lists of plain strings.
            normalized = []
            for item in candidates:
                if isinstance(item, str):
                    normalized.append({
                        "word": item,
                        "visual_score": 0.0,
                    })
                else:
                    normalized.append(item)

            # Normalize visual scores within this word position.
            scores = [float(c["visual_score"]) for c in normalized]
            max_score = max(scores)
            visual_log_norm = max_score + math.log(
                sum(math.exp(s - max_score) for s in scores)
            )

            next_beams = []

            for previous_score, history in beams:
                for candidate in normalized:
                    word = candidate["word"].lower()
                    visual_score = (
                        float(candidate["visual_score"]) - visual_log_norm
                    )
                    language_score = self.score(history, word)

                    combined = (
                        previous_score
                        + self.visual_weight * visual_score
                        + self.language_weight * language_score
                    )

                    next_beams.append((combined, history + [word]))

            next_beams.sort(key=lambda item: item[0], reverse=True)
            beams = next_beams[:self.beam_size]

        return beams[0][1] if beams else []


if __name__ == "__main__":
    decoder = SentenceDecoder("outputs/ngram_model.json", beam_size=10)
    tests = [
        [["the", "there"], ["ill", "were"], ["one"]],
        [["the", "there"], ["ill", "were"], ["only", "one"]],
        [["how", "out"], ["of"], ["the"]],
        [["idea", "idly"], ["of"], ["the"]],
    ]

    for candidates in tests:
        print("CANDIDATES:", candidates)
        print("DECODED:", decoder.decode(candidates))
        print()