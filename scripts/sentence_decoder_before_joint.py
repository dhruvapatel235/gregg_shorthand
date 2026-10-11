import json
import math


class SentenceDecoder:

    def __init__(self, ngram_path, beam_size=10):

        with open(ngram_path, "r", encoding="utf-8") as f:
            model = json.load(f)

        self.unigrams = model["unigrams"]
        self.bigrams = model["bigrams"]
        self.trigrams = model["trigrams"]

        self.total_uni = sum(self.unigrams.values())
        self.total_bi = sum(self.bigrams.values())
        self.total_tri = sum(self.trigrams.values())

        self.vocab_size = len(self.unigrams)

        self.beam_size = beam_size


    def score(self, history, word):

        word = word.lower()

        uni = self.unigrams.get(word, 0)

        if len(history) >= 1:
            bi = self.bigrams.get(
                f"{history[-1]}\t{word}",
                0
            )
        else:
            bi = 0

        if len(history) >= 2:
            tri = self.trigrams.get(
                f"{history[-2]}\t{history[-1]}\t{word}",
                0
            )
        else:
            tri = 0

        uni_p = (
            uni + 1
        ) / (
            self.total_uni + self.vocab_size
        )

        bi_p = (
            bi + 1
        ) / (
            self.total_bi + self.vocab_size
        )

        tri_p = (
            tri + 1
        ) / (
            self.total_tri + self.vocab_size
        )

        return (
            0.2 * math.log(uni_p)
            + 0.4 * math.log(bi_p)
            + 0.4 * math.log(tri_p)
        )


    def decode(self, candidate_lists):

        """
        candidate_lists:

        [
            ["there", "the"],
            ["were", "ill"],
            ["one"]
        ]

        Returns the highest-scoring complete sentence.
        """

        # Each beam item:
        # (score, words)

        beams = [
            (0.0, [])
        ]

        for candidates in candidate_lists:

            if not candidates:

                candidates = ["[UNKNOWN]"]

            next_beams = []

            for score, history in beams:

                for word in candidates:

                    new_score = (
                        score
                        + self.score(history, word)
                    )

                    new_history = (
                        history + [word]
                    )

                    next_beams.append(
                        (
                            new_score,
                            new_history
                        )
                    )

            # Keep only best complete hypotheses
            next_beams.sort(
                key=lambda x: x[0],
                reverse=True
            )

            beams = next_beams[
                :self.beam_size
            ]

        if not beams:

            return []

        return beams[0][1]


if __name__ == "__main__":

    decoder = SentenceDecoder(
        "outputs/ngram_model.json",
        beam_size=10
    )

    tests = [

        [
            ["the", "there"],
            ["ill", "were"],
            ["one"]
        ],

        [
            ["the", "there"],
            ["ill", "were"],
            ["only", "one"]
        ],

        [
            ["how", "out"],
            ["of"],
            ["the"]
        ],

        [
            ["idea", "idly"],
            ["of"],
            ["the"]
        ]
    ]

    for candidates in tests:

        result = decoder.decode(
            candidates
        )

        print(
            "CANDIDATES:",
            candidates
        )

        print(
            "DECODED:",
            result
        )

        print()