import json
import math


class JointDecoder:

    def __init__(self, ngram_path, beam_size=20):

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


    def language_score(self, history, word):

        word = word.lower()

        uni = self.unigrams.get(word, 0)

        if history:
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
            0.1 * math.log(uni_p)
            + 0.3 * math.log(bi_p)
            + 0.6 * math.log(tri_p)
        )


    def decode(self, word_candidates):

        """
        word_candidates:

        [
            [
                (english_word, recognition_score),
                ...
            ],
            ...
        ]

        Returns:

            [
                (english_word, grascii_score, total_score),
                ...
            ]
        """

        beams = [
            (0.0, [])
        ]

        for candidates in word_candidates:

            if not candidates:
                candidates = [
                    ("[UNKNOWN]", -20.0)
                ]

            next_beams = []

            for old_score, history in beams:

                for english, recognition_score in candidates:

                    lm_score = self.language_score(
                        history,
                        english
                    )

                    total_score = (
                        old_score
                        + recognition_score
                        + lm_score
                    )

                    next_beams.append(
                        (
                            total_score,
                            history + [
                                (
                                    english,
                                    recognition_score
                                )
                            ]
                        )
                    )

            next_beams.sort(
                key=lambda x: x[0],
                reverse=True
            )

            beams = next_beams[
                :self.beam_size
            ]

        if not beams:
            return []

        best = beams[0][1]

        return best