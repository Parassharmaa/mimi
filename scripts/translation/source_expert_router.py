#!/usr/bin/env python3
"""Dependency-free inference for a frozen Mimi source-only expert router."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


WHITE_SPACES = re.compile(r"\s\s+")


class SourceExpertRouter:
    """Evaluate the serialized TF-IDF/ridge router without scikit-learn."""

    def __init__(self, payload: dict) -> None:
        if (
            payload.get("schemaVersion") != 1
            or payload.get("format") != "mimi-source-expert-router-v1"
        ):
            raise ValueError("unsupported source expert router")
        features = payload.get("features", {})
        if features != {
            "analyzer": "unicode-codepoint-character",
            "ngramRange": [2, 5],
            "lowercase": True,
            "minimumDocumentFrequency": 2,
            "sublinearTermFrequency": True,
            "inverseDocumentFrequency": "smooth-idf",
            "normalization": "l2",
            "sourceLengthBin": (
                "append newline then __MIMI_LENGTH_BIN_{min(chars//20,20)}__"
            ),
        }:
            raise ValueError("unsupported source expert router feature contract")
        self.direction = str(payload["direction"])
        if self.direction not in {"en-ja", "ja-en"}:
            raise ValueError("unsupported source expert router direction")
        self.vocabulary = {
            str(term): int(index) for term, index in payload["vocabulary"].items()
        }
        self.idf = [float(value) for value in payload["inverseDocumentFrequency"]]
        ridge = payload["ridge"]
        self.coefficients = [float(value) for value in ridge["coefficients"]]
        self.intercept = float(ridge["intercept"])
        if not (
            len(self.vocabulary) == len(self.idf) == len(self.coefficients)
            and set(self.vocabulary.values()) == set(range(len(self.vocabulary)))
        ):
            raise ValueError("invalid source expert router vector dimensions")
        routing = payload["routing"]
        self.minimum_source_characters = int(routing["minimumSourceCharacters"])
        self.score_threshold = float(routing["scoreThreshold"])

    @classmethod
    def load(cls, path: Path) -> SourceExpertRouter:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def feature_text(source: str) -> str:
        length_bin = min(len(source) // 20, 20)
        return f"{source}\n__MIMI_LENGTH_BIN_{length_bin}__"

    def score(self, source: str) -> float:
        text = WHITE_SPACES.sub(" ", self.feature_text(source).lower())
        term_counts: Counter[int] = Counter()
        for width in range(2, 6):
            for offset in range(max(0, len(text) - width + 1)):
                index = self.vocabulary.get(text[offset : offset + width])
                if index is not None:
                    term_counts[index] += 1
        weighted = {
            index: (1.0 + math.log(count)) * self.idf[index]
            for index, count in term_counts.items()
        }
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        if norm == 0.0:
            return self.intercept
        return self.intercept + sum(
            self.coefficients[index] * value / norm
            for index, value in weighted.items()
        )

    def routes_to_expert(self, source: str) -> bool:
        return (
            len(source) >= self.minimum_source_characters
            and self.score(source) >= self.score_threshold
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("source")
    args = parser.parse_args()
    router = SourceExpertRouter.load(args.model)
    score = router.score(args.source)
    print(
        json.dumps(
            {
                "direction": router.direction,
                "score": score,
                "routesToExpert": router.routes_to_expert(args.source),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
