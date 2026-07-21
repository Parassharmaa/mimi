#!/usr/bin/env python3
"""Focused contracts for the source-only Marian expert router."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from evaluate_expert_router import (
    bootstrap_interval,
    feature_text,
    group_id,
    routed_summary,
    sentence_chrf,
    split_name,
)
from source_expert_router import SourceExpertRouter


def run() -> None:
    assert group_id("public:jlt:law-123:tu-1:en-ja") == group_id(
        "public:jlt:law-123:tu-99:en-ja"
    )
    assert group_id("public:alt:SNT.456.1:en-ja") == group_id(
        "public:alt:SNT.456.99:en-ja"
    )
    assert group_id("public:tatoeba:1:en-ja") != group_id(
        "public:tatoeba:2:en-ja"
    )
    assert split_name("public:jlt:law-123:tu-1:en-ja", "en-ja") == split_name(
        "public:jlt:law-123:tu-99:en-ja", "en-ja"
    )
    assert "__MIMI_LENGTH_BIN_" in feature_text("fixture source")
    assert sentence_chrf("identical fixture", ["identical fixture"]) == 100.0

    rows = [
        {
            "id": "long-positive",
            "source": "x" * 80,
            "references": ["reference"],
            "baselineHypothesis": "base",
            "expertHypothesis": "expert",
            "expertDelta": 6.0,
        },
        {
            "id": "short-negative",
            "source": "x" * 30,
            "references": ["reference"],
            "baselineHypothesis": "base",
            "expertHypothesis": "expert",
            "expertDelta": -10.0,
        },
    ]
    predictions = np.array([1.0, 1.0])
    summary = routed_summary(rows, predictions, 0.5, 60)
    assert summary["routedCases"] == 1
    assert summary["routedIDs"] == ["long-positive"]
    assert summary["meanSentenceChrFPlusPlusDelta"] == 3.0
    interval = bootstrap_interval(
        rows,
        predictions,
        0.5,
        60,
        samples=1_000,
        seed=314159,
    )
    assert interval[0] >= 0.0 and interval[1] > 0.0

    sources = [
        "This agreement enters into force on July 19.",
        "The parties shall comply with applicable law.",
        "Mimi translates the final sentence.",
        "今日は良い天気です。",
    ]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform([feature_text(source) for source in sources])
    regressor = Ridge(alpha=1.0, solver="lsqr").fit(
        matrix,
        np.array([1.0, 0.5, -0.5, -1.0]),
    )
    portable = SourceExpertRouter(
        {
            "schemaVersion": 1,
            "format": "mimi-source-expert-router-v1",
            "direction": "en-ja",
            "features": {
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
            },
            "vocabulary": {
                term: int(index) for term, index in vectorizer.vocabulary_.items()
            },
            "inverseDocumentFrequency": [float(value) for value in vectorizer.idf_],
            "ridge": {
                "alpha": 1.0,
                "coefficients": [float(value) for value in regressor.coef_],
                "intercept": float(regressor.intercept_),
            },
            "routing": {"minimumSourceCharacters": 0, "scoreThreshold": 0.0},
        }
    )
    expected = regressor.predict(vectorizer.transform([feature_text(s) for s in sources]))
    actual = np.array([portable.score(source) for source in sources])
    assert np.allclose(actual, expected, atol=1e-6)

    print("expert router contracts passed")


if __name__ == "__main__":
    run()
