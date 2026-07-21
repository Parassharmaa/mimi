#!/usr/bin/env python3
"""Deterministic uncertainty-plus-diversity selection for DQRD teacher sources."""

from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np


STRATA = ("low", "medium", "high")
DEFAULT_SHARES = {"low": 0.15, "medium": 0.35, "high": 0.50}


def stable_rank(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


def _validate(rows: list[dict], count: int) -> None:
    if not 0 < count <= len(rows):
        raise ValueError(f"selection count must be in [1, {len(rows)}]")
    identifiers = [str(row.get("id", "")).strip() for row in rows]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("selection rows need non-empty unique IDs")
    dimensions: set[int] = set()
    for row in rows:
        uncertainty = row.get("student_sequence_nll")
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)):
            raise ValueError(f"row has invalid uncertainty: {row['id']}")
        if not math.isfinite(float(uncertainty)) or float(uncertainty) < 0:
            raise ValueError(f"row has non-finite or negative uncertainty: {row['id']}")
        embedding = row.get("_selection_embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"row has no encoder embedding: {row['id']}")
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError(f"row has invalid encoder embedding: {row['id']}")
        dimensions.add(len(vector))
    if len(dimensions) != 1:
        raise ValueError("encoder embeddings must have one shared dimension")


def _stratify(rows: list[dict], seed: str) -> dict[str, list[dict]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["student_sequence_nll"]),
            stable_rank(seed, str(row["id"])),
        ),
    )
    output = {stratum: [] for stratum in STRATA}
    for index, row in enumerate(ordered):
        bucket = min(len(STRATA) - 1, index * len(STRATA) // len(ordered))
        output[STRATA[bucket]].append(row)
    return output


def _allocate(count: int, buckets: dict[str, list[dict]]) -> dict[str, int]:
    shares = {
        name: DEFAULT_SHARES[name]
        for name in STRATA
        if buckets[name]
    }
    share_total = sum(shares.values())
    targets = {name: count * shares[name] / share_total for name in shares}
    allocated = {
        name: min(len(buckets[name]), math.floor(targets[name]))
        for name in shares
    }
    while sum(allocated.values()) < count:
        eligible = [name for name in shares if allocated[name] < len(buckets[name])]
        if not eligible:
            raise ValueError("stratum capacity cannot satisfy selection count")
        name = max(
            eligible,
            key=lambda value: (
                targets[value] - allocated[value],
                shares[value],
                -STRATA.index(value),
            ),
        )
        allocated[name] += 1
    return {name: allocated.get(name, 0) for name in STRATA}


def _kcenter(rows: list[dict], count: int, seed: str, stratum: str) -> list[dict]:
    if count == 0:
        return []
    matrix = np.asarray(
        [row["_selection_embedding"] for row in rows],
        dtype=np.float32,
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        bad = rows[int(np.flatnonzero(norms[:, 0] <= 1e-12)[0])]["id"]
        raise ValueError(f"row has a zero encoder embedding: {bad}")
    matrix /= norms
    tie_order = {
        index: rank
        for rank, index in enumerate(
            sorted(
                range(len(rows)),
                key=lambda value: stable_rank(seed, f"{stratum}:{rows[value]['id']}"),
            )
        )
    }
    first = max(
        range(len(rows)),
        key=lambda index: (
            float(rows[index]["student_sequence_nll"]),
            -tie_order[index],
        ),
    )
    selected = [first]
    selected_set = {first}
    selection_distance = {first: 1.0}
    minimum_distance = 1.0 - matrix @ matrix[first]
    while len(selected) < count:
        candidate = max(
            (index for index in range(len(rows)) if index not in selected_set),
            key=lambda index: (
                float(minimum_distance[index]),
                float(rows[index]["student_sequence_nll"]),
                -tie_order[index],
            ),
        )
        selected.append(candidate)
        selected_set.add(candidate)
        selection_distance[candidate] = max(0.0, float(minimum_distance[candidate]))
        minimum_distance = np.minimum(
            minimum_distance,
            1.0 - matrix @ matrix[candidate],
        )
    output: list[dict] = []
    for index in selected:
        row = {key: value for key, value in rows[index].items() if key != "_selection_embedding"}
        row["selection_uncertainty_stratum"] = stratum
        row["selection_diversity_distance"] = selection_distance[index]
        output.append(row)
    return output


def hybrid_select(rows: list[dict], count: int, seed: str) -> list[dict]:
    """Select across uncertainty thirds, then maximize encoder cosine coverage."""

    _validate(rows, count)
    buckets = _stratify(rows, seed)
    allocation = _allocate(count, buckets)
    selected = [
        row
        for stratum in STRATA
        for row in _kcenter(
            buckets[stratum],
            allocation[stratum],
            seed,
            stratum,
        )
    ]
    selected.sort(key=lambda row: stable_rank(seed, f"output:{row['id']}"))
    for index, row in enumerate(selected, start=1):
        row["selection_rank"] = index
    return selected


def selection_summary(selected: list[dict]) -> dict:
    distances = sorted(float(row["selection_diversity_distance"]) for row in selected)
    uncertainties = sorted(float(row["student_sequence_nll"]) for row in selected)
    return {
        "algorithm": "uncertainty-thirds-plus-greedy-cosine-k-center-v1",
        "stratumShares": DEFAULT_SHARES,
        "selectedByStratum": dict(
            sorted(Counter(row["selection_uncertainty_stratum"] for row in selected).items())
        ),
        "sequenceNLL": {
            "minimum": uncertainties[0],
            "median": uncertainties[len(uncertainties) // 2],
            "maximum": uncertainties[-1],
        },
        "selectionDiversityDistance": {
            "minimum": distances[0],
            "median": distances[len(distances) // 2],
            "maximum": distances[-1],
        },
    }
