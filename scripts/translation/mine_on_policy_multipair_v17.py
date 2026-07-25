#!/usr/bin/env python3
"""Mine reproducible safe-parent candidate lists for the V17 diagnostic.

This is a pre-semantic, training-free stage. Generated strings are negative or
ranking evidence only. The script never authorizes training and deliberately
stops before COMET or Claude judgments.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_translation_structures import critical_tokens, tokens
from filter_training_dataset_against_protected import (
    ProtectedIndex,
    normalized,
    sha256,
)
from typed_critical_token_policy import typed_preserves

EXPERIMENT = "faithful-on-policy-multipair-v17-prediagnostic"
DIRECTION_CONFIG = {
    "en-ja": {
        "source_language": "en-US",
        "target_language": "ja-JP",
        "dataset_manifest_sha256": (
            "c046ddef6633d04fbf9c0067a38a5f3bd26adbddd747bf493de6c7ca87e05cf8"
        ),
        "dataset_train_sha256": (
            "f44673e170b45e957c9577389534e6a062f54ee35b4ca41c6dcb1fc552069afa"
        ),
        "dataset_train_rows": 7_546,
        "model_sha256": (
            "eb9aa8db0e99d371036b0c55635cfdb3a5ee4d5715e32ab58bebe6173aa17ee5"
        ),
        "model_manifest_sha256": (
            "35bf93c0bc7da7de519ce34b51b5ee8ba7ebbf67bcb36afc61d1cea075a0c548"
        ),
    },
    "ja-en": {
        "source_language": "ja-JP",
        "target_language": "en-US",
        "dataset_manifest_sha256": (
            "6267c223805f082d0bdfc2dd827453df7aae976cbb309557126fc3a6609aa3eb"
        ),
        "dataset_train_sha256": (
            "81fef0184aa0e9a71bebdcf923c844ada05014cf30486bd4373aaf18db1e630d"
        ),
        "dataset_train_rows": 9_235,
        "model_sha256": (
            "8e7f7eff76d74b343884fe9a170b6dbad55d42f20ac5f526b6e8ec71e6c94f71"
        ),
        "model_manifest_sha256": (
            "0d195dc163250a9fa9312fb7ad8ba3341ab65167b90926d46f0a65d76047bd38"
        ),
    },
}
PROTECTED_SUITES = {
    "Research/translation/benchmark/canary.jsonl": (
        "957a6feffea2542e9ea5d5f345db6bfde4228226f40884244b8226d991fbeb70"
    ),
    "Research/translation/benchmark/public-stress-v1.jsonl": (
        "f345e151403579143f2c7143372eb5498c3a10aca5d87d791a140239ffdece5a"
    ),
    "Research/translation/benchmark/public-stress-v2.jsonl": (
        "2a29b15a0d06b8eb271c094c7cedec0206a0c4f6284bff4bf2618d7641115f1f"
    ),
    "Research/translation/benchmark/public-stress-v3.jsonl": (
        "ffff66c6cd6b7458785c2217b931e9aec9155b7441769700fc082bb97fe9db06"
    ),
    "Research/translation/benchmark/legal-safety-validation-v1.jsonl": (
        "352b04c12a17480ffd3e41ea89afef6caf00f0b0aae640050398898a3e81bc91"
    ),
    "Research/translation/benchmark/legal-safety-test-v1.jsonl": (
        "ea27ac27bb23e99dd3d4fe29b70bab7ebb660fcac9309b3d06f08e9124ca91ca"
    ),
    "Research/translation/benchmark/m2m100-418m-feasibility-v1.jsonl": (
        "344d7f460704ac61620270122dbdf8e3bf0ba1ba5fc940562e7b0d226bda3519"
    ),
    "Research/translation/benchmark/development-accuracy-v1.jsonl": (
        "0684350a4c941a7fc87801444c027e0f5c02ba6f77418c792428d4200b521605"
    ),
    "Research/translation/benchmark/development-accuracy-v1.segments.jsonl": (
        "b464dfbfe128b3e0cbac2652d8db6ddb976536638676a1dca6e07507eb32a58f"
    ),
    "Research/translation/benchmark/automated-claim-v1.sources.jsonl": (
        "f039ce456c55f051e8bbcc13ed9bc8270a722819308e008b39da7f30327ec16c"
    ),
}
SAMPLING_ARMS = (
    {
        "name": "sample-t0.8-p0.9",
        "temperature": 0.8,
        "top_p": 0.9,
        "count": 2,
    },
    {
        "name": "sample-t1.1-p0.95",
        "temperature": 1.1,
        "top_p": 0.95,
        "count": 2,
    },
)
ALLOWED_LICENSES = {
    "CC-BY-2.0-FR",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "project-owned",
}


def repeated_token_loop(ids: list[int]) -> bool:
    for width in range(3, min(16, len(ids) // 3) + 1):
        for start in range(len(ids) - width * 3 + 1):
            phrase = ids[start : start + width]
            if (
                ids[start + width : start + width * 2] == phrase
                and ids[start + width * 2 : start + width * 3] == phrase
            ):
                return True
    return False


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise SystemExit(f"expected non-empty JSONL: {path}")
    return values


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def stable_digest(seed: int, role: str, *values: str) -> str:
    return hashlib.sha256(
        "\0".join((str(seed), role, *values)).encode("utf-8")
    ).hexdigest()


def output_record(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def validate_direction_inputs(
    direction: str,
    dataset: Path,
    model: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = DIRECTION_CONFIG[direction]
    manifest_path = dataset / "manifest.json"
    train_path = dataset / "train.jsonl"
    valid_path = dataset / "valid.jsonl"
    model_path = model / "model.safetensors"
    model_manifest_path = model / "mimi_training_manifest.json"
    for path in (
        manifest_path,
        train_path,
        valid_path,
        model_path,
        model_manifest_path,
    ):
        if not path.is_file():
            raise SystemExit(f"missing V17 input: {path}")
    manifest = load_json(manifest_path)
    if (
        sha256(manifest_path) != config["dataset_manifest_sha256"]
        or sha256(train_path) != config["dataset_train_sha256"]
        or manifest.get("direction") != direction
        or manifest.get("target_source") != "licensed-human-reference"
        or manifest.get("promotion_eligible") is not False
        or manifest.get("counts", {}).get("train") != config["dataset_train_rows"]
        or manifest.get("outputs", {}).get("train", {}).get("sha256")
        != sha256(train_path)
        or manifest.get("outputs", {}).get("valid", {}).get("sha256")
        != sha256(valid_path)
    ):
        raise SystemExit(f"V17 {direction} dataset identity or rights state differs")
    model_manifest = load_json(model_manifest_path)
    if (
        sha256(model_path) != config["model_sha256"]
        or sha256(model_manifest_path) != config["model_manifest_sha256"]
        or model_manifest.get("direction") != direction
        or model_manifest.get("license") != "CC-BY-SA-4.0"
    ):
        raise SystemExit(f"V17 {direction} safe parent differs")
    train_rows = load_jsonl(train_path)
    valid_rows = load_jsonl(valid_path)
    if len(train_rows) != config["dataset_train_rows"]:
        raise SystemExit(f"V17 {direction} training row count differs")
    expected_languages = (
        config["source_language"],
        config["target_language"],
    )
    for row in train_rows:
        if (
            (row.get("source_language"), row.get("target_language"))
            != expected_languages
            or str(row.get("source_license", "")) not in ALLOWED_LICENSES
            or not str(row.get("source_provenance", "")).strip()
            or not str(row.get("source", "")).strip()
            or not str(row.get("target", "")).strip()
        ):
            raise SystemExit(f"V17 {direction} row lacks licensed provenance")
    return train_rows, valid_rows, manifest


def validate_generation_model(
    direction: str,
    generation_model: Path,
) -> dict[str, Any]:
    manifest_path = generation_model / "manifest.json"
    weights_path = generation_model / "model.safetensors"
    tokenizer_path = generation_model / "tokenizer.json"
    for path in (manifest_path, weights_path, tokenizer_path):
        if not path.is_file():
            raise SystemExit(f"missing V17 MLX generation input: {path}")
    manifest = load_json(manifest_path)
    config = DIRECTION_CONFIG[direction]
    declared_weights = manifest.get("files", {}).get("model.safetensors", {})
    declared_tokenizer = manifest.get("files", {}).get("tokenizer.json", {})
    if (
        manifest.get("format") != "mimi-mlx-marian-v1"
        or manifest.get("direction") != direction
        or manifest.get("source_weights_sha256") != config["model_sha256"]
        or manifest.get("license") != "CC-BY-SA-4.0"
        or manifest.get("bits") != 4
        or manifest.get("group_size") != 64
        or declared_weights.get("sha256") != sha256(weights_path)
        or declared_weights.get("bytes") != weights_path.stat().st_size
        or declared_tokenizer.get("sha256") != sha256(tokenizer_path)
        or declared_tokenizer.get("bytes") != tokenizer_path.stat().st_size
    ):
        raise SystemExit(
            f"V17 {direction} MLX model is not an authenticated q4 conversion "
            "of the exact full-precision safe parent"
        )
    return manifest


def validate_protected(root: Path) -> list[Path]:
    paths = []
    for relative, expected in PROTECTED_SUITES.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"V17 protected suite differs: {relative}")
        paths.append(path)
    return paths


def select_sources(
    rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    protected: ProtectedIndex,
    direction: str,
    row_limit: int,
    maximum_source_tokens: int,
    maximum_target_tokens: int,
    seed: int,
    maximum_jaccard: float,
    ngram_size: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    blocked = {
        normalized(str(row.get(field, "")))
        for row in valid_rows
        for field in ("source", "target")
        if str(row.get(field, "")).strip()
    }
    rejected: Counter[str] = Counter()
    seen_sources: set[str] = set()
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain") or "unknown")].append(row)
    for domain, domain_rows in by_domain.items():
        domain_rows.sort(
            key=lambda row: (
                stable_digest(
                    seed,
                    f"source:{direction}:{domain}",
                    str(row["id"]),
                    str(row["source"]),
                ),
                str(row["id"]),
            )
        )
    selected: list[dict[str, Any]] = []
    domains = sorted(by_domain)
    while len(selected) < row_limit:
        progressed = False
        for domain in domains:
            while by_domain[domain]:
                row = by_domain[domain].pop(0)
                progressed = True
                source = str(row["source"])
                target = str(row["target"])
                source_key = normalized(source)
                if source_key in seen_sources:
                    rejected["duplicate-source"] += 1
                    continue
                if source_key in blocked or normalized(target) in blocked:
                    rejected["validation-overlap"] += 1
                    continue
                if (
                    protected.match(source, maximum_jaccard, ngram_size)
                    is not None
                ):
                    rejected["protected-source-overlap"] += 1
                    continue
                if (
                    protected.match(target, maximum_jaccard, ngram_size)
                    is not None
                ):
                    rejected["protected-target-overlap"] += 1
                    continue
                source_tokens = tokenizer(
                    source,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
                target_tokens = tokenizer(
                    text_target=target,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
                if len(source_tokens) > maximum_source_tokens:
                    rejected["source-too-long"] += 1
                    continue
                if len(target_tokens) > maximum_target_tokens:
                    rejected["target-too-long"] += 1
                    continue
                seen_sources.add(source_key)
                selected.append(
                    {
                        **row,
                        "direction": direction,
                        "source_token_count": len(source_tokens),
                        "reference_token_count": len(target_tokens),
                        "long_source": (
                            row.get("origin") == "human-alt-document-window"
                            or len(source_tokens) >= 64
                        ),
                    }
                )
                break
            if len(selected) == row_limit:
                break
        if not progressed:
            break
    if len(selected) != row_limit:
        raise SystemExit(
            f"V17 {direction} has {len(selected)} eligible rows; "
            f"{row_limit} required"
        )
    selected.sort(
        key=lambda row: (
            stable_digest(
                seed,
                f"selected:{direction}",
                str(row["id"]),
                str(row["source"]),
            ),
            str(row["id"]),
        )
    )
    return selected, rejected


def sample_mlx_candidate(
    model: Any,
    input_ids: list[int],
    *,
    temperature: float,
    top_p: float,
    seed: int,
    maximum_target_tokens: int,
) -> tuple[list[int], bool]:
    import mlx.core as mx
    import numpy as np

    if not 0 < top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    rng = np.random.default_rng(seed)
    encoder_states = model.encode(mx.array([input_ids], dtype=mx.int32))
    decoder_id = 32_000
    caches = None
    output: list[int] = []
    for position in range(maximum_target_tokens):
        logits, caches = model.decode_step(
            decoder_id,
            encoder_states,
            caches,
            position,
        )
        values = np.asarray(logits[0, -1].astype(mx.float32))
        values[32_000] = -math.inf
        values = values / temperature
        shifted = values - np.max(values)
        all_probabilities = np.exp(shifted)
        total_probability = all_probabilities.sum()
        shortlist_size = min(256, len(values))
        while True:
            shortlist = np.argpartition(
                all_probabilities,
                -shortlist_size,
            )[-shortlist_size:]
            order = shortlist[
                np.argsort(all_probabilities[shortlist])[::-1]
            ]
            probabilities = all_probabilities[order]
            cumulative = np.cumsum(probabilities) / total_probability
            if cumulative[-1] >= top_p or shortlist_size == len(values):
                break
            shortlist_size = min(len(values), shortlist_size * 2)
        keep = int(np.searchsorted(cumulative, top_p, side="left")) + 1
        order = order[:keep]
        probabilities = probabilities[:keep]
        probabilities = probabilities / probabilities.sum()
        token = int(rng.choice(order, p=probabilities))
        if token == 0:
            return output, True
        output.append(token)
        decoder_id = token
    return output, False


def generated_rows_mlx(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    direction: str,
    maximum_target_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for row in rows:
        input_ids = tokenizer(
            str(row["source"]),
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        candidates: list[dict[str, Any]] = []
        for position, (token_ids, _score) in enumerate(
            model.generate_beam_nbest(
                input_ids,
                beam_size=4,
                maximum_tokens=maximum_target_tokens,
                length_penalty=1.0,
                num_return_sequences=4,
            )
        ):
            candidates.append(
                {
                    "generation_origin": "beam-4",
                    "origin_position": position,
                    "generation_seed": None,
                    "token_ids": token_ids,
                    "hypothesis": tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                    ),
                    "terminated": bool(
                        len(token_ids) < maximum_target_tokens
                    ),
                    "reached_generation_limit": bool(
                        len(token_ids) >= maximum_target_tokens
                    ),
                    "repeated_token_loop": repeated_token_loop(token_ids),
                }
            )
        for arm in SAMPLING_ARMS:
            for position in range(int(arm["count"])):
                candidate_seed = int(
                    stable_digest(
                        seed,
                        f"mlx-sample:{direction}:{arm['name']}",
                        str(row["id"]),
                        str(position),
                    )[:16],
                    16,
                )
                token_ids, terminated = sample_mlx_candidate(
                    model,
                    input_ids,
                    temperature=float(arm["temperature"]),
                    top_p=float(arm["top_p"]),
                    seed=candidate_seed,
                    maximum_target_tokens=maximum_target_tokens,
                )
                candidates.append(
                    {
                        "generation_origin": arm["name"],
                        "origin_position": position,
                        "generation_seed": candidate_seed,
                        "token_ids": token_ids,
                        "hypothesis": tokenizer.decode(
                            token_ids,
                            skip_special_tokens=True,
                        ),
                        "terminated": terminated,
                        "reached_generation_limit": (
                            not terminated
                            and len(token_ids) >= maximum_target_tokens
                        ),
                        "repeated_token_loop": repeated_token_loop(token_ids),
                    }
                )

        by_text: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            key = normalized(str(candidate["hypothesis"]))
            if key in by_text:
                by_text[key]["generation_origins"].append(
                    {
                        "name": candidate["generation_origin"],
                        "position": candidate["origin_position"],
                        "seed": candidate["generation_seed"],
                    }
                )
                by_text[key]["reached_generation_limit"] = bool(
                    by_text[key]["reached_generation_limit"]
                    or candidate["reached_generation_limit"]
                )
                by_text[key]["repeated_token_loop"] = bool(
                    by_text[key]["repeated_token_loop"]
                    or candidate["repeated_token_loop"]
                )
                continue
            candidate_id = (
                "v17-candidate-"
                + stable_digest(
                    seed,
                    f"candidate:{direction}",
                    str(row["id"]),
                    str(candidate["hypothesis"]),
                )[:20]
            )
            by_text[key] = {
                "candidate_id": candidate_id,
                "hypothesis": candidate["hypothesis"],
                "token_ids": candidate["token_ids"],
                "terminated": candidate["terminated"],
                "reached_generation_limit": candidate[
                    "reached_generation_limit"
                ],
                "repeated_token_loop": candidate["repeated_token_loop"],
                "generation_origins": [
                    {
                        "name": candidate["generation_origin"],
                        "position": candidate["origin_position"],
                        "seed": candidate["generation_seed"],
                    }
                ],
            }
        outputs.append(
            {
                **row,
                "candidate_list": sorted(
                    by_text.values(),
                    key=lambda candidate: str(candidate["candidate_id"]),
                ),
            }
        )
    return outputs


def sequence_scores(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    device: Any,
    batch_size: int,
) -> dict[tuple[str, str], dict[str, float | int]]:
    import torch
    import torch.nn.functional as torch_functional

    flat: list[tuple[str, str, str, str]] = []
    for row in rows:
        row_id = str(row["id"])
        flat.append((row_id, "reference", str(row["source"]), str(row["target"])))
        flat.extend(
            (
                row_id,
                str(candidate["candidate_id"]),
                str(row["source"]),
                str(candidate["hypothesis"]),
            )
            for candidate in row["candidate_list"]
        )
    result: dict[tuple[str, str], dict[str, float | int]] = {}
    with torch.inference_mode():
        for offset in range(0, len(flat), batch_size):
            values = flat[offset : offset + batch_size]
            batch = tokenizer(
                [value[2] for value in values],
                text_target=[value[3] for value in values],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            labels = batch["labels"]
            labels[labels == int(tokenizer.pad_token_id)] = -100
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            labels = batch["labels"]
            mask = labels.ne(-100)
            selected = (
                torch_functional.log_softmax(logits.float(), dim=-1)
                .gather(2, labels.clamp_min(0).unsqueeze(2))
                .squeeze(2)
            )
            totals = (selected * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp_min(1)
            means = totals / lengths
            for value, total, mean, length in zip(
                values,
                totals.cpu(),
                means.cpu(),
                lengths.cpu(),
                strict=True,
            ):
                result[(value[0], value[1])] = {
                    "total_log_probability": float(total),
                    "mean_log_probability": float(mean),
                    "token_count": int(length),
                }
    return result


def deterministic_risk_tags(
    row: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    source = str(row["source"])
    reference = str(row["target"])
    hypothesis = str(candidate["hypothesis"])
    source_language = str(row["source_language"])
    target_language = str(row["target_language"])
    tags = []
    if critical_tokens(source) != critical_tokens(hypothesis):
        tags.append("exact-critical-mismatch")
    if not typed_preserves(
        source,
        hypothesis,
        source_language,
        target_language,
    ):
        tags.append("typed-critical-mismatch")
    if tokens(source)["negative"] != tokens(hypothesis)["negative"]:
        tags.append("negation-marker-mismatch")
    if candidate["repeated_token_loop"]:
        tags.append("repetition-risk")
    if candidate["reached_generation_limit"] or not hypothesis.strip():
        tags.append("generation-risk")
    reference_tokens = int(row["reference_score"]["token_count"])
    candidate_tokens = int(candidate["score"]["token_count"])
    ratio = candidate_tokens / max(1, reference_tokens)
    if ratio <= 0.75:
        tags.append("omission-risk")
    if ratio >= 1.35:
        tags.append("addition-risk")
    if (
        target_language == "ja-JP"
        and (
            "negation-marker-mismatch" in tags
            or "exact-critical-mismatch" in tags
            or any(
                marker in reference
                for marker in (
                    "ない",
                    "ません",
                    "禁止",
                    "不可",
                    "ください",
                    "です",
                    "ます",
                    "ございます",
                    "いたします",
                )
            )
        )
    ):
        tags.append("japanese-sensitive-audit")
    return sorted(set(tags))


def preselect_scored_rows(
    rows: list[dict[str, Any]],
    *,
    near_margin: float,
    maximum_pairs_per_source: int,
) -> list[dict[str, Any]]:
    preselected = []
    for row in rows:
        row_id = str(row["id"])
        reference_score = row["reference_score"]
        candidates = row["candidate_list"]
        eligible = [
            candidate
            for candidate in candidates
            if (
                not candidate["exact_reference_match"]
                and float(candidate["reference_minus_candidate_margin"])
                <= near_margin
                and (
                    float(candidate["chrf_plus_plus"]) < 95.0
                    or bool(candidate["deterministic_risk_tags"])
                )
            )
        ]
        eligible.sort(
            key=lambda candidate: (
                0 if candidate["deterministic_risk_tags"] else 1,
                float(candidate["reference_minus_candidate_margin"]),
                float(candidate["chrf_plus_plus"]),
                str(candidate["candidate_id"]),
            )
        )
        for candidate in eligible[:maximum_pairs_per_source]:
            pair_id = "v17-pair-" + hashlib.sha256(
                f"{row_id}\0{candidate['candidate_id']}".encode()
            ).hexdigest()[:20]
            preselected.append(
                {
                    "pair_id": pair_id,
                    "source_id": row_id,
                    "candidate_id": candidate["candidate_id"],
                    "direction": row["direction"],
                    "domain": row.get("domain"),
                    "origin": row.get("origin"),
                    "source": row["source"],
                    "reference": row["target"],
                    "hypothesis": candidate["hypothesis"],
                    "source_language": row["source_language"],
                    "target_language": row["target_language"],
                    "source_license": row["source_license"],
                    "source_provenance": row["source_provenance"],
                    "reference_mean_log_probability": reference_score[
                        "mean_log_probability"
                    ],
                    "candidate_mean_log_probability": candidate["score"][
                        "mean_log_probability"
                    ],
                    "reference_minus_candidate_margin": candidate[
                        "reference_minus_candidate_margin"
                    ],
                    "chrf_plus_plus": candidate["chrf_plus_plus"],
                    "sacrebleu_intl": candidate["sacrebleu_intl"],
                    "deterministic_risk_tags": candidate[
                        "deterministic_risk_tags"
                    ],
                    "positive_target_source": "licensed-human-reference",
                    "generated_strings_are_positive_targets": False,
                    "semantic_status": "not-yet-judged",
                }
            )
    preselected.sort(key=lambda pair: str(pair["pair_id"]))
    return preselected


def score_and_preselect(
    rows: list[dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, float | int]],
    *,
    near_margin: float,
    maximum_pairs_per_source: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import sacrebleu

    output_rows = []
    for row in rows:
        row_id = str(row["id"])
        reference_score = scores[(row_id, "reference")]
        current = {
            **row,
            "reference_score": reference_score,
        }
        candidates = []
        for candidate in row["candidate_list"]:
            candidate_id = str(candidate["candidate_id"])
            hypothesis = str(candidate["hypothesis"])
            reference = str(row["target"])
            score = scores[(row_id, candidate_id)]
            candidate_value = {
                **candidate,
                "score": score,
                "reference_minus_candidate_margin": (
                    float(reference_score["mean_log_probability"])
                    - float(score["mean_log_probability"])
                ),
                "chrf_plus_plus": float(
                    sacrebleu.sentence_chrf(
                        hypothesis,
                        [reference],
                        word_order=2,
                    ).score
                ),
                "sacrebleu_intl": float(
                    sacrebleu.sentence_bleu(
                        hypothesis,
                        [reference],
                        tokenize="intl",
                    ).score
                ),
            }
            candidate_value["deterministic_risk_tags"] = (
                deterministic_risk_tags(current, candidate_value)
            )
            candidate_value["exact_reference_match"] = (
                normalized(hypothesis) == normalized(reference)
            )
            candidates.append(candidate_value)
        candidates.sort(
            key=lambda candidate: (
                -float(candidate["score"]["mean_log_probability"]),
                str(candidate["candidate_id"]),
            )
        )
        current["candidate_list"] = candidates
        output_rows.append(current)
    return output_rows, preselect_scored_rows(
        output_rows,
        near_margin=near_margin,
        maximum_pairs_per_source=maximum_pairs_per_source,
    )


def build_comet_inputs(
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suite_rows = []
    results = []
    for pair in pairs:
        case_id = str(pair["pair_id"])
        case = {
            "id": case_id,
            "sourceLanguage": pair["source_language"],
            "targetLanguage": pair["target_language"],
            "domain": pair.get("domain") or "unknown",
            "source": pair["source"],
            "references": [pair["reference"]],
        }
        suite_rows.append(case)
        results.append(
            {
                **case,
                "caseID": case_id,
                "hypothesis": pair["hypothesis"],
            }
        )
    return suite_rows, {
        "schemaVersion": 1,
        "engine": "v17-safe-parent-preselected-candidates",
        "purpose": "COMET candidate-mining input; not an evaluation result",
        "results": results,
    }


def count_tags(pairs: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                tag
                for pair in pairs
                for tag in pair["deterministic_risk_tags"]
            ).items()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--en-ja-dataset", type=Path, required=True)
    parser.add_argument("--ja-en-dataset", type=Path, required=True)
    parser.add_argument("--en-ja-model", type=Path, required=True)
    parser.add_argument("--ja-en-model", type=Path, required=True)
    parser.add_argument("--en-ja-generation-model", type=Path, required=True)
    parser.add_argument("--ja-en-generation-model", type=Path, required=True)
    parser.add_argument("--row-limit-per-direction", type=int, default=1_024)
    parser.add_argument("--score-batch-size", type=int, default=24)
    parser.add_argument("--maximum-source-tokens", type=int, default=192)
    parser.add_argument("--maximum-target-tokens", type=int, default=192)
    parser.add_argument("--near-margin", type=float, default=0.25)
    parser.add_argument("--maximum-pairs-per-source", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--character-ngram-size", type=int, default=5)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    parser.add_argument(
        "--device",
        choices=("mps", "cpu", "cuda"),
        default="mps",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if min(
        args.row_limit_per_direction,
        args.score_batch_size,
        args.maximum_source_tokens,
        args.maximum_target_tokens,
        args.maximum_pairs_per_source,
        args.character_ngram_size,
    ) < 1:
        raise SystemExit("V17 sizes must be positive")
    if args.near_margin <= 0:
        raise SystemExit("near-margin must be positive")
    if not 0 <= args.maximum_jaccard < 1:
        raise SystemExit("maximum-jaccard must be in [0, 1)")

    import torch
    from marian_mlx import load_model as load_mlx_marian
    from transformers import MarianMTModel, MarianTokenizer

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device(args.device)
    root = Path(__file__).resolve().parents[2]
    protected_paths = validate_protected(root)
    protected = ProtectedIndex(
        protected_paths,
        args.character_ngram_size,
    )
    datasets = {
        "en-ja": args.en_ja_dataset,
        "ja-en": args.ja_en_dataset,
    }
    models = {
        "en-ja": args.en_ja_model,
        "ja-en": args.ja_en_model,
    }
    generation_models = {
        "en-ja": args.en_ja_generation_model,
        "ja-en": args.ja_en_generation_model,
    }
    selected_all = []
    candidate_all = []
    preselected_all = []
    rejected_by_direction: dict[str, dict[str, int]] = {}
    input_manifests = {}
    for direction in ("en-ja", "ja-en"):
        train_rows, valid_rows, dataset_manifest = validate_direction_inputs(
            direction,
            datasets[direction],
            models[direction],
        )
        tokenizer = MarianTokenizer.from_pretrained(models[direction])
        generation_manifest = validate_generation_model(
            direction,
            generation_models[direction],
        )
        selected, rejected = select_sources(
            train_rows,
            valid_rows,
            tokenizer=tokenizer,
            protected=protected,
            direction=direction,
            row_limit=args.row_limit_per_direction,
            maximum_source_tokens=args.maximum_source_tokens,
            maximum_target_tokens=args.maximum_target_tokens,
            seed=args.seed,
            maximum_jaccard=args.maximum_jaccard,
            ngram_size=args.character_ngram_size,
        )
        mlx_model = load_mlx_marian(
            generation_models[direction] / "model.safetensors",
            quantization_bits=4,
            quantization_group_size=64,
        )
        generated = generated_rows_mlx(
            mlx_model,
            tokenizer,
            selected,
            direction=direction,
            maximum_target_tokens=args.maximum_target_tokens,
            seed=args.seed,
        )
        del mlx_model
        gc.collect()
        model = MarianMTModel.from_pretrained(models[direction]).to(device).eval()
        scores = sequence_scores(
            model,
            tokenizer,
            generated,
            device=device,
            batch_size=args.score_batch_size,
        )
        candidate_rows, preselected = score_and_preselect(
            generated,
            scores,
            near_margin=args.near_margin,
            maximum_pairs_per_source=args.maximum_pairs_per_source,
        )
        selected_all.extend(selected)
        candidate_all.extend(candidate_rows)
        preselected_all.extend(preselected)
        rejected_by_direction[direction] = dict(sorted(rejected.items()))
        input_manifests[direction] = {
            "dataset_manifest_sha256": sha256(
                datasets[direction] / "manifest.json"
            ),
            "dataset_train_sha256": sha256(
                datasets[direction] / "train.jsonl"
            ),
            "dataset_valid_sha256": sha256(
                datasets[direction] / "valid.jsonl"
            ),
            "model_sha256": sha256(models[direction] / "model.safetensors"),
            "model_manifest_sha256": sha256(
                models[direction] / "mimi_training_manifest.json"
            ),
            "generation_model_manifest_sha256": sha256(
                generation_models[direction] / "manifest.json"
            ),
            "generation_model_sha256": sha256(
                generation_models[direction] / "model.safetensors"
            ),
            "generation_model_source_weights_sha256": generation_manifest[
                "source_weights_sha256"
            ],
            "dataset_effective_licenses": dataset_manifest[
                "effective_licenses"
            ]["train"],
        }
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    selected_all.sort(
        key=lambda row: (str(row["direction"]), str(row["id"]))
    )
    candidate_all.sort(
        key=lambda row: (str(row["direction"]), str(row["id"]))
    )
    preselected_all.sort(key=lambda row: str(row["pair_id"]))
    args.output.mkdir(parents=True, exist_ok=True)
    selected_path = args.output / "selected-sources.jsonl"
    candidate_path = args.output / "candidate-lists.jsonl"
    preselected_path = args.output / "preselected-pairs.jsonl"
    comet_suite_path = args.output / "comet-suite.jsonl"
    comet_report_path = args.output / "comet-engine-report.json"
    write_jsonl(selected_path, selected_all)
    write_jsonl(candidate_path, candidate_all)
    write_jsonl(preselected_path, preselected_all)
    comet_suite, comet_report = build_comet_inputs(preselected_all)
    write_jsonl(comet_suite_path, comet_suite)
    comet_report_path.write_text(
        json.dumps(
            comet_report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    selected_counts = Counter(str(row["direction"]) for row in selected_all)
    pair_counts = Counter(str(row["direction"]) for row in preselected_all)
    origin_counts = Counter(str(row["origin"]) for row in selected_all)
    domain_counts = Counter(
        f"{row['direction']}:{row.get('domain') or 'unknown'}"
        for row in selected_all
    )
    candidate_counts = Counter(
        str(row["direction"])
        for row in candidate_all
        for _candidate in row["candidate_list"]
    )
    stage_one_gates = {
        "total_preselected_at_least_1500": len(preselected_all) >= 1_500,
        "en_ja_preselected_at_least_600": pair_counts["en-ja"] >= 600,
        "ja_en_preselected_at_least_600": pair_counts["ja-en"] >= 600,
    }
    stage_one_passed = all(stage_one_gates.values())
    manifest_path = args.output / "manifest.json"
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": (
            "presemantic-candidate-availability-passed"
            if stage_one_passed
            else "presemantic-candidate-availability-rejected"
        ),
        "training_authorized": False,
        "semantic_audit_authorized": stage_one_passed,
        "comet_candidate_scoring_authorized": stage_one_passed,
        "generated_strings_are_positive_targets": False,
        "positive_target_source": "licensed-human-reference",
        "private_reasoning_traces_used": False,
        "selection": {
            "seed": args.seed,
            "rows_per_direction": args.row_limit_per_direction,
            "balanced_round_robin_by_domain": True,
            "maximum_source_tokens": args.maximum_source_tokens,
            "maximum_target_tokens": args.maximum_target_tokens,
            "candidate_generation": {
                "beam": {
                    "num_beams": 4,
                    "num_return_sequences": 4,
                    "length_penalty": 1.0,
                },
                "sampling_arms": list(SAMPLING_ARMS),
                "per_candidate_sha256_seed": True,
                "rollout_precision": "MLX affine q4 group-64",
                "confidence_precision": "full-precision safe parent",
            },
            "near_margin_maximum": args.near_margin,
            "maximum_pairs_per_source": args.maximum_pairs_per_source,
        },
        "decontamination": {
            "normalization": "Unicode NFKC, casefold, remove whitespace",
            "character_ngram_size": args.character_ngram_size,
            "maximum_jaccard_exclusive": args.maximum_jaccard,
            "protected_suites": [
                {
                    "path": str(path.relative_to(root)),
                    "sha256": sha256(path),
                }
                for path in protected_paths
            ],
            "rejected": rejected_by_direction,
        },
        "inputs": input_manifests,
        "counts": {
            "selected_sources": {
                **dict(sorted(selected_counts.items())),
                "total": len(selected_all),
                "long": sum(bool(row["long_source"]) for row in selected_all),
            },
            "generated_unique_candidates": {
                **dict(sorted(candidate_counts.items())),
                "total": sum(candidate_counts.values()),
            },
            "preselected_pairs": {
                **dict(sorted(pair_counts.items())),
                "total": len(preselected_all),
                "negative_preferred": sum(
                    float(row["reference_minus_candidate_margin"]) <= 0
                    for row in preselected_all
                ),
            },
            "selected_origins": dict(sorted(origin_counts.items())),
            "selected_direction_domains": dict(sorted(domain_counts.items())),
            "deterministic_risk_tags": count_tags(preselected_all),
        },
        "stage_one_gates": stage_one_gates,
        "next_gate": (
            "pinned COMET plus exact independent Claude Sonnet 5 and Opus 5 "
            "faithfulness audit, followed by safety-versus-retention gradient audit"
            if stage_one_passed
            else "stop V17 without COMET, judges, gradients, contract, or training"
        ),
        "reproducibility": {
            "python": platform.python_version(),
            "packages": {
                package: importlib.metadata.version(package)
                for package in (
                    "mlx",
                    "numpy",
                    "sacrebleu",
                    "torch",
                    "transformers",
                )
            },
            "device_type": device.type,
            "output_manifest_excludes_wall_clock_time_and_absolute_output_path": True,
        },
        "outputs": {
            "selected_sources": output_record(selected_path),
            "candidate_lists": output_record(candidate_path),
            "preselected_pairs": output_record(preselected_path),
            "comet_suite": output_record(comet_suite_path),
            "comet_engine_report": output_record(comet_report_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected": manifest["counts"]["selected_sources"],
                "candidates": manifest["counts"]["generated_unique_candidates"],
                "preselected": manifest["counts"]["preselected_pairs"],
                "risk_tags": manifest["counts"]["deterministic_risk_tags"],
                "manifest_sha256": sha256(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
