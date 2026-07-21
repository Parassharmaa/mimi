#!/usr/bin/env python3
"""Build direction-safe Qwen-target or matched human-reference Marian data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import unicodedata
from collections import Counter
from pathlib import Path


DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
ALLOWED_LICENSES = {
    "Apache-2.0",
    "CC-BY-2.0-FR",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC0-1.0",
    "MIT",
    "project-owned",
}
TEACHER_MODEL = "mlx-community/Qwen3-8B-4bit"
TEACHER_REVISION = "545dc4251c05440727734bcd94334791f6ab0192"
TEACHER_LICENSE = "Apache-2.0"


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing JSON input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text)
    if not value:
        return set()
    if len(value) < size:
        return {value}
    return {value[index:index + size] for index in range(len(value) - size + 1)}


def near(text: str, protected: list[set[str]], maximum: float) -> bool:
    candidate = ngrams(text)
    return bool(candidate) and any(
        len(candidate & heldout) / max(1, len(candidate | heldout)) > maximum
        for heldout in protected
    )


def validate_direction(values: list[dict], expected: tuple[str, str], label: str) -> None:
    for row in values:
        actual = (row.get("source_language"), row.get("target_language"))
        if actual != expected:
            raise SystemExit(f"{label} has wrong direction {actual}: {row.get('id')}")


def effective_license(row: dict) -> str:
    return str(row.get("source_license") or row.get("license") or "").strip()


def policy_at_least(policy: dict, key: str, minimum: float) -> bool:
    try:
        value = float(policy[key])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(value) and value >= minimum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("accepted_teacher_rows", type=Path)
    parser.add_argument("reference_suite", type=Path)
    parser.add_argument("base_dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=tuple(DIRECTIONS), required=True)
    parser.add_argument("--target-source", choices=("qwen", "human-reference"), default="qwen")
    parser.add_argument("--protected-suite", type=Path, action="append", required=True)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    parser.add_argument("--maximum-teacher-rows", type=int)
    parser.add_argument("--seed", default="mimi-local-reference-teacher-ablation-v1")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.maximum_teacher_rows is not None and args.maximum_teacher_rows < 1:
        raise SystemExit("maximum-teacher-rows must be positive")
    if not 0 <= args.maximum_jaccard <= 1:
        raise SystemExit("maximum-jaccard must be between zero and one")

    expected = DIRECTIONS[args.direction]
    base_train_path = args.base_dataset / "train.jsonl"
    base_valid_path = args.base_dataset / "valid.jsonl"
    base_train, base_valid = rows(base_train_path), rows(base_valid_path)
    validate_direction(base_train, expected, "base train")
    validate_direction(base_valid, expected, "base validation")
    for row in [*base_train, *base_valid]:
        if effective_license(row) not in ALLOWED_LICENSES:
            raise SystemExit(f"base row has a non-approved or missing license: {row.get('id')}")

    accepted_all = rows(args.accepted_teacher_rows)
    accepted = [
        row
        for row in accepted_all
        if (row.get("source_language"), row.get("target_language")) == expected
    ]
    if not accepted:
        raise SystemExit(f"accepted teacher rows have no {args.direction} cases")
    manifest_path = args.accepted_teacher_rows.with_suffix(
        args.accepted_teacher_rows.suffix + ".manifest.json"
    )
    filter_manifest = load_json(manifest_path)
    suite_manifest_path = args.reference_suite.with_suffix(
        args.reference_suite.suffix + ".manifest.json"
    )
    suite_manifest = load_json(suite_manifest_path)
    filter_policy = filter_manifest.get("policy", {})
    if (
        filter_manifest.get("promotion_eligible") is not False
        or filter_manifest.get("purpose")
        != "reviewer-free hidden-reference local teacher filter; never promotion evidence"
        or filter_manifest.get("inputs", {}).get("suite", {}).get("sha256")
        != sha256(args.reference_suite)
        or filter_manifest.get("inputs", {}).get("suite_manifest", {}).get("sha256")
        != sha256(suite_manifest_path)
        or suite_manifest.get("outputs", {}).get("suite", {}).get("sha256")
        != sha256(args.reference_suite)
        or set(suite_manifest.get("allowed_licenses", [])) != ALLOWED_LICENSES
        or filter_manifest.get("outputs", {}).get("accepted", {}).get("sha256")
        != sha256(args.accepted_teacher_rows)
        or filter_manifest.get("teacher") != {
            "repository": TEACHER_MODEL,
            "revision": TEACHER_REVISION,
            "license": TEACHER_LICENSE,
        }
        or not policy_at_least(filter_policy, "minimum_teacher_comet_22", 0.85)
        or not policy_at_least(filter_policy, "minimum_comet_22_delta", 0.01)
        or not policy_at_least(filter_policy, "minimum_teacher_chrf_pp", 25.0)
        or not policy_at_least(filter_policy, "minimum_chrf_pp_delta", 2.0)
    ):
        raise SystemExit("accepted rows lack the exact hidden-reference filter manifest")

    prepared_jaccard = float(
        suite_manifest.get("maximum_protected_five_gram_jaccard", -1)
    )
    if args.maximum_jaccard > prepared_jaccard:
        raise SystemExit("builder contamination threshold is weaker than preparation")

    prepared_protected_hashes = {
        str(item.get("sha256", ""))
        for item in suite_manifest.get("inputs", {}).get("protected_suites", [])
    }
    build_protected_hashes = {sha256(path) for path in args.protected_suite}
    if not prepared_protected_hashes or build_protected_hashes != prepared_protected_hashes:
        raise SystemExit("builder protected suites differ from the preparation contract")
    prepared_exclusions = {
        str(Path(str(item.get("path", ""))).resolve()): str(item.get("sha256", ""))
        for item in suite_manifest.get("inputs", {}).get("excluded_datasets", [])
    }
    expected_base_files = {
        str(base_train_path.resolve()): sha256(base_train_path),
        str(base_valid_path.resolve()): sha256(base_valid_path),
    }
    if any(prepared_exclusions.get(path) != digest for path, digest in expected_base_files.items()):
        raise SystemExit("builder base dataset was not authenticated during preparation")

    reference_rows = rows(args.reference_suite)
    references = {str(row.get("id", "")): row for row in reference_rows}
    if not references or len(references) != len(reference_rows) or "" in references:
        raise SystemExit("reference suite has missing or duplicate IDs")
    protected = [
        ngrams(text)
        for path in args.protected_suite
        for row in rows(path)
        for text in [row.get("source", ""), *row.get("references", [])]
        if str(text).strip()
    ]
    base_sources = {normalized(str(row["source"])) for row in [*base_train, *base_valid]}

    teacher: list[dict] = []
    seen_sources: set[str] = set()
    for row in accepted:
        identifier = str(row.get("id", ""))
        source_id = str(row.get("source_id", ""))
        source = str(row.get("source", "")).strip()
        candidate = str(row.get("target", "")).strip()
        evidence = row.get("quality_control", {})
        if (
            not identifier
            or not source_id
            or not source
            or not candidate
            or row.get("origin") != "strict-local-qwen-reference-filtered"
            or row.get("review_status") != "hidden-reference-metric-filtered-provisional"
            or row.get("training_only") is not True
            or row.get("promotion_eligible") is not False
            or evidence.get("reference_exposed_to_teacher") is not False
            or evidence.get("reasoning_trace_requested_or_stored") is not False
            or effective_license(row) not in ALLOWED_LICENSES
        ):
            raise SystemExit(f"accepted row lacks strict reference-filter evidence: {identifier}")
        reference = references.get(source_id)
        if reference is None:
            raise SystemExit(f"accepted row is absent from reference suite: {identifier}")
        if (
            reference.get("source") != source
            or (reference.get("sourceLanguage"), reference.get("targetLanguage")) != expected
            or len(reference.get("references", [])) != 1
        ):
            raise SystemExit(f"accepted/reference text or direction mismatch: {identifier}")
        source_norm = normalized(source)
        if source_norm in base_sources:
            raise SystemExit(f"accepted source overlaps base train/validation: {identifier}")
        if source_norm in seen_sources:
            raise SystemExit(f"duplicate accepted source: {identifier}")
        if near(source, protected, args.maximum_jaccard) or near(
            candidate, protected, args.maximum_jaccard
        ):
            raise SystemExit(f"accepted row is near protected evaluation: {identifier}")
        seen_sources.add(source_norm)
        human_reference = str(reference["references"][0]).strip()
        target = candidate if args.target_source == "qwen" else human_reference
        teacher.append({
            **row,
            "target": target,
            "qwen_candidate": candidate,
            "origin": (
                "strict-local-qwen-reference-distillation"
                if args.target_source == "qwen"
                else "matched-licensed-human-reference-control"
            ),
            "attribution": row.get("source_provenance"),
        })

    ranked = sorted(
        teacher,
        key=lambda row: hashlib.sha256(
            f"{args.seed}\0{row['source_id']}".encode()
        ).hexdigest(),
    )
    if args.maximum_teacher_rows is not None:
        ranked = ranked[:args.maximum_teacher_rows]
    teacher_train = ranked
    if not teacher_train:
        raise SystemExit("deterministic teacher split left no training rows")

    train = [*base_train, *teacher_train]
    # Accepted rows were selected using their hidden references, so none of
    # them can be a clean checkpoint-selection slice. Preserve base validation
    # membership and order and use it as the only training-time dev set.
    valid = list(base_valid)
    random.Random(f"{args.seed}:{args.direction}:{args.target_source}:train").shuffle(train)
    args.output.mkdir(parents=True, exist_ok=True)
    train_path, valid_path = args.output / "train.jsonl", args.output / "valid.jsonl"
    for path, values in ((train_path, train), (valid_path, valid)):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values),
            encoding="utf-8",
        )
    manifest = {
        "schema_version": 1,
        "experiment": "strict local Qwen reference-filtered Marian ablation",
        "promotion_eligible": False,
        "direction": args.direction,
        "target_source": args.target_source,
        "seed": args.seed,
        "maximum_protected_five_gram_jaccard": args.maximum_jaccard,
        "validation_policy": "unchanged base validation; no reference-filtered row used for model selection",
        "maximum_teacher_rows": args.maximum_teacher_rows,
        "counts": {
            "base_train": len(base_train),
            "base_valid": len(base_valid),
            "accepted_all_directions": len(accepted_all),
            "accepted_direction": len(accepted),
            "teacher_selected": len(ranked),
            "teacher_train": len(teacher_train),
            "teacher_dev": 0,
            "train": len(train),
            "valid": len(valid),
        },
        "origins": {
            "train": dict(sorted(Counter(str(row.get("origin", "unknown")) for row in train).items())),
            "valid": dict(sorted(Counter(str(row.get("origin", "unknown")) for row in valid).items())),
        },
        "effective_licenses": {
            "train": dict(sorted(Counter(effective_license(row) for row in train).items())),
            "valid": dict(sorted(Counter(effective_license(row) for row in valid).items())),
        },
        "teacher_models": {
            TEACHER_MODEL: {"revision": TEACHER_REVISION, "license": TEACHER_LICENSE},
        },
        "inputs": {
            "accepted": {"path": str(args.accepted_teacher_rows.resolve()), "sha256": sha256(args.accepted_teacher_rows)},
            "accepted_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
            "reference_suite": {"path": str(args.reference_suite.resolve()), "sha256": sha256(args.reference_suite)},
            "reference_suite_manifest": {"path": str(suite_manifest_path.resolve()), "sha256": sha256(suite_manifest_path)},
            "base_train": {"path": str(base_train_path.resolve()), "sha256": sha256(base_train_path)},
            "base_valid": {"path": str(base_valid_path.resolve()), "sha256": sha256(base_valid_path)},
            "protected_suites": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in args.protected_suite
            ],
        },
        "outputs": {
            "train": {"path": str(train_path.resolve()), "sha256": sha256(train_path)},
            "valid": {"path": str(valid_path.resolve()), "sha256": sha256(valid_path)},
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
