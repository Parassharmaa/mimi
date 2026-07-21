#!/usr/bin/env python3
"""Build a licensed hard-source Marian arm from every balanced-suite reference."""

from __future__ import annotations

import argparse
import hashlib
import json
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
SUITE_PURPOSE = "reference-hidden local Qwen teacher training suite; never evaluation evidence"


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


def effective_license(row: dict) -> str:
    return str(row.get("source_license") or row.get("license") or "").strip()


def validate_direction(values: list[dict], expected: tuple[str, str], label: str) -> None:
    for row in values:
        actual = (row.get("source_language"), row.get("target_language"))
        if actual != expected:
            raise SystemExit(f"{label} has wrong direction {actual}: {row.get('id')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_suite", type=Path)
    parser.add_argument("base_dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=tuple(DIRECTIONS), required=True)
    parser.add_argument("--protected-suite", type=Path, action="append", required=True)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    parser.add_argument("--maximum-human-rows", type=int)
    parser.add_argument("--seed", default="mimi-balanced-human-reference-v1")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.maximum_human_rows is not None and args.maximum_human_rows < 1:
        raise SystemExit("maximum-human-rows must be positive")
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

    suite_manifest_path = args.reference_suite.with_suffix(
        args.reference_suite.suffix + ".manifest.json"
    )
    suite_manifest = load_json(suite_manifest_path)
    if (
        suite_manifest.get("purpose") != SUITE_PURPOSE
        or suite_manifest.get("promotion_eligible") is not False
        or suite_manifest.get("reference_exposed_to_teacher") is not False
        or set(suite_manifest.get("allowed_licenses", [])) != ALLOWED_LICENSES
        or suite_manifest.get("outputs", {}).get("suite", {}).get("sha256")
        != sha256(args.reference_suite)
    ):
        raise SystemExit("reference suite lacks an authentic preparation/license manifest")
    prepared_jaccard = float(
        suite_manifest.get("maximum_protected_five_gram_jaccard", -1)
    )
    if args.maximum_jaccard > prepared_jaccard:
        raise SystemExit("builder contamination threshold is weaker than preparation")
    prepared_protected = {
        str(item.get("sha256", ""))
        for item in suite_manifest.get("inputs", {}).get("protected_suites", [])
    }
    actual_protected = {sha256(path) for path in args.protected_suite}
    if not prepared_protected or actual_protected != prepared_protected:
        raise SystemExit("builder protected suites differ from suite preparation")
    prepared_exclusions = {
        str(Path(str(item.get("path", ""))).resolve()): str(item.get("sha256", ""))
        for item in suite_manifest.get("inputs", {}).get("excluded_datasets", [])
    }
    for path in (base_train_path, base_valid_path):
        if prepared_exclusions.get(str(path.resolve())) != sha256(path):
            raise SystemExit("base dataset was not authenticated during suite preparation")

    protected = [
        ngrams(text)
        for path in args.protected_suite
        for row in rows(path)
        for text in [row.get("source", ""), *row.get("references", [])]
        if str(text).strip()
    ]
    base_sources = {normalized(str(row["source"])) for row in [*base_train, *base_valid]}
    suite_rows = rows(args.reference_suite)
    selected: list[dict] = []
    seen_sources: set[str] = set()
    for row in suite_rows:
        if (row.get("sourceLanguage"), row.get("targetLanguage")) != expected:
            continue
        identifier = str(row.get("id", ""))
        source = str(row.get("source", "")).strip()
        references = [str(value).strip() for value in row.get("references", [])]
        license_name = str(row.get("sourceLicense", "")).strip()
        provenance = str(row.get("sourceProvenance", "")).strip()
        reference_provenance = str(row.get("referenceProvenance", "")).strip()
        if (
            not identifier
            or not source
            or len(references) != 1
            or not references[0]
            or license_name not in ALLOWED_LICENSES
            or not provenance
            or not reference_provenance
            or row.get("claimEligible") is not False
            or row.get("referenceExposedToTeacher") is not False
        ):
            raise SystemExit(f"suite row lacks licensed human-reference evidence: {identifier}")
        source_norm = normalized(source)
        if source_norm in base_sources:
            raise SystemExit(f"suite source overlaps base train/validation: {identifier}")
        if source_norm in seen_sources:
            raise SystemExit(f"duplicate suite source: {identifier}")
        target = references[0]
        if near(source, protected, args.maximum_jaccard) or near(
            target, protected, args.maximum_jaccard
        ):
            raise SystemExit(f"suite row is near protected evaluation: {identifier}")
        seen_sources.add(source_norm)
        selected.append({
            "id": f"balanced-human:{identifier}",
            "source_id": identifier,
            "source_language": expected[0],
            "target_language": expected[1],
            "source": source,
            "target": target,
            "domain": row.get("domain", "unknown"),
            "origin": "human-balanced-hard-reference",
            "source_license": license_name,
            "source_provenance": provenance,
            "reference_provenance": reference_provenance,
            "attribution": provenance,
            "review_status": "licensed-human-reference",
            "training_only": True,
            "promotion_eligible": False,
        })
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}\0{row['source_id']}".encode()
        ).hexdigest()
    )
    if args.maximum_human_rows is not None:
        selected = selected[:args.maximum_human_rows]
    if not selected:
        raise SystemExit(f"reference suite has no {args.direction} rows")

    train = [*base_train, *selected]
    valid = list(base_valid)
    random.Random(f"{args.seed}:{args.direction}:train").shuffle(train)
    args.output.mkdir(parents=True, exist_ok=True)
    train_path, valid_path = args.output / "train.jsonl", args.output / "valid.jsonl"
    for path, values in ((train_path, train), (valid_path, valid)):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values),
            encoding="utf-8",
        )
    manifest = {
        "schema_version": 1,
        "experiment": "balanced licensed human-reference hard-source ablation",
        "promotion_eligible": False,
        "direction": args.direction,
        "target_source": "licensed-human-reference",
        "seed": args.seed,
        "maximum_protected_five_gram_jaccard": args.maximum_jaccard,
        "maximum_human_rows": args.maximum_human_rows,
        "validation_policy": "unchanged base validation; selected hard rows remain training-only",
        "counts": {
            "base_train": len(base_train),
            "base_valid": len(base_valid),
            "human_reference_train": len(selected),
            "synthetic_train": 0,
            "train": len(train),
            "valid": len(valid),
        },
        "origins": {
            "train": dict(sorted(Counter(str(row.get("origin", "unknown")) for row in train).items())),
            "valid": dict(sorted(Counter(str(row.get("origin", "unknown")) for row in valid).items())),
        },
        "domains": dict(sorted(Counter(str(row["domain"]) for row in selected).items())),
        "effective_licenses": {
            "train": dict(sorted(Counter(effective_license(row) for row in train).items())),
            "valid": dict(sorted(Counter(effective_license(row) for row in valid).items())),
        },
        "inputs": {
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
