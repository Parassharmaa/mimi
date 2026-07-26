#!/usr/bin/env python3
"""Build a licensed human-only curriculum for bilingual critical-token retention."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from typed_critical_token_policy import (
    strict_tokens,
    typed_preserves,
    typed_signature,
)


ALLOWED_LICENSES = {
    "CC-BY-2.0-FR",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "PDL-1.0-compatible-CC-BY-4.0",
    "project-owned",
}
LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_parent(
    parent: Path,
    direction: str,
) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("direction") != direction:
        raise SystemExit("parent dataset manifest has the wrong schema or direction")
    paths = (parent / "train.jsonl", parent / "valid.jsonl")
    for split, path in zip(("train", "valid"), paths, strict=True):
        record = manifest.get("outputs", {}).get(split, {})
        if record.get("sha256") != sha256(path):
            raise SystemExit(f"parent manifest does not authenticate {split}")
    return manifest, *paths


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    direction: str,
    path: Path,
) -> None:
    if not rows:
        raise SystemExit(f"parent split is empty: {path}")
    expected_languages = LANGUAGES[direction]
    identifiers: set[str] = set()
    for row in rows:
        identifier = str(row.get("id", ""))
        if not identifier or identifier in identifiers:
            raise SystemExit(f"missing or duplicate parent ID: {identifier}")
        identifiers.add(identifier)
        if (
            row.get("source_language"),
            row.get("target_language"),
        ) != expected_languages:
            raise SystemExit(f"parent row has the wrong direction: {identifier}")
        for field in (
            "source",
            "target",
            "origin",
            "source_license",
            "source_provenance",
        ):
            if not str(row.get(field, "")).strip():
                raise SystemExit(f"parent row lacks {field}: {identifier}")
        if row["source_license"] not in ALLOWED_LICENSES:
            raise SystemExit(
                f"parent row has a non-distributable license: "
                f"{identifier} / {row['source_license']}"
            )
        if (
            row["source_license"] != "project-owned"
            and not str(row.get("attribution", "")).strip()
        ):
            raise SystemExit(f"licensed parent row lacks attribution: {identifier}")


def benchmark_texts(row: dict[str, Any]) -> list[str]:
    values = [row.get("source")]
    values.extend(row.get("segments") or [])
    values.extend(row.get("references") or [])
    values.extend(row.get("referenceSegments") or [])
    return [str(value) for value in values if str(value or "").strip()]


def contamination_index(
    suite_paths: list[Path],
) -> tuple[set[str], list[dict[str, Any]]]:
    texts: set[str] = set()
    records = []
    for path in suite_paths:
        rows = load_jsonl(path)
        before = len(texts)
        for row in rows:
            texts.update(normalize_text(value) for value in benchmark_texts(row))
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows": len(rows),
                "unique_normalized_texts_added": len(texts) - before,
            }
        )
    return texts, records


def has_typed_payload(value: str, language: str) -> bool:
    signature = typed_signature(value, language)
    return bool(
        signature.protected
        or signature.percentages
        or signature.numbers
        or signature.opaque_numbers
    )


def curriculum_classes(
    row: dict[str, Any],
    direction: str,
) -> list[str]:
    source_language, target_language = LANGUAGES[direction]
    source = str(row["source"])
    target = str(row["target"])
    if not has_typed_payload(source, source_language):
        return []
    if not typed_preserves(source, target, source_language, target_language):
        return []
    classes = ["typed-critical-aligned"]
    if strict_tokens(source) == strict_tokens(target):
        classes.append("strict-surface-aligned")
    else:
        classes.append("bilingual-surface-transformation")
    signature = typed_signature(source, source_language)
    if signature.numbers:
        classes.append("numeric-value")
    if signature.opaque_numbers:
        classes.append("opaque-numeric-surface")
    if signature.percentages:
        classes.append("percentage")
    if signature.protected:
        classes.append("protected-structure")
    return classes


def derive(
    rows: list[dict[str, Any]],
    *,
    direction: str,
    excluded_texts: set[str],
) -> tuple[list[dict[str, Any]], Counter[str], list[str]]:
    output = []
    counts: Counter[str] = Counter()
    excluded_ids = []
    for row in rows:
        source = normalize_text(str(row["source"]))
        target = normalize_text(str(row["target"]))
        if source in excluded_texts or target in excluded_texts:
            excluded_ids.append(str(row["id"]))
            counts["excluded:registered-benchmark-exact-text"] += 1
            continue
        classes = curriculum_classes(row, direction)
        focus = bool(classes)
        original_origin = str(row["origin"])
        output.append(
            {
                **row,
                "origin": (
                    "typed-numeric-target"
                    if focus
                    else "base-preservation-replay"
                ),
                "source_origin": original_origin,
                "constraint_classes": classes,
                "text_derived_from_parent_without_modification": True,
            }
        )
        counts["focus" if focus else "preservation"] += 1
        counts.update(f"class:{value}" for value in classes)
        counts[f"source-origin:{original_origin}"] += 1
    return output, counts, excluded_ids


def validate_split_separation(
    train: list[dict[str, Any]],
    valid: list[dict[str, Any]],
) -> None:
    train_pairs = {
        (normalize_text(str(row["source"])), normalize_text(str(row["target"])))
        for row in train
    }
    valid_pairs = {
        (normalize_text(str(row["source"])), normalize_text(str(row["target"])))
        for row in valid
    }
    overlap = train_pairs & valid_pairs
    if overlap:
        raise SystemExit(
            f"derived train and validation contain {len(overlap)} exact normalized pairs"
        )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument(
        "--screen-suite",
        type=Path,
        action="append",
        default=[],
        help="Repeat for every registered evaluation suite excluded by exact text.",
    )
    parser.add_argument("--minimum-focus-train-rows", type=int, default=1000)
    parser.add_argument(
        "--minimum-surface-transform-train-rows",
        type=int,
        default=500,
    )
    args = parser.parse_args()

    if args.minimum_focus_train_rows < 1:
        raise SystemExit("minimum focus train rows must be positive")
    if args.minimum_surface_transform_train_rows < 1:
        raise SystemExit("minimum surface-transform train rows must be positive")
    if not args.screen_suite:
        raise SystemExit("at least one --screen-suite is required")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    parent_manifest, train_path, valid_path = validate_parent(
        args.parent,
        args.direction,
    )
    parent_train = load_jsonl(train_path)
    parent_valid = load_jsonl(valid_path)
    validate_rows(parent_train, direction=args.direction, path=train_path)
    validate_rows(parent_valid, direction=args.direction, path=valid_path)
    excluded_texts, suite_records = contamination_index(args.screen_suite)
    train, train_counts, train_excluded = derive(
        parent_train,
        direction=args.direction,
        excluded_texts=excluded_texts,
    )
    valid, valid_counts, valid_excluded = derive(
        parent_valid,
        direction=args.direction,
        excluded_texts=excluded_texts,
    )
    validate_split_separation(train, valid)

    if train_counts["focus"] < args.minimum_focus_train_rows:
        raise SystemExit(
            f"typed-numeric curriculum is too small: {train_counts['focus']} train rows"
        )
    if (
        train_counts["class:bilingual-surface-transformation"]
        < args.minimum_surface_transform_train_rows
    ):
        raise SystemExit(
            "typed-numeric bilingual surface-transformation slice is too small: "
            f"{train_counts['class:bilingual-surface-transformation']} train rows"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    output_train = args.output / "train.jsonl"
    output_valid = args.output / "valid.jsonl"
    write_jsonl(output_train, train)
    write_jsonl(output_valid, valid)
    license_counts = {
        split: dict(
            sorted(Counter(str(row["source_license"]) for row in rows).items())
        )
        for split, rows in (("train", train), ("valid", valid))
    }
    manifest = {
        "schema_version": 1,
        "experiment": "typed-numeric-preservation-v20-curriculum",
        "direction": args.direction,
        "status": "training-only",
        "promotion_eligible": False,
        "target_source": "licensed human references from authenticated parent",
        "private_reasoning_traces_used": False,
        "synthetic_rows": 0,
        "text_changes_from_parent": 0,
        "derivation": (
            "exclude registered benchmark exact-text matches, then relabel parent "
            "origins; source and target strings remain unchanged"
        ),
        "training_policy": {
            "focus_origin": "typed-numeric-target",
            "preservation_origin": "base-preservation-replay",
            "focus_definition": (
                "non-empty bilingual typed signature preserved by the human target"
            ),
            "recommended_objective": (
                "bounded low-rate continuation with focus loss weighting, "
                "frozen-parent KL on preservation replay, and L2-to-parent"
            ),
        },
        "parent": {
            "directory": str(args.parent),
            "manifest": {
                "path": str(args.parent / "manifest.json"),
                "sha256": sha256(args.parent / "manifest.json"),
            },
            "manifest_promotion_eligible": parent_manifest.get(
                "promotion_eligible"
            ),
            "outputs": {
                "train": {
                    "path": str(train_path),
                    "sha256": sha256(train_path),
                },
                "valid": {
                    "path": str(valid_path),
                    "sha256": sha256(valid_path),
                },
            },
        },
        "contamination_screen": {
            "normalization": "Unicode NFKC, whitespace collapsed, exact text",
            "fields": [
                "source",
                "segments",
                "references",
                "referenceSegments",
            ],
            "suites": suite_records,
            "excluded": {
                "train": len(train_excluded),
                "valid": len(valid_excluded),
            },
            "excluded_id_sha256": {
                "train": hashlib.sha256(
                    "\n".join(sorted(train_excluded)).encode()
                ).hexdigest(),
                "valid": hashlib.sha256(
                    "\n".join(sorted(valid_excluded)).encode()
                ).hexdigest(),
            },
        },
        "counts": {
            "train": dict(sorted(train_counts.items())),
            "valid": dict(sorted(valid_counts.items())),
            "output_train": len(train),
            "output_valid": len(valid),
        },
        "effective_licenses": license_counts,
        "outputs": {
            "train": {
                "path": str(output_train),
                "bytes": output_train.stat().st_size,
                "sha256": sha256(output_train),
            },
            "valid": {
                "path": str(output_valid),
                "bytes": output_valid.stat().st_size,
                "sha256": sha256(output_valid),
            },
        },
        "does_not_authorize_app_integration": True,
        "does_not_authorize_public_upload": True,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "direction": args.direction,
                "train": len(train),
                "valid": len(valid),
                "focusTrain": train_counts["focus"],
                "surfaceTransformTrain": train_counts[
                    "class:bilingual-surface-transformation"
                ],
                "excludedTrain": len(train_excluded),
                "excludedValid": len(valid_excluded),
                "manifestSHA256": sha256(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
