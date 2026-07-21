#!/usr/bin/env python3
"""Derive a text-identical critical-preservation curriculum from licensed data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from typed_critical_token_policy import single_percentage_preserves


STRICT_RE = re.compile(
    r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"
    r"|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>|%"
    r"|(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"
)
EN_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|cannot|can't|won't|don't|doesn't|didn't|"
    r"isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|shouldn't|wouldn't|"
    r"couldn't|mustn't)\b",
    re.IGNORECASE,
)
JA_NEGATION_RE = re.compile(
    r"ない|なく|なけれ|ません|ず|ぬ|禁止|不可|不要|未|なし|無い|できません"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value)
    return sorted(token.replace(",", "") for token in STRICT_RE.findall(normalized))


def preserves(source: str, target: str) -> bool:
    return strict_tokens(source) == strict_tokens(target) or single_percentage_preserves(
        source, target
    )


def constraint_classes(row: dict[str, Any], direction: str) -> list[str]:
    source = str(row["source"])
    target = str(row["target"])
    classes = []
    if strict_tokens(source) and preserves(source, target):
        classes.append("exact-protected-structure")
    source_negation = EN_NEGATION_RE.search(source) if direction == "en-ja" else JA_NEGATION_RE.search(source)
    target_negation = JA_NEGATION_RE.search(target) if direction == "en-ja" else EN_NEGATION_RE.search(target)
    if source_negation and target_negation:
        classes.append("bilingual-negation")
    return classes


def load_rows(path: Path, direction: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = {"en-ja": ("en-US", "ja-JP"), "ja-en": ("ja-JP", "en-US")}[direction]
    identifiers = [str(row.get("id", "")) for row in rows]
    if not rows or "" in identifiers or len(identifiers) != len(set(identifiers)):
        raise SystemExit(f"parent split is empty or has invalid IDs: {path}")
    for row in rows:
        if (row.get("source_language"), row.get("target_language")) != expected:
            raise SystemExit(f"parent split has wrong direction: {row.get('id')}")
        if not all(str(row.get(field, "")).strip() for field in ("source", "target", "origin")):
            raise SystemExit(f"parent row lacks source, target, or origin: {row.get('id')}")
    return rows


def validate_parent(parent: Path, direction: str) -> tuple[dict, Path, Path]:
    manifest_path = parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("direction") != direction:
        raise SystemExit("parent dataset manifest has the wrong schema or direction")
    train_path = parent / "train.jsonl"
    valid_path = parent / "valid.jsonl"
    for split, path in (("train", train_path), ("valid", valid_path)):
        record = manifest.get("outputs", {}).get(split, {})
        if record.get("sha256") != sha256(path):
            raise SystemExit(f"parent manifest does not authenticate {split}")
    return manifest, train_path, valid_path


def validate_exposure(
    exposure_path: Path, parent_paths: tuple[Path, Path]
) -> list[dict[str, str]]:
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    entries = exposure.get("assets")
    if not isinstance(entries, list):
        raise SystemExit("exposure manifest has no asset inventory")
    records = []
    for parent_path in parent_paths:
        expected_path = parent_path.resolve()
        match = None
        for entry in entries:
            raw_path = entry.get("path")
            if not isinstance(raw_path, str):
                continue
            candidate = (exposure_path.parent / raw_path).resolve()
            if candidate == expected_path and entry.get("sha256") == sha256(parent_path):
                match = entry
                break
        if match is None:
            raise SystemExit(f"parent split is absent from frozen exposure: {parent_path}")
        records.append({"path": str(parent_path), "sha256": sha256(parent_path)})
    return records


def derive(rows: list[dict[str, Any]], direction: str) -> tuple[list[dict], Counter[str]]:
    output = []
    counts: Counter[str] = Counter()
    for row in rows:
        classes = constraint_classes(row, direction)
        original_origin = str(row["origin"])
        derived = {
            **row,
            "origin": (
                "critical-preservation-target" if classes else "base-preservation-replay"
            ),
            "source_origin": original_origin,
            "constraint_classes": classes,
            "text_derived_from_parent_without_modification": True,
        }
        output.append(derived)
        counts["critical" if classes else "preservation"] += 1
        counts.update(f"class:{value}" for value in classes)
        counts[f"source-origin:{original_origin}"] += 1
    return output, counts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path)
    parser.add_argument("exposure_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=("en-ja", "ja-en"), required=True)
    parser.add_argument("--minimum-critical-train-rows", type=int, default=100)
    args = parser.parse_args()
    if args.minimum_critical_train_rows < 1:
        raise SystemExit("minimum critical train rows must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    parent_manifest, train_path, valid_path = validate_parent(args.parent, args.direction)
    exposure_records = validate_exposure(
        args.exposure_manifest, (train_path, valid_path)
    )
    train, train_counts = derive(load_rows(train_path, args.direction), args.direction)
    valid, valid_counts = derive(load_rows(valid_path, args.direction), args.direction)
    if train_counts["critical"] < args.minimum_critical_train_rows:
        raise SystemExit(
            f"critical curriculum is too small: {train_counts['critical']} train rows"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    output_train = args.output / "train.jsonl"
    output_valid = args.output / "valid.jsonl"
    write_jsonl(output_train, train)
    write_jsonl(output_valid, valid)
    license_counts = {
        split: dict(
            sorted(Counter(str(row.get("source_license", "unknown")) for row in rows).items())
        )
        for split, rows in (("train", train), ("valid", valid))
    }
    manifest = {
        "schema_version": 1,
        "experiment": "licensed critical-preservation curriculum v1",
        "direction": args.direction,
        "promotion_eligible": False,
        "training_only": True,
        "target_source": "text-identical licensed human references from exposed parent",
        "private_reasoning_traces_used": False,
        "synthetic_rows": 0,
        "text_changes_from_parent": 0,
        "derivation": (
            "relabel parent origins only; exact source and target strings are unchanged"
        ),
        "training_policy": {
            "criticalOrigin": "critical-preservation-target",
            "preservationOrigin": "base-preservation-replay",
            "criticalClasses": [
                "exact-protected-structure",
                "bilingual-negation",
            ],
            "recommendedFrozenBaseKLOrigin": "base-preservation-replay",
        },
        "parent": {
            "directory": str(args.parent),
            "manifest": {
                "path": str(args.parent / "manifest.json"),
                "sha256": sha256(args.parent / "manifest.json"),
            },
            "outputs": exposure_records,
        },
        "frozenExposure": {
            "path": str(args.exposure_manifest),
            "sha256": sha256(args.exposure_manifest),
            "parentSplitsPresent": True,
            "newTextIntroduced": False,
            "note": "Any final claim must still rebuild the exposure contract after training.",
        },
        "counts": {
            "train": dict(sorted(train_counts.items())),
            "valid": dict(sorted(valid_counts.items())),
        },
        "effective_licenses": license_counts,
        "parent_manifest_promotion_eligible": parent_manifest.get("promotion_eligible"),
        "outputs": {
            "train": {"path": str(output_train), "sha256": sha256(output_train)},
            "valid": {"path": str(output_valid), "sha256": sha256(output_valid)},
        },
        "does_not_authorize_app_integration": True,
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
                "criticalTrain": train_counts["critical"],
                "criticalValid": valid_counts["critical"],
                "manifestSHA256": sha256(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
