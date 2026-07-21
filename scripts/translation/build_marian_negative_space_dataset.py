#!/usr/bin/env python3
"""Build deterministic, licensed negative-space pairs for Marian adaptation.

The correct side is always the authenticated human/project-owned reference.  The
rejected side is a mechanically corrupted copy used only as negative evidence;
it is never presented as a translation target.  This is deliberately narrower
than free-form synthetic translation and does not use or contain reasoning
traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
CRITICAL_TYPES = {
    "negation-reversal",
    "number-substitution",
    "placeholder-substitution",
    "url-substitution",
    "unit-substitution",
}
ASCII_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])\d+(?:[.,]\d+)*(?![A-Za-z0-9_])")
FULLWIDTH_NUMBER_RE = re.compile(r"[０-９]+")
PLACEHOLDER_RE = re.compile(
    r"\{\{[^{}\n]+\}\}|\{[^{}\n]+\}|%\([^)]+\)[a-zA-Z]|%\d*\$?[a-zA-Z]|<[/!]?[A-Za-z][^>\n]*>"
)
URL_RE = re.compile(r"https?://[^\s<>()]+")
UNIT_RE = re.compile(
    r"(?<![A-Za-z])(?:km|cm|mm|kg|mg|GB|MB|KB|ms|Hz|USD|JPY)(?![A-Za-z])|"
    r"キロメートル|センチメートル|ミリメートル|キログラム|グラム|"
    r"時間|分間|秒間|米ドル|ドル|円"
)
UNIT_REPLACEMENTS = {
    "km": "m",
    "cm": "km",
    "mm": "cm",
    "kg": "g",
    "mg": "kg",
    "GB": "MB",
    "MB": "KB",
    "KB": "GB",
    "ms": "s",
    "Hz": "kHz",
    "USD": "JPY",
    "JPY": "USD",
    "キロメートル": "メートル",
    "センチメートル": "キロメートル",
    "ミリメートル": "センチメートル",
    "キログラム": "グラム",
    "グラム": "キログラム",
    "時間": "分間",
    "分間": "秒間",
    "秒間": "分間",
    "米ドル": "円",
    "ドル": "円",
    "円": "ドル",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_digest(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.split())


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"input is empty: {path}")
    return rows


def replace_match(text: str, match: re.Match[str], replacement: str) -> str:
    return text[: match.start()] + replacement + text[match.end() :]


def mutate_number(value: str) -> str:
    digits = "０１２３４５６７８９" if any("０" <= char <= "９" for char in value) else "0123456789"
    for index, char in enumerate(value):
        if char in digits:
            replacement = digits[(digits.index(char) + 1) % 10]
            return value[:index] + replacement + value[index + 1 :]
    raise AssertionError("number regex produced no digit")


def negation_corruption(text: str, target_language: str) -> str | None:
    if target_language == "en-US":
        replacements = (
            (re.compile(r"\bcannot\b", re.IGNORECASE), "can"),
            (re.compile(r"\bcan't\b", re.IGNORECASE), "can"),
            (re.compile(r"\bwon't\b", re.IGNORECASE), "will"),
            (re.compile(r"\bnever\b", re.IGNORECASE), "always"),
            (re.compile(r"\bwithout\b", re.IGNORECASE), "with"),
            (re.compile(r"\bnot\b", re.IGNORECASE), ""),
            (re.compile(r"\bno\b", re.IGNORECASE), "some"),
        )
    else:
        replacements = tuple(
            (re.compile(re.escape(source)), target)
            for source, target in (
                ("ではありませんでした", "でした"),
                ("ではありません", "です"),
                ("じゃなかった", "だった"),
                ("じゃない", "だ"),
                ("ませんでした", "ました"),
                ("ません", "ます"),
                ("できない", "できる"),
                ("なかった", "あった"),
                ("ない", "ある"),
            )
        )
    for pattern, replacement in replacements:
        match = pattern.search(text)
        if match:
            return normalized(replace_match(text, match, replacement))
    return None


def split_units(text: str) -> tuple[str, str, str] | None:
    match = UNIT_RE.search(text)
    if not match:
        return None
    return text[: match.start()], match.group(), text[match.end() :]


def omission_variants(text: str, target_language: str) -> list[tuple[str, str, float]]:
    stripped = text.strip()
    if target_language == "en-US":
        units = stripped.split()
        if len(units) < 6:
            return []
        cut = max(2, len(units) // 3)
        tail = " ".join(units[:-cut]).rstrip(",;:") + "."
        head = " ".join(units[cut:])
    else:
        units = list(stripped)
        if len(units) < 12:
            return []
        cut = max(3, len(units) // 3)
        tail = "".join(units[:-cut]).rstrip("、，；：。！？") + "。"
        head = "".join(units[cut:])
    return [
        (tail, "content-omission-tail", 1.0),
        (head, "content-omission-head", 0.95),
    ]


def duplication_variants(text: str, target_language: str) -> list[tuple[str, str, float]]:
    stripped = text.strip()
    if not stripped:
        return []
    separator = "、" if target_language == "ja-JP" else " "
    full = stripped.rstrip("。！？.!?") + separator + stripped
    if target_language == "en-US":
        units = stripped.split()
        tail_units = units[-max(2, len(units) // 3) :]
        tail = stripped + separator + " ".join(tail_units)
    else:
        tail_units = list(stripped)[-max(3, len(stripped) // 3) :]
        tail = stripped + separator + "".join(tail_units)
    return [
        (full, "content-duplication-full", 0.90),
        (tail, "content-duplication-tail", 0.85),
    ]


def violations(target: str, target_language: str) -> list[dict]:
    candidates: list[tuple[str, str, float]] = []

    match = URL_RE.search(target)
    if match:
        candidates.append(
            (replace_match(target, match, "https://invalid.example"), "url-substitution", 1.0)
        )
    match = PLACEHOLDER_RE.search(target)
    if match:
        candidates.append(
            (replace_match(target, match, "{MIMI_WRONG_PLACEHOLDER}"), "placeholder-substitution", 1.0)
        )
    match = ASCII_NUMBER_RE.search(target) or FULLWIDTH_NUMBER_RE.search(target)
    if match:
        candidates.append(
            (replace_match(target, match, mutate_number(match.group())), "number-substitution", 1.0)
        )
    unit = split_units(target)
    if unit:
        before, value, after = unit
        candidates.append((before + UNIT_REPLACEMENTS[value] + after, "unit-substitution", 1.0))
    negated = negation_corruption(target, target_language)
    if negated:
        candidates.append((negated, "negation-reversal", 1.0))
    candidates.extend(omission_variants(target, target_language))
    candidates.extend(duplication_variants(target, target_language))

    output: list[dict] = []
    seen = {normalized(target)}
    for rejected, violation_type, severity in candidates:
        rejected = normalized(rejected)
        if not rejected or rejected in seen:
            continue
        seen.add(rejected)
        output.append(
            {
                "rejected": rejected,
                "violation_type": violation_type,
                "severity": severity,
            }
        )
    return output


def validate_parent(directory: Path, direction: str) -> tuple[list[dict], list[dict], dict]:
    train_path, valid_path = directory / "train.jsonl", directory / "valid.jsonl"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("parent dataset lacks manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("direction") != direction:
        raise SystemExit("parent manifest schema or direction differs")
    outputs = manifest.get("outputs", {})
    for name, path in (("train", train_path), ("valid", valid_path)):
        if outputs.get(name, {}).get("sha256") != sha256(path):
            raise SystemExit(f"parent manifest does not authenticate {name}")
    expected = LANGUAGES[direction]

    def validate(rows: list[dict], split: str) -> list[dict]:
        seen: set[str] = set()
        for row in rows:
            identifier = str(row.get("id", ""))
            if not identifier or identifier in seen:
                raise SystemExit(f"parent {split} IDs are empty or duplicated")
            seen.add(identifier)
            if (row.get("source_language"), row.get("target_language")) != expected:
                raise SystemExit(f"parent {split} contains wrong-direction rows")
            if not all(str(row.get(field, "")).strip() for field in ("source", "target", "source_license")):
                raise SystemExit(f"parent {split} row lacks text or license")
        return rows

    train, valid = validate(load_jsonl(train_path), "train"), validate(load_jsonl(valid_path), "valid")
    if {normalized(row["source"]) for row in train} & {normalized(row["source"]) for row in valid}:
        raise SystemExit("parent sources leak across train and validation")
    return train, valid, manifest


def unique_rows(rows: list[dict]) -> list[dict]:
    selected: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (normalized(row["source"]), normalized(row["target"]))
        current = selected.get(key)
        if current is None or row["id"] < current["id"]:
            selected[key] = row
    return list(selected.values())


def select_positives(rows: list[dict], limit: int, seed: str) -> list[tuple[dict, list[dict]]]:
    eligible: list[tuple[dict, list[dict]]] = []
    for row in unique_rows(rows):
        generated = violations(row["target"], row["target_language"])
        if len(generated) < 3:
            continue
        has_critical = any(value["violation_type"] in CRITICAL_TYPES for value in generated)
        eligible.append((row, generated, has_critical))
    eligible.sort(
        key=lambda item: (
            not item[2],
            stable_digest(seed, item[0]["id"]),
        )
    )
    return [(row, generated) for row, generated, _ in eligible[:limit]]


def expand(selected: list[tuple[dict, list[dict]]], seed: str, maximum_violations: int) -> list[dict]:
    output: list[dict] = []
    for row, generated in selected:
        generated = sorted(
            generated,
            key=lambda value: (
                value["violation_type"] not in CRITICAL_TYPES,
                stable_digest(seed, row["id"], value["violation_type"], value["rejected"]),
            ),
        )[:maximum_violations]
        for index, negative in enumerate(generated):
            output.append(
                {
                    "id": f"negative-space:{row['id']}:{index}:{negative['violation_type']}",
                    "source_id": row.get("source_id", row["id"]),
                    "parent_id": row["id"],
                    "source": row["source"],
                    "chosen": row["target"],
                    "rejected": negative["rejected"],
                    "source_language": row["source_language"],
                    "target_language": row["target_language"],
                    "violation_type": negative["violation_type"],
                    "severity": negative["severity"],
                    "origin": row.get("origin"),
                    "domain": row.get("domain"),
                    "source_license": row["source_license"],
                    "source_provenance": row.get("source_provenance"),
                    "attribution": row.get("attribution"),
                    "negative_generation": "deterministic-target-corruption-used-only-as-negative-evidence",
                }
            )
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--train-positives", type=int, default=8000)
    parser.add_argument("--valid-positives", type=int, default=256)
    parser.add_argument("--maximum-violations", type=int, default=4)
    parser.add_argument("--seed", default="mimi-negative-space-v1")
    args = parser.parse_args()

    if min(args.train_positives, args.valid_positives) < 1:
        raise SystemExit("positive limits must be positive")
    if not 3 <= args.maximum_violations <= 5:
        raise SystemExit("maximum-violations must be in the literature-reviewed range 3...5")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")

    parent_train, parent_valid, parent_manifest = validate_parent(
        args.parent_directory, args.direction
    )
    train_selected = select_positives(parent_train, args.train_positives, args.seed + ":train")
    valid_selected = select_positives(parent_valid, args.valid_positives, args.seed + ":valid")
    if len(train_selected) < args.train_positives or len(valid_selected) < args.valid_positives:
        raise SystemExit("not enough eligible unique parent rows for the requested limits")
    train_rows = expand(train_selected, args.seed + ":train", args.maximum_violations)
    valid_rows = expand(valid_selected, args.seed + ":valid", args.maximum_violations)
    if {normalized(row["source"]) for row in train_rows} & {
        normalized(row["source"]) for row in valid_rows
    }:
        raise SystemExit("generated sources leak across train and validation")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    train_path, valid_path = args.output_directory / "train.jsonl", args.output_directory / "valid.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": "deterministic token-local negative-space Marian adaptation",
        "direction": args.direction,
        "purpose": "training-only negative evidence; rejected strings are never positive translation targets",
        "seed": args.seed,
        "private_reasoning_traces_used": False,
        "free_form_synthetic_translations_used": False,
        "human_review_required": False,
        "promotion_eligible": False,
        "parent": {
            "directory": str(args.parent_directory),
            "manifest_sha256": sha256(args.parent_directory / "manifest.json"),
            "train_sha256": parent_manifest["outputs"]["train"]["sha256"],
            "valid_sha256": parent_manifest["outputs"]["valid"]["sha256"],
        },
        "selection": {
            "train_positive_rows": len(train_selected),
            "valid_positive_rows": len(valid_selected),
            "maximum_violations_per_positive": args.maximum_violations,
            "policy": "critical-bearing rows first, then deterministic SHA-256 rank; duplicate source-target pairs removed",
        },
        "counts": {
            "train_pairs": len(train_rows),
            "valid_pairs": len(valid_rows),
            "train_violation_types": dict(sorted(Counter(row["violation_type"] for row in train_rows).items())),
            "valid_violation_types": dict(sorted(Counter(row["violation_type"] for row in valid_rows).items())),
            "train_licenses": dict(sorted(Counter(row["source_license"] for row in train_rows).items())),
            "valid_licenses": dict(sorted(Counter(row["source_license"] for row in valid_rows).items())),
        },
        "outputs": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "valid": {"path": str(valid_path), "sha256": sha256(valid_path)},
        },
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
