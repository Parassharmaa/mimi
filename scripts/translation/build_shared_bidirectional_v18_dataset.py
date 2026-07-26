#!/usr/bin/env python3
"""Build a promotion-eligible, protected-screened dataset for V18."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_bidirectional_dataset import interleave, normalized, repeat_to_count
from filter_training_dataset_against_protected import (
    ProtectedIndex,
    sha256,
    text_fields,
)


ALLOWED_LICENSES = {
    "CC-BY-2.0-FR",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "project-owned",
}
DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def authenticated_dataset(root: Path) -> tuple[dict, dict[str, Path]]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    paths = {split: root / f"{split}.jsonl" for split in ("train", "valid")}
    for split, path in paths.items():
        expected = manifest.get("outputs", {}).get(split, {}).get("sha256")
        if not expected or expected != sha256(path):
            raise SystemExit(f"{root} {split} does not match its manifest")
    return manifest, paths


def validate_row(row: dict, direction: str, path: Path) -> None:
    expected_languages = DIRECTIONS[direction]
    if (row.get("source_language"), row.get("target_language")) != expected_languages:
        raise SystemExit(f"wrong language direction in {path}: {row.get('id')}")
    if row.get("source_license") not in ALLOWED_LICENSES:
        raise SystemExit(f"unapproved license in {path}: {row.get('id')}")
    if not str(row.get("attribution") or row.get("source_provenance") or "").strip():
        raise SystemExit(f"missing attribution/provenance in {path}: {row.get('id')}")
    if not str(row.get("source", "")).strip() or not str(row.get("target", "")).strip():
        raise SystemExit(f"missing source/target in {path}: {row.get('id')}")
    origin = str(row.get("origin", ""))
    if not (origin.startswith("human-") or origin == "mimi-shipped-ui-pair"):
        raise SystemExit(f"non-human target origin in {path}: {row.get('id')}")


def exclusion_reason(row: dict) -> str | None:
    if row.get("promotion_eligible") is False:
        return "explicitly-promotion-ineligible"
    if row.get("training_only") is True:
        return "explicitly-training-only"
    return None


def protected_matches(
    row: dict,
    index: ProtectedIndex,
    maximum_jaccard: float,
    ngram_size: int,
) -> list[dict]:
    output = []
    for field, text in text_fields(row, protected=False):
        match = index.match(text, maximum_jaccard, ngram_size)
        if match is not None:
            output.append({"dataset_field": field, **match})
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("en_ja_dataset", type=Path)
    parser.add_argument("ja_en_dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protected-suite", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--character-ngram-size", type=int, default=5)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.character_ngram_size < 1:
        raise SystemExit("character-ngram-size must be positive")
    if not 0 <= args.maximum_jaccard < 1:
        raise SystemExit("maximum-jaccard must be at least zero and below one")

    roots = {
        "en-ja": args.en_ja_dataset,
        "ja-en": args.ja_en_dataset,
    }
    parent_records = {}
    source_rows: dict[str, dict[str, list[dict]]] = {}
    for direction, root in roots.items():
        manifest, paths = authenticated_dataset(root)
        parent_records[direction] = {
            "directory": str(root),
            "manifest": {
                "path": str(root / "manifest.json"),
                "sha256": sha256(root / "manifest.json"),
            },
            **{
                split: {
                    "path": str(path),
                    "sha256": sha256(path),
                    "rows": len(load_jsonl(path)),
                }
                for split, path in paths.items()
            },
        }
        source_rows[direction] = {
            split: load_jsonl(path) for split, path in paths.items()
        }
        for split, rows in source_rows[direction].items():
            identifiers: set[str] = set()
            pairs: set[tuple[str, str]] = set()
            for row in rows:
                validate_row(row, direction, paths[split])
                identifier = str(row.get("id", ""))
                if not identifier or identifier in identifiers:
                    raise SystemExit(
                        f"missing or duplicate ID in {paths[split]}: {identifier}"
                    )
                identifiers.add(identifier)
                pair = (normalized(str(row["source"])), normalized(str(row["target"])))
                if pair in pairs:
                    raise SystemExit(
                        f"duplicate normalized pair in {paths[split]}: {identifier}"
                    )
                pairs.add(pair)
        train_pairs = {
            (normalized(str(row["source"])), normalized(str(row["target"])))
            for row in source_rows[direction]["train"]
        }
        valid_pairs = {
            (normalized(str(row["source"])), normalized(str(row["target"])))
            for row in source_rows[direction]["valid"]
        }
        if train_pairs & valid_pairs:
            raise SystemExit(f"{root} has exact normalized train/valid overlap")

    protected = ProtectedIndex(args.protected_suite, args.character_ngram_size)
    retained: dict[str, dict[str, list[dict]]] = {
        direction: {"train": [], "valid": []} for direction in DIRECTIONS
    }
    exclusions = []
    for direction in DIRECTIONS:
        for split in ("train", "valid"):
            for row in source_rows[direction][split]:
                reason = exclusion_reason(row)
                matches = protected_matches(
                    row,
                    protected,
                    args.maximum_jaccard,
                    args.character_ngram_size,
                )
                if reason or matches:
                    exclusions.append(
                        {
                            "direction": direction,
                            "split": split,
                            "id": row["id"],
                            "reason": reason or "protected-overlap",
                            "matches": matches,
                        }
                    )
                    continue
                retained[direction][split].append({**row, "direction": direction})
            if not retained[direction][split]:
                raise SystemExit(f"no retained {direction} {split} rows")

    balanced_count = max(
        len(retained[direction]["train"]) for direction in DIRECTIONS
    )
    balanced_train = {
        direction: repeat_to_count(
            retained[direction]["train"],
            balanced_count,
            args.seed + index,
        )
        for index, direction in enumerate(DIRECTIONS)
    }
    validation = {
        direction: [
            {
                **row,
                "id": f"{direction}:{row['id']}",
                "original_id": row["id"],
                "balance_repeat_index": 0,
            }
            for row in sorted(
                retained[direction]["valid"], key=lambda item: str(item["id"])
            )
        ]
        for direction in DIRECTIONS
    }
    outputs = {
        "train": interleave(balanced_train["en-ja"], balanced_train["ja-en"]),
        "valid": interleave(validation["en-ja"], validation["ja-en"]),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output_paths = {
        split: args.output / f"{split}.jsonl" for split in ("train", "valid")
    }
    for split, path in output_paths.items():
        write_jsonl(path, outputs[split])

    exclusion_counts = Counter(item["reason"] for item in exclusions)
    manifest = {
        "schema_version": 1,
        "experiment": "shared-bidirectional-v18-licensed-human-dataset",
        "status": "dataset-ready-training-not-authorized",
        "operation": "promotion-eligible-protected-screened-balanced-bidirectional-mixture",
        "seed": args.seed,
        "source_prefixes": {"en-ja": "<2ja> ", "ja-en": "<2en> "},
        "inputs": {
            "parents": parent_records,
            "protected_suites": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.protected_suite
            ],
        },
        "filter": {
            "normalization": "Unicode NFKC, casefold, remove whitespace",
            "character_ngram_size": args.character_ngram_size,
            "maximum_jaccard_exclusive": args.maximum_jaccard,
            "dataset_fields": ["source", "target"],
            "protected_fields": [
                "source",
                "target",
                "references",
                "segments",
                "referenceSegments",
            ],
            "explicit_promotion_ineligible_rows_excluded": True,
            "explicit_training_only_rows_excluded": True,
        },
        "counts": {
            "input": {
                direction: {
                    split: len(source_rows[direction][split])
                    for split in ("train", "valid")
                }
                for direction in DIRECTIONS
            },
            "retained": {
                direction: {
                    split: len(retained[direction][split])
                    for split in ("train", "valid")
                }
                for direction in DIRECTIONS
            },
            "balanced_train_per_direction": balanced_count,
            "train": len(outputs["train"]),
            "valid": len(outputs["valid"]),
            "repeated_train_rows": sum(
                row["balance_repeat_index"] > 0 for row in outputs["train"]
            ),
            "excluded": len(exclusions),
            "exclusions_by_reason": dict(sorted(exclusion_counts.items())),
        },
        "licenses": {
            "allowed": sorted(ALLOWED_LICENSES),
            **{
                split: dict(
                    sorted(Counter(row["source_license"] for row in rows).items())
                )
                for split, rows in outputs.items()
            },
        },
        "origins": {
            split: dict(
                sorted(Counter(str(row["origin"]) for row in rows).items())
            )
            for split, rows in outputs.items()
        },
        "outputs": {
            split: {
                "path": str(path),
                "sha256": sha256(path),
                "rows": len(outputs[split]),
            }
            for split, path in output_paths.items()
        },
        "licensed_human_targets_only": True,
        "private_reasoning_traces_used": False,
        "v17_generated_candidates_used": False,
        "promotion_eligible": True,
        "training_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "exclusions": exclusions,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
