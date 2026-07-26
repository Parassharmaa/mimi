#!/usr/bin/env python3
"""Build the licensed, protected-independent v12 JA-to-EN repair dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_canonical_sequence_v10_dataset import (
    TEACHER_ORIGIN,
    legal_category,
)
from filter_training_dataset_against_protected import (
    ProtectedIndex,
    normalized,
    sha256,
)

EXPERIMENT = "canonical-safety-repair-v12-ja-en"
LEGAL_QUOTAS = {
    "negation": 160,
    "critical": 160,
    "repetition-risk": 8,
    "terminology-risk": 128,
    "omission-risk": 64,
    "long": 240,
    "general": 328,
}
GENERAL_QUOTAS = {
    "human-alt-test": 128,
    "human-kftt-test": 216,
    "human-tatoeba-test": 104,
}
OPEN_LICENSES = {
    "CC-BY-2.0-FR",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "PDL-1.0-compatible-CC-BY-4.0",
    "project-owned",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": display_path(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def stable_rank(seed: int, bucket: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{seed}\0{bucket}\0{row.get('id', '')}\0"
            f"{row.get('source', '')}\0{row.get('target', '')}"
        ).encode()
    ).hexdigest()


def ensure_translation_row(row: dict[str, Any], label: str) -> None:
    if (row.get("source_language"), row.get("target_language")) != (
        "ja-JP",
        "en-US",
    ):
        raise SystemExit(f"{label} contains wrong-direction row: {row.get('id')}")
    for field in (
        "id",
        "source",
        "target",
        "origin",
        "source_license",
        "source_provenance",
    ):
        if not str(row.get(field, "")).strip():
            raise SystemExit(f"{label} row lacks {field}: {row.get('id')}")
    if row["source_license"] not in OPEN_LICENSES:
        raise SystemExit(
            f"{label} row has an unregistered license: "
            f"{row.get('id')}: {row.get('source_license')}"
        )


def protected_matches(
    index: ProtectedIndex,
    row: dict[str, Any],
    *,
    maximum_jaccard: float,
    ngram_size: int,
) -> list[dict[str, Any]]:
    matches = []
    for field in ("source", "target"):
        match = index.match(
            str(row[field]),
            maximum_jaccard,
            ngram_size,
        )
        if match is not None:
            matches.append({"datasetField": field, **match})
    return matches


def normalize_messages_row(
    row: dict[str, Any],
    *,
    origin: str,
) -> dict[str, Any] | None:
    metadata = row.get("metadata", {})
    if metadata.get("direction") != "ja-en":
        return None
    messages = row.get("messages", [])
    by_role = {
        str(message.get("role")): str(message.get("content", "")).strip()
        for message in messages
        if isinstance(message, dict)
    }
    source = by_role.get("user", "")
    target = by_role.get("assistant", "")
    attribution = str(metadata.get("attribution", "")).strip()
    source_id = str(metadata.get("source_id", "")).strip()
    license_name = str(metadata.get("license", "")).strip()
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"v12:{origin}:{source_id}:{source_digest}",
        "source_id": source_id,
        "source": source,
        "target": target,
        "source_language": "ja-JP",
        "target_language": "en-US",
        "domain": "v12-fresh-general-test",
        "origin": origin,
        "source_license": license_name,
        "source_provenance": attribution,
        "attribution": attribution,
        "sequence_role": "licensed-human-fresh-test-reference",
    }


def normalize_alt_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if (row.get("source_language"), row.get("target_language")) != (
        "ja-JP",
        "en-US",
    ):
        return None
    return {
        **row,
        "id": f"v12:human-alt-test:{row['id']}",
        "domain": "v12-fresh-general-test",
        "origin": "human-alt-test",
        "sequence_role": "licensed-human-fresh-test-reference",
    }


def candidate_is_clean(
    row: dict[str, Any],
    *,
    protected: ProtectedIndex,
    blocked_sources: set[str],
    maximum_jaccard: float,
    ngram_size: int,
) -> tuple[bool, str]:
    source_key = normalized(str(row["source"]))
    if source_key in blocked_sources:
        return False, "source-overlap"
    if protected_matches(
        protected,
        row,
        maximum_jaccard=maximum_jaccard,
        ngram_size=ngram_size,
    ):
        return False, "protected-overlap"
    return True, ""


def select_bucket(
    rows: list[dict[str, Any]],
    *,
    quota: int,
    seed: int,
    bucket: str,
    protected: ProtectedIndex,
    blocked_sources: set[str],
    maximum_jaccard: float,
    ngram_size: int,
    rejected: Counter[str],
) -> list[dict[str, Any]]:
    selected = []
    for row in sorted(
        rows,
        key=lambda item: (stable_rank(seed, bucket, item), str(item["id"])),
    ):
        clean, reason = candidate_is_clean(
            row,
            protected=protected,
            blocked_sources=blocked_sources,
            maximum_jaccard=maximum_jaccard,
            ngram_size=ngram_size,
        )
        if not clean:
            rejected[f"{bucket}:{reason}"] += 1
            continue
        blocked_sources.add(normalized(str(row["source"])))
        selected.append(row)
        if len(selected) == quota:
            break
    if len(selected) != quota:
        raise SystemExit(
            f"insufficient clean rows for {bucket}: {len(selected)} < {quota}"
        )
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v10_directory", type=Path)
    parser.add_argument("legal_directory", type=Path)
    parser.add_argument("alt_directory", type=Path)
    parser.add_argument("kftt_directory", type=Path)
    parser.add_argument("tatoeba_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protected-suite",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--character-ngram-size", type=int, default=5)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.character_ngram_size < 1 or not 0 <= args.maximum_jaccard < 1:
        raise SystemExit("invalid protected-overlap settings")

    root = Path(__file__).resolve().parents[2]
    v10_manifest_path = args.v10_directory / "manifest.json"
    v10_manifest = load_json(v10_manifest_path)
    v10_paths = {
        split: args.v10_directory / f"{split}.jsonl" for split in ("train", "valid")
    }
    if (
        v10_manifest.get("experiment")
        != "canonical-sequence-v10-ja-en-error-stratified"
        or v10_manifest.get("direction") != "ja-en"
        or any(
            v10_manifest.get("outputs", {}).get(split, {}).get("sha256") != sha256(path)
            for split, path in v10_paths.items()
        )
    ):
        raise SystemExit("v10 dataset identity or authentication differs")
    v10_rows = {split: load_jsonl(path) for split, path in v10_paths.items()}
    all_v10_sources = {
        normalized(str(row["source"])) for rows in v10_rows.values() for row in rows
    }
    train_rows = [
        {
            **row,
            "id": f"v12-train:{row['id']}",
            "domain": f"v12-repair:{row.get('domain', 'unknown')}",
            "v12_stratum": str(
                row.get("v10_stratum") or row.get("sequence_role", "licensed-human")
            ),
            "sequence_role": "licensed-human-safety-repair-reference",
        }
        for row in v10_rows["train"]
        if row.get("origin") != TEACHER_ORIGIN
    ]
    if len(train_rows) != 7_104:
        raise SystemExit(
            f"unexpected v12 licensed-human train count: {len(train_rows)}"
        )

    protected = ProtectedIndex(
        args.protected_suite,
        args.character_ngram_size,
    )
    train_sources: set[str] = set()
    for row in train_rows:
        ensure_translation_row(row, "v12 train")
        source_key = normalized(str(row["source"]))
        if source_key in train_sources:
            raise SystemExit(f"v12 train repeats a source: {row['id']}")
        train_sources.add(source_key)
        matches = protected_matches(
            protected,
            row,
            maximum_jaccard=args.maximum_jaccard,
            ngram_size=args.character_ngram_size,
        )
        if matches:
            raise SystemExit(
                f"v12 inherited train row overlaps protected data: {row['id']}"
            )

    legal_manifest_path = args.legal_directory / "manifest.json"
    legal_manifest = load_json(legal_manifest_path)
    legal_test_path = args.legal_directory / "test.jsonl"
    if (
        legal_manifest.get("outputs", {}).get("test", {}).get("sha256")
        != sha256(legal_test_path)
        or legal_manifest.get("license") != "PDL-1.0-compatible-CC-BY-4.0"
        or legal_manifest.get("private_reasoning_traces_used") is not False
    ):
        raise SystemExit("legal test input is not authenticated or distributable")
    legal_by_category: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(legal_test_path):
        category = legal_category(row)
        if category is None:
            continue
        selected = {
            **row,
            "id": f"v12-valid:{category}:{row['id']}",
            "domain": "v12-fresh-legal-test",
            "v12_stratum": f"legal:{category}",
            "sequence_role": "licensed-human-fresh-test-reference",
        }
        ensure_translation_row(selected, "v12 legal validation")
        legal_by_category[category].append(selected)

    general_sources = {
        "human-alt-test": (
            args.alt_directory / "test.jsonl",
            normalize_alt_row,
        ),
        "human-kftt-test": (
            args.kftt_directory / "test.jsonl",
            lambda row: normalize_messages_row(
                row,
                origin="human-kftt-test",
            ),
        ),
        "human-tatoeba-test": (
            args.tatoeba_directory / "test.jsonl",
            lambda row: normalize_messages_row(
                row,
                origin="human-tatoeba-test",
            ),
        ),
    }
    general_by_origin: dict[str, list[dict[str, Any]]] = {}
    for origin, (path, converter) in general_sources.items():
        rows = []
        for raw in load_jsonl(path):
            row = converter(raw)
            if row is None:
                continue
            row["v12_stratum"] = f"general:{origin}"
            ensure_translation_row(row, f"v12 {origin} validation")
            rows.append(row)
        general_by_origin[origin] = rows

    rejected: Counter[str] = Counter()
    blocked_sources = set(all_v10_sources)
    valid_rows = []
    for category, quota in LEGAL_QUOTAS.items():
        valid_rows.extend(
            select_bucket(
                legal_by_category[category],
                quota=quota,
                seed=args.seed,
                bucket=f"legal:{category}",
                protected=protected,
                blocked_sources=blocked_sources,
                maximum_jaccard=args.maximum_jaccard,
                ngram_size=args.character_ngram_size,
                rejected=rejected,
            )
        )
    for origin, quota in GENERAL_QUOTAS.items():
        valid_rows.extend(
            select_bucket(
                general_by_origin[origin],
                quota=quota,
                seed=args.seed,
                bucket=f"general:{origin}",
                protected=protected,
                blocked_sources=blocked_sources,
                maximum_jaccard=args.maximum_jaccard,
                ngram_size=args.character_ngram_size,
                rejected=rejected,
            )
        )
    if len(valid_rows) != 1_536:
        raise SystemExit(f"unexpected v12 validation count: {len(valid_rows)}")
    valid_rows.sort(key=lambda row: stable_rank(args.seed, "valid", row))
    train_rows.sort(key=lambda row: stable_rank(args.seed, "train", row))

    valid_sources = {normalized(str(row["source"])) for row in valid_rows}
    if train_sources & valid_sources:
        raise SystemExit("v12 train and validation sources overlap")
    if len(valid_sources) != len(valid_rows):
        raise SystemExit("v12 validation contains duplicate sources")

    args.output.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": args.output / "train.jsonl",
        "valid": args.output / "valid.jsonl",
    }
    write_jsonl(output_paths["train"], train_rows)
    write_jsonl(output_paths["valid"], valid_rows)

    attribution_counts: Counter[tuple[str, str, str]] = Counter()
    for row in [*train_rows, *valid_rows]:
        attribution_counts[
            (
                str(row["origin"]),
                str(row["source_license"]),
                str(row["source_provenance"]),
            )
        ] += 1
    attribution_path = args.output / "attribution.jsonl"
    write_jsonl(
        attribution_path,
        [
            {
                "origin": origin,
                "license": license_name,
                "provenance": provenance,
                "rows": count,
            }
            for (origin, license_name, provenance), count in sorted(
                attribution_counts.items()
            )
        ],
    )

    source_inputs = {
        "v10_manifest": record(v10_manifest_path, root),
        "v10_train": record(v10_paths["train"], root),
        "v10_valid": record(v10_paths["valid"], root),
        "legal_manifest": record(legal_manifest_path, root),
        "legal_test": record(legal_test_path, root),
    }
    for name, directory in (
        ("alt", args.alt_directory),
        ("kftt", args.kftt_directory),
        ("tatoeba", args.tatoeba_directory),
    ):
        source_inputs[f"{name}_manifest"] = record(
            directory / "manifest.json",
            root,
        )
        source_inputs[f"{name}_test"] = record(
            directory / "test.jsonl",
            root,
        )
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "frozen-ready-for-negative-generation",
        "direction": "ja-en",
        "purpose": (
            "licensed-human safety repair with a fresh source-disjoint "
            "error-stratified validation suite"
        ),
        "target_source": "licensed human references only",
        "promotion_eligible": False,
        "training_only": True,
        "private_reasoning_traces_used": False,
        "free_form_synthetic_translations_used": False,
        "human_reviewer_required": False,
        "selection": {
            "seed": args.seed,
            "train_policy": (
                "all unique licensed-human v10 train rows; remove every "
                "GPT-5.6 teacher-repeat row"
            ),
            "validation_policy": (
                "deterministic SHA-256 rank inside frozen legal-risk and "
                "general-corpus quotas after excluding every v10 source and "
                "screening source plus target against protected suites"
            ),
            "legal_quotas": LEGAL_QUOTAS,
            "general_quotas": GENERAL_QUOTAS,
        },
        "counts": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "unique_train_sources": len(train_sources),
            "unique_valid_sources": len(valid_sources),
            "train_origins": dict(
                sorted(Counter(str(row["origin"]) for row in train_rows).items())
            ),
            "valid_origins": dict(
                sorted(Counter(str(row["origin"]) for row in valid_rows).items())
            ),
            "train_strata": dict(
                sorted(Counter(str(row["v12_stratum"]) for row in train_rows).items())
            ),
            "valid_strata": dict(
                sorted(Counter(str(row["v12_stratum"]) for row in valid_rows).items())
            ),
            "rejected": dict(sorted(rejected.items())),
        },
        "effective_licenses": {
            split: dict(
                sorted(Counter(str(row["source_license"]) for row in rows).items())
            )
            for split, rows in (
                ("train", train_rows),
                ("valid", valid_rows),
            )
        },
        "distribution_provenance": {
            "all_rows_have_source_license": True,
            "all_rows_have_source_provenance": True,
            "licenses_are_open_or_project_owned": sorted(
                {str(row["source_license"]) for row in [*train_rows, *valid_rows]}
            ),
            "attribution_sidecar": record(attribution_path, root),
        },
        "decontamination": {
            "normalization": "Unicode NFKC, casefold, remove whitespace",
            "character_ngram_size": args.character_ngram_size,
            "maximum_jaccard_exclusive": args.maximum_jaccard,
            "screened_fields": ["source", "target"],
            "protected_suites": [record(path, root) for path in args.protected_suite],
            "all_v10_train_and_validation_sources_excluded_from_fresh_validation": True,
            "train_valid_source_overlap": False,
            "protected_hits_in_outputs": 0,
        },
        "inputs": source_inputs,
        "outputs": {
            split: {
                **record(path, root),
                "rows": len(rows),
            }
            for split, path, rows in (
                ("train", output_paths["train"], train_rows),
                ("valid", output_paths["valid"], valid_rows),
            )
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": display_path(args.output, root),
                "manifest_sha256": sha256(manifest_path),
                "train_rows": len(train_rows),
                "valid_rows": len(valid_rows),
                "valid_strata": manifest["counts"]["valid_strata"],
                "rejected": manifest["counts"]["rejected"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
