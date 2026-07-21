#!/usr/bin/env python3
"""Freeze a reference-hidden, reviewer-free local translation-teacher suite.

Only train-split, distributable reference-backed seeds that are novel relative to
the supplied student datasets and distant from protected evaluations are kept.
The reference and current-student output remain in the suite for later independent
scoring, but the generation runner is contractually forbidden from prompting with
either field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


LANGUAGES = {"en-US", "ja-JP"}
ALLOWED_LICENSES = {
    "Apache-2.0",
    "CC-BY-2.0-FR",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC0-1.0",
    "MIT",
    "project-owned",
}


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def dataset_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = [path / name for name in ("train.jsonl", "valid.jsonl")]
        if not all(item.is_file() for item in files):
            raise SystemExit(f"excluded dataset needs train.jsonl and valid.jsonl: {path}")
        return files
    raise SystemExit(f"missing excluded dataset: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", type=Path)
    parser.add_argument("suite_output", type=Path)
    parser.add_argument("baseline_report_output", type=Path)
    parser.add_argument("--protected-suite", type=Path, action="append", required=True)
    parser.add_argument("--exclude-dataset", type=Path, action="append", required=True)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    for output in (args.suite_output, args.baseline_report_output):
        if output.exists() and output.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    if not 0 <= args.maximum_jaccard <= 1:
        raise SystemExit("maximum-jaccard must be between zero and one")

    seed_rows = rows(args.seeds)
    protected = [
        ngrams(text)
        for path in args.protected_suite
        for row in rows(path)
        for text in [row.get("source", ""), *row.get("references", [])]
        if str(text).strip()
    ]
    exclusion_files = [
        item
        for path in args.exclude_dataset
        for item in dataset_files(path)
    ]
    excluded_sources = {
        normalized(str(row.get("source", "")))
        for path in exclusion_files
        for row in rows(path)
        if str(row.get("source", "")).strip()
    }
    excluded_validation_texts = {
        normalized(str(row.get(field, "")))
        for path in exclusion_files
        if path.name == "valid.jsonl"
        for row in rows(path)
        for field in ("source", "target")
        if str(row.get(field, "")).strip()
    }
    excluded_validation_grams = [
        ngrams(value)
        for value in excluded_validation_texts
        if value
    ]

    accepted: list[dict] = []
    rejected: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    for seed in seed_rows:
        identifier = str(seed.get("id", "")).strip()
        source = str(seed.get("source", "")).strip()
        reference = str(seed.get("reference_translation", "")).strip()
        source_language = str(seed.get("source_language", ""))
        target_language = str(seed.get("target_language", ""))
        source_key = (source_language, normalized(source))
        if not identifier or identifier in seen_ids:
            raise SystemExit(f"seed has a missing or duplicate ID: {identifier!r}")
        seen_ids.add(identifier)
        if not source or not reference:
            rejected["missing-reference"] += 1
            continue
        if seed.get("split") != "train":
            rejected["non-training-split"] += 1
            continue
        if (
            source_language not in LANGUAGES
            or target_language not in LANGUAGES
            or source_language == target_language
        ):
            raise SystemExit(f"unsupported seed language pair: {identifier}")
        license_name = str(seed.get("license", "")).strip()
        if license_name not in ALLOWED_LICENSES:
            raise SystemExit(f"seed has a non-approved or missing license: {identifier}: {license_name}")
        if not str(seed.get("provenance", "")).strip() or not str(
            seed.get("reference_provenance", "")
        ).strip():
            raise SystemExit(f"seed lacks source/reference provenance: {identifier}")
        if normalized(source) in excluded_sources:
            rejected["existing-student-source"] += 1
            continue
        if normalized(source) in excluded_validation_texts or normalized(reference) in excluded_validation_texts:
            rejected["student-validation-overlap"] += 1
            continue
        if near(source, excluded_validation_grams, args.maximum_jaccard) or near(
            reference, excluded_validation_grams, args.maximum_jaccard
        ):
            rejected["near-student-validation"] += 1
            continue
        if near(source, protected, args.maximum_jaccard) or near(
            reference, protected, args.maximum_jaccard
        ):
            rejected["near-protected-evaluation"] += 1
            continue
        if source_key in seen_sources:
            rejected["duplicate-source"] += 1
            continue
        student = str(seed.get("student_hypothesis", "")).strip()
        student_chrf = seed.get("student_chrf_pp")
        if not student or not isinstance(student_chrf, (int, float)):
            rejected["missing-student-baseline"] += 1
            continue
        seen_sources.add(source_key)
        accepted.append({
            "id": identifier,
            "sourceLanguage": source_language,
            "targetLanguage": target_language,
            "domain": seed.get("domain", "unknown"),
            "source": source,
            "references": [reference],
            "split": "local-reference-teacher-train-only",
            "claimEligible": False,
            "sourceLicense": license_name,
            "sourceProvenance": seed["provenance"],
            "referenceProvenance": seed["reference_provenance"],
            "studentHypothesis": student,
            "studentChrFPlusPlus": float(student_chrf),
            "referenceExposedToTeacher": False,
        })

    if not accepted:
        raise SystemExit("no novel reference-backed teacher cases survived")
    accepted.sort(key=lambda row: str(row["id"]))
    args.suite_output.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_report_output.parent.mkdir(parents=True, exist_ok=True)
    args.suite_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted),
        encoding="utf-8",
    )
    baseline_report = {
        "schemaVersion": 1,
        "engine": "frozen-seed-student-baseline",
        "purpose": "training-only baseline for local reference teacher filtering",
        "claimEligible": False,
        "referenceExposedToTeacher": False,
        "results": [
            {
                "caseID": row["id"],
                "sourceLanguage": row["sourceLanguage"],
                "targetLanguage": row["targetLanguage"],
                "domain": row["domain"],
                "source": row["source"],
                "references": row["references"],
                "claimEligible": False,
                "hypothesis": row["studentHypothesis"],
                "latencySeconds": 0.0,
                "warmLatencySeconds": [],
            }
            for row in accepted
        ],
    }
    args.baseline_report_output.write_text(
        json.dumps(baseline_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts_by_direction = Counter(
        f"{row['sourceLanguage']}>{row['targetLanguage']}" for row in accepted
    )
    counts_by_domain = Counter(str(row["domain"]) for row in accepted)
    manifest = {
        "schema_version": 1,
        "purpose": "reference-hidden local Qwen teacher training suite; never evaluation evidence",
        "promotion_eligible": False,
        "reference_exposed_to_teacher": False,
        "allowed_licenses": sorted(ALLOWED_LICENSES),
        "licenses": dict(sorted(Counter(str(row["sourceLicense"]) for row in accepted).items())),
        "maximum_protected_five_gram_jaccard": args.maximum_jaccard,
        "counts": {
            "input": len(seed_rows),
            "accepted": len(accepted),
            "rejected": dict(sorted(rejected.items())),
            "by_direction": dict(sorted(counts_by_direction.items())),
            "by_domain": dict(sorted(counts_by_domain.items())),
        },
        "inputs": {
            "seeds": {"path": str(args.seeds.resolve()), "sha256": sha256(args.seeds)},
            "protected_suites": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in args.protected_suite
            ],
            "excluded_datasets": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in exclusion_files
            ],
        },
        "outputs": {
            "suite": {"path": str(args.suite_output.resolve()), "sha256": sha256(args.suite_output)},
            "baseline_report": {
                "path": str(args.baseline_report_output.resolve()),
                "sha256": sha256(args.baseline_report_output),
            },
        },
    }
    manifest_path = args.suite_output.with_suffix(args.suite_output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
