#!/usr/bin/env python3
"""Build a blinded canonical-teacher/reference/incumbent review queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from filter_synthetic_batch import normalized, response_payload, valid_candidate
from filter_training_dataset_against_protected import ProtectedIndex


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            for value in values:
                handle.write(
                    json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
                )
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing file: {path}") from error


def exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing file: {path}") from error


def candidate_id(source_id: str, origin: str, translation: str) -> str:
    return hashlib.sha256(
        f"{source_id}\0{origin}\0{normalized(translation)}".encode()
    ).hexdigest()[:24]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_seeds", type=Path)
    parser.add_argument("teacher_output", type=Path)
    parser.add_argument("incumbent_suite", type=Path)
    parser.add_argument("incumbent_report", type=Path)
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("queue_manifest", type=Path)
    parser.add_argument(
        "--additional-protected-suite",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    parser.add_argument("--character-ngram-size", type=int, default=5)
    args = parser.parse_args()

    seeds = rows(args.pilot_seeds)
    seed_by_id = {str(row["id"]): row for row in seeds}
    if not seeds or len(seed_by_id) != len(seeds):
        raise SystemExit("pilot seeds are empty or contain duplicate IDs")
    suite = rows(args.incumbent_suite)
    suite_by_id = {str(row["id"]): row for row in suite}
    if set(suite_by_id) != set(seed_by_id):
        raise SystemExit("incumbent suite does not cover the exact pilot seeds")

    incumbent_report = json.loads(args.incumbent_report.read_text(encoding="utf-8"))
    if (
        incumbent_report.get("caseSuiteSHA256") is not None
        and incumbent_report.get("caseSuiteSHA256") != sha256(args.incumbent_suite)
    ):
        raise SystemExit("incumbent report is not bound to the supplied suite")
    incumbent_rows = incumbent_report.get("results")
    if not isinstance(incumbent_rows, list):
        raise SystemExit("incumbent report has no results")
    incumbent_by_id = {
        str(row.get("caseID")): row for row in incumbent_rows
    }
    if len(incumbent_by_id) != len(incumbent_rows) or set(incumbent_by_id) != set(seed_by_id):
        raise SystemExit("incumbent report coverage is incomplete or duplicated")

    teacher_by_id: dict[str, tuple[dict, dict]] = {}
    for batch_row in rows(args.teacher_output):
        source_id = str(batch_row.get("custom_id", ""))
        if source_id not in seed_by_id or source_id in teacher_by_id:
            raise SystemExit(f"invalid or duplicate teacher output: {source_id}")
        try:
            payload, body = response_payload(batch_row)
        except (ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid teacher response for {source_id}: {error}") from error
        if (
            set(payload) != {"source_id", "canonical_translation", "risk_tags"}
            or payload["source_id"] != source_id
            or not isinstance(payload["canonical_translation"], str)
            or not payload["canonical_translation"].strip()
            or not isinstance(payload["risk_tags"], list)
        ):
            raise SystemExit(f"teacher canonical payload is invalid: {source_id}")
        teacher_by_id[source_id] = (payload, body)
    if set(teacher_by_id) != set(seed_by_id):
        raise SystemExit("teacher output does not cover the exact pilot seeds")

    protected_paths = [
        args.protected_benchmark,
        *args.additional_protected_suite,
    ]
    protected = ProtectedIndex(protected_paths, args.character_ngram_size)
    review_queue: list[dict] = []
    rejected = Counter()
    admitted_directions = Counter()
    critical_modes = Counter()
    for source_id in sorted(seed_by_id):
        seed = seed_by_id[source_id]
        suite_row = suite_by_id[source_id]
        incumbent = incumbent_by_id[source_id]
        payload, teacher_body = teacher_by_id[source_id]
        source = str(seed["source"])
        reference = str(seed["reference_translation"]).strip()
        canonical = str(payload["canonical_translation"]).strip()
        baseline = str(incumbent.get("hypothesis", "")).strip()
        if (
            suite_row.get("source") != source
            or suite_row.get("references") != [reference]
            or incumbent.get("source") != source
            or incumbent.get("references") != [reference]
            or incumbent.get("sourceLanguage") != seed["source_language"]
            or incumbent.get("targetLanguage") != seed["target_language"]
        ):
            raise SystemExit(f"source/reference lineage mismatch: {source_id}")
        if protected.match(
            source, args.maximum_jaccard, args.character_ngram_size
        ) is not None:
            rejected["source-near-heldout"] += 1
            continue

        proposed = [
            ("teacher", canonical),
            ("licensed-reference", reference),
            ("current-mimi-baseline", baseline),
        ]
        normalized_targets = [normalized(value) for _, value in proposed]
        if len(set(normalized_targets)) != 3:
            if normalized_targets[0] == normalized_targets[1]:
                rejected["teacher-reference-equivalent-no-new-target"] += 1
            else:
                rejected["duplicate-anonymous-candidate"] += 1
            continue

        accepted_candidates: list[tuple[str, str, str]] = []
        rejection_reason: str | None = None
        for origin, translation in proposed:
            accepted, reason, critical_mode = valid_candidate(
                source,
                translation,
                seed["source_language"],
                seed["target_language"],
                reference,
                True,
            )
            if accepted and protected.match(
                translation, args.maximum_jaccard, args.character_ngram_size
            ) is not None:
                accepted, reason = False, "target-near-heldout"
            if not accepted or critical_mode is None:
                rejection_reason = f"{origin}:{reason}"
                break
            accepted_candidates.append((origin, translation, critical_mode))
        if rejection_reason is not None:
            rejected[rejection_reason] += 1
            continue

        admitted_directions[
            f"{seed['source_language']}->{seed['target_language']}"
        ] += 1
        for origin, translation, critical_mode in accepted_candidates:
            critical_modes[f"{origin}:{critical_mode}"] += 1
            is_teacher = origin == "teacher"
            review_queue.append({
                "candidate_id": candidate_id(source_id, origin, translation),
                "candidate_origin": origin,
                "source_id": source_id,
                "source_language": seed["source_language"],
                "target_language": seed["target_language"],
                "split": seed.get("split", "train"),
                "domain": seed["domain"],
                "source": source,
                "translation": translation,
                "style": (
                    "canonical"
                    if origin == "teacher"
                    else origin
                ),
                "risk_tags": payload["risk_tags"] if is_teacher else [],
                "translation_brief": None,
                "critical_token_admission": (
                    "licensed-reference"
                    if origin == "licensed-reference"
                    else critical_mode
                ),
                "source_license": seed["license"],
                "source_provenance": seed["provenance"],
                "licensed_reference": reference,
                "reference_provenance": seed.get("reference_provenance"),
                # The queue-level generator identity is shared source metadata.
                # Candidate origin remains separate and is removed from judge input.
                "teacher_model": teacher_body.get("model"),
                "teacher_response_id": teacher_body.get("id"),
                "teacher_system_fingerprint": teacher_body.get(
                    "system_fingerprint"
                ),
                "baseline_engine": (
                    incumbent_report.get("engine")
                    if origin == "current-mimi-baseline"
                    else None
                ),
                "baseline_model_revision": (
                    incumbent_report.get("modelRevision")
                    if origin == "current-mimi-baseline"
                    else None
                ),
                "review_status": "pending-two-independent-reviews",
            })

    source_count = len({row["source_id"] for row in review_queue})
    if len(review_queue) != source_count * 3:
        raise SystemExit("canonical review queue must have exactly three candidates per source")
    exclusive_jsonl(args.review_queue, review_queue)
    manifest = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "blinded canonical-teacher/reference/current-Mimi comparison",
        "claimEligible": False,
        "candidateOriginExposedToJudges": False,
        "reasoningTraceStored": False,
        "sources": source_count,
        "candidateRows": len(review_queue),
        "admittedDirections": dict(sorted(admitted_directions.items())),
        "rejected": dict(sorted(rejected.items())),
        "criticalAdmissionModes": dict(sorted(critical_modes.items())),
        "inputs": {
            "pilotSeeds": {"path": str(args.pilot_seeds), "sha256": sha256(args.pilot_seeds)},
            "teacherOutput": {"path": str(args.teacher_output), "sha256": sha256(args.teacher_output)},
            "incumbentSuite": {"path": str(args.incumbent_suite), "sha256": sha256(args.incumbent_suite)},
            "incumbentReport": {"path": str(args.incumbent_report), "sha256": sha256(args.incumbent_report)},
            "protectedSuites": [
                {"path": str(path), "sha256": sha256(path)}
                for path in protected_paths
            ],
        },
        "output": {
            "path": str(args.review_queue),
            "sha256": sha256(args.review_queue),
        },
        "appChangeAuthorized": False,
        "trainingAuthorized": False,
        "publicUploadAuthorized": False,
    }
    exclusive_json(args.queue_manifest, manifest)
    print(json.dumps({
        "sources": source_count,
        "candidate_rows": len(review_queue),
        "admitted_directions": dict(sorted(admitted_directions.items())),
        "rejected": dict(sorted(rejected.items())),
        "review_queue_sha256": sha256(args.review_queue),
        "manifest": str(args.queue_manifest),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
