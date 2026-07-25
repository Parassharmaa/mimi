#!/usr/bin/env python3
"""Turn GPT batch output into a blinded review queue; never approve training rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from filter_training_dataset_against_protected import ProtectedIndex
from typed_critical_token_policy import (
    reference_validated_critical_preservation_mode,
)


TOKEN_RE = re.compile(r"https?://\S+|\{[^{}]+\}|%\w|\b\d[\d,.:/-]*\b")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def response_payload(batch_row: dict) -> tuple[dict, dict]:
    body = batch_row.get("response", {}).get("body", batch_row.get("body", {}))
    for output in body.get("output", []):
        if output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                return json.loads(content["text"]), body
    if "output_text" in body:
        return json.loads(body["output_text"]), body
    raise ValueError("response has no Structured Outputs text")


def valid_candidate(
    source: str,
    target: str,
    source_language: str,
    target_language: str,
    licensed_reference: str,
    allow_reference_validated_critical_equivalence: bool,
) -> tuple[bool, str, str | None]:
    source_norm, target_norm = normalized(source), normalized(target)
    if not target_norm or target_norm == source_norm:
        return False, "empty-or-copied", None
    if len(target_norm) > max(24, len(source_norm) * 4) or len(target_norm) * 4 < len(source_norm):
        return False, "length-ratio", None
    critical_mode = reference_validated_critical_preservation_mode(
        source,
        target,
        licensed_reference if allow_reference_validated_critical_equivalence else "",
        source_language,
        target_language,
    )
    if critical_mode is None:
        return False, "protected-token-mismatch", None
    if target_language == "ja-JP" and len(JAPANESE_RE.findall(target)) < 2:
        return False, "target-script", None
    if target_language == "en-US" and len(LATIN_RE.findall(target)) < 2:
        return False, "target-script", None
    return True, "accepted-for-review", critical_mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", type=Path)
    parser.add_argument("batch_output", type=Path)
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    parser.add_argument(
        "--additional-protected-suite",
        type=Path,
        action="append",
        default=[],
        help="additional held-out JSONL suite; may be repeated",
    )
    parser.add_argument("--character-ngram-size", type=int, default=5)
    parser.add_argument(
        "--include-licensed-reference-candidate",
        action="store_true",
        help=(
            "blindly add each available licensed human reference to the judge "
            "candidate set; an exact teacher duplicate is marked reference-equivalent"
        ),
    )
    parser.add_argument(
        "--reference-validated-critical-equivalence",
        action="store_true",
        help=(
            "allow narrowly scoped bilingual critical-token equivalence only "
            "when the row has a licensed human reference validating the same "
            "target critical surfaces; source-only rows remain strict"
        ),
    )
    args = parser.parse_args()

    if args.review_queue.exists() and args.review_queue.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.review_queue}")

    seed_by_id = {str(row["id"]): row for row in rows(args.seeds)}
    protected_paths = [
        args.protected_benchmark,
        *args.additional_protected_suite,
    ]
    protected = ProtectedIndex(protected_paths, args.character_ngram_size)
    queue: list[dict] = []
    rejects: dict[str, int] = {}
    seen_results: set[str] = set()
    for batch_row in rows(args.batch_output):
        custom_id = str(batch_row["custom_id"])
        if custom_id in seen_results:
            raise SystemExit(f"batch output contains duplicate custom_id: {custom_id}")
        seen_results.add(custom_id)
        seed = seed_by_id.get(custom_id)
        if seed is None:
            raise SystemExit(f"batch result has unknown custom_id: {custom_id}")
        payload, body = response_payload(batch_row)
        if str(payload.get("source_id")) != custom_id:
            raise SystemExit(f"Structured Output source_id mismatch: {custom_id}")
        source = seed["source"]
        if protected.match(
            source, args.maximum_jaccard, args.character_ngram_size
        ) is not None:
            rejects["source-near-heldout"] = rejects.get("source-near-heldout", 0) + 1
            continue
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 3:
            raise SystemExit(f"Structured Output must contain exactly three candidates: {custom_id}")
        styles = [candidate.get("style") for candidate in candidates]
        if set(styles) != {"natural-spoken", "concise-caption", "meaning-conservative"}:
            raise SystemExit(f"Structured Output has missing or duplicate styles: {custom_id}")
        source_queue: list[dict] = []
        normalized_targets: set[str] = set()
        rejection_reason: str | None = None
        licensed_reference = str(seed.get("reference_translation") or "").strip()
        for candidate in candidates:
            translation = candidate["translation"].strip()
            accepted, reason, critical_mode = valid_candidate(
                source,
                translation,
                seed["source_language"],
                seed["target_language"],
                licensed_reference,
                args.reference_validated_critical_equivalence,
            )
            if accepted and protected.match(
                translation, args.maximum_jaccard, args.character_ngram_size
            ) is not None:
                accepted, reason = False, "target-near-heldout"
            normalized_translation = normalized(translation)
            if accepted and normalized_translation in normalized_targets:
                accepted, reason = False, "duplicate-candidate"
            if not accepted:
                rejection_reason = reason
                break
            normalized_targets.add(normalized_translation)
            candidate_id = hashlib.sha256(
                f"{custom_id}\0{candidate['style']}\0{normalized_translation}".encode()
            ).hexdigest()[:24]
            source_queue.append({
                "candidate_id": candidate_id,
                "candidate_origin": "teacher",
                "source_id": custom_id,
                "source_language": seed["source_language"],
                "target_language": seed["target_language"],
                "split": seed.get("split", "train"),
                "domain": seed.get("domain", "unknown"),
                "source": source,
                "translation": translation,
                "style": candidate["style"],
                "risk_tags": candidate["risk_tags"],
                "translation_brief": payload["translation_brief"],
                "critical_token_admission": critical_mode,
                "source_license": seed["license"],
                "source_provenance": seed["provenance"],
                "licensed_reference": seed.get("reference_translation"),
                "reference_provenance": seed.get("reference_provenance"),
                "teacher_model": body.get("model"),
                "teacher_response_id": body.get("id"),
                "teacher_system_fingerprint": body.get("system_fingerprint"),
                "review_status": "pending-two-independent-reviews",
            })
        if rejection_reason is not None:
            rejects[rejection_reason] = rejects.get(rejection_reason, 0) + 1
            continue
        if args.include_licensed_reference_candidate:
            if licensed_reference:
                normalized_reference = normalized(licensed_reference)
                matching_teacher = next(
                    (
                        row
                        for row in source_queue
                        if normalized(row["translation"]) == normalized_reference
                    ),
                    None,
                )
                if matching_teacher is not None:
                    matching_teacher["candidate_origin"] = "teacher-reference-equivalent"
                    matching_teacher["reference_anchor_status"] = "exact-teacher-match"
                else:
                    reference_id = hashlib.sha256(
                        f"{custom_id}\0licensed-reference\0{normalized_reference}".encode()
                    ).hexdigest()[:24]
                    source_queue.append({
                        "candidate_id": reference_id,
                        "candidate_origin": "licensed-reference",
                        "source_id": custom_id,
                        "source_language": seed["source_language"],
                        "target_language": seed["target_language"],
                        "split": seed.get("split", "train"),
                        "domain": seed.get("domain", "unknown"),
                        "source": source,
                        "translation": licensed_reference,
                        "style": "licensed-reference",
                        "risk_tags": [],
                        "translation_brief": None,
                        "critical_token_admission": "licensed-reference",
                        "source_license": seed["license"],
                        "source_provenance": seed["provenance"],
                        "licensed_reference": licensed_reference,
                        "reference_provenance": seed.get("reference_provenance"),
                        "teacher_model": body.get("model"),
                        "teacher_response_id": body.get("id"),
                        "teacher_system_fingerprint": body.get("system_fingerprint"),
                        "reference_anchor_status": "blinded-fourth-candidate",
                        "review_status": "pending-two-independent-reviews",
                    })
        queue.extend(source_queue)

    missing_results = set(seed_by_id) - seen_results
    if missing_results:
        raise SystemExit(
            f"batch output is missing {len(missing_results)} seed results; "
            f"first: {next(iter(missing_results))}"
        )

    args.review_queue.parent.mkdir(parents=True, exist_ok=True)
    args.review_queue.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue),
        encoding="utf-8",
    )
    candidate_counts_by_source = Counter(row["source_id"] for row in queue)
    print(json.dumps({
        "queued": len(queue),
        "sources": len(candidate_counts_by_source),
        "candidate_counts": {
            str(count): sum(value == count for value in candidate_counts_by_source.values())
            for count in (3, 4)
        },
        "licensed_reference_candidates": sum(
            row.get("candidate_origin") == "licensed-reference" for row in queue
        ),
        "teacher_reference_equivalents": sum(
            row.get("candidate_origin") == "teacher-reference-equivalent"
            for row in queue
        ),
        "critical_token_admission": dict(
            sorted(Counter(
                row.get("critical_token_admission", "unknown")
                for row in queue
                if row.get("candidate_origin") != "licensed-reference"
            ).items())
        ),
        "reference_validated_critical_equivalence": (
            args.reference_validated_critical_equivalence
        ),
        "protected_suites": len(protected_paths),
        "rejected": rejects,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
