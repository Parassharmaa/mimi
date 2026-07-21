#!/usr/bin/env python3
"""Fail-closed validation for Mimi's claim-ready held-out benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int) -> set[str]:
    value = normalized(text).replace(" ", "")
    return {value[index:index + size] for index in range(max(1, len(value) - size + 1))}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_digest(row: dict) -> str:
    protected = {
        key: row[key]
        for key in (
            "id",
            "documentID",
            "sourceLanguage",
            "targetLanguage",
            "domain",
            "source",
            "references",
            "sourceAuthorID",
            "referenceAuthorIDs",
            "sourceGeneratedByAI",
            "referenceGeneratedByAI",
            "split",
            "license",
            "provenance",
        )
    }
    payload = json.dumps(
        protected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def apportioned(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {domain: total * weight for domain, weight in weights.items()}
    counts = {domain: math.floor(value) for domain, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(weights, key=lambda domain: (-(raw[domain] - counts[domain]), domain))
    for domain in order[:remaining]:
        counts[domain] += 1
    return counts


def review_map(path: Path) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in load_jsonl(path):
        case_id = str(row.get("caseID", "")).strip()
        if not case_id:
            raise SystemExit("review record is missing caseID")
        if case_id in output:
            raise SystemExit(f"duplicate review record: {case_id}")
        output[case_id] = row
    return output


def iter_training_text(row: dict):
    for field in (
        "source",
        "target",
        "translation",
        "reference_translation",
        "student_hypothesis",
    ):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            yield field, value
    for field in ("references", "targets", "translations", "candidates"):
        values = row.get(field, [])
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values):
            if isinstance(value, str) and value.strip():
                yield f"{field}[{index}]", value
                continue
            if not isinstance(value, dict):
                continue
            for nested_field in ("text", "target", "translation", "hypothesis"):
                nested = value.get(nested_field)
                if isinstance(nested, str) and nested.strip():
                    yield f"{field}[{index}].{nested_field}", nested
    for message in row.get("messages", []):
        value = message.get("content") if isinstance(message, dict) else None
        if isinstance(value, str) and value.strip():
            yield "messages.content", value


def training_document_ids(row: dict) -> set[str]:
    values: set[str] = set()
    for container in (row, row.get("metadata", {})):
        if not isinstance(container, dict):
            continue
        for field in ("documentID", "document_id", "documentId"):
            value = container.get(field)
            if isinstance(value, str) and value.strip():
                values.add(value.strip())
    return values


def scan_training(
    paths: list[Path],
    heldout: list[tuple[str, str]],
    heldout_document_ids: set[str],
    size: int,
    maximum: float,
    forbid_document_overlap: bool,
) -> int:
    heldout_ngrams = [(case_id, ngrams(text, size)) for case_id, text in heldout]
    index: dict[str, set[int]] = defaultdict(set)
    heldout_exact: dict[str, str] = {}
    for index_value, (case_id, grams) in enumerate(heldout_ngrams):
        for gram in grams:
            index[gram].add(index_value)
        heldout_exact[normalized(heldout[index_value][1])] = case_id

    scanned = 0
    for path in paths:
        for row_number, row in enumerate(load_jsonl(path), start=1):
            overlapping_documents = training_document_ids(row) & heldout_document_ids
            if forbid_document_overlap and overlapping_documents:
                raise SystemExit(
                    "training document-level contamination: "
                    f"{path}:{row_number} -> {sorted(overlapping_documents)[0]}"
                )
            for field, text in iter_training_text(row):
                scanned += 1
                exact_case = heldout_exact.get(normalized(text))
                if exact_case is not None:
                    raise SystemExit(
                        f"training exact-match contamination: {path}:{row_number} {field} -> {exact_case}"
                    )
                candidate = ngrams(text, size)
                possible = set().union(*(index[gram] for gram in candidate if gram in index))
                for heldout_index in possible:
                    case_id, heldout_gram = heldout_ngrams[heldout_index]
                    similarity = len(candidate & heldout_gram) / max(1, len(candidate | heldout_gram))
                    if similarity > maximum:
                        raise SystemExit(
                            "training near-match contamination: "
                            f"{path}:{row_number} {field} -> {case_id} ({similarity:.4f})"
                        )
    return scanned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("review_records", type=Path)
    parser.add_argument("--training-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    suite = load_jsonl(args.suite)
    reviews = review_map(args.review_records)
    expected_directions = set(manifest["directions"])
    minimum = int(manifest["minimumCasesPerDirection"])
    exact = int(manifest.get("exactCasesPerDirection", minimum))
    minimum_references = int(manifest["referencePolicy"]["minimumReferencesPerCase"])
    minimum_reference_authors = int(
        manifest["referencePolicy"]["minimumIndependentReferenceAuthors"]
    )
    minimum_reviewers = int(
        manifest["referencePolicy"]["minimumIndependentBilingualReviewers"]
    )
    allowed_domains = set(manifest["domains"])

    ids = [str(row.get("id", "")) for row in suite]
    if not all(ids) or len(ids) != len(set(ids)):
        raise SystemExit("suite IDs must be non-empty and unique")
    if set(reviews) != set(ids):
        missing, extra = set(ids) - set(reviews), set(reviews) - set(ids)
        raise SystemExit(f"review records do not match suite; missing={len(missing)} extra={len(extra)}")

    by_direction: dict[str, list[dict]] = defaultdict(list)
    seen_sources: dict[str, str] = {}
    heldout_document_ids: set[str] = set()
    heldout_text: list[tuple[str, str]] = []
    for row in suite:
        case_id = str(row["id"])
        direction = f"{row.get('sourceLanguage')}>{row.get('targetLanguage')}"
        if direction not in expected_directions:
            raise SystemExit(f"unsupported direction: {case_id} {direction}")
        if row.get("domain") not in allowed_domains:
            raise SystemExit(f"unsupported domain: {case_id}")
        if row.get("split") != "heldout" or row.get("reviewStatus") != "adjudicated":
            raise SystemExit(f"claim-ready case must be heldout and adjudicated: {case_id}")
        if row.get("claimEligible") is not True:
            raise SystemExit(f"claim-ready case is not claimEligible: {case_id}")
        if row.get("sourceGeneratedByAI") is not False or row.get("referenceGeneratedByAI") is not False:
            raise SystemExit(f"source/reference synthetic declaration is missing or true: {case_id}")
        document_id = str(row.get("documentID", "")).strip()
        if not document_id:
            raise SystemExit(f"case is missing documentID: {case_id}")
        heldout_document_ids.add(document_id)
        if not str(row.get("license", "")).strip() or not str(row.get("provenance", "")).strip():
            raise SystemExit(f"case is missing license/provenance: {case_id}")
        source = str(row.get("source", "")).strip()
        references = [str(value).strip() for value in row.get("references", [])]
        if not source or len(references) < minimum_references or not all(references):
            raise SystemExit(f"case has insufficient source/references: {case_id}")
        if len({normalized(value) for value in references}) != len(references):
            raise SystemExit(f"case has duplicate references: {case_id}")
        source_author = str(row.get("sourceAuthorID", "")).strip()
        reference_authors = [
            str(value).strip() for value in row.get("referenceAuthorIDs", [])
        ]
        if (
            not source_author
            or len(reference_authors) != len(references)
            or len(set(reference_authors)) < minimum_reference_authors
            or not all(reference_authors)
            or source_author in set(reference_authors)
        ):
            raise SystemExit(f"case lacks independent source/reference authors: {case_id}")
        normalized_source = normalized(source)
        if normalized_source in seen_sources:
            raise SystemExit(f"duplicate source across cases: {seen_sources[normalized_source]} and {case_id}")
        seen_sources[normalized_source] = case_id

        review = reviews[case_id]
        reviewer_ids = [str(value).strip() for value in review.get("reviewerIDs", [])]
        adjudicator = str(review.get("adjudicatorID", "")).strip()
        if review.get("blinded") is not True or review.get("decision") != "approved":
            raise SystemExit(f"review is not blind and approved: {case_id}")
        if len(reviewer_ids) < minimum_reviewers or len(set(reviewer_ids)) != len(reviewer_ids):
            raise SystemExit(f"review lacks distinct bilingual reviewers: {case_id}")
        if not adjudicator or adjudicator in set(reviewer_ids):
            raise SystemExit(f"review lacks an independent adjudicator: {case_id}")
        if {source_author, *reference_authors} & {*reviewer_ids, adjudicator}:
            raise SystemExit(f"case authors overlap reviewers or adjudicator: {case_id}")
        if review.get("approvedReferences") != references:
            raise SystemExit(f"review references do not match suite: {case_id}")
        expected_attestations = {
            "human": True,
            "bilingualQualified": True,
            "independent": True,
            "noAIAssistance": True,
        }
        reviewer_attestations = review.get("reviewerAttestations", {})
        if set(reviewer_attestations) != set(reviewer_ids) or any(
            value != expected_attestations for value in reviewer_attestations.values()
        ):
            raise SystemExit(f"reviewer attestations are missing or invalid: {case_id}")
        if review.get("adjudicatorAttestations") != expected_attestations:
            raise SystemExit(f"adjudicator attestations are missing or invalid: {case_id}")
        if review.get("suiteCaseSHA256") != case_digest(row):
            raise SystemExit(f"review record hash does not match suite case: {case_id}")

        by_direction[direction].append(row)
        heldout_text.extend([(case_id, source), *((case_id, value) for value in references)])

    if set(by_direction) != expected_directions:
        raise SystemExit("suite does not contain every required direction")
    direction_counts: dict[str, dict] = {}
    for direction, direction_rows in by_direction.items():
        if len(direction_rows) != exact:
            raise SystemExit(f"{direction} has {len(direction_rows)} cases; need exactly {exact}")
        expected_domains = apportioned(len(direction_rows), manifest["domains"])
        actual_domains = Counter(str(row["domain"]) for row in direction_rows)
        if dict(actual_domains) != expected_domains:
            raise SystemExit(
                f"{direction} domain quota mismatch; actual={dict(actual_domains)} expected={expected_domains}"
            )
        direction_counts[direction] = {
            "cases": len(direction_rows),
            "domains": dict(sorted(actual_domains.items())),
        }

    contamination = manifest["contaminationPolicy"]
    scanned = scan_training(
        args.training_jsonl,
        heldout_text,
        heldout_document_ids,
        int(contamination["characterNgramSize"]),
        float(contamination["maximumTrainHeldoutJaccard"]),
        bool(contamination.get("forbidTrainingDocumentIDOverlap", False)),
    )
    output = {
        "schemaVersion": 1,
        "status": "claim-ready-suite-validated",
        "suiteID": manifest["suiteID"],
        "suite": {"path": str(args.suite), "sha256": sha256(args.suite)},
        "reviewRecords": {
            "path": str(args.review_records),
            "sha256": sha256(args.review_records),
        },
        "directions": direction_counts,
        "trainingFilesScanned": len(args.training_jsonl),
        "trainingTextsScanned": scanned,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
