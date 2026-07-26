#!/usr/bin/env python3
"""Freeze Mimi's non-claimable sentence and segmented-document development suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
SENTENCE_CORPORA = {
    "tatoeba": "everyday-conversation",
    "alt": "human-translated-news",
    "kftt": "professional-wikipedia",
    "jlt": "ministry-published-legal",
}
DOCUMENT_CORPORA = {
    "alt": "long-document-news",
    "kftt": "long-document-wikipedia",
    "jlt": "long-document-legal",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank(seed: str, *parts: str) -> str:
    return hashlib.sha256("\0".join((seed, *parts)).encode()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("source suite is empty")
    identifiers = [str(row.get("id", "")) for row in rows]
    if "" in identifiers or len(identifiers) != len(set(identifiers)):
        raise SystemExit("source suite has missing or duplicate IDs")
    return rows


def paired_rows(rows: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    pairs: dict[tuple[str, str], dict[str, dict]] = {}
    ambiguous: set[tuple[str, str]] = set()
    languages_to_direction = {languages: name for name, languages in DIRECTIONS.items()}
    for row in rows:
        corpus = str(row.get("sourceCorpus", ""))
        source_id = str(row.get("sourceID", ""))
        direction = languages_to_direction.get(
            (row.get("sourceLanguage"), row.get("targetLanguage"))
        )
        if (
            corpus not in SENTENCE_CORPORA
            or not source_id
            or direction is None
            or not isinstance(row.get("references"), list)
            or len(row["references"]) != 1
        ):
            raise SystemExit(f"invalid source row: {row.get('id')}")
        key = (corpus, source_id)
        current = pairs.setdefault(key, {})
        if direction in current:
            ambiguous.add(key)
        current[direction] = row
    return {
        key: directions
        for key, directions in pairs.items()
        if key not in ambiguous and set(directions) == set(DIRECTIONS)
    }


def numeric_suffix(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", value)
    return (int(match.group(1)) if match else -1, value)


def natural_component_order(value: str) -> tuple[int, str]:
    return numeric_suffix(value)


def document_groups(
    corpus: str,
    pairs: dict[tuple[str, str], dict[str, dict]],
    segment_count: int,
) -> dict[str, list[str]]:
    source_ids = sorted(
        (source_id for pair_corpus, source_id in pairs if pair_corpus == corpus),
        key=natural_component_order,
    )
    if corpus == "alt":
        grouped: dict[str, list[str]] = defaultdict(list)
        for source_id in source_ids:
            parts = source_id.split(".")
            if len(parts) < 3:
                continue
            grouped[".".join(parts[:2])].append(source_id)
        return {
            group: sorted(values, key=natural_component_order)
            for group, values in grouped.items()
            if len(values) >= segment_count
        }
    if corpus == "jlt":
        grouped = defaultdict(list)
        for source_id in source_ids:
            group = source_id.split(":tu-", 1)[0]
            grouped[group].append(source_id)
        return {
            group: sorted(values, key=natural_component_order)
            for group, values in grouped.items()
            if len(values) >= segment_count
        }
    if corpus == "kftt":
        return {
            f"kftt-block-{index // segment_count:04d}": source_ids[
                index : index + segment_count
            ]
            for index in range(0, len(source_ids), segment_count)
            if len(source_ids[index : index + segment_count]) == segment_count
        }
    raise SystemExit(f"unsupported document corpus: {corpus}")


def select_document_components(
    *,
    seed: str,
    corpus: str,
    groups: dict[str, list[str]],
    document_count: int,
    segment_count: int,
) -> list[tuple[str, list[str]]]:
    selected_groups = sorted(
        groups,
        key=lambda group: stable_rank(seed, "document-group", corpus, group),
    )[:document_count]
    if len(selected_groups) != document_count:
        raise SystemExit(
            f"need {document_count} {corpus} document groups, found {len(selected_groups)}"
        )
    output = []
    for group in selected_groups:
        values = groups[group]
        available_windows = len(values) - segment_count + 1
        offset = (
            int(stable_rank(seed, "document-window", corpus, group)[:16], 16)
            % available_windows
        )
        output.append((group, values[offset : offset + segment_count]))
    return output


def combined_text(rows: list[dict], key: str) -> str:
    if key == "source":
        return "\n".join(str(row["source"]).strip() for row in rows)
    return "\n".join(str(row["references"][0]).strip() for row in rows)


def combined_metadata(rows: list[dict], field: str) -> str:
    values = list(dict.fromkeys(str(row.get(field, "")).strip() for row in rows))
    values = [value for value in values if value]
    if not values:
        raise SystemExit(f"document components lack {field}")
    return values[0] if len(values) == 1 else " | ".join(values)


def flat_segment(
    *,
    benchmark_id: str,
    parent_id: str,
    row: dict,
    domain: str,
    source_unit: str,
    segment_index: int,
    segment_count: int,
) -> dict:
    return {
        "id": benchmark_id,
        "sourceLanguage": row["sourceLanguage"],
        "targetLanguage": row["targetLanguage"],
        "domain": domain,
        "source": row["source"],
        "references": row["references"],
        "claimEligible": False,
        "split": "development-accuracy-v1-segments",
        "license": row["license"],
        "provenance": row["provenance"],
        "reviewStatus": row["reviewStatus"],
        "sourceCorpus": row["sourceCorpus"],
        "sourceID": row["sourceID"],
        "sourceUnit": source_unit,
        "parentCaseID": parent_id,
        "segmentIndex": segment_index,
        "segmentCount": segment_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("segment_output", type=Path)
    parser.add_argument("--seed", default="mimi-development-accuracy-v1-20260725")
    parser.add_argument("--sentence-pairs-per-corpus", type=int, default=20)
    parser.add_argument("--document-segments", type=int, default=6)
    parser.add_argument("--alt-documents", type=int, default=8)
    parser.add_argument("--kftt-documents", type=int, default=6)
    parser.add_argument("--jlt-documents", type=int, default=6)
    args = parser.parse_args()

    if args.output.exists() or args.segment_output.exists():
        raise SystemExit("refusing to overwrite an existing development suite")
    if args.sentence_pairs_per_corpus < 1 or args.document_segments < 2:
        raise SystemExit("invalid sentence or document-segment count")
    document_counts = {
        "alt": args.alt_documents,
        "kftt": args.kftt_documents,
        "jlt": args.jlt_documents,
    }
    if any(count < 0 for count in document_counts.values()) or not sum(
        document_counts.values()
    ):
        raise SystemExit("at least one document case is required")

    source_rows = load_rows(args.source_suite)
    pairs = paired_rows(source_rows)
    used_components: set[tuple[str, str]] = set()
    suite: list[dict] = []
    segment_suite: list[dict] = []

    for corpus, document_count in document_counts.items():
        groups = document_groups(corpus, pairs, args.document_segments)
        selections = select_document_components(
            seed=args.seed,
            corpus=corpus,
            groups=groups,
            document_count=document_count,
            segment_count=args.document_segments,
        )
        for group, source_ids in selections:
            for source_id in source_ids:
                key = (corpus, source_id)
                if key in used_components:
                    raise SystemExit(f"document component reused: {corpus}/{source_id}")
                used_components.add(key)
            for direction, languages in DIRECTIONS.items():
                component_rows = [
                    pairs[(corpus, source_id)][direction] for source_id in source_ids
                ]
                parent_id = (
                    f"development-accuracy-v1:document:{corpus}:{group}:{direction}"
                )
                segment_ids = []
                for index, row in enumerate(component_rows):
                    benchmark_id = f"{parent_id}:segment-{index + 1:02d}"
                    segment_ids.append(benchmark_id)
                    segment_suite.append(
                        flat_segment(
                            benchmark_id=benchmark_id,
                            parent_id=parent_id,
                            row=row,
                            domain=DOCUMENT_CORPORA[corpus],
                            source_unit="document-segment",
                            segment_index=index,
                            segment_count=len(component_rows),
                        )
                    )
                source = combined_text(component_rows, "source")
                reference = combined_text(component_rows, "reference")
                suite.append(
                    {
                        "id": parent_id,
                        "sourceLanguage": languages[0],
                        "targetLanguage": languages[1],
                        "domain": DOCUMENT_CORPORA[corpus],
                        "source": source,
                        "references": [reference],
                        "claimEligible": False,
                        "split": "development-accuracy-v1",
                        "license": combined_metadata(component_rows, "license"),
                        "provenance": combined_metadata(component_rows, "provenance"),
                        "reviewStatus": "composed-human-reference-unreviewed",
                        "sourceCorpus": corpus,
                        "sourceID": group,
                        "sourceUnit": "document",
                        "segmentCount": len(component_rows),
                        "segments": [row["source"] for row in component_rows],
                        "referenceSegments": [
                            row["references"][0] for row in component_rows
                        ],
                        "sourceComponentIDs": [row["id"] for row in component_rows],
                        "segmentBenchmarkIDs": segment_ids,
                        "documentEvaluationMode": (
                            "segment-then-join-no-cross-segment-context"
                        ),
                        "sourceCharacterCount": len(source),
                        "referenceCharacterCount": len(reference),
                    }
                )

    for corpus, domain in SENTENCE_CORPORA.items():
        available = [
            source_id
            for pair_corpus, source_id in pairs
            if pair_corpus == corpus and (corpus, source_id) not in used_components
        ]
        selected = sorted(
            available,
            key=lambda source_id: stable_rank(args.seed, "sentence", corpus, source_id),
        )[: args.sentence_pairs_per_corpus]
        if len(selected) != args.sentence_pairs_per_corpus:
            raise SystemExit(
                f"need {args.sentence_pairs_per_corpus} {corpus} sentence pairs, "
                f"found {len(selected)}"
            )
        for source_id in selected:
            for direction, languages in DIRECTIONS.items():
                row = pairs[(corpus, source_id)][direction]
                parent_id = (
                    f"development-accuracy-v1:sentence:{corpus}:{source_id}:{direction}"
                )
                benchmark_id = f"{parent_id}:segment-01"
                segment_suite.append(
                    flat_segment(
                        benchmark_id=benchmark_id,
                        parent_id=parent_id,
                        row=row,
                        domain=domain,
                        source_unit="sentence",
                        segment_index=0,
                        segment_count=1,
                    )
                )
                suite.append(
                    {
                        "id": parent_id,
                        "sourceLanguage": languages[0],
                        "targetLanguage": languages[1],
                        "domain": domain,
                        "source": row["source"],
                        "references": row["references"],
                        "claimEligible": False,
                        "split": "development-accuracy-v1",
                        "license": row["license"],
                        "provenance": row["provenance"],
                        "reviewStatus": row["reviewStatus"],
                        "sourceCorpus": corpus,
                        "sourceID": source_id,
                        "sourceUnit": "sentence",
                        "segmentCount": 1,
                        "segments": [row["source"]],
                        "referenceSegments": row["references"],
                        "sourceComponentIDs": [row["id"]],
                        "segmentBenchmarkIDs": [benchmark_id],
                        "documentEvaluationMode": None,
                        "sourceCharacterCount": len(row["source"]),
                        "referenceCharacterCount": len(row["references"][0]),
                    }
                )

    suite.sort(key=lambda row: row["id"])
    segment_suite.sort(key=lambda row: row["id"])
    expected_direction_cases = len(
        SENTENCE_CORPORA
    ) * args.sentence_pairs_per_corpus + sum(document_counts.values())
    expected_cases = expected_direction_cases * len(DIRECTIONS)
    expected_direction_segments = (
        len(SENTENCE_CORPORA) * args.sentence_pairs_per_corpus
        + sum(document_counts.values()) * args.document_segments
    )
    expected_segments = expected_direction_segments * len(DIRECTIONS)
    if len(suite) != expected_cases or len(segment_suite) != expected_segments:
        raise SystemExit(
            f"unexpected suite sizes: cases={len(suite)} segments={len(segment_suite)}"
        )
    for languages in DIRECTIONS.values():
        if (
            sum(
                (row["sourceLanguage"], row["targetLanguage"]) == languages
                for row in suite
            )
            != expected_direction_cases
        ):
            raise SystemExit(f"direction case-count mismatch: {languages}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.segment_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in suite
        ),
        encoding="utf-8",
    )
    args.segment_output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in segment_suite
        ),
        encoding="utf-8",
    )
    manifest = {
        "schemaVersion": 1,
        "suite": "development-accuracy-v1",
        "purpose": (
            "non-claimable model-development selection with sentence and "
            "segmented-document coverage"
        ),
        "seed": args.seed,
        "claimEligible": False,
        "sourceSuite": {
            "path": str(args.source_suite),
            "sha256": sha256(args.source_suite),
        },
        "outputs": {
            "caseSuite": {
                "path": str(args.output),
                "sha256": sha256(args.output),
                "cases": len(suite),
            },
            "segmentSuite": {
                "path": str(args.segment_output),
                "sha256": sha256(args.segment_output),
                "cases": len(segment_suite),
            },
        },
        "caseCounts": {
            "perDirection": expected_direction_cases,
            "sentencesPerDirection": (
                len(SENTENCE_CORPORA) * args.sentence_pairs_per_corpus
            ),
            "documentsPerDirection": sum(document_counts.values()),
            "total": len(suite),
        },
        "segmentCounts": {
            "perDirection": expected_direction_segments,
            "total": len(segment_suite),
            "segmentsPerDocument": args.document_segments,
        },
        "sentenceQuotasPerDirection": {
            corpus: args.sentence_pairs_per_corpus for corpus in SENTENCE_CORPORA
        },
        "documentQuotasPerDirection": document_counts,
        "documentEvaluationMode": "segment-then-join-no-cross-segment-context",
        "selection": (
            "ascending SHA-256(seed, kind, corpus, identifier); coherent ALT/JLT "
            "groups and deterministic contiguous KFTT blocks; deterministic "
            "component window"
        ),
        "trainingOverlapControls": {
            "localDatasetSplit": (
                "all components come from public-stress-v3 declared test splits"
            ),
            "knownLimitation": (
                "public corpora may overlap opaque upstream ElanMT pretraining or "
                "prior development/model selection"
            ),
        },
        "limitations": [
            "one licensed human reference per source segment",
            "composed documents have no independent document-level bilingual review",
            "segment-then-join evaluation measures app behavior but supplies no cross-segment context",
            "public development data cannot replace the sealed 400+400 promotion suite",
        ],
        "sealedPromotionSuiteTouched": False,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
