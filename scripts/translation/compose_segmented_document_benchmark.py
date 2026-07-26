#!/usr/bin/env python3
"""Compose sentence-level engine outputs into benchmark document units."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summed_warm_latencies(segments: list[dict], case_id: str) -> list[float]:
    lengths = {len(segment.get("warmLatencySeconds", [])) for segment in segments}
    if len(lengths) != 1:
        raise SystemExit(f"segment warm-run mismatch: {case_id}")
    length = lengths.pop()
    return [
        sum(float(segment["warmLatencySeconds"][index]) for segment in segments)
        for index in range(length)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_suite", type=Path)
    parser.add_argument("segment_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--direction",
        choices=("en-ja", "ja-en"),
        help="Compose only one direction from a direction-filtered segment report.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    cases = load_jsonl(args.case_suite)
    requested_direction = {
        "en-ja": ("en-US", "ja-JP"),
        "ja-en": ("ja-JP", "en-US"),
    }.get(args.direction)
    if requested_direction is not None:
        cases = [
            case
            for case in cases
            if (case.get("sourceLanguage"), case.get("targetLanguage"))
            == requested_direction
        ]
        if not cases:
            raise SystemExit(f"case suite has no {args.direction} cases")
    report = load_json(args.segment_report)
    segment_results = report.get("results")
    if not isinstance(segment_results, list) or not segment_results:
        raise SystemExit("segment report has no results")
    indexed = {}
    for result in segment_results:
        case_id = str(result.get("caseID", ""))
        if not case_id or case_id in indexed:
            raise SystemExit(f"missing or duplicate segment result: {case_id}")
        indexed[case_id] = result

    composed_results = []
    consumed: set[str] = set()
    for case in cases:
        case_id = str(case.get("id", ""))
        segment_ids = case.get("segmentBenchmarkIDs")
        source_segments = case.get("segments")
        reference_segments = case.get("referenceSegments")
        if (
            not case_id
            or not isinstance(segment_ids, list)
            or not isinstance(source_segments, list)
            or not isinstance(reference_segments, list)
            or len(segment_ids) != int(case.get("segmentCount", -1))
            or len(source_segments) != len(segment_ids)
            or len(reference_segments) != len(segment_ids)
        ):
            raise SystemExit(f"invalid composed case: {case_id}")
        missing = [
            segment_id for segment_id in segment_ids if segment_id not in indexed
        ]
        if missing:
            raise SystemExit(f"missing segment results for {case_id}: {missing}")
        segments = [indexed[segment_id] for segment_id in segment_ids]
        for index, segment in enumerate(segments):
            if (
                segment.get("sourceLanguage") != case.get("sourceLanguage")
                or segment.get("targetLanguage") != case.get("targetLanguage")
                or segment.get("source") != source_segments[index]
                or segment.get("references") != [reference_segments[index]]
            ):
                raise SystemExit(
                    f"segment report disagrees with composed case: {case_id}/{index}"
                )
        consumed.update(segment_ids)
        hypotheses = [
            str(segment.get("hypothesis", "")).strip() for segment in segments
        ]
        empty_segment_indexes = [
            index for index, hypothesis in enumerate(hypotheses) if not hypothesis
        ]
        result = {
            "caseID": case_id,
            "sourceLanguage": case["sourceLanguage"],
            "targetLanguage": case["targetLanguage"],
            "domain": case["domain"],
            "source": case["source"],
            "references": case["references"],
            "claimEligible": bool(case["claimEligible"]),
            "sourceUnit": case["sourceUnit"],
            "segmentCount": case["segmentCount"],
            "hypothesis": "\n".join(hypotheses),
            "segmentHypotheses": hypotheses,
            "emptySegmentCount": len(empty_segment_indexes),
            "emptySegmentIndexes": empty_segment_indexes,
            "segmentBenchmarkIDs": segment_ids,
            "latencySeconds": sum(
                float(segment["latencySeconds"]) for segment in segments
            ),
            "warmLatencySeconds": summed_warm_latencies(segments, case_id),
            "segmentLatencySeconds": [
                float(segment["latencySeconds"]) for segment in segments
            ],
            "segmentWarmLatencySeconds": [
                [float(value) for value in segment.get("warmLatencySeconds", [])]
                for segment in segments
            ],
        }
        if all("outputTokenIDs" in segment for segment in segments):
            result["segmentOutputTokenIDs"] = [
                segment["outputTokenIDs"] for segment in segments
            ]
        composed_results.append(result)

    if consumed != set(indexed):
        extra = sorted(set(indexed) - consumed)
        raise SystemExit(f"segment report contains unreferenced cases: {extra[:3]}")

    output = {
        key: value
        for key, value in report.items()
        if key not in {"results", "benchmarkConfiguration"}
    }
    output["engine"] = f"{report.get('engine', 'unknown')}:segment-then-join"
    output["sourceSegmentEngine"] = report.get("engine")
    output["sourceSegmentReportSHA256"] = sha256(args.segment_report)
    output["caseSuiteSHA256"] = sha256(args.case_suite)
    output["documentAggregation"] = {
        "strategy": "translate-segments-independently-then-join-with-newline",
        "crossSegmentContext": False,
        "latencyAggregation": "sum",
        "warmLatencyAggregation": "sum-by-run-index",
    }
    output["benchmarkConfiguration"] = {
        **report.get("benchmarkConfiguration", {}),
        "caseUnits": len(cases),
        "segmentUnits": len(segment_results),
    }
    output["results"] = composed_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(composed_results)} composed cases to {args.output}")


if __name__ == "__main__":
    main()
