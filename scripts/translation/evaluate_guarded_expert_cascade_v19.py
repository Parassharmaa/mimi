#!/usr/bin/env python3
"""Compose and score V19's loop-guarded local Marian expert cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sacrebleu


EXPERIMENT = "guarded-expert-cascade-v19"
DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def repeated_token_loop(token_ids: list[int]) -> bool:
    for width in range(3, min(16, len(token_ids) // 3) + 1):
        for start in range(0, len(token_ids) - width * 3 + 1):
            phrase = token_ids[start : start + width]
            if (
                token_ids[start + width : start + width * 2] == phrase
                and token_ids[start + width * 2 : start + width * 3] == phrase
            ):
                return True
    return False


def record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(root))
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def validate_record(item: dict[str, Any], root: Path) -> Path:
    path = root / str(item.get("path", ""))
    if (
        not path.is_file()
        or item.get("bytes") != path.stat().st_size
        or item.get("sha256") != sha256(path)
    ):
        raise SystemExit(f"authenticated input differs: {path}")
    return path


def result_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = report.get("results")
    if not isinstance(values, list) or not values:
        raise SystemExit("benchmark report has no results")
    mapped = {str(row.get("caseID", "")): row for row in values}
    if "" in mapped or len(mapped) != len(values):
        raise SystemExit("benchmark report has empty or duplicate case IDs")
    return mapped


def matching_case(candidate: dict[str, Any], bundled: dict[str, Any]) -> None:
    for key in (
        "caseID",
        "sourceLanguage",
        "targetLanguage",
        "domain",
        "source",
        "references",
        "segmentCount",
        "segmentBenchmarkIDs",
    ):
        if candidate.get(key) != bundled.get(key):
            raise SystemExit(
                f"candidate/bundled case metadata differs: "
                f"{candidate.get('caseID')} {key}"
            )
    segment_count = int(candidate.get("segmentCount", 0))
    for row, label in ((candidate, "candidate"), (bundled, "bundled")):
        if not (
            len(row.get("segmentHypotheses", []))
            == len(row.get("segmentOutputTokenIDs", []))
            == segment_count
        ):
            raise SystemExit(
                f"{label} segment evidence differs: {candidate.get('caseID')}"
            )


def references(rows: list[dict[str, Any]]) -> list[list[str]]:
    maximum = max(len(row["references"]) for row in rows)
    return [
        [
            row["references"][min(index, len(row["references"]) - 1)]
            for row in rows
        ]
        for index in range(maximum)
    ]


def metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    hypotheses = [str(row["hypothesis"]) for row in rows]
    return {
        "cases": len(rows),
        "chrFPlusPlus": float(
            sacrebleu.corpus_chrf(
                hypotheses,
                references(rows),
                word_order=2,
            ).score
        ),
        "BLEU": float(
            sacrebleu.corpus_bleu(
                hypotheses,
                references(rows),
                tokenize="intl",
            ).score
        ),
    }


def gate(
    name: str,
    actual: float | int,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, Any]:
    if (minimum is None) == (maximum is None):
        raise ValueError("gate requires exactly one bound")
    passed = actual >= minimum if minimum is not None else actual <= maximum
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "minimum" if minimum is not None else "maximum": (
            minimum if minimum is not None else maximum
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("composed_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for path in (args.composed_report, args.output):
        if path.exists():
            raise SystemExit(f"refusing to overwrite output: {path}")

    root = Path(__file__).resolve().parents[2]
    contract = load(args.contract)
    if (
        contract.get("schema_version") != 1
        or contract.get("experiment") != EXPERIMENT
        or contract.get("status") != "frozen-public-development-screen"
        or contract.get("implementation", {}).get("sha256")
        != sha256(Path(__file__).resolve())
    ):
        raise SystemExit("V19 public-screen contract differs")

    candidate_path = validate_record(contract["inputs"]["candidate_report"], root)
    bundled_path = validate_record(contract["inputs"]["bundled_report"], root)
    suite_path = validate_record(contract["inputs"]["suite"], root)
    runtime_path = validate_record(contract["runtime"]["implementation"], root)
    bundle_manifest_path = validate_record(
        contract["bundle"]["developer_pack_manifest"],
        root,
    )
    candidate = load(candidate_path)
    bundled = load(bundled_path)
    if (
        candidate.get("caseSuiteSHA256") != sha256(suite_path)
        or bundled.get("caseSuiteSHA256") != sha256(suite_path)
        or candidate.get("modelRevision")
        != contract["inputs"]["candidate_model_revision"]
        or bundled.get("modelRevision")
        != contract["inputs"]["bundled_model_revision"]
    ):
        raise SystemExit("V19 report lineage differs")

    candidate_rows = result_map(candidate)
    bundled_rows = result_map(bundled)
    if set(candidate_rows) != set(bundled_rows):
        raise SystemExit("candidate and bundled reports cover different cases")

    composed_rows: list[dict[str, Any]] = []
    fallback_segments: list[str] = []
    invalid_fallback_segments: list[str] = []
    candidate_segments = 0
    for case_id in candidate_rows:
        candidate_row = candidate_rows[case_id]
        bundled_row = bundled_rows[case_id]
        matching_case(candidate_row, bundled_row)
        segment_ids = candidate_row["segmentBenchmarkIDs"]
        chosen_segments = []
        chosen_token_ids = []
        for index, (candidate_text, bundled_text) in enumerate(
            zip(
                candidate_row["segmentHypotheses"],
                bundled_row["segmentHypotheses"],
            )
        ):
            candidate_ids = candidate_row["segmentOutputTokenIDs"][index]
            bundled_ids = bundled_row["segmentOutputTokenIDs"][index]
            candidate_failed = (
                not str(candidate_text).strip()
                or repeated_token_loop(candidate_ids)
            )
            if not candidate_failed:
                chosen_segments.append(candidate_text)
                chosen_token_ids.append(candidate_ids)
                candidate_segments += 1
                continue
            fallback_segments.append(str(segment_ids[index]))
            if (
                not str(bundled_text).strip()
                or repeated_token_loop(bundled_ids)
            ):
                invalid_fallback_segments.append(str(segment_ids[index]))
            chosen_segments.append(bundled_text)
            chosen_token_ids.append(bundled_ids)
        composed_rows.append(
            {
                **candidate_row,
                "hypothesis": "\n".join(chosen_segments),
                "segmentHypotheses": chosen_segments,
                "segmentOutputTokenIDs": chosen_token_ids,
            }
        )

    direction_metrics: dict[str, Any] = {}
    candidate_metrics: dict[str, Any] = {}
    bundled_metrics: dict[str, Any] = {}
    for direction, (source_language, _) in DIRECTIONS.items():
        composed_direction = [
            row
            for row in composed_rows
            if row["sourceLanguage"] == source_language
        ]
        candidate_direction = [
            row
            for row in candidate_rows.values()
            if row["sourceLanguage"] == source_language
        ]
        bundled_direction = [
            row
            for row in bundled_rows.values()
            if row["sourceLanguage"] == source_language
        ]
        direction_metrics[direction] = metrics(composed_direction)
        candidate_metrics[direction] = metrics(candidate_direction)
        bundled_metrics[direction] = metrics(bundled_direction)
        direction_metrics[direction]["deltaFromCandidate"] = {
            name: direction_metrics[direction][name]
            - candidate_metrics[direction][name]
            for name in ("chrFPlusPlus", "BLEU")
        }
        direction_metrics[direction]["deltaFromBundled"] = {
            name: direction_metrics[direction][name]
            - bundled_metrics[direction][name]
            for name in ("chrFPlusPlus", "BLEU")
        }

    total_segments = sum(
        len(row["segmentHypotheses"]) for row in composed_rows
    )
    bundle_directory = bundle_manifest_path.parent
    bundle_bytes = sum(
        path.stat().st_size for path in bundle_directory.rglob("*") if path.is_file()
    )
    requirements = contract["requirements"]
    gates = [
        gate(
            f"{direction}-{metric_name}-noninferior-to-candidate",
            direction_metrics[direction][metric_name],
            minimum=candidate_metrics[direction][metric_name],
        )
        for direction in DIRECTIONS
        for metric_name in ("chrFPlusPlus", "BLEU")
    ]
    gates.extend(
        [
            gate(
                "fallback-segments",
                len(fallback_segments),
                minimum=int(requirements["minimum_fallback_segments"]),
            ),
            gate(
                "invalid-fallback-segments",
                len(invalid_fallback_segments),
                maximum=0,
            ),
            gate(
                "bundle-bytes",
                bundle_bytes,
                maximum=int(requirements["maximum_bundle_bytes"]),
            ),
        ]
    )
    passed = all(item["passed"] for item in gates)
    composed = {
        "schemaVersion": 1,
        "engine": "mlx:guarded-expert-cascade-v19:stored-output-composition",
        "modelRevision": (
            "guarded-cascade-manifest-sha256:"
            + sha256(bundle_manifest_path)
        ),
        "caseSuiteSHA256": sha256(suite_path),
        "runtimeImplementationSHA256": sha256(runtime_path),
        "latencyEvidenceStatus": "not-measured-for-cascade",
        "results": composed_rows,
    }
    args.composed_report.parent.mkdir(parents=True, exist_ok=True)
    args.composed_report.write_text(
        json.dumps(composed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": (
            "public-screen-passed"
            if passed
            else "public-screen-rejected"
        ),
        "contract": record(args.contract, root),
        "implementation": record(Path(__file__).resolve(), root),
        "runtime_implementation": record(runtime_path, root),
        "inputs": {
            "candidate_report": record(candidate_path, root),
            "bundled_report": record(bundled_path, root),
            "suite": record(suite_path, root),
        },
        "bundle": {
            "manifest": record(bundle_manifest_path, root),
            "bytes": bundle_bytes,
            "format": load(bundle_manifest_path).get("format"),
            "distribution_status": load(bundle_manifest_path).get(
                "distributionStatus"
            ),
        },
        "cases": len(composed_rows),
        "segments": total_segments,
        "candidate_segments": candidate_segments,
        "fallback_segments": fallback_segments,
        "fallback_segment_count": len(fallback_segments),
        "invalid_fallback_segments": invalid_fallback_segments,
        "metrics": direction_metrics,
        "candidate_metrics": candidate_metrics,
        "bundled_metrics": bundled_metrics,
        "gates": gates,
        "public_screen_passed": passed,
        "composed_report": record(args.composed_report, root),
        "comet_required": True,
        "long_document_evaluation_required": True,
        "runtime_latency_memory_required": True,
        "protected_evaluation_authorized": False,
        "app_resource_change_authorized": False,
        "bundle_replacement_authorized": False,
        "public_upload_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "status": result["status"],
                "bundle_bytes": bundle_bytes,
                "fallback_segments": len(fallback_segments),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
