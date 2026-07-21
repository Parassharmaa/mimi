#!/usr/bin/env python3
"""Score one frozen Mimi engine report with an exactly pinned COMET model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL = "Unbabel/wmt22-comet-da"
DEFAULT_REVISION = "371e9839ca4e213dde891b066cf3080f75ec7e72"
DEFAULT_PACKAGE_VERSION = "2.2.7"
DEFAULT_SETUPTOOLS_VERSION = "80.9.0"
MODEL_LICENSE = "Apache-2.0"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float:
    if not values:
        raise SystemExit("cannot average an empty learned-metric slice")
    return sum(values) / len(values)


def validate_inputs(suite_rows: list[dict], engine_report: dict) -> list[dict]:
    suite = {str(row.get("id", "")): row for row in suite_rows}
    results = {str(row.get("caseID", "")): row for row in engine_report.get("results", [])}
    if not suite or len(suite) != len(suite_rows) or set(suite) != set(results):
        raise SystemExit("suite and engine report must have identical non-empty case IDs")
    ordered: list[dict] = []
    for case_id in sorted(suite):
        case, result = suite[case_id], results[case_id]
        for field in ("sourceLanguage", "targetLanguage", "domain", "source", "references"):
            if result.get(field) != case.get(field):
                raise SystemExit(f"engine result disagrees with suite {field}: {case_id}")
        hypothesis = str(result.get("hypothesis", "")).strip()
        references = [str(value).strip() for value in case.get("references", [])]
        if not hypothesis or not references or not all(references):
            raise SystemExit(f"case lacks a hypothesis or reference: {case_id}")
        ordered.append({**case, "hypothesis": hypothesis})
    return ordered


def build_report(
    suite_path: Path,
    engine_report_path: Path,
    rows: list[dict],
    scores: list[float],
    *,
    model_repository: str,
    model_revision: str,
    package_version: str,
    setuptools_version: str,
    torch_version: str,
) -> dict:
    expected_scores = sum(len(row["references"]) for row in rows)
    if len(scores) != expected_scores or not all(isinstance(value, float) for value in scores):
        raise SystemExit("COMET did not return exactly one float score per case/reference pair")
    offset = 0
    result_rows: list[dict] = []
    by_direction: dict[str, list[float]] = defaultdict(list)
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        reference_count = len(row["references"])
        reference_scores = scores[offset:offset + reference_count]
        offset += reference_count
        score = mean(reference_scores)
        direction = f"{row['sourceLanguage']}>{row['targetLanguage']}"
        result_rows.append(
            {
                "caseID": row["id"],
                "sourceLanguage": row["sourceLanguage"],
                "targetLanguage": row["targetLanguage"],
                "domain": row["domain"],
                "score": score,
                "referenceScores": reference_scores,
            }
        )
        by_direction[direction].append(score)
        by_domain[f"{direction}/{row['domain']}"].append(score)
    signature_value = {
        "metric": "COMET-22",
        "modelRepository": model_repository,
        "modelRevision": model_revision,
        "modelLicense": MODEL_LICENSE,
        "package": "unbabel-comet",
        "packageVersion": package_version,
        "setuptoolsVersion": setuptools_version,
        "precision": "float32",
        "multipleReferenceAggregation": "mean",
    }
    signature = hashlib.sha256(
        json.dumps(signature_value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **signature_value,
        "signatureSHA256": signature,
        "engine": json.loads(engine_report_path.read_text(encoding="utf-8"))["engine"],
        "suiteSHA256": sha256(suite_path),
        "engineReportSHA256": sha256(engine_report_path),
        "hardware": platform.machine(),
        "torchVersion": torch_version,
        "directions": {
            key: {"cases": len(values), "meanScore": mean(values)}
            for key, values in sorted(by_direction.items())
        },
        "domains": {
            key: {"cases": len(values), "meanScore": mean(values)}
            for key, values in sorted(by_domain.items())
        },
        "results": result_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("engine_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-repository", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    parser.add_argument("--package-version", default=DEFAULT_PACKAGE_VERSION)
    parser.add_argument("--setuptools-version", default=DEFAULT_SETUPTOOLS_VERSION)
    parser.add_argument("--cache-directory", type=Path, default=Path("Research/translation/models/hf-cache"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.batch_size < 1:
        raise SystemExit("batch-size must be positive")
    if args.num_workers < 1:
        raise SystemExit("num-workers must be positive for COMET 2.2.7")
    installed = importlib.metadata.version("unbabel-comet")
    if installed != args.package_version:
        raise SystemExit(
            f"unbabel-comet version mismatch: installed={installed} required={args.package_version}"
        )
    installed_setuptools = importlib.metadata.version("setuptools")
    if installed_setuptools != args.setuptools_version:
        raise SystemExit(
            "setuptools version mismatch: "
            f"installed={installed_setuptools} required={args.setuptools_version}"
        )

    import torch
    from comet import load_from_checkpoint
    from huggingface_hub import snapshot_download

    suite_rows = load_jsonl(args.suite)
    engine_report = json.loads(args.engine_report.read_text(encoding="utf-8"))
    rows = validate_inputs(suite_rows, engine_report)
    snapshot = Path(
        snapshot_download(
            repo_id=args.model_repository,
            revision=args.model_revision,
            cache_dir=args.cache_directory,
        )
    )
    checkpoints = sorted(snapshot.rglob("*.ckpt"))
    if len(checkpoints) != 1:
        raise SystemExit(f"expected one COMET checkpoint at pinned revision; found {len(checkpoints)}")
    model = load_from_checkpoint(str(checkpoints[0]))
    model.float()
    inputs = [
        {"src": row["source"], "mt": row["hypothesis"], "ref": reference}
        for row in rows
        for reference in row["references"]
    ]
    prediction = model.predict(
        inputs,
        batch_size=args.batch_size,
        gpus=0,
        num_workers=args.num_workers,
    )
    scores = [float(value) for value in prediction.scores]
    report = build_report(
        args.suite,
        args.engine_report,
        rows,
        scores,
        model_repository=args.model_repository,
        model_revision=args.model_revision,
        package_version=installed,
        setuptools_version=installed_setuptools,
        torch_version=torch.__version__,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "signatureSHA256": report["signatureSHA256"]}))


if __name__ == "__main__":
    main()
