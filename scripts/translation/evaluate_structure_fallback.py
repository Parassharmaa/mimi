#!/usr/bin/env python3
"""Evaluate a frozen two-checkpoint structure fallback on an independent suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import sacrebleu

from audit_translation_structures import critical_tokens, tokens


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def authenticate(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise SystemExit(f"{label} hash mismatch: {path}")


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def suite_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (row.get("sourceLanguage"), row.get("targetLanguage")) != (
            "en-US",
            "ja-JP",
        ):
            continue
        case_id = str(row.get("id", ""))
        if not case_id or case_id in rows or row.get("claimEligible") is not False:
            raise SystemExit(f"invalid independent-suite row: {case_id}")
        rows[case_id] = row
    if not rows:
        raise SystemExit("independent suite has no EN-to-JA cases")
    return rows


def model_record(path: Path, expected: dict, label: str) -> dict:
    manifest_path = path / "manifest.json"
    authenticate(manifest_path, expected["manifestSha256"], f"{label} manifest")
    manifest = load(manifest_path)
    weights = path / "model.safetensors"
    record = manifest.get("files", {}).get("model.safetensors")
    if (
        manifest.get("source_weights_sha256") != expected["sourceWeightsSha256"]
        or not isinstance(record, dict)
        or record.get("sha256") != expected["quantizedWeightsSha256"]
        or record.get("bytes") != weights.stat().st_size
        or sha256(weights) != expected["quantizedWeightsSha256"]
    ):
        raise SystemExit(f"{label} model integrity differs")
    return {
        "path": str(path),
        "manifestSha256": sha256(manifest_path),
        "sourceWeightsSha256": manifest["source_weights_sha256"],
        "quantizedWeightsSha256": record["sha256"],
    }


def report_rows(
    path: Path,
    suite: dict[str, dict],
    expected_model: dict,
    implementation: dict,
    label: str,
) -> tuple[dict, dict[str, dict]]:
    report = load(path)
    runtime = report.get("runtimeImplementation", {})
    if (
        runtime.get("benchmarkScriptSha256")
        != implementation["benchmarkScriptSha256"]
        or runtime.get("marianRuntimeSha256")
        != implementation["marianRuntimeSha256"]
        or runtime.get("packages", {}).get("mlx") != implementation["mlxVersion"]
    ):
        raise SystemExit(f"{label} report runtime differs")
    declared = report.get("declaredModels", {}).get("en-ja")
    if not isinstance(declared, dict) or any(
        declared.get(key) != expected_model[key]
        for key in (
            "manifestSha256",
            "sourceWeightsSha256",
            "quantizedWeightsSha256",
        )
    ):
        raise SystemExit(f"{label} report model binding differs")
    rows = {}
    for row in report.get("results", []):
        case_id = str(row.get("caseID", ""))
        if not case_id or case_id in rows:
            raise SystemExit(f"{label} report has duplicate case: {case_id}")
        rows[case_id] = row
    if set(rows) != set(suite):
        raise SystemExit(f"{label} report coverage differs")
    for case_id, expected in suite.items():
        for field in (
            "sourceLanguage",
            "targetLanguage",
            "domain",
            "source",
            "references",
            "claimEligible",
        ):
            if rows[case_id].get(field) != expected.get(field):
                raise SystemExit(f"{label} report disagrees on {field}: {case_id}")
    return report, rows


def flags(row: dict) -> tuple[bool, bool]:
    source = str(row["source"])
    hypothesis = str(row["hypothesis"])
    critical = critical_tokens(source) != critical_tokens(hypothesis)
    negative = tokens(source)["negative"] != tokens(hypothesis)["negative"]
    return critical, negative


def preference(row: dict) -> tuple[int, int, int]:
    critical, negative = flags(row)
    return int(critical) + int(negative), int(critical), int(negative)


def sentence_chrf(row: dict) -> float:
    return float(
        sacrebleu.sentence_chrf(
            row["hypothesis"], row["references"], word_order=2
        ).score
    )


def bootstrap(values: list[float], samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    return {
        "mean": sum(values) / len(values),
        "lower": means[math.floor(samples * 0.025)],
        "upper": means[min(samples - 1, math.ceil(samples * 0.975) - 1)],
        "samples": samples,
        "seed": seed,
        "confidence": 0.95,
    }


def corpus_chrf(rows: list[dict]) -> float:
    return float(
        sacrebleu.corpus_chrf(
            [row["hypothesis"] for row in rows],
            [[row["references"][0] for row in rows]],
            word_order=2,
        ).score
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("primary_report", type=Path)
    parser.add_argument("alternate_report", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    contract = load(args.contract)
    if (
        contract.get("selectionUsesReferences") is not False
        or contract.get("doesNotAuthorizeModelPromotion") is not True
        or contract.get("doesNotAuthorizeAppIntegration") is not True
    ):
        raise SystemExit("fallback contract lacks fail-closed policy")
    implementation = contract["implementation"]
    current_script = Path(__file__).resolve()
    authenticate(current_script, implementation["evaluatorScriptSha256"], "evaluator")
    script_directory = current_script.parent
    authenticate(
        script_directory / "run_mlx_marian_benchmark.py",
        implementation["benchmarkScriptSha256"],
        "benchmark",
    )
    authenticate(
        script_directory / "marian_mlx.py",
        implementation["marianRuntimeSha256"],
        "Marian runtime",
    )
    authenticate(
        script_directory / "audit_translation_structures.py",
        implementation["structureAuditScriptSha256"],
        "structure audit",
    )

    suite_path = Path(contract["suite"]["path"])
    suite_manifest_path = Path(contract["suite"]["manifestPath"])
    authenticate(suite_path, contract["suite"]["sha256"], "suite")
    authenticate(
        suite_manifest_path, contract["suite"]["manifestSha256"], "suite manifest"
    )
    suite = suite_rows(suite_path)
    if len(suite) != contract["suite"]["casesPerDirection"]:
        raise SystemExit("independent suite case count differs")

    model_records = {
        role: model_record(Path(spec["path"]), spec, role)
        for role, spec in contract["models"].items()
    }
    _, primary = report_rows(
        args.primary_report, suite, model_records["primary"], implementation, "primary"
    )
    _, alternate = report_rows(
        args.alternate_report,
        suite,
        model_records["alternate"],
        implementation,
        "alternate",
    )
    _, baseline = report_rows(
        args.baseline_report,
        suite,
        model_records["baseline"],
        implementation,
        "baseline",
    )

    selected = []
    fallback_ids = []
    for case_id in sorted(suite):
        first = primary[case_id]
        second = alternate[case_id]
        use_alternate = preference(second) < preference(first)
        chosen = dict(second if use_alternate else first)
        chosen["selectedEngine"] = (
            "alternate-structure-fallback" if use_alternate else "primary"
        )
        chosen["structureFallbackUsed"] = use_alternate
        if use_alternate:
            fallback_ids.append(case_id)
            chosen["latencySeconds"] = float(first["latencySeconds"]) + float(
                second["latencySeconds"]
            )
            chosen["warmLatencySeconds"] = [
                left + right
                for left, right in zip(
                    first.get("warmLatencySeconds") or [first["latencySeconds"]],
                    second.get("warmLatencySeconds") or [second["latencySeconds"]],
                    strict=True,
                )
            ]
        selected.append(chosen)

    candidate_critical = sum(flags(row)[0] for row in selected)
    candidate_negation = sum(flags(row)[1] for row in selected)
    baseline_rows = [baseline[case_id] for case_id in sorted(suite)]
    baseline_critical = sum(flags(row)[0] for row in baseline_rows)
    baseline_negation = sum(flags(row)[1] for row in baseline_rows)
    deltas = [
        sentence_chrf(row) - sentence_chrf(baseline[row["caseID"]])
        for row in selected
    ]
    bootstrap_spec = contract["gates"]["pairedSentenceChrFPlusPlus"]
    interval = bootstrap(
        deltas, int(bootstrap_spec["samples"]), int(bootstrap_spec["seed"])
    )
    warm_latencies = [
        float(value)
        for row in selected
        for value in (row.get("warmLatencySeconds") or [row["latencySeconds"]])
    ]
    pack_path = Path(contract["distribution"]["currentPackPath"])
    authenticate(
        pack_path / "manifest.json",
        contract["distribution"]["currentPackManifestSha256"],
        "current pack",
    )
    conservative_bytes = directory_bytes(pack_path) + directory_bytes(
        Path(contract["models"]["primary"]["path"])
    )
    gates = {
        "positivePairedQuality": interval["lower"]
        > float(bootstrap_spec["minimumLowerBound"]),
        "criticalTokenNonRegression": candidate_critical <= baseline_critical,
        "negationNonRegression": candidate_negation <= baseline_negation,
        "maximumFallbackRate": len(fallback_ids) / len(selected)
        <= float(contract["gates"]["maximumFallbackRate"]),
        "maximumWarmP95Seconds": percentile(warm_latencies, 0.95)
        <= float(contract["gates"]["maximumWarmP95Seconds"]),
        "maximumConservativeBundleBytes": conservative_bytes
        <= int(contract["distribution"]["maximumBytes"]),
    }
    payload = {
        "schemaVersion": 1,
        "status": (
            "independent-legal-safety-test-passed"
            if all(gates.values())
            else "structure-fallback-rejected"
        ),
        "contract": {"path": str(args.contract), "sha256": sha256(args.contract)},
        "implementation": {
            "path": str(current_script),
            "sha256": sha256(current_script),
        },
        "inputs": {
            "primaryReport": {
                "path": str(args.primary_report),
                "sha256": sha256(args.primary_report),
            },
            "alternateReport": {
                "path": str(args.alternate_report),
                "sha256": sha256(args.alternate_report),
            },
            "baselineReport": {
                "path": str(args.baseline_report),
                "sha256": sha256(args.baseline_report),
            },
        },
        "models": model_records,
        "policy": {
            "selection": "minimum (critical+negation, critical, negation); primary on tie",
            "selectionUsesReferences": False,
            "fallbackCases": len(fallback_ids),
            "fallbackRate": len(fallback_ids) / len(selected),
            "fallbackCaseIDs": fallback_ids,
        },
        "quality": {
            "candidateCorpusChrFPlusPlus": corpus_chrf(selected),
            "baselineCorpusChrFPlusPlus": corpus_chrf(baseline_rows),
            "pairedSentenceChrFPlusPlusDelta": interval,
        },
        "structure": {
            "candidate": {
                "exactCriticalTokenMismatches": candidate_critical,
                "negationMarkerMismatches": candidate_negation,
            },
            "baseline": {
                "exactCriticalTokenMismatches": baseline_critical,
                "negationMarkerMismatches": baseline_negation,
            },
        },
        "runtime": {"warmP95Seconds": percentile(warm_latencies, 0.95)},
        "distribution": {
            "conservativeBytes": conservative_bytes,
            "maximumBytes": contract["distribution"]["maximumBytes"],
            "physicalModelCount": 5,
        },
        "gates": gates,
        "claimEligible": False,
        "doesNotAuthorizeModelPromotion": True,
        "doesNotAuthorizeAppIntegration": True,
        "results": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "quality": payload["quality"],
                "structure": payload["structure"],
                "fallbackCases": len(fallback_ids),
                "warmP95Seconds": payload["runtime"]["warmP95Seconds"],
                "conservativeBytes": conservative_bytes,
                "gates": gates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
