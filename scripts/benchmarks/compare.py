#!/usr/bin/env python3
"""Compare matching benchmark runs with paired group-level latency deltas."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from metrics import quantile
from report import GROUP_FIELDS, metric_values
from integrity import verify_run
from quality import paired_quality


def validate_cohort(baseline: dict, candidate: dict, allow_model_change: bool = False) -> None:
    if baseline.get("unicodeDataVersion") != candidate.get("unicodeDataVersion"):
        raise ValueError("Incomparable Unicode normalization versions")
    for key in ("protocolSha256", "suiteSha256", "evaluatorFiles"):
        if baseline[key] != candidate[key]: raise ValueError("Incomparable runs: " + key + " differs")
    for key in ("hardware", "physicalMemoryBytes", "operatingSystem", "architecture", "buildConfiguration", "processorCount", "lowPowerMode"):
        if baseline["appEnvironment"].get(key) != candidate["appEnvironment"].get(key):
            raise ValueError("Incomparable runtime environment: " + key)
    if not baseline["performanceEligible"] or not candidate["performanceEligible"]:
        raise ValueError("Diagnostic/debug or stale-source runs cannot establish performance changes")
    plan_key = lambda item: (item["caseID"], item["caseSha256"], item["lane"], item["engine"])
    if {plan_key(item) for item in baseline["plan"]} != {plan_key(item) for item in candidate["plan"]}:
        raise ValueError("Incomparable runs: selected cases/lanes/engines differ")
    if not allow_model_change and baseline["models"] != candidate["models"]:
        raise ValueError("Model artifacts differ; explicitly use --allow-model-change for a model comparison")


def compare_rows(baseline: list[dict], candidate: list[dict], protocol: dict) -> list[dict]:
    def grouped(rows):
        result = defaultdict(list)
        for row in rows:
            if row["phase"] == "warm": result[tuple(row[field] for field in GROUP_FIELDS) + (row["groupID"],)].append(row)
        return result
    left, right = grouped(baseline), grouped(candidate)
    if left.keys() != right.keys(): raise ValueError("Missing paired units")
    groups = defaultdict(list)
    for key in left: groups[key[:-1]].append(key)
    output = []
    for group, keys in sorted(groups.items()):
        paired = {"bothSuccess": 0, "baselineOnly": 0, "candidateOnly": 0, "neitherSuccess": 0}
        deltas = []
        primary = protocol["primaryLatency"][group[0]]
        for key in keys:
            a, b = left[key], right[key]
            identities = lambda rows: {(row["caseID"], row["attemptIndex"]) for row in rows}
            if identities(a) != identities(b): raise ValueError("Missing or mismatched repetitions")
            good_a = all(row["outcome"] == "success" for row in a)
            good_b = all(row["outcome"] == "success" for row in b)
            paired["bothSuccess" if good_a and good_b else "baselineOnly" if good_a else "candidateOnly" if good_b else "neitherSuccess"] += 1
            av = [value for row in a for value in metric_values(row, primary)]
            bv = [value for row in b for value in metric_values(row, primary)]
            if good_a and good_b and av and bv:
                deltas.append({"groupID": key[-1], "baselineMedian": statistics.median(av),
                               "candidateMedian": statistics.median(bv), "delta": statistics.median(bv) - statistics.median(av)})
        values = [row["delta"] for row in deltas]
        interval = None
        if len(values) >= 2:
            rng = random.Random(protocol["seed"])
            estimates = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(protocol["bootstrapReplicates"])]
            alpha = (1 - protocol["confidenceLevel"]) / 2
            interval = [quantile(estimates, alpha), quantile(estimates, 1 - alpha)]
        all_a = [row for key in keys for row in left[key]]
        all_b = [row for key in keys for row in right[key]]
        drop_observations = [[row["metrics"].get("droppedAudioSeconds") for row in rows
                              if row["metrics"].get("droppedAudioSeconds") is not None] for rows in (all_a, all_b)]
        baseline_drops, candidate_drops = [sum(values) if values else None for values in drop_observations]
        warnings = [sum(bool(row.get("speechWarnings")) for row in rows) for rows in (all_a, all_b)]
        more_loss = baseline_drops is not None and candidate_drops is not None and candidate_drops > baseline_drops
        output.append({**dict(zip(GROUP_FIELDS, group)), "primaryMetric": primary, "pairedOutcomeUnits": paired,
                       "latencyPairedUnits": len(values), "meanUnitMedianDelta": statistics.mean(values) if values else None,
                       "clusterBootstrapCI95": interval, "pairedDeltas": deltas,
                       "quality": paired_quality(all_a, all_b),
                       "baselineDroppedAudioSeconds": baseline_drops, "candidateDroppedAudioSeconds": candidate_drops,
                       "queueObservationCounts": [len(values) for values in drop_observations],
                       "warningAttemptCounts": warnings,
                       "reliabilityRegression": paired["baselineOnly"] > 0 or more_loss or warnings[1] > warnings[0]
                           or len(drop_observations[1]) < len(drop_observations[0])})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-model-change", action="store_true")
    parser.add_argument("--check", action="store_true", help="Exit nonzero for lost success units or increased audio loss")
    args = parser.parse_args()
    for root in (args.baseline, args.candidate): verify_run(root)
    metadata = [json.loads((root / "metadata.json").read_text()) for root in (args.baseline, args.candidate)]
    validate_cohort(*metadata, args.allow_model_change)
    rows = [[json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()] for root in (args.baseline, args.candidate)]
    groups = compare_rows(*rows, metadata[0]["protocol"])
    result = {"schemaVersion": 1, "groups": groups, "claimEligible": False,
              "interpretation": "Positive latency deltas are slower. Confidence intervals resample independent groups, retaining all repeats. No multiplicity-adjusted model ranking or automatic latency/quality gate is claimed.",
              "quality": "Use the coverage-qualified WER/CER and chrF2 summaries alongside paired completion counts; latency alone cannot establish improvement.",
              "thermalStates": [value["appEnvironment"].get("thermalState") for value in metadata]}
    with args.output.open("x") as output: output.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.check and any(group["reliabilityRegression"] for group in groups): raise SystemExit(1)


if __name__ == "__main__": main()
