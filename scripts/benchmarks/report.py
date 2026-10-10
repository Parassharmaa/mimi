#!/usr/bin/env python3
"""Summarize frozen runs with explicit coverage, clustered uncertainty, and quality."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from metrics import LATENCY_METRICS, quantile
from prepare_suite import sha256
from integrity import verify_run
from quality import chrf_quality

GROUP_FIELDS = ("lane", "engine", "language", "condition", "phase")


def metric_values(row: dict, metric: str) -> list[float]:
    if metric == "translationQueueMs":
        return [r["queueMs"] for r in row["translationRequests"] if r["queueMs"] is not None]
    if metric == "translationServiceMs":
        return [r["serviceMs"] for r in row["translationRequests"] if r["outcome"] == "succeeded"]
    if metric == "sourceToPublicationMs":
        return [r["sourceToPublicationMs"] for r in row["publications"] if r["sourceToPublicationMs"] is not None]
    if metric == "deadlineCappedAttemptWallMs":
        value = row.get(metric)
    else: value = row["metrics"].get(metric)
    return [] if value is None else [value]


def statistics(rows: list[dict], metric: str, protocol: dict) -> dict:
    clusters = defaultdict(list)
    observed_attempts = 0
    for row in rows:
        values = metric_values(row, metric)
        if values:
            observed_attempts += 1
            clusters[row["groupID"]].extend(values)
    values = [value for group in clusters.values() for value in group]
    tail_ok = len(clusters) >= protocol["tailMinimumIndependentUnits"]
    interval = None
    if len(clusters) >= 2:
        rng = random.Random(protocol["seed"])
        groups = list(clusters.values())
        estimates = [quantile([value for group in rng.choices(groups, k=len(groups)) for value in group], 0.5)
                     for _ in range(protocol["bootstrapReplicates"])]
        alpha = (1 - protocol["confidenceLevel"]) / 2
        interval = [quantile(estimates, alpha), quantile(estimates, 1 - alpha)]
    return {"observationCount": len(values), "observedAttemptCount": observed_attempts,
            "independentUnitCount": len(clusters), "p50": quantile(values, 0.5),
            "p90": quantile(values, 0.9) if tail_ok else None,
            "p95": quantile(values, 0.95) if tail_ok else None,
            "max": max(values) if values else None, "p50ClusterBootstrapCI95": interval,
            "tailSuppressed": not tail_ok, "tailExploratory": len(clusters) < 100,
            "quantileMethod": protocol["quantileMethod"]}


def summarize(run: Path) -> dict:
    verify_run(run)
    metadata = json.loads((run / "metadata.json").read_text())
    rows = [json.loads(line) for line in (run / "attempts.jsonl").read_text().splitlines()]
    expected = {(item["caseID"], item["lane"], item["engine"], index)
                for item in metadata["plan"]
                for index in range(1 + metadata["protocol"]["warmups"] + metadata["protocol"]["repetitions"])}
    actual = {(row["caseID"], row["lane"], row["engine"], row["attemptIndex"]) for row in rows}
    if actual != expected or len(actual) != len(rows):
        raise ValueError("Run is incomplete or contains duplicate/unplanned attempts; do not summarize partial coverage as a complete run")
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[field] for field in GROUP_FIELDS)].append(row)
    summaries = []
    for key, attempts in sorted(groups.items()):
        counts = Counter(row["outcome"] for row in attempts)
        scored = [row["speechQuality"] for row in attempts if row["speechQuality"] is not None]
        reference_units = sum(row["referenceUnits"] for row in scored)
        speech_quality = {name: sum(row[name] for row in scored) for name in ("edits", "substitutions", "deletions", "insertions", "referenceUnits")}
        speech_quality.update(scoredAttempts=len(scored), corpusErrorRate=speech_quality["edits"] / reference_units if reference_units else None,
                              metric="CER" if key[2] == "ja-JP" else "WER")
        translation_rows = [row for row in attempts if row["lane"] != "speech" and row["references"] and
                            row["result"]["translationStatus"] in {"success", "empty_output"}]
        chrf = chrf_quality(translation_rows)
        observed = sum(row["observedAttempt"] for row in attempts)
        summaries.append({**dict(zip(GROUP_FIELDS, key)), "plannedAttempts": len(attempts), "observedAttempts": observed,
                          "independentUnits": len({row["groupID"] for row in attempts}), "outcomes": dict(counts),
                          "successRate": counts["success"] / len(attempts),
                          "unavailableAttempts": sum(row.get("unavailableReason") is not None for row in attempts),
                          "warningAttempts": sum(bool(row["speechWarnings"]) for row in attempts),
                          "queueObservedAttempts": sum(row["metrics"].get("droppedAudioSeconds") is not None for row in attempts),
                          "lossyAttempts": sum((row["metrics"].get("droppedAudioSeconds") or 0) > 0 for row in attempts),
                          "speechQuality": speech_quality, "translationQuality": chrf,
                          "translationFailureReasons": dict(Counter(request["error"] for row in attempts for request in row["translationRequests"] if request.get("error"))),
                          "routes": dict(sum((Counter(row["routeCounts"]) for row in attempts), Counter())),
                          "metrics": {metric: statistics(attempts, metric, metadata["protocol"]) for metric in
                                      (*LATENCY_METRICS, "translationQueueMs", "translationServiceMs", "sourceToPublicationMs", "deadlineCappedAttemptWallMs")}})
    return {"schemaVersion": 1, "metadataSha256": sha256(run / "metadata.json"), "attemptsSha256": sha256(run / "attempts.jsonl"),
            "performanceEligible": metadata["performanceEligible"], "claimEligible": False,
            "plannedAttempts": len(rows), "observedAttempts": sum(row["observedAttempt"] for row in rows),
            "outcomes": dict(Counter(row["outcome"] for row in rows)), "groups": summaries,
            "metricDefinitions": "docs/BENCHMARKS.md", "bootstrapUnit": "groupID, retaining every repeat and segment within the group",
            "limitations": ["Synthetic speech does not establish human-speech accuracy.",
                            "chrF2 is lexical overlap; these references do not authorize semantic-quality claims or promotion.",
                            "Cold means fresh process; filesystem and driver caches are uncontrolled.",
                            "Latency ends at model publication and excludes microphone capture and compositor paint.",
                            "Missing observations are null. Tails require at least 20 independent groups and remain exploratory below 100.",
                            "Process peak RSS is cumulative within a process, not isolated peak memory for each warm attempt."]}


def markdown(report: dict) -> str:
    lines = ["# Mimi benchmark baseline", "", f"Observed {report['observedAttempts']}/{report['plannedAttempts']} planned attempts, including cold runs and warmups.",
             f"Outcomes: `{json.dumps(report['outcomes'], sort_keys=True)}`.", "",
             "Warm latency uses observed events. Failed and missing results remain in the coverage denominator. A dash is unmeasured or insufficient independent units.", ""]
    for lane, title, primary in [("speech", "Speech recognition", "audioStartToFirstTextMs"),
                                 ("translation", "Reference text to translation", "referenceSubmitToFinalTranslationMs"),
                                 ("pipeline", "Speech to translation", "audioStartToFirstTranslationMs")]:
        lines += ["## " + title, "", "| Engine | Language | Condition | Units | Success / planned | Primary p50 ms | p95 ms | WER / CER % | chrF2 |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        def number(value): return "—" if value is None else f"{value:.2f}"
        for group in report["groups"]:
            if group["lane"] != lane or group["phase"] != "warm": continue
            timing = group["metrics"][primary]
            rate = group["speechQuality"]["corpusErrorRate"]
            quality = group["translationQuality"]
            lines.append(f"| {group['engine']} | {group['language']} | {group['condition']} | {group['independentUnits']} | {group['outcomes'].get('success', 0)}/{group['plannedAttempts']} | {number(timing['p50'])} | {number(timing['p95'])} | {number(rate * 100 if rate is not None else None)} | {number(quality['score'] if quality else None)} |")
        lines += [""]
    lines += ["## Interpretation", ""] + ["- " + item for item in report["limitations"]]
    lines += ["", "Detailed preparation, queue/service/publication latency, RTF, memory, churn, route counts, confidence intervals, failures, and denominators are in `summary.json`. Raw attempts are in `attempts.jsonl`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = summarize(args.run)
    (args.run / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.run / "summary.md").write_text(markdown(report))
    print(args.run / "summary.md")


if __name__ == "__main__": main()
