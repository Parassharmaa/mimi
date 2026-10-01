#!/usr/bin/env python3
"""Summarize paired latency runs, resampling whole fixtures with their repeats."""
import argparse
import json
import math
import random
import statistics


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("instant")
    args = parser.parse_args()
    baseline = json.load(open(args.baseline))
    instant = json.load(open(args.instant))
    assert instant["status"] == "passed", "candidate failed the output/outcome gate"
    assert baseline["modelManifestSHA256"] == instant["modelManifestSHA256"]
    print("Publication-ready text, milliseconds. ASR and paint excluded.")
    print("8 owned fixtures, 3 dependent repetitions. Tail percentiles are exploratory.")
    for label, attempts, policy in [
        ("Initial final-only replay", baseline["attempts"], "final-only"),
        ("Initial caption replay", baseline["attempts"], "caption-debounce-140ms"),
        ("Paired caption replay", instant["pairedAttempts"], "caption-debounce-140ms"),
        ("Immediate dispatch", instant["pairedAttempts"], "instant"),
    ]:
        rows = [row for row in attempts if row["policy"] == policy]
        completed = [row["eventToPublicationMS"] for row in rows if not row.get("failure")]
        print(f"{label}: {len(completed)}/{len(rows)} successful, p50={statistics.median(completed):.1f}, p95={percentile(completed, .95):.1f}")
    pairs = {}
    for row in instant["pairedAttempts"]:
        pairs.setdefault((row["fixtureID"], row["repetition"]), {})[row["policy"]] = row
    differences = {}
    for (fixture, _), policies in pairs.items():
        if not any(row.get("failure") for row in policies.values()):
            differences.setdefault(fixture, []).append(policies["caption-debounce-140ms"]["eventToPublicationMS"] - policies["instant"]["eventToPublicationMS"])
    task_means = [statistics.mean(repeats) for repeats in differences.values()]
    rng = random.Random(42)
    bootstrap = [statistics.mean(rng.choices(task_means, k=len(task_means))) for _ in range(10_000)]
    print(f"Paired mean reduction={statistics.mean(task_means):.1f}ms, exploratory fixture bootstrap 95% interval [{percentile(bootstrap, .025):.1f}, {percentile(bootstrap, .975):.1f}]ms")
    replays = instant["replays"]
    first = [row["firstTranslationMS"] for row in replays if row["firstTranslationMS"] is not None]
    print(f"Production stream first publication: {len(first)}/{len(replays)} replays, p50={statistics.median(first):.1f}, p95={percentile(first, .95):.1f}ms")
    print(f"Final parity: {sum(row['finalMatchesBaseline'] for row in replays)}/{len(replays)} successful matching outputs, {sum(row['finalFailureMatchesBaseline'] for row in replays)} matching baseline failures")
    print(f"Production inference: {sum(row['inferenceCount'] for row in replays)} calls, {sum(len(row['failures']) for row in replays)} failures, no retries")


if __name__ == "__main__":
    main()
