#!/usr/bin/env python3
"""Apply an exact-source translation memory to an authenticated engine report."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import unicodedata
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("memory", type=Path)
    parser.add_argument("engine_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pack", type=Path)
    args = parser.parse_args()
    memory = json.loads(args.memory.read_text(encoding="utf-8"))
    report = json.loads(args.engine_report.read_text(encoding="utf-8"))
    if memory.get("schemaVersion") != 1 or not isinstance(memory.get("entries"), dict):
        raise SystemExit("unsupported translation-memory schema")
    if not isinstance(report.get("results"), list):
        raise SystemExit("engine report lacks results")
    directions = {"en-US": "en-ja", "ja-JP": "ja-en"}
    results = []
    hits = {"en-ja": 0, "ja-en": 0}
    lookup_latencies = []
    for source_row in report["results"]:
        row = dict(source_row)
        current_direction = directions[str(row["sourceLanguage"])]
        started = time.perf_counter_ns()
        target = memory["entries"][current_direction].get(normalize(str(row["source"])))
        latency = (time.perf_counter_ns() - started) / 1_000_000_000
        lookup_latencies.append(latency)
        if target is not None:
            row["selectedNeuralEngine"] = row.get("selectedEngine")
            row["selectedEngine"] = "exact-translation-memory"
            row["hypothesis"] = target
            row["latencySeconds"] = latency
            row["warmLatencySeconds"] = [latency]
            row["outputTokenIDs"] = None
            hits[current_direction] += 1
        results.append(row)
    memory_hash = sha256(args.memory)
    pack_manifest = None
    if args.pack is not None:
        pack_manifest_path = args.pack / "manifest.json"
        pack_manifest = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
        memory_metadata = pack_manifest.get("translationMemory", {})
        memory_record = pack_manifest.get("files", {}).get(
            memory_metadata.get("path", ""),
            {},
        )
        if memory_record.get("sha256") != memory_hash:
            raise SystemExit("pack does not authenticate the supplied translation memory")
        model_bytes = sum(
            path.stat().st_size for path in args.pack.rglob("*") if path.is_file()
        )
    else:
        model_bytes = (report.get("modelBytes") or 0) + args.memory.stat().st_size
    output = {
        **{key: value for key, value in report.items() if key != "results"},
        "engine": f"{report['engine']}+exact-translation-memory",
        "modelRevision": f"{report.get('modelRevision')}+memory-sha256:{memory_hash}",
        "modelBytes": model_bytes,
        "translationMemory": {
            "path": str(args.memory),
            "sha256": memory_hash,
            "entries": sum(len(values) for values in memory["entries"].values()),
            "hits": hits,
            "lookupP95Seconds": sorted(lookup_latencies)[int((len(lookup_latencies) - 1) * 0.95)],
            "claimEligible": False,
        },
        "doesNotAuthorizeAppIntegration": True,
        "results": results,
    }
    if pack_manifest is not None:
        output["translationMemory"]["pack"] = {
            "path": str(args.pack),
            "manifestSHA256": sha256(args.pack / "manifest.json"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["translationMemory"]))


if __name__ == "__main__":
    main()
