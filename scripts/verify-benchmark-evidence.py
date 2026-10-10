#!/usr/bin/env python3
"""Check the archived baseline's hashes and coverage without replaying models."""
import hashlib
import json
import math
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
from integrity import verify_run
from prepare_suite import sha256


def main():
    archive = ROOT / "docs/benchmarks/foundation-v1/evidence.tar.gz"
    expected = Path(str(archive) + ".sha256").read_text().split()[0]
    if sha256(archive) != expected: raise ValueError("Benchmark archive checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="mimi-benchmark-evidence-") as temporary:
        root = Path(temporary)
        seen = set()
        total_bytes = 0
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                name = PurePosixPath(member.name)
                if not member.isfile() or name.is_absolute() or ".." in name.parts or name.parts[0] != "baseline" or member.name in seen:
                    raise ValueError("Unsafe or duplicate archive member")
                seen.add(member.name)
                total_bytes += member.size
                if total_bytes > 100_000_000: raise ValueError("Unexpected evidence size")
                target = root / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as data, target.open("xb") as output:
                    shutil.copyfileobj(data, output)
        collection_root = root / "baseline"
        collection = json.loads((collection_root / "collection.json").read_text())
        total = 0
        for name, original_hash in collection["runReceipts"].items():
            run = collection_root / name
            if not run.resolve().is_relative_to(collection_root.resolve()): raise ValueError("Run escapes collection")
            verify_run(run)
            if sha256(run / "original-receipt.json") != original_hash: raise ValueError("Original receipt mismatch")
            metadata = json.loads((run / "metadata.json").read_text())
            if metadata["schemaVersion"] != 1 or metadata["claimEligible"] is not False: raise ValueError("Unexpected benchmark scope")
            if metadata["suiteSha256"] != sha256(run / "suite.jsonl"): raise ValueError("Suite hash mismatch")
            protocol = metadata["protocol"]
            expected_rows = {(entry["caseID"], entry["lane"], entry["engine"], index)
                             for entry in metadata["plan"]
                             for index in range(1 + protocol["warmups"] + protocol["repetitions"])}
            rows = [json.loads(line) for line in (run / "attempts.jsonl").read_text().splitlines()]
            actual_rows = {(row["caseID"], row["lane"], row["engine"], row["attemptIndex"]) for row in rows}
            if actual_rows != expected_rows or len(actual_rows) != len(rows): raise ValueError("Missing or duplicate attempts")
            for row in rows:
                if row["schemaVersion"] != 1: raise ValueError("Unsupported attempt schema")
                for value in row["metrics"].values():
                    if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                        raise ValueError("Invalid numeric measurement")
                quality = row.get("speechQuality")
                if quality and quality["edits"] != sum(quality[key] for key in ("substitutions", "deletions", "insertions")):
                    raise ValueError("Inconsistent ASR edit counts")
            summary = json.loads((run / "summary.json").read_text())
            if summary["metadataSha256"] != sha256(run / "metadata.json") or summary["attemptsSha256"] != sha256(run / "attempts.jsonl"):
                raise ValueError("Summary does not describe these raw artifacts")
            if summary["plannedAttempts"] != len(rows): raise ValueError("Summary denominator mismatch")
            for group in summary["groups"]:
                for metric in group["metrics"].values():
                    if metric["independentUnitCount"] < protocol["tailMinimumIndependentUnits"] and metric["p95"] is not None:
                        raise ValueError("Unsupported tail estimate")
            total += len(rows)
        print(f"Benchmark evidence verified: {len(collection['runReceipts'])} frozen runs, {total} retained attempts, no model inference.")


if __name__ == "__main__": main()
