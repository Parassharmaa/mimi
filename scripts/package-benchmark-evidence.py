#!/usr/bin/env python3
"""Export portable benchmark evidence without machine-specific requests or console logs."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
from integrity import seal_run, verify_run
from prepare_suite import sha256


def export_run(source: Path, destination: Path) -> None:
    verify_run(source)
    destination.mkdir()
    for name in ("metadata.json", "attempts.jsonl", "suite.jsonl", "summary.json", "summary.md"):
        shutil.copyfile(source / name, destination / name)
    shutil.copyfile(source / "receipt.json", destination / "original-receipt.json")
    registered = json.loads((source / "receipt.json").read_text())["files"]
    for case in sorted(source.glob("case-*")):
        if not case.is_dir(): continue
        if not case.resolve().is_relative_to(source.resolve()): raise ValueError("Case directory escapes the run")
        target = destination / case.name
        target.mkdir()
        for name in ("trace.jsonl", "process.json"):
            if (case / name).is_file():
                relative = str((case / name).relative_to(source))
                if relative not in registered or not (case / name).resolve().is_relative_to(source.resolve()):
                    raise ValueError("Unregistered case artifact")
                shutil.copyfile(case / name, target / name)
    (destination / "export.json").write_text(json.dumps({
        "schemaVersion": 1, "originalReceiptSha256": sha256(source / "receipt.json"),
        "omitted": ["request.json", "stdout.log", "stderr.log"],
        "note": "Model paths and console logs stay in the original local run. Measurement rows and native event traces are copied byte-for-byte. The original receipt retains their source hashes."
    }, indent=2) + "\n")
    seal_run(destination)
    verify_run(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    collection = json.loads((args.collection / "collection.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("profile.json", "collection.json"):
        shutil.copyfile(args.collection / name, args.output / name)
    for name, expected in collection["runReceipts"].items():
        source = (args.collection / name).resolve()
        if not source.is_relative_to(args.collection.resolve()) or sha256(source / "receipt.json") != expected:
            raise ValueError("Collection receipt mismatch")
        export_run(source, args.output / name)
    print(args.output)


if __name__ == "__main__": main()
