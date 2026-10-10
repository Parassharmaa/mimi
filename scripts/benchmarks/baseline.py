#!/usr/bin/env python3
"""Collect the declared initial baseline serially using existing frozen inputs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from prepare_suite import sha256

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/benchmarks/baseline-v1.json")
    for suite in ("synthetic", "human-en", "human-ja", "translation"):
        parser.add_argument("--" + suite, type=Path, required=True)
    for model in ("whisper", "phonon", "parakeet"):
        parser.add_argument("--" + model + "-model", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    receipts = {}
    for run in profile["runs"]:
        output = args.output / run["name"]
        suite = getattr(args, run["suite"].replace("-", "_"))
        command = [sys.executable, str(ROOT / "scripts/benchmarks/run.py"), str(suite), str(output),
                   "--repetitions", str(profile["repetitions"]), "--warmups", str(profile["warmups"]), "--lanes", *run["lanes"]]
        for model in ("whisper", "phonon", "parakeet"):
            command += ["--" + model + "-model", str(getattr(args, model + "_model"))]
        for case in run.get("cases", []): command += ["--case", case]
        if run.get("condition"): command += ["--condition", run["condition"]]
        print("Collecting " + run["name"], flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        subprocess.run([sys.executable, str(ROOT / "scripts/benchmarks/report.py"), str(output)], check=True, cwd=ROOT)
        receipts[run["name"]] = sha256(output / "receipt.json")
    (args.output / "collection.json").write_text(json.dumps({"schemaVersion": 1, "profileSha256": sha256(args.profile), "runReceipts": receipts}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__": main()
