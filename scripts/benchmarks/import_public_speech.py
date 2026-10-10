#!/usr/bin/env python3
"""Freeze an existing hash-bound public ASR screen in the common suite format."""
import argparse
import json
import shutil
import subprocess
import wave
from pathlib import Path
from prepare_suite import sha256, activity_estimate
import array
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", required=True, choices=["en-US", "ja-JP"])
    args = parser.parse_args()
    original = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not original or len({row["caseID"] for row in original}) != len(original): raise ValueError("Empty or duplicate source cases")
    for row in original:
        if row.get("dataset") != "google/fleurs" or row.get("datasetRevision") != "70bb2e84b976b7e960aa89f1c648e09c59f894dd" or row.get("license") != "CC-BY-4.0":
            raise ValueError("This adapter requires the repository's registered public FLEURS screen")
        expected_config = "en_us" if args.language == "en-US" else "ja_jp"
        if row.get("datasetConfig") != expected_config: raise ValueError("Language does not match the registered dataset")
        audio = (args.manifest.parent / row["audio"]).resolve()
        if not audio.is_relative_to(args.manifest.parent.resolve()) or sha256(audio) != row["audioSha256"]:
            raise ValueError("Source audio is outside the suite or has changed")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, row in enumerate(original):
        source = args.manifest.parent / row["audio"]
        audio = args.output / f"case-{index:04d}.wav"
        shutil.copyfile(source, audio)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(audio)
        ], text=True))
        duration = float(probe["format"]["duration"])
        estimate = None
        try:
            with wave.open(str(audio), "rb") as handle:
                if handle.getnchannels() == 1 and handle.getsampwidth() == 2:
                    samples = array.array("h", handle.readframes(handle.getnframes()))
                    if sys.byteorder != "little": samples.byteswap()
                    estimate = activity_estimate(list(samples), handle.getframerate())
        except wave.Error:
            pass  # Keep the original float WAV bytes; no invented speech-onset annotation.
        rows.append({"caseID": row["caseID"], "groupID": row["caseID"], "language": args.language,
                     "source": row["reference"], "references": [], "condition": "public-human-speech", "domain": "read-speech",
                     "fixtureClass": "public-development", "license": row["license"],
                     "provenance": f"{row['dataset']}@{row['datasetRevision']}:{row['datasetConfig']}:{row['datasetSplit']}:{row['datasetPosition']}",
                     "claimEligible": False, "lanes": ["speech", "pipeline"], "audio": audio.name,
                     "audioSha256": sha256(audio), "audioDurationSeconds": duration,
                     "acousticActivityEstimate": estimate or {}, "speechStartSeconds": None, "speechEndSeconds": None,
                     "protectedTerms": row.get("protectedTerms", []), "protectedTermAliases": row.get("protectedTermAliases", {})})
    output = args.output / "manifest.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    (args.output / "suite.json").write_text(json.dumps({"schemaVersion": 1, "manifestSha256": sha256(output),
        "sourceManifestSha256": sha256(args.manifest), "caseCount": len(rows),
        "note": "Public development screen already used in model selection. It is not a sealed holdout; translation references are unavailable."}, indent=2) + "\n")
    print(output)


if __name__ == "__main__": main()
