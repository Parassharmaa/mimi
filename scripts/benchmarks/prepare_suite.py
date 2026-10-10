#!/usr/bin/env python3
"""Freeze owned synthetic speech probes and/or an explicitly supplied public text suite."""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_audio(path: Path, samples: list[int], rate: int = 16000) -> None:
    values = array.array("h", samples)
    if sys.byteorder != "little":
        values.byteswap()
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(values.tobytes())


def activity_estimate(samples: list[int], rate: int) -> dict:
    width = rate // 100
    active = [i for i in range(0, len(samples), width)
              if math.sqrt(sum((x / 32768) ** 2 for x in samples[i:i + width]) / max(1, len(samples[i:i + width]))) >= 0.01]
    return {"method": "10ms RMS >= 0.01; acoustic activity estimate, not human speech annotation",
            "startSeconds": active[0] / rate if active else None,
            "endSeconds": min(len(samples), active[-1] + width) / rate if active else None}


def noise_variant(samples: list[int], seed: int, snr_db: float = 15) -> tuple[list[int], float]:
    rng = random.Random(seed)
    noise = [rng.gauss(0, 1) for _ in samples]
    signal_rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    noise_rms = math.sqrt(sum(x * x for x in noise) / len(noise))
    scale = signal_rms / (10 ** (snr_db / 20) * noise_rms)
    mixed = [x + n * scale for x, n in zip(samples, noise)]
    gain = min(1, 32760 / max(1, max(abs(x) for x in mixed)))
    return [round(x * gain) for x in mixed], gain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--utterances", type=Path, default=ROOT / "configs/benchmarks/utterances-v1.json")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--public-translations", type=Path)
    args = parser.parse_args()
    # Keep promotion/claim material out of this development runner.
    if args.public_translations and ("automated-claim" in str(args.public_translations) or "held-out" in str(args.public_translations)):
        parser.error("Use an explicitly public development suite, never sealed promotion material.")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    fixtures = json.loads(args.utterances.read_text())
    for item in fixtures:
        case_id = item["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", case_id):
            raise ValueError("Unsafe fixture ID")
        row = {"caseID": case_id, "groupID": case_id, "language": item["language"],
               "source": item["source"], "references": item["references"], "domain": item["domain"],
               "condition": "long" if item["domain"] == "long-utterance" else "clean",
               "fixtureClass": "owned-synthetic", "license": "project-owned prompts",
               "provenance": "Mimi regression prompts; unreviewed bilingual reference text",
               "claimEligible": False, "reviewStatus": "bootstrap-unreviewed", "lanes": ["translation"],
               "speechStartSeconds": None, "speechEndSeconds": None}
        if not args.text_only:
            aiff = args.output / (case_id + ".aiff")
            audio = args.output / (case_id + ".wav")
            subprocess.run(["say", "-v", item["voice"], "-o", str(aiff), item["source"]], check=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(audio)], check=True)
            with wave.open(str(audio), "rb") as handle:
                assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, 16000)
                samples = array.array("h", handle.readframes(handle.getnframes()))
                if sys.byteorder != "little": samples.byteswap()
            samples = [0] * 6400 + list(samples) + [0] * 12800
            write_audio(audio, samples)
            row.update(audio=audio.name, audioSha256=sha256(audio), audioDurationSeconds=len(samples) / 16000,
                       acousticActivityEstimate=activity_estimate(samples, 16000), voice=item["voice"],
                       lanes=["speech", "translation", "pipeline"])
            aiff.unlink()  # Only our intermediate synthesis file.
            if item["domain"] == "meeting":
                noisy, gain = noise_variant(samples, 20261010)
                noise_audio = args.output / (case_id + "-noise.wav")
                write_audio(noise_audio, noisy)
                rows.append({**row, "caseID": case_id + "-noise", "condition": "noise",
                             "audio": noise_audio.name, "audioSha256": sha256(noise_audio),
                             "noise": {"kind": "white", "seed": 20261010, "wholeClipSNRDb": 15, "gain": gain},
                             "lanes": ["speech", "pipeline"]})
        rows.append(row)
    if not args.text_only:
        silence = args.output / "silence.wav"
        write_audio(silence, [0] * 64000)
        for language in ("en-US", "ja-JP"):
            rows.append({"caseID": language + "-silence", "groupID": language + "-silence", "language": language,
                         "source": "", "references": [], "domain": "silence", "condition": "silence",
                         "fixtureClass": "owned-synthetic", "license": "project-owned", "provenance": "digital silence",
                         "claimEligible": False, "audio": silence.name, "audioSha256": sha256(silence),
                         "audioDurationSeconds": 4, "lanes": ["speech", "pipeline"],
                         "speechStartSeconds": None, "speechEndSeconds": None})
    if args.public_translations:
        for line in args.public_translations.read_text().splitlines():
            item = json.loads(line)
            if item.get("claimEligible") is not False:
                raise ValueError("Only explicitly claim-ineligible public development rows are accepted")
            rows.append({"caseID": item["id"], "groupID": item.get("parentCaseID", item["id"]),
                         "language": item["sourceLanguage"], "source": item["source"], "references": item["references"],
                         "domain": item["domain"], "condition": "public-development", "fixtureClass": "public-development",
                         "license": item["license"], "provenance": item["provenance"], "claimEligible": False,
                         "lanes": ["translation"], "reviewStatus": item["reviewStatus"]})
    if len({row["caseID"] for row in rows}) != len(rows): raise ValueError("Duplicate case IDs")
    manifest = args.output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    metadata = {"schemaVersion": 1, "manifestSha256": sha256(manifest), "caseCount": len(rows),
                "utterancesSha256": sha256(args.utterances), "platform": platform.platform(), "python": sys.version,
                "publicTranslationsSha256": sha256(args.public_translations) if args.public_translations else None,
                "note": "Audio hashes define comparability. Resynthesis on another OS/voice version creates a new cohort. Synthetic probes do not establish human-speech accuracy."}
    (args.output / "suite.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(manifest)


if __name__ == "__main__": main()
