#!/usr/bin/env python3
"""Deterministic contracts for Mimi's speech benchmark utilities."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOISE_BUILDER = REPOSITORY_ROOT / "scripts/speech/build_noisy_speech_fixture.py"
COMPARATOR = REPOSITORY_ROOT / "scripts/speech/compare_asr_benchmarks.py"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tone(path: Path, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        int(
            0.2
            * 32767
            * math.sin(2 * math.pi * frequency * sample / 16_000)
        )
        for sample in range(16_000)
    ]
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(
            b"".join(struct.pack("<h", sample) for sample in samples)
        )


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class SpeechBenchmarkToolTests(unittest.TestCase):
    def test_noise_fixture_is_hash_bound_collision_safe_and_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-noise-test-") as temporary:
            root = Path(temporary)
            sources = root / "sources"
            first_audio = sources / "first/same.wav"
            second_audio = sources / "second/same.wav"
            write_tone(first_audio, 440)
            write_tone(second_audio, 660)
            rows = [
                {
                    "caseID": "case:first",
                    "audio": "first/same.wav",
                    "audioSha256": sha256_file(first_audio),
                    "reference": "Mimi first",
                    "license": "CC0-1.0",
                },
                {
                    "caseID": "case:second",
                    "audio": "second/same.wav",
                    "audioSha256": sha256_file(second_audio),
                    "reference": "Mimi second",
                    "license": "CC0-1.0",
                },
            ]
            suite = sources / "manifest.jsonl"
            suite.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )

            outputs = [root / "output-a", root / "output-b"]
            for output in outputs:
                subprocess.run(
                    [
                        sys.executable,
                        str(NOISE_BUILDER),
                        str(suite),
                        str(output),
                        "--noise-color",
                        "pink",
                        "--snr-db",
                        "15",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            manifests = [
                (output / "manifest.jsonl").read_bytes()
                for output in outputs
            ]
            self.assertEqual(manifests[0], manifests[1])
            derived = [
                json.loads(line)
                for line in manifests[0].decode().splitlines()
            ]
            self.assertEqual(len({row["audio"] for row in derived}), 2)
            self.assertTrue(
                all(row["noiseLicense"] == "CC0-1.0" for row in derived)
            )
            for row in derived:
                toolchain = row["noiseTransform"]["toolchain"]
                for name in ("ffmpeg", "ffprobe"):
                    self.assertRegex(
                        toolchain[name]["sha256"],
                        r"^[0-9a-f]{64}$",
                    )
                self.assertEqual(
                    (outputs[0] / row["audio"]).read_bytes(),
                    (outputs[1] / row["audio"]).read_bytes(),
                )

    def test_comparison_aligns_clean_and_derived_case_ids(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mimi-compare-test-") as temporary:
            root = Path(temporary)
            suite = root / "suite.jsonl"
            suite_rows = [
                {
                    "caseID": "base:first:noisy",
                    "derivedFromCaseID": "base:first",
                    "protectedTerms": ["Mimi"],
                    "protectedTermAliases": {"Mimi": []},
                },
                {
                    "caseID": "base:second:noisy",
                    "derivedFromCaseID": "base:second",
                    "protectedTerms": [],
                    "protectedTermAliases": {},
                },
            ]
            suite.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in suite_rows
                ),
                encoding="utf-8",
            )
            left = root / "left.json"
            right = root / "right.json"
            output = root / "comparison.json"
            write_json(
                left,
                {
                    "engine": "raw-clean",
                    "metric": "wer",
                    "corpusErrorRate": 0.0,
                    "meanPacedWallRealTimeFactor": 1.01,
                    "results": [
                        {
                            "caseID": "base:first",
                            "hypothesis": "Mimi",
                            "editDistance": 0,
                            "referenceUnits": 1,
                            "errorRate": 0.0,
                        },
                        {
                            "caseID": "base:second",
                            "hypothesis": "second",
                            "editDistance": 0,
                            "referenceUnits": 1,
                            "errorRate": 0.0,
                        },
                    ],
                },
            )
            write_json(
                right,
                {
                    "engine": "raw-noisy",
                    "metric": "wer",
                    "corpusErrorRate": 0.5,
                    "meanPacedWallRealTimeFactor": 1.02,
                    "results": [
                        {
                            "caseID": "base:first:noisy",
                            "hypothesis": "missing",
                            "editDistance": 1,
                            "referenceUnits": 1,
                            "errorRate": 1.0,
                        },
                        {
                            "caseID": "base:second:noisy",
                            "hypothesis": "second",
                            "editDistance": 0,
                            "referenceUnits": 1,
                            "errorRate": 0.0,
                        },
                    ],
                },
            )

            subprocess.run(
                [
                    sys.executable,
                    str(COMPARATOR),
                    str(suite),
                    str(left),
                    str(right),
                    str(output),
                    "--left-label",
                    "Clean",
                    "--right-label",
                    "Noise",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            comparison = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(comparison["conditionAlignment"], "derived-source")
            self.assertEqual(comparison["pairwise"]["leftEngine"], "Clean")
            self.assertEqual(comparison["pairwise"]["rightEngine"], "Noise")
            self.assertEqual(comparison["pairwise"]["leftWins"], 1)
            self.assertEqual(comparison["pairwise"]["ties"], 1)
            self.assertEqual(comparison["engines"][0]["rawEngine"], "raw-clean")
            self.assertEqual(comparison["engines"][1]["meanRealTimeFactor"], 1.02)


if __name__ == "__main__":
    unittest.main()
