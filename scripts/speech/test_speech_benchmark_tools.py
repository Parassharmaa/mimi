#!/usr/bin/env python3
"""Deterministic contracts for Mimi's speech benchmark utilities."""

from __future__ import annotations

import hashlib
import importlib.util
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
KOKORO_BUILDER = (
    REPOSITORY_ROOT
    / "scripts/speech/prepare_kokoro_natural_long_form_v1.py"
)
AMI_BUILDER = (
    REPOSITORY_ROOT / "scripts/speech/prepare_ami_natural_meeting_v1.py"
)
NATURAL_SUMMARIZER = (
    REPOSITORY_ROOT
    / "scripts/speech/summarize_natural_speech_evidence.py"
)


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


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
    def test_natural_summary_edit_breakdown_is_exact(self) -> None:
        module = load_module(
            "mimi_natural_summary",
            NATURAL_SUMMARIZER,
        )

        result = module.edit_breakdown(
            ["a", "b", "c"],
            ["a", "x", "c", "e"],
        )

        self.assertEqual(result["matches"], 2)
        self.assertEqual(result["substitutions"], 1)
        self.assertEqual(result["deletions"], 0)
        self.assertEqual(result["insertions"], 1)
        self.assertEqual(result["edits"], 2)
        self.assertEqual(result["errorRate"], 2 / 3)

    def test_ami_natural_fixture_parses_nonscenario_overlap(
        self,
    ) -> None:
        module = load_module("mimi_ami_fixture", AMI_BUILDER)
        meeting = module.parse_meeting_metadata(
            b"""<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <meeting observation="EN2001a" type="nonscenario"
    name="Natural meeting" topic="A topic" duration="5250.251">
    <speaker nxt_agent="A"/>
    <speaker nxt_agent="B"/>
  </meeting>
</nite:root>"""
        )
        first = module.parse_word_member(
            b"""<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <w nite:id="a0" starttime="10.0" endtime="11.0">hello</w>
  <w nite:id="a1" starttime="11.0" endtime="11.0" punc="true">.</w>
</nite:root>""",
            "A",
        )
        second = module.parse_word_member(
            b"""<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <w nite:id="b0" starttime="10.5" endtime="11.2">there</w>
</nite:root>""",
            "B",
        )

        self.assertEqual(meeting["type"], "nonscenario")
        self.assertEqual(meeting["speakers"], ["A", "B"])
        self.assertEqual([word.text for word in first], ["hello"])
        self.assertEqual(module.count_overlapping_words(first + second), 2)
        self.assertEqual(
            module.SIGNALS["headset"]["description"],
            "close-talking headset microphone mix",
        )
        for signal in module.SIGNALS.values():
            self.assertRegex(str(signal["sha256"]), r"^[0-9a-f]{64}$")

    def test_kokoro_natural_fixture_selection_is_pinned_and_ordered(
        self,
    ) -> None:
        module = load_module("mimi_kokoro_fixture", KOKORO_BUILDER)
        payload = "\n".join(
            [
                (
                    "gongitsune-by-nankichi-niimi-00001|"
                    "gongitsune_01_niimi_64kb.mp3|0|100|title|reading"
                ),
                (
                    f"{module.FIRST_CASE_ID}|{module.SOURCE_AUDIO_MEMBER}|"
                    "200|300|first words|reading"
                ),
                (
                    f"{module.LAST_CASE_ID}|{module.SOURCE_AUDIO_MEMBER}|"
                    "350|500|last words|reading"
                ),
                (
                    "gongitsune-by-nankichi-niimi-00064|"
                    "gongitsune_02_niimi_64kb.mp3|0|100|next|reading"
                ),
            ]
        ).encode()

        selected = module.select_registered_rows(
            module.parse_metadata(payload)
        )

        self.assertEqual(
            [row.case_id for row in selected],
            [module.FIRST_CASE_ID, module.LAST_CASE_ID],
        )
        self.assertEqual(
            module.DATASET_REVISION,
            "88377525a85728c79cf78f6bfcb81396f9c23827",
        )
        for digest in (
            module.METADATA_SHA256,
            module.SOURCE_ARCHIVE_SHA256,
            module.SOURCE_AUDIO_SHA256,
        ):
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

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
