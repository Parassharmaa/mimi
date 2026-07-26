#!/usr/bin/env python3
"""Verify Mimi's selected corrected paced-speech evidence without audio files."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "Research/speech/work"
EXPECTED_EXECUTABLE_SHA256 = (
    "0a71d7179b4a993137b817cf9863db47689e3bbcfb075a64dff9ce51f9af9c52"
)
EXPECTED_WEIGHTS_SHA256 = (
    "45298f6dc48df8c11e0a8d1dc5e0197c688bfa530646fa21f1a0238d2b0ecda3"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_report(path: Path, suite_path: Path, *, paced: bool) -> dict:
    report = load_json(path)
    require(
        report["suiteSha256"] == sha256_file(suite_path),
        f"{path}: suite hash does not match {suite_path}",
    )
    require(
        report["executableSha256"] == EXPECTED_EXECUTABLE_SHA256,
        f"{path}: unexpected executable hash",
    )
    require(
        report["modelWeightsSha256"] == EXPECTED_WEIGHTS_SHA256,
        f"{path}: unexpected model weights hash",
    )
    result_ids = [row["caseID"] for row in report["results"]]
    require(
        result_ids == report["selectedCaseIDs"],
        f"{path}: selected IDs and result order differ",
    )
    require(
        len(result_ids) == report["caseCount"] == len(set(result_ids)),
        f"{path}: case count or uniqueness mismatch",
    )
    if paced:
        require(
            report["replayMode"] == "paced-production-queue",
            f"{path}: report is not the production queue replay",
        )
        require(
            report["queueCapacitySeconds"] == 8,
            f"{path}: report did not use the eight-second queue",
        )
        for key in (
            "totalDroppedAudioSamples",
            "totalAudioDropEventCount",
            "totalBackpressureEventCount",
        ):
            require(report[key] == 0, f"{path}: nonzero {key}")
        require(
            0 < report["maximumInputScheduleLatenessSeconds"] < 0.1,
            f"{path}: wake lateness is absent or implausible",
        )
        require(
            0.99 <= report["meanInputDeliveryRealTimeFactor"] <= 1.01,
            f"{path}: input was not delivered at real-time pace",
        )
    else:
        require(
            report["replayMode"] == "direct-compute",
            f"{path}: direct control has the wrong replay mode",
        )
    return report


def verify_noise_manifest(path: Path, source_case_ids: set[str]) -> None:
    rows = load_jsonl(path)
    require(len(rows) == 24, f"{path}: expected 24 noisy cases")
    derived_ids = [row["derivedFromCaseID"] for row in rows]
    require(
        set(derived_ids) == source_case_ids and len(set(derived_ids)) == 24,
        f"{path}: noisy cases do not align one-to-one with clean sources",
    )
    for row in rows:
        require(row["noiseLicense"] == "CC0-1.0", f"{path}: noise is not CC0")
        require(
            row["sourceAudioSha256"] != row["audioSha256"],
            f"{path}: noisy output unexpectedly matches source audio",
        )
        for executable in ("ffmpeg", "ffprobe"):
            metadata = row["noiseTransform"]["toolchain"][executable]
            require(
                SHA256_PATTERN.fullmatch(metadata["sha256"]) is not None,
                f"{path}: invalid {executable} hash",
            )
            require(metadata["version"], f"{path}: missing {executable} version")


def main() -> None:
    clean_reports: dict[str, tuple[dict, dict]] = {}
    for language in ("ja", "en"):
        suite = EVIDENCE_ROOT / f"fleurs-{language}-screen-v1/manifest.jsonl"
        direct = verify_report(
            EVIDENCE_ROOT
            / (
                "streaming-profile-sweep-v1/"
                f"{language}-product-direct-corrected-v2.json"
            ),
            suite,
            paced=False,
        )
        paced = verify_report(
            EVIDENCE_ROOT
            / (
                "streaming-profile-sweep-v1/"
                f"{language}-product-paced-queue-8s-corrected-v2.json"
            ),
            suite,
            paced=True,
        )
        direct_hypotheses = [
            (row["caseID"], row["hypothesis"]) for row in direct["results"]
        ]
        paced_hypotheses = [
            (row["caseID"], row["hypothesis"]) for row in paced["results"]
        ]
        require(
            direct_hypotheses == paced_hypotheses,
            f"{language}: paced hypotheses differ from same-hash direct control",
        )
        clean_reports[language] = (direct, paced)

    for language in ("ja", "en"):
        clean_suite = EVIDENCE_ROOT / f"fleurs-{language}-screen-v1/manifest.jsonl"
        clean_case_ids = {row["caseID"] for row in load_jsonl(clean_suite)}
        noise_root = EVIDENCE_ROOT / f"noisy-{language}-pink-15db-v2"
        noise_suite = noise_root / "manifest.jsonl"
        verify_noise_manifest(noise_suite, clean_case_ids)
        verify_report(
            noise_root / "mimi-paced-queue-8s-corrected-v2.json",
            noise_suite,
            paced=True,
        )
        comparison = load_json(
            noise_root / "clean-vs-noisy-8s-corrected-v2.json"
        )
        require(
            comparison["conditionAlignment"] == "derived-source"
            and comparison["caseCount"] == 24,
            f"{noise_root}: invalid paired comparison",
        )

    for language in ("ja", "en"):
        long_root = EVIDENCE_ROOT / f"long-form-{language}-v1"
        verify_report(
            long_root
            / "mimi-product-paced-queue-silence-1s-8s-corrected-v2.json",
            long_root / "manifest-silence-1s.jsonl",
            paced=True,
        )
        verify_report(
            long_root / "mimi-product-paced-queue-gapless-8s-corrected-v2.json",
            long_root / "manifest.jsonl",
            paced=True,
        )

    print(
        "Mimi corrected paced-speech evidence passed: same-hash direct parity, "
        "hash-bound noise, and zero-loss clean/noisy/long queue reports."
    )


if __name__ == "__main__":
    main()
