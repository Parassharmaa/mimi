#!/usr/bin/env python3
"""Verify Mimi's pinned natural long-form speech evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from evidence_identity import (
    ADAPTIVE_SEGMENTATION_IMPLEMENTATION_FILES,
    adaptive_segmentation_implementation_sha256,
    executable_adaptive_segmentation_identity,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "Research/speech/work"
BUILT_EXECUTABLE = REPOSITORY_ROOT / ".build/debug/Mimi"
EXPECTED_WEIGHTS_SHA256 = (
    "45298f6dc48df8c11e0a8d1dc5e0197c688bfa530646fa21f1a0238d2b0ecda3"
)
JAPANESE_ROOT = EVIDENCE_ROOT / "natural-ja-kokoro-v1"
ENGLISH_ROOT = EVIDENCE_ROOT / "natural-en-ami-v1"
JAPANESE_REPORTS = {
    "product": "mimi-product-paced-v1.json",
    "adaptive": "mimi-adaptive-ja30-6-paced-v1.json",
}
ENGLISH_REPORTS = {
    "headset": {
        "product": "mimi-product-headset-paced-v1.json",
        "hard24": "mimi-hard24-headset-paced-v1.json",
        "adaptive": "mimi-adaptive-en24-6-headset-paced-v1.json",
    },
    "array1-01": {
        "product": "mimi-product-array1-01-paced-v1.json",
        "hard24": "mimi-hard24-array1-01-paced-v1.json",
        "adaptive": "mimi-adaptive-en24-6-array1-01-paced-v1.json",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest_registries() -> None:
    japanese = load_jsonl(JAPANESE_ROOT / "manifest.jsonl")
    require(len(japanese) == 1, "Japanese natural manifest must have one case")
    japanese_case = japanese[0]
    require(
        japanese_case["caseID"]
        == "natural-ja-v1:kokoro:gongitsune:chapter-01-body",
        "unexpected Japanese natural case",
    )
    require(
        "international status requires jurisdiction review"
        in japanese_case["license"],
        "Japanese license caveat is missing",
    )
    require(
        "local and uncommitted"
        in japanese_case["redistributionStatus"],
        "Japanese redistribution restriction is missing",
    )
    require(
        "held-out evaluation only" in japanese_case["benchmarkUse"],
        "Japanese held-out registry is missing",
    )
    for key in (
        "sourceMetadataArchiveSha256",
        "sourceMetadataMemberSha256",
        "sourceArchiveSha256",
        "sourceAudioSha256",
        "audioSha256",
        "referenceSha256",
    ):
        require(
            len(japanese_case[key]) == 64,
            f"Japanese manifest has invalid {key}",
        )

    combined = load_jsonl(ENGLISH_ROOT / "manifest.jsonl")
    require(
        [row["signalCondition"] for row in combined]
        == ["headset", "array1-01"],
        "English combined manifest condition order changed",
    )
    for condition in ("headset", "array1-01"):
        condition_rows = load_jsonl(
            ENGLISH_ROOT / f"manifest-{condition}.jsonl"
        )
        require(
            condition_rows == [
                row
                for row in combined
                if row["signalCondition"] == condition
            ],
            f"English {condition} manifest differs from combined registry",
        )
        row = condition_rows[0]
        require(row["license"] == "CC-BY-4.0", "unexpected AMI license")
        require(
            row["meetingType"] == "nonscenario",
            "AMI case is not registered as a natural meeting",
        )
        require(
            row["speakerCount"] == 5
            and row["referenceSpeakers"] == ["A", "B", "C", "D", "E"],
            "AMI speaker registry changed",
        )
        require(
            row["referenceWordCount"] == 926,
            "AMI lexical reference changed",
        )
        require(
            row["referenceOverlappingWordCount"] == 135,
            "AMI overlap registry changed",
        )
        require(
            "held-out evaluation only" in row["benchmarkUse"],
            "AMI held-out registry is missing",
        )


def verify_report(
    path: Path,
    suite_path: Path,
    expected_profile: tuple[float, float],
    implementation_sha256: str,
) -> dict:
    report = load_json(path)
    require(
        report["suiteSha256"] == sha256_file(suite_path),
        f"{path}: suite hash mismatch",
    )
    require(
        report["implementationSha256"] == implementation_sha256
        and report["executableImplementationSha256"]
        == implementation_sha256,
        f"{path}: implementation identity mismatch",
    )
    require(
        report["implementationFiles"]
        == list(ADAPTIVE_SEGMENTATION_IMPLEMENTATION_FILES),
        f"{path}: implementation file inventory mismatch",
    )
    require(
        report["modelWeightsSha256"] == EXPECTED_WEIGHTS_SHA256,
        f"{path}: unexpected model weights",
    )
    profile = report["effectiveProfile"]
    require(
        (
            profile["maximumUtteranceSeconds"],
            profile["forcedBoundaryLookbackSeconds"],
        )
        == expected_profile,
        f"{path}: unexpected profile",
    )
    require(
        report["replayMode"] == "paced-production-queue",
        f"{path}: expected paced production queue",
    )
    require(
        report["queueCapacitySeconds"] == 8,
        f"{path}: unexpected queue capacity",
    )
    for key in (
        "totalDroppedAudioSamples",
        "totalAudioDropEventCount",
        "totalBackpressureEventCount",
    ):
        require(report[key] == 0, f"{path}: nonzero {key}")
    require(
        0.99 <= report["meanInputDeliveryRealTimeFactor"] <= 1.01,
        f"{path}: input delivery was not real time",
    )
    require(
        report["meanPacedWallRealTimeFactor"] <= 1.02,
        f"{path}: paced wall RTF exceeds 1.02",
    )
    require(
        report["maximumQueuedAudioSeconds"]
        <= report["queueCapacitySeconds"],
        f"{path}: peak queue exceeds capacity",
    )
    require(
        report["postAudioFinalizationP95Seconds"] <= 1.5,
        f"{path}: post-audio finalization exceeds 1.5 seconds",
    )
    require(
        report["firstTextP95Seconds"] <= 4.2,
        f"{path}: first text exceeds 4.2 seconds",
    )
    require(
        report["peakRSSBytes"] <= 1_300_000_000,
        f"{path}: peak RSS exceeds 1.3 GB",
    )
    require(
        report["selectedCaseIDs"]
        == [row["caseID"] for row in report["results"]],
        f"{path}: selected case order mismatch",
    )
    return report


def verify_segments(
    report: dict,
    path: Path,
    language: str,
) -> Counter:
    reasons: Counter = Counter()
    separator = "" if language == "ja" else " "
    for result in report["results"]:
        segments = result.get("finalizedSegments")
        require(
            isinstance(segments, list) and segments,
            f"{path}: missing finalized segment telemetry",
        )
        previous_end = 0.0
        rendered: list[str] = []
        for index, segment in enumerate(segments):
            audio_end = segment.get("audioEndSeconds")
            require(
                isinstance(audio_end, (int, float))
                and audio_end > previous_end,
                f"{path}: segment {index} has a non-monotonic audio end",
            )
            require(
                audio_end <= result["audioDurationSeconds"] + 0.000_001,
                f"{path}: segment {index} exceeds source duration",
            )
            previous_end = audio_end
            reason = segment["reason"]
            reasons[reason] += 1
            carry = segment.get("forcedBoundaryLookbackSeconds")
            if reason == "adaptive-low-energy":
                require(
                    isinstance(carry, (int, float))
                    and 0.2 <= carry <= 6.000_001,
                    f"{path}: adaptive carry is outside bounds",
                )
            else:
                require(
                    carry is None,
                    f"{path}: non-adaptive segment has carry telemetry",
                )
            rendered.append(segment["text"].strip())
        require(
            (
                result["audioDurationSeconds"] - previous_end
                <= result["feedChunkSeconds"] * 2 + 0.000_001
            ),
            f"{path}: excessive residual audio tail",
        )
        require(
            separator.join(rendered) == result["hypothesis"],
            f"{path}: segments do not reconstruct the final hypothesis",
        )
    return reasons


def main() -> None:
    verify_manifest_registries()
    implementation_sha256 = adaptive_segmentation_implementation_sha256()
    require(
        BUILT_EXECUTABLE.is_file(),
        "build Mimi before verifying natural speech evidence",
    )
    require(
        executable_adaptive_segmentation_identity(BUILT_EXECUTABLE)
        == implementation_sha256,
        "built Mimi executable does not match the current implementation",
    )

    reports: list[dict] = []
    japanese_suite = JAPANESE_ROOT / "manifest.jsonl"
    japanese_product = verify_report(
        JAPANESE_ROOT / JAPANESE_REPORTS["product"],
        japanese_suite,
        (30, 0),
        implementation_sha256,
    )
    japanese_adaptive = verify_report(
        JAPANESE_ROOT / JAPANESE_REPORTS["adaptive"],
        japanese_suite,
        (30, 6),
        implementation_sha256,
    )
    reports.extend((japanese_product, japanese_adaptive))
    japanese_product_reasons = verify_segments(
        japanese_product,
        JAPANESE_ROOT / JAPANESE_REPORTS["product"],
        "ja",
    )
    japanese_adaptive_reasons = verify_segments(
        japanese_adaptive,
        JAPANESE_ROOT / JAPANESE_REPORTS["adaptive"],
        "ja",
    )
    require(
        japanese_product["corpusErrorRate"] <= 0.25,
        "Japanese natural raw CER exceeds 25%",
    )
    require(
        japanese_product["results"][0]["hypothesis"]
        == japanese_adaptive["results"][0]["hypothesis"],
        "Japanese natural hypothesis changed under inactive adaptive profile",
    )
    require(
        japanese_product["results"][0]["finalizedSegments"]
        == japanese_adaptive["results"][0]["finalizedSegments"],
        "Japanese natural segment ledger changed under inactive profile",
    )
    require(
        japanese_product_reasons == Counter({"endpoint": 32})
        and japanese_adaptive_reasons == Counter({"endpoint": 32}),
        "Japanese natural boundary reasons changed",
    )

    english_summary: dict[str, dict[str, float]] = {}
    for condition, names in ENGLISH_REPORTS.items():
        suite_path = ENGLISH_ROOT / f"manifest-{condition}.jsonl"
        product = verify_report(
            ENGLISH_ROOT / names["product"],
            suite_path,
            (30, 0),
            implementation_sha256,
        )
        hard24 = verify_report(
            ENGLISH_ROOT / names["hard24"],
            suite_path,
            (24, 0),
            implementation_sha256,
        )
        adaptive = verify_report(
            ENGLISH_ROOT / names["adaptive"],
            suite_path,
            (24, 6),
            implementation_sha256,
        )
        reports.extend((product, hard24, adaptive))
        product_reasons = verify_segments(
            product,
            ENGLISH_ROOT / names["product"],
            "en",
        )
        hard24_reasons = verify_segments(
            hard24,
            ENGLISH_ROOT / names["hard24"],
            "en",
        )
        adaptive_reasons = verify_segments(
            adaptive,
            ENGLISH_ROOT / names["adaptive"],
            "en",
        )
        require(
            product_reasons["maximum-duration"] >= 1,
            f"{condition}: product maximum boundary never fired",
        )
        require(
            hard24_reasons["maximum-duration"] >= 1
            and hard24_reasons["adaptive-low-energy"] == 0,
            f"{condition}: hard-24 ablation is not a hard-cut control",
        )
        require(
            adaptive_reasons["adaptive-low-energy"] >= 1
            and adaptive_reasons["maximum-duration"] == 0,
            f"{condition}: adaptive profile did not replace hard cuts",
        )
        require(
            adaptive["corpusErrorRate"]
            <= product["corpusErrorRate"] - 0.05,
            f"{condition}: adaptive profile lacks a five-point product gain",
        )
        require(
            adaptive["corpusErrorRate"]
            <= hard24["corpusErrorRate"] - 0.03,
            f"{condition}: low-energy placement lacks a three-point gain",
        )
        require(
            adaptive["corpusErrorRate"] <= 0.35,
            f"{condition}: adaptive WER exceeds 35%",
        )
        english_summary[condition] = {
            "productWER": product["corpusErrorRate"],
            "hard24WER": hard24["corpusErrorRate"],
            "adaptiveWER": adaptive["corpusErrorRate"],
        }

    require(
        len({report["executableSha256"] for report in reports}) == 1,
        "natural reports were not produced by one executable",
    )
    require(
        len({report["modelWeightsSha256"] for report in reports}) == 1,
        "natural reports were not produced by one model",
    )
    print(
        "Mimi natural speech evidence passed: pinned license and "
        "contamination registries, exact Japanese inactive parity, English "
        "headset and far-field adaptive gains beyond a hard-24 ablation, "
        "complete segment reconstruction, real-time delivery, and zero "
        "queue loss."
    )
    print(json.dumps(english_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
