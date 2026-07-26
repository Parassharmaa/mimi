#!/usr/bin/env python3
"""Verify Mimi's selected adaptive long-form segmentation evidence."""

from __future__ import annotations

import hashlib
import json
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
LANGUAGE_PROFILE = {
    "ja": {
        "maximumUtteranceSeconds": 30,
        "forcedBoundaryLookbackSeconds": 6,
        "gaplessMaximumErrorRate": 0.07,
    },
    "en": {
        "maximumUtteranceSeconds": 24,
        "forcedBoundaryLookbackSeconds": 6,
        "gaplessMaximumErrorRate": 0.13,
    },
}
SELECTED_REPORTS = {
    "ja": {
        "gapless": "mimi-paced-gapless-adaptive-ja30-6-final-v2.json",
        "paused": "mimi-direct-paused-adaptive-ja30-6-final-v2.json",
    },
    "en": {
        "gapless": "mimi-paced-gapless-adaptive-en24-6-final-v2.json",
        "paused": "mimi-direct-paused-adaptive-en24-6-final-v2.json",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_candidate_report(
    path: Path,
    suite_path: Path,
    language: str,
    implementation_sha256: str,
    *,
    paced: bool,
) -> dict:
    report = load_json(path)
    profile = LANGUAGE_PROFILE[language]
    require(
        report["suiteSha256"] == sha256_file(suite_path),
        f"{path}: suite hash mismatch",
    )
    require(
        report["implementationSha256"] == implementation_sha256,
        f"{path}: implementation hash does not match this checkout",
    )
    require(
        report["executableImplementationSha256"]
        == implementation_sha256,
        f"{path}: executable was not built from selected implementation",
    )
    require(
        report["implementationFiles"]
        == list(ADAPTIVE_SEGMENTATION_IMPLEMENTATION_FILES),
        f"{path}: implementation input list mismatch",
    )
    require(
        report["modelWeightsSha256"] == EXPECTED_WEIGHTS_SHA256,
        f"{path}: unexpected model hash",
    )
    require(
        report["effectiveProfile"]["maximumUtteranceSeconds"]
        == profile["maximumUtteranceSeconds"],
        f"{path}: unexpected maximum utterance",
    )
    require(
        report["effectiveProfile"]["forcedBoundaryLookbackSeconds"]
        == profile["forcedBoundaryLookbackSeconds"],
        f"{path}: unexpected forced-boundary lookback",
    )
    require(
        report["selectedCaseIDs"]
        == [row["caseID"] for row in report["results"]],
        f"{path}: selected case order mismatch",
    )
    if paced:
        require(
            report["replayMode"] == "paced-production-queue",
            f"{path}: expected production queue replay",
        )
        require(
            report["queueCapacitySeconds"] == 8,
            f"{path}: expected eight-second queue",
        )
        for key in (
            "totalDroppedAudioSamples",
            "totalAudioDropEventCount",
            "totalBackpressureEventCount",
        ):
            require(report[key] == 0, f"{path}: nonzero {key}")
        require(
            0.99 <= report["meanInputDeliveryRealTimeFactor"] <= 1.01,
            f"{path}: input was not delivered at real-time pace",
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
            report["postAudioFinalizationP95Seconds"] <= 1.1,
            f"{path}: finalization exceeds 1.1 seconds",
        )
    else:
        require(
            report["replayMode"] == "direct-compute",
            f"{path}: expected direct compute control",
        )
    return report


def verify_segment_telemetry(
    report: dict,
    path: Path,
    language: str,
    *,
    adaptive_required: bool,
) -> None:
    adaptive_count = 0
    separator = "" if language == "ja" else " "
    configured_lookback = LANGUAGE_PROFILE[language][
        "forcedBoundaryLookbackSeconds"
    ]
    for result in report["results"]:
        segments = result.get("finalizedSegments")
        require(
            isinstance(segments, list) and segments,
            f"{path}: {result['caseID']} lacks finalized segment telemetry",
        )
        previous_audio_end = 0.0
        rendered_segments = []
        for index, segment in enumerate(segments):
            audio_end = segment.get("audioEndSeconds")
            require(
                isinstance(audio_end, (int, float)),
                f"{path}: segment {index} lacks numeric audio end",
            )
            require(
                audio_end > previous_audio_end,
                f"{path}: segment audio ends are not strictly increasing",
            )
            require(
                audio_end <= result["audioDurationSeconds"] + 0.11,
                f"{path}: segment audio end exceeds recording duration",
            )
            previous_audio_end = audio_end
            rendered_segments.append(segment["text"].strip())
            lookback = segment.get("forcedBoundaryLookbackSeconds")
            if segment["reason"] == "adaptive-low-energy":
                adaptive_count += 1
                require(
                    isinstance(lookback, (int, float)),
                    f"{path}: adaptive segment lacks carry duration",
                )
                require(
                    0.2 <= lookback <= configured_lookback + 0.000_001,
                    f"{path}: adaptive carry is outside configured bounds",
                )
            else:
                require(
                    lookback is None,
                    f"{path}: non-adaptive segment unexpectedly has carry",
                )
        final_coverage_tolerance = (
            result["feedChunkSeconds"] * 2 + 0.000_001
            if segments[-1]["reason"] == "endpoint"
            else result["feedChunkSeconds"] + 0.01
        )
        require(
            abs(previous_audio_end - result["audioDurationSeconds"])
            <= final_coverage_tolerance,
            f"{path}: segment telemetry leaves an excessive residual tail",
        )
        require(
            separator.join(rendered_segments) == result["hypothesis"],
            f"{path}: segment telemetry does not reconstruct hypothesis",
        )
    if adaptive_required:
        require(
            adaptive_count >= 1,
            f"{path}: adaptive low-energy boundary never fired",
        )
    else:
        require(
            adaptive_count == 0,
            f"{path}: adaptive low-energy boundary fired unexpectedly",
        )


def main() -> None:
    implementation_sha256 = adaptive_segmentation_implementation_sha256()
    require(
        BUILT_EXECUTABLE.is_file(),
        "build Mimi before verifying adaptive segmentation evidence",
    )
    require(
        executable_adaptive_segmentation_identity(BUILT_EXECUTABLE)
        == implementation_sha256,
        "built Mimi executable does not match the current implementation",
    )
    executable_sha256s = set()
    for language in ("ja", "en"):
        profile = LANGUAGE_PROFILE[language]
        root = EVIDENCE_ROOT / f"long-form-{language}-v1"
        gapless_suite = root / "manifest.jsonl"
        paused_suite = root / "manifest-silence-1s.jsonl"
        baseline_gapless = load_json(
            root / "mimi-product-paced-queue-gapless-8s-corrected-v2.json"
        )
        baseline_paused = load_json(
            root
            / "mimi-product-paced-queue-silence-1s-8s-corrected-v2.json"
        )
        candidate_gapless = verify_candidate_report(
            root / SELECTED_REPORTS[language]["gapless"],
            gapless_suite,
            language,
            implementation_sha256,
            paced=True,
        )
        candidate_paused = verify_candidate_report(
            root / SELECTED_REPORTS[language]["paused"],
            paused_suite,
            language,
            implementation_sha256,
            paced=False,
        )
        executable_sha256s.add(candidate_gapless["executableSha256"])
        executable_sha256s.add(candidate_paused["executableSha256"])
        verify_segment_telemetry(
            candidate_gapless,
            root / SELECTED_REPORTS[language]["gapless"],
            language,
            adaptive_required=True,
        )
        verify_segment_telemetry(
            candidate_paused,
            root / SELECTED_REPORTS[language]["paused"],
            language,
            adaptive_required=False,
        )

        require(
            candidate_gapless["corpusErrorRate"]
            < baseline_gapless["corpusErrorRate"],
            f"{language}: adaptive gapless result did not improve",
        )
        require(
            candidate_gapless["corpusErrorRate"]
            <= profile["gaplessMaximumErrorRate"],
            f"{language}: adaptive gapless result misses its quality gate",
        )
        require(
            candidate_paused["corpusErrorRate"]
            == baseline_paused["corpusErrorRate"],
            f"{language}: paused error rate changed",
        )
        require(
            candidate_paused["results"][0]["hypothesis"]
            == baseline_paused["results"][0]["hypothesis"],
            f"{language}: paused hypothesis changed",
        )
        clean = load_json(
            EVIDENCE_ROOT
            / (
                "streaming-profile-sweep-v1/"
                f"{language}-product-direct-corrected-v2.json"
            )
        )
        noisy = load_json(
            EVIDENCE_ROOT
            / (
                f"noisy-{language}-pink-15db-v2/"
                "mimi-paced-queue-8s-corrected-v2.json"
            )
        )
        maximum_registered_clip = max(
            row["audioDurationSeconds"]
            for report in (clean, noisy)
            for row in report["results"]
        )
        require(
            maximum_registered_clip
            < profile["maximumUtteranceSeconds"],
            f"{language}: registered short case can activate forced boundary",
        )

    require(
        len(executable_sha256s) == 1,
        "selected adaptive reports were not produced by one executable",
    )
    print(
        "Mimi adaptive segmentation evidence passed: portable source identity, "
        "shared local binary, complete paced boundary telemetry, zero-loss "
        "gapless gains, exact paused parity, and inactive clean/noisy "
        "short-case surfaces."
    )


if __name__ == "__main__":
    main()
