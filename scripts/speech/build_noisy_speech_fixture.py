#!/usr/bin/env python3
"""Build a hash-bound deterministic additive-noise ASR fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


VOLUME_PATTERN = re.compile(
    r"(?P<kind>mean|max)_volume:\s+(?P<value>-?(?:\d+(?:\.\d+)?|inf)) dB"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def executable_metadata(name: str) -> tuple[Path, dict[str, str]]:
    executable = shutil.which(name)
    if executable is None:
        raise SystemExit(f"required executable is unavailable: {name}")
    path = Path(executable).resolve()
    completed = subprocess.run(
        [str(path), "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return path, {
        "version": completed.stdout.splitlines()[0],
        "sha256": sha256_file(path),
    }


def audio_duration(path: Path, ffprobe: Path) -> float:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def volume_levels(path: Path, ffmpeg: Path) -> tuple[float, float]:
    completed = subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    levels = {
        match.group("kind"): float(match.group("value"))
        for match in VOLUME_PATTERN.finditer(completed.stderr)
    }
    if set(levels) != {"mean", "max"}:
        raise RuntimeError(f"could not measure audio levels for {path}")
    return levels["mean"], levels["max"]


def deterministic_seed(namespace: str, case_id: str) -> int:
    digest = hashlib.sha256(f"{namespace}\0{case_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--snr-db", type=float, default=15.0)
    parser.add_argument(
        "--noise-color",
        choices=("white", "pink", "brown"),
        default="pink",
    )
    parser.add_argument("--seed", default="mimi-noisy-speech-v1")
    parser.add_argument("--source-attenuation-db", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not math.isfinite(args.snr_db) or not 0 <= args.snr_db <= 40:
        raise SystemExit("--snr-db must be between 0 and 40")
    if (
        not math.isfinite(args.source_attenuation_db)
        or not -30 <= args.source_attenuation_db <= 0
    ):
        raise SystemExit("--source-attenuation-db must be between -30 and 0")

    rows = [
        json.loads(line)
        for line in args.suite.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("suite is empty")
    case_ids = [row["caseID"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise SystemExit("suite contains duplicate caseID values")

    source_root = args.suite.parent
    output_audio = args.output_directory / "audio"
    output_audio.mkdir(parents=True, exist_ok=True)
    ffmpeg, ffmpeg_build = executable_metadata("ffmpeg")
    ffprobe, ffprobe_build = executable_metadata("ffprobe")
    snr_label = f"{args.snr_db:g}".replace(".", "_")
    manifest_rows: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="mimi-noisy-speech-") as temporary:
        temporary_directory = Path(temporary)
        for row in rows:
            source_path = (source_root / row["audio"]).resolve()
            if not source_path.is_file():
                raise SystemExit(f"audio does not exist: {source_path}")
            expected_sha256 = row.get("audioSha256")
            if not expected_sha256:
                raise SystemExit(f"{row['caseID']} does not declare audioSha256")
            actual_sha256 = sha256_file(source_path)
            if actual_sha256 != expected_sha256:
                raise SystemExit(
                    f"{row['caseID']} audio hash mismatch: expected "
                    f"{expected_sha256}, got {actual_sha256}"
                )

            duration = audio_duration(source_path, ffprobe)
            case_seed = deterministic_seed(args.seed, row["caseID"])
            noise_path = temporary_directory / f"{case_seed}.wav"
            subprocess.run(
                [
                    str(ffmpeg),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    (
                        f"anoisesrc=r=16000:c={args.noise_color}:"
                        f"s={case_seed}:d={duration:.9f}"
                    ),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_f32le",
                    str(noise_path),
                ],
                check=True,
            )
            source_mean_db, source_max_db = volume_levels(source_path, ffmpeg)
            noise_mean_db, noise_max_db = volume_levels(noise_path, ffmpeg)
            unattenuated_noise_gain_db = (
                source_mean_db - args.snr_db - noise_mean_db
            )
            combined_peak_amplitude = (
                10 ** (source_max_db / 20)
                + 10 ** (
                    (noise_max_db + unattenuated_noise_gain_db) / 20
                )
            )
            headroom_attenuation_db = (
                -1.0 - 20 * math.log10(combined_peak_amplitude)
            )
            effective_source_attenuation_db = min(
                args.source_attenuation_db,
                headroom_attenuation_db,
            )
            target_noise_mean_db = (
                source_mean_db
                + effective_source_attenuation_db
                - args.snr_db
            )
            noise_gain_db = target_noise_mean_db - noise_mean_db
            case_digest = hashlib.sha256(row["caseID"].encode()).hexdigest()[:12]
            output_name = (
                f"{source_path.stem}-{case_digest}-"
                f"{args.noise_color}-{snr_label}db.wav"
            )
            output_path = output_audio / output_name
            subprocess.run(
                [
                    str(ffmpeg),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_path),
                    "-i",
                    str(noise_path),
                    "-filter_complex",
                    (
                        "[0:a]volume="
                        f"{effective_source_attenuation_db:.9f}dB[speech];"
                        f"[1:a]volume={noise_gain_db:.9f}dB[noise];"
                        "[speech][noise]amix=inputs=2:duration=first:"
                        "dropout_transition=0:normalize=0[out]"
                    ),
                    "-map",
                    "[out]",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_f32le",
                    str(output_path),
                ],
                check=True,
            )
            output_mean_db, output_max_db = volume_levels(output_path, ffmpeg)
            if output_max_db >= 0:
                raise SystemExit(
                    f"{row['caseID']} noisy fixture clips at {output_max_db} dB"
                )
            output_duration = audio_duration(output_path, ffprobe)
            if abs(output_duration - duration) > 1 / 16_000:
                raise SystemExit(
                    f"{row['caseID']} duration changed from {duration} "
                    f"to {output_duration}"
                )

            derived = dict(row)
            derived.update(
                {
                    "caseID": (
                        f"{row['caseID']}:{args.noise_color}-noise-"
                        f"{snr_label}db"
                    ),
                    "audio": f"audio/{output_name}",
                    "audioSha256": sha256_file(output_path),
                    "derivedFromCaseID": row["caseID"],
                    "sourceAudioSha256": actual_sha256,
                    "noiseLicense": "CC0-1.0",
                    "noiseProvenance": (
                        "Deterministic synthetic signal generated with "
                        "FFmpeg anoisesrc; no external recording"
                    ),
                    "noiseTransform": {
                        "format": "mimi-additive-noise-v1",
                        "toolchain": {
                            "ffmpeg": ffmpeg_build,
                            "ffprobe": ffprobe_build,
                        },
                        "noiseColor": args.noise_color,
                        "noiseSeed": case_seed,
                        "seedNamespace": args.seed,
                        "targetSNRDB": args.snr_db,
                        "requestedSourceAttenuationDB": (
                            args.source_attenuation_db
                        ),
                        "sourceAttenuationDB": (
                            effective_source_attenuation_db
                        ),
                        "sourceMeanDB": source_mean_db,
                        "sourceMaxDB": source_max_db,
                        "noiseMeanDBBeforeGain": noise_mean_db,
                        "noiseMaxDBBeforeGain": noise_max_db,
                        "noiseGainDB": noise_gain_db,
                        "expectedSNRDB": (
                            source_mean_db
                            + effective_source_attenuation_db
                            - (noise_mean_db + noise_gain_db)
                        ),
                        "mixedMeanDB": output_mean_db,
                        "mixedMaxDB": output_max_db,
                    },
                }
            )
            manifest_rows.append(derived)

    manifest_path = args.output_directory / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in manifest_rows
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "caseCount": len(manifest_rows),
                "noiseColor": args.noise_color,
                "targetSNRDB": args.snr_db,
                "manifestSha256": sha256_file(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
