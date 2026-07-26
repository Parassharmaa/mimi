#!/usr/bin/env python3
"""Concatenate a registered speech suite into one reproducible long-form case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dynamic-normalize", action="store_true")
    parser.add_argument("--inter-clip-silence", type=float, default=0)
    args = parser.parse_args()

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
    if not math.isfinite(args.inter_clip_silence) or args.inter_clip_silence < 0:
        raise SystemExit("--inter-clip-silence must be a nonnegative number")

    source_root = args.suite.parent
    audio_paths = [(source_root / row["audio"]).resolve() for row in rows]
    missing = [str(path) for path in audio_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing source audio: {', '.join(missing)}")
    for row, path in zip(rows, audio_paths, strict=True):
        expected_sha256 = row.get("audioSha256")
        if not expected_sha256:
            raise SystemExit(f"{row['caseID']} does not declare audioSha256")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"{row['caseID']} audio hash mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    suffix_parts: list[str] = []
    if args.dynamic_normalize:
        suffix_parts.append("normalized")
    if args.inter_clip_silence > 0:
        silence_label = f"{args.inter_clip_silence:g}".replace(".", "_")
        suffix_parts.append(f"silence-{silence_label}s")
    suffix = f"-{'-'.join(suffix_parts)}" if suffix_parts else ""
    audio_output = args.output_directory / f"long-form{suffix}.wav"
    with tempfile.TemporaryDirectory(prefix="mimi-long-form-") as temporary:
        temporary_directory = Path(temporary)
        silence_path: Path | None = None
        if args.inter_clip_silence > 0:
            silence_path = temporary_directory / "silence.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=16000:cl=mono",
                    "-t",
                    str(args.inter_clip_silence),
                    "-c:a",
                    "pcm_f32le",
                    str(silence_path),
                ],
                check=True,
            )

        concat_path = temporary_directory / "concat.txt"
        concat_lines: list[str] = []
        for index, path in enumerate(audio_paths):
            escaped = str(path).replace("'", r"'\''")
            concat_lines.append(f"file '{escaped}'")
            if silence_path is not None and index < len(audio_paths) - 1:
                concat_lines.append(f"file '{silence_path}'")
        concat_path.write_text(
            "\n".join(concat_lines) + "\n",
            encoding="utf-8",
        )
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
        ]
        if args.dynamic_normalize:
            command.extend(
                [
                    "-af",
                    "dynaudnorm=f=500:g=15:p=0.95",
                ]
            )
        command.extend(
            [
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_f32le",
                str(audio_output),
            ]
        )
        subprocess.run(
            command,
            check=True,
        )

    digest = sha256_file(audio_output)
    licenses = sorted({str(row["license"]) for row in rows})
    long_row = {
        "caseID": (
            f"long-form:{args.suite.parent.name}:{len(rows)}"
            f"{':' + ':'.join(suffix_parts) if suffix_parts else ''}"
        ),
        "sourceCaseIDs": [row["caseID"] for row in rows],
        "dataset": "concatenated registered speech screen",
        "datasetRevision": rows[0].get("datasetRevision"),
        "license": licenses[0] if len(licenses) == 1 else licenses,
        "audio": audio_output.name,
        "audioSha256": digest,
        "reference": " ".join(row["reference"] for row in rows),
        "dynamicNormalization": args.dynamic_normalize,
        "interClipSilenceSeconds": args.inter_clip_silence,
    }
    manifest_output = args.output_directory / f"manifest{suffix}.jsonl"
    manifest_output.write_text(
        json.dumps(long_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audio": str(audio_output),
                "manifest": str(manifest_output),
                "sourceCaseCount": len(rows),
                "audioSha256": digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
