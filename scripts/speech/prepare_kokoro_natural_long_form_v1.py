#!/usr/bin/env python3
"""Build Mimi's pinned natural Japanese audiobook benchmark case."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


DATASET_REPOSITORY = "https://github.com/kaiidams/Kokoro-Speech-Dataset"
DATASET_REVISION = "88377525a85728c79cf78f6bfcb81396f9c23827"
DATASET_LICENSE_URL = (
    "https://github.com/kaiidams/Kokoro-Speech-Dataset"
    "/blob/88377525a85728c79cf78f6bfcb81396f9c23827/README.md#license"
)
METADATA_URL = (
    "https://github.com/kaiidams/Kokoro-Speech-Dataset"
    "/releases/download/1.3/kokoro-speech-v1_3.zip"
)
METADATA_SHA256 = (
    "5a4a290672016ebe70372ed3d47063f846d86e2b96aa9e9b9d35161670d9f666"
)
METADATA_MEMBER = "gongitsune-by-nankichi-niimi.metadata.txt"
SOURCE_ARCHIVE_URL = (
    "https://archive.org/download/gongitsune_um_librivox/"
    "gongitsune_um_librivox_64kb_mp3.zip"
)
SOURCE_ARCHIVE_SHA256 = (
    "06fb200a3cbe65902b22fbd8febb6869e522b4413c0ffc0bec65d0c6e8983431"
)
SOURCE_AUDIO_MEMBER = "gongitsune_01_niimi_64kb.mp3"
SOURCE_AUDIO_SHA256 = (
    "f2dd16a2e9400d54819f0967ccd77a7948b1437dcf739d702f792a3e933fc141"
)
SOURCE_SAMPLE_RATE = 22_050
FIRST_CASE_ID = "gongitsune-by-nankichi-niimi-00002"
LAST_CASE_ID = "gongitsune-by-nankichi-niimi-00063"
BOUNDARY_PADDING_SAMPLES = SOURCE_SAMPLE_RATE // 2


@dataclass(frozen=True)
class Alignment:
    case_id: str
    audio_member: str
    start_sample: int
    end_sample: int
    text: str
    reading: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_metadata(payload: bytes) -> list[Alignment]:
    rows: list[Alignment] = []
    for line_number, line in enumerate(
        payload.decode("utf-8").splitlines(),
        start=1,
    ):
        fields = line.split("|")
        if len(fields) != 6:
            raise ValueError(
                f"{METADATA_MEMBER}:{line_number} has {len(fields)} fields"
            )
        case_id, audio_member, start, end, text, reading = fields
        start_sample = int(start)
        end_sample = int(end)
        if not case_id or not audio_member or not text:
            raise ValueError(
                f"{METADATA_MEMBER}:{line_number} has an empty required field"
            )
        if start_sample < 0 or end_sample <= start_sample:
            raise ValueError(
                f"{METADATA_MEMBER}:{line_number} has invalid sample bounds"
            )
        rows.append(
            Alignment(
                case_id=case_id,
                audio_member=audio_member,
                start_sample=start_sample,
                end_sample=end_sample,
                text=text,
                reading=reading,
            )
        )
    return rows


def select_registered_rows(rows: list[Alignment]) -> list[Alignment]:
    matching = [
        row for row in rows if row.audio_member == SOURCE_AUDIO_MEMBER
    ]
    identifiers = [row.case_id for row in matching]
    try:
        first = identifiers.index(FIRST_CASE_ID)
        last = identifiers.index(LAST_CASE_ID)
    except ValueError as error:
        raise ValueError("registered Kokoro case IDs are missing") from error
    selected = matching[first : last + 1]
    if not selected:
        raise ValueError("registered Kokoro selection is empty")
    for previous, current in zip(selected, selected[1:]):
        if current.start_sample < previous.end_sample:
            raise ValueError(
                f"overlapping alignments: {previous.case_id}, {current.case_id}"
            )
    return selected


def download_registered(
    url: str,
    expected_sha256: str,
    destination: Path,
) -> Path:
    if destination.is_file():
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise ValueError(
                f"cached source hash mismatch for {destination}: "
                f"expected {expected_sha256}, got {actual}"
            )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        actual = sha256_file(temporary)
        if actual != expected_sha256:
            raise ValueError(
                f"download hash mismatch for {url}: "
                f"expected {expected_sha256}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def executable_identity(name: str) -> dict[str, str]:
    path_text = shutil.which(name)
    if path_text is None:
        raise ValueError(f"{name} is required")
    path = Path(path_text).resolve()
    version = subprocess.run(
        [str(path), "-version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "version": version,
    }


def probe_audio(path: Path, ffprobe: str) -> dict:
    output = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=sample_rate,channels,codec_name",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--cache-directory",
        type=Path,
        help="download cache; defaults beside the ignored output directory",
    )
    args = parser.parse_args()

    output_directory = args.output_directory.resolve()
    cache_directory = (
        args.cache_directory.resolve()
        if args.cache_directory is not None
        else output_directory.parent / ".natural-source-cache"
    )
    metadata_archive = download_registered(
        METADATA_URL,
        METADATA_SHA256,
        cache_directory / "kokoro-speech-v1_3.zip",
    )
    source_archive = download_registered(
        SOURCE_ARCHIVE_URL,
        SOURCE_ARCHIVE_SHA256,
        cache_directory / "gongitsune_um_librivox_64kb_mp3.zip",
    )

    with zipfile.ZipFile(metadata_archive) as archive:
        metadata_payload = archive.read(METADATA_MEMBER)
    rows = select_registered_rows(parse_metadata(metadata_payload))

    with zipfile.ZipFile(source_archive) as archive:
        source_payload = archive.read(SOURCE_AUDIO_MEMBER)
    actual_audio_sha256 = sha256_bytes(source_payload)
    if actual_audio_sha256 != SOURCE_AUDIO_SHA256:
        raise ValueError(
            f"{SOURCE_AUDIO_MEMBER} hash mismatch: "
            f"expected {SOURCE_AUDIO_SHA256}, got {actual_audio_sha256}"
        )

    ffmpeg_identity = executable_identity("ffmpeg")
    ffprobe_identity = executable_identity("ffprobe")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_audio = output_directory / "kokoro-gongitsune-natural-v1.wav"
    with tempfile.TemporaryDirectory(prefix="mimi-kokoro-natural-") as temporary:
        source_path = Path(temporary) / SOURCE_AUDIO_MEMBER
        source_path.write_bytes(source_payload)
        source_probe = probe_audio(source_path, ffprobe_identity["path"])
        streams = source_probe.get("streams", [])
        if len(streams) != 1:
            raise ValueError("registered Kokoro source must have one audio stream")
        stream = streams[0]
        if (
            int(stream["sample_rate"]) != SOURCE_SAMPLE_RATE
            or int(stream["channels"]) != 1
        ):
            raise ValueError(
                "registered Kokoro source must be 22.05 kHz mono audio"
            )

        selection_start = max(
            0,
            rows[0].start_sample - BOUNDARY_PADDING_SAMPLES,
        )
        selection_end = rows[-1].end_sample + BOUNDARY_PADDING_SAMPLES
        subprocess.run(
            [
                ffmpeg_identity["path"],
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-af",
                (
                    f"atrim=start_sample={selection_start}:"
                    f"end_sample={selection_end},"
                    "asetpts=PTS-STARTPTS,aresample=16000"
                ),
                "-ac",
                "1",
                "-c:a",
                "pcm_f32le",
                str(output_audio),
            ],
            check=True,
        )

    output_probe = probe_audio(output_audio, ffprobe_identity["path"])
    reference = " ".join(row.text for row in rows)
    manifest = {
        "caseID": "natural-ja-v1:kokoro:gongitsune:chapter-01-body",
        "dataset": "Kokoro Speech Dataset v1.3 reconstructed source",
        "datasetRepository": DATASET_REPOSITORY,
        "datasetRevision": DATASET_REVISION,
        "datasetLicenseURL": DATASET_LICENSE_URL,
        "license": (
            "Public domain in the USA per the dataset author; "
            "international status requires jurisdiction review"
        ),
        "redistributionStatus": (
            "benchmark audio remains local and uncommitted pending "
            "international distribution review"
        ),
        "benchmarkUse": (
            "held-out evaluation only; registered before any later Mimi "
            "speech-model training"
        ),
        "pretrainingContaminationRisk": (
            "public audiobook and alignments may have appeared in unknown "
            "pretraining corpora; results are reported as natural-speech "
            "stress evidence, not a contamination-free model-quality claim"
        ),
        "sourceBook": "ごん狐 (Gon gitsune) by Nankichi Niimi",
        "sourceReader": "ekzemplaro (LibriVox reader 7044)",
        "sourceMetadataURL": METADATA_URL,
        "sourceMetadataArchiveSha256": METADATA_SHA256,
        "sourceMetadataMember": METADATA_MEMBER,
        "sourceMetadataMemberSha256": sha256_bytes(metadata_payload),
        "sourceArchiveURL": SOURCE_ARCHIVE_URL,
        "sourceArchiveSha256": SOURCE_ARCHIVE_SHA256,
        "sourceAudioMember": SOURCE_AUDIO_MEMBER,
        "sourceAudioSha256": SOURCE_AUDIO_SHA256,
        "sourceSampleRate": SOURCE_SAMPLE_RATE,
        "sourceCaseIDs": [row.case_id for row in rows],
        "sourceAlignmentStartSample": rows[0].start_sample,
        "sourceAlignmentEndSample": rows[-1].end_sample,
        "sourceSelectionStartSample": selection_start,
        "sourceSelectionEndSample": selection_end,
        "boundaryPaddingSamples": BOUNDARY_PADDING_SAMPLES,
        "audio": output_audio.name,
        "audioSha256": sha256_file(output_audio),
        "audioProbe": output_probe,
        "reference": reference,
        "referenceSha256": sha256_bytes(reference.encode("utf-8")),
        "referenceConstruction": (
            "ordered Kokoro v1.3 aligned text rows 00002 through 00063; "
            "spacing is preserved and ignored by Mimi's Japanese CER"
        ),
        "toolchain": {
            "ffmpeg": ffmpeg_identity,
            "ffprobe": ffprobe_identity,
        },
    }
    manifest_path = output_directory / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audio": str(output_audio),
                "audioSha256": manifest["audioSha256"],
                "durationSeconds": float(
                    output_probe["format"]["duration"]
                ),
                "manifest": str(manifest_path),
                "referenceCharacters": len(reference),
                "sourceRows": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
