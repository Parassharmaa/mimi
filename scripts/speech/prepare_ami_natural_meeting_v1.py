#!/usr/bin/env python3
"""Build Mimi's pinned natural English AMI meeting benchmark cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import urllib.request
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


CORPUS_URL = "https://groups.inf.ed.ac.uk/ami/corpus/"
DOWNLOAD_URL = "https://groups.inf.ed.ac.uk/ami/download/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ANNOTATION_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
    "ami_public_manual_1.6.2.zip"
)
ANNOTATION_SHA256 = (
    "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"
)
MEETING_ID = "EN2001a"
MEETING_TYPE = "nonscenario"
MEETING_METADATA_MEMBER = "corpusResources/meetings.xml"
REFERENCE_START_SECONDS = 5.57
REFERENCE_END_SECONDS = 306.14
BOUNDARY_PADDING_SECONDS = 1.0
SELECTION_START_SECONDS = REFERENCE_START_SECONDS - BOUNDARY_PADDING_SECONDS
SELECTION_END_SECONDS = REFERENCE_END_SECONDS + BOUNDARY_PADDING_SECONDS
SOURCE_SAMPLE_RATE = 16_000
SOURCE_SAMPLE_WIDTH = 2
SIGNALS = {
    "headset": {
        "member": "EN2001a.Mix-Headset.wav",
        "description": "close-talking headset microphone mix",
        "sha256": (
            "81e06be816e9d94d0bee410bef1b158b2cdfba8e2b80f44a6e62cf6b9fd780f9"
        ),
    },
    "array1-01": {
        "member": "EN2001a.Array1-01.wav",
        "description": "far-field table microphone array 1 channel 01",
        "sha256": (
            "20cf2243fff84361cc5c755529350d23c544580d58f00959e71945d8fdc8a8f7"
        ),
    },
}
SIGNAL_BASE_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
    f"{MEETING_ID}/audio"
)


@dataclass(frozen=True)
class AMIWord:
    identifier: str
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def parse_meeting_metadata(payload: bytes) -> dict[str, object]:
    root = ElementTree.fromstring(payload)
    for element in root.iter():
        if (
            local_name(element.tag) == "meeting"
            and element.attrib.get("observation") == MEETING_ID
        ):
            speakers = [
                child.attrib["nxt_agent"]
                for child in element
                if local_name(child.tag) == "speaker"
            ]
            return {
                "type": element.attrib["type"],
                "name": element.attrib["name"],
                "topic": element.attrib["topic"],
                "durationSeconds": float(element.attrib["duration"]),
                "speakers": speakers,
            }
    raise ValueError(f"{MEETING_ID} is missing from {MEETING_METADATA_MEMBER}")


def parse_word_member(
    payload: bytes,
    speaker: str,
) -> list[AMIWord]:
    root = ElementTree.fromstring(payload)
    words: list[AMIWord] = []
    for element in root:
        if local_name(element.tag) != "w":
            continue
        if element.attrib.get("punc") == "true":
            continue
        text = (element.text or "").strip()
        start_text = element.attrib.get("starttime")
        end_text = element.attrib.get("endtime")
        if not text or start_text is None or end_text is None:
            raise ValueError(f"untimed lexical word in {MEETING_ID}.{speaker}")
        start = float(start_text)
        end = float(end_text)
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end < start
        ):
            raise ValueError(f"invalid lexical timing in {MEETING_ID}.{speaker}")
        words.append(
            AMIWord(
                identifier=element.attrib[
                    "{http://nite.sourceforge.net/}id"
                ],
                speaker=speaker,
                start_seconds=start,
                end_seconds=end,
                text=text,
            )
        )
    return words


def registered_words(
    archive: zipfile.ZipFile,
    speakers: list[str],
) -> tuple[list[AMIWord], dict[str, str]]:
    words: list[AMIWord] = []
    member_hashes: dict[str, str] = {}
    for speaker in sorted(speakers):
        member = f"words/{MEETING_ID}.{speaker}.words.xml"
        payload = archive.read(member)
        member_hashes[member] = sha256_bytes(payload)
        words.extend(parse_word_member(payload, speaker))
    selected = [
        word
        for word in words
        if word.start_seconds >= REFERENCE_START_SECONDS
        and word.end_seconds <= REFERENCE_END_SECONDS
    ]
    selected.sort(
        key=lambda word: (
            word.start_seconds,
            word.end_seconds,
            word.speaker,
            word.identifier,
        )
    )
    if not selected:
        raise ValueError("registered AMI word selection is empty")
    return selected, member_hashes


def count_overlapping_words(words: list[AMIWord]) -> int:
    overlapping: set[int] = set()
    for left_index, left in enumerate(words):
        for right_index in range(left_index + 1, len(words)):
            right = words[right_index]
            if right.start_seconds >= left.end_seconds:
                break
            if (
                right.speaker != left.speaker
                and right.end_seconds > left.start_seconds
            ):
                overlapping.add(left_index)
                overlapping.add(right_index)
    return len(overlapping)


def extract_registered_audio(source: Path, output: Path) -> dict[str, object]:
    with wave.open(str(source), "rb") as input_audio:
        parameters = input_audio.getparams()
        if (
            parameters.nchannels != 1
            or parameters.sampwidth != SOURCE_SAMPLE_WIDTH
            or parameters.framerate != SOURCE_SAMPLE_RATE
            or parameters.comptype != "NONE"
        ):
            raise ValueError(
                "registered AMI source must be 16 kHz mono 16-bit PCM"
            )
        start_sample = round(SELECTION_START_SECONDS * SOURCE_SAMPLE_RATE)
        end_sample = round(SELECTION_END_SECONDS * SOURCE_SAMPLE_RATE)
        if end_sample > parameters.nframes:
            raise ValueError("registered AMI selection exceeds source audio")
        input_audio.setpos(start_sample)
        frames = input_audio.readframes(end_sample - start_sample)

    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as output_audio:
        output_audio.setnchannels(1)
        output_audio.setsampwidth(SOURCE_SAMPLE_WIDTH)
        output_audio.setframerate(SOURCE_SAMPLE_RATE)
        output_audio.writeframes(frames)
    return {
        "sourceFrames": parameters.nframes,
        "sourceDurationSeconds": parameters.nframes / SOURCE_SAMPLE_RATE,
        "selectionStartSample": start_sample,
        "selectionEndSample": end_sample,
        "selectionFrames": end_sample - start_sample,
        "selectionDurationSeconds": (
            (end_sample - start_sample) / SOURCE_SAMPLE_RATE
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--cache-directory",
        type=Path,
        help="download cache; defaults beside the ignored output directory",
    )
    parser.add_argument(
        "--condition",
        choices=(*SIGNALS, "all"),
        default="all",
    )
    args = parser.parse_args()

    output_directory = args.output_directory.resolve()
    cache_directory = (
        args.cache_directory.resolve()
        if args.cache_directory is not None
        else output_directory.parent / ".natural-source-cache"
    )
    annotation_archive = download_registered(
        ANNOTATION_URL,
        ANNOTATION_SHA256,
        cache_directory / "ami_public_manual_1.6.2.zip",
    )
    with zipfile.ZipFile(annotation_archive) as archive:
        meeting_metadata_payload = archive.read(MEETING_METADATA_MEMBER)
        meeting_metadata = parse_meeting_metadata(meeting_metadata_payload)
        if meeting_metadata["type"] != MEETING_TYPE:
            raise ValueError(
                f"{MEETING_ID} is not registered as {MEETING_TYPE}"
            )
        words, word_member_hashes = registered_words(
            archive,
            meeting_metadata["speakers"],
        )

    reference = " ".join(word.text for word in words)
    conditions = (
        list(SIGNALS) if args.condition == "all" else [args.condition]
    )
    rows: list[dict] = []
    for condition in conditions:
        signal = SIGNALS[condition]
        member = str(signal["member"])
        source_url = f"{SIGNAL_BASE_URL}/{member}"
        source_path = download_registered(
            source_url,
            str(signal["sha256"]),
            cache_directory / member,
        )
        output_audio = (
            output_directory / f"ami-en2001a-{condition}-natural-v1.wav"
        )
        audio_selection = extract_registered_audio(
            source_path,
            output_audio,
        )
        rows.append(
            {
                "caseID": f"natural-en-v1:ami:{MEETING_ID}:{condition}",
                "dataset": "AMI Meeting Corpus",
                "corpusURL": CORPUS_URL,
                "downloadURL": DOWNLOAD_URL,
                "license": "CC-BY-4.0",
                "licenseURL": LICENSE_URL,
                "attribution": (
                    "AMI Meeting Corpus contributors; five-minute excerpt "
                    "resampled only by exact PCM frame selection for Mimi"
                ),
                "benchmarkUse": (
                    "held-out evaluation only; registered before any later "
                    "Mimi speech-model training"
                ),
                "pretrainingContaminationRisk": (
                    "AMI is a public ASR corpus and may have appeared in "
                    "unknown pretraining corpora; this natural-meeting result "
                    "is reported separately from contamination-screened claims"
                ),
                "meetingID": MEETING_ID,
                "meetingType": meeting_metadata["type"],
                "meetingName": meeting_metadata["name"],
                "meetingTopic": meeting_metadata["topic"],
                "speakerCount": len(meeting_metadata["speakers"]),
                "speakers": sorted(meeting_metadata["speakers"]),
                "signalCondition": condition,
                "signalDescription": signal["description"],
                "sourceAudioURL": source_url,
                "sourceAudioSha256": signal["sha256"],
                "sourceAudioMember": member,
                "sourceAudioSampleRate": SOURCE_SAMPLE_RATE,
                "sourceAnnotationURL": ANNOTATION_URL,
                "sourceAnnotationSha256": ANNOTATION_SHA256,
                "sourceAnnotationDownloadLabel": (
                    "AMI manual annotations v1.6.2"
                ),
                "sourceAnnotationInternalReadmeRelease": "1.7",
                "sourceMeetingMetadataMember": MEETING_METADATA_MEMBER,
                "sourceMeetingMetadataSha256": sha256_bytes(
                    meeting_metadata_payload
                ),
                "sourceWordMemberSha256": word_member_hashes,
                "sourceReferenceStartSeconds": REFERENCE_START_SECONDS,
                "sourceReferenceEndSeconds": REFERENCE_END_SECONDS,
                "sourceSelectionStartSeconds": SELECTION_START_SECONDS,
                "sourceSelectionEndSeconds": SELECTION_END_SECONDS,
                "boundaryPaddingSeconds": BOUNDARY_PADDING_SECONDS,
                "sourceWordIDs": [word.identifier for word in words],
                "referenceWordCount": len(words),
                "referenceOverlappingWordCount": count_overlapping_words(
                    words
                ),
                "referenceSpeakers": sorted(
                    {word.speaker for word in words}
                ),
                "referenceConstruction": (
                    "lexical w elements from all five manual AMI speaker "
                    "transcripts, excluding punctuation, sorted by start time; "
                    "overlapping speech is retained and linearized "
                    "deterministically"
                ),
                "reference": reference,
                "referenceSha256": sha256_bytes(
                    reference.encode("utf-8")
                ),
                "audio": output_audio.name,
                "audioSha256": sha256_file(output_audio),
                "audioSelection": audio_selection,
            }
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    for row in rows:
        condition = row["signalCondition"]
        (output_directory / f"manifest-{condition}.jsonl").write_text(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    combined_manifest = output_directory / "manifest.jsonl"
    combined_manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "conditions": {
                    str(row["signalCondition"]): {
                        "audio": row["audio"],
                        "audioSha256": row["audioSha256"],
                    }
                    for row in rows
                },
                "durationSeconds": rows[0]["audioSelection"][
                    "selectionDurationSeconds"
                ],
                "manifest": str(combined_manifest),
                "overlappingWords": rows[0][
                    "referenceOverlappingWordCount"
                ],
                "referenceWords": len(words),
                "speakers": rows[0]["referenceSpeakers"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
