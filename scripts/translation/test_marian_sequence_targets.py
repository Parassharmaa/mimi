#!/usr/bin/env python3
"""Offline contracts for source-only Marian sequence-distillation datasets."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from distill_marian_sequence_targets import (
    load_authenticated_dataset,
    materialize_dataset,
    text_sha256,
    unique_training_sources,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def row(identifier: str, source: str, target: str, original_id: str | None = None) -> dict:
    value = {
        "id": identifier,
        "source": source,
        "target": target,
        "source_language": "en-US",
        "target_language": "ja-JP",
        "source_license": "CC-BY-SA-3.0",
        "source_provenance": "fixture",
        "attribution": "fixture",
        "origin": "human-kftt-replay",
        "domain": "wikipedia",
    }
    if original_id is not None:
        value["original_id"] = original_id
    return value


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-sequence-targets-") as temporary:
        directory = Path(temporary)
        train = [
            row("a", "Source A", "Reference A", "original-a"),
            row("a-repeat", "Source A", "Reference A", "original-a"),
            row("b", "Source B", "Reference B"),
        ]
        valid = [row("valid", "Validation", "Validation reference")]
        write_jsonl(directory / "train.jsonl", train)
        write_jsonl(directory / "valid.jsonl", valid)
        manifest = {
            "direction": "en-ja",
            "outputs": {
                "train": {"sha256": sha256(directory / "train.jsonl")},
                "valid": {"sha256": sha256(directory / "valid.jsonl")},
            },
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )
        loaded_train, loaded_valid, _, _ = load_authenticated_dataset(
            directory, "en-ja"
        )
        sources = unique_training_sources(loaded_train)
        assert sources == [
            {"id": "b", "source": "Source B"},
            {"id": "original-a", "source": "Source A"},
        ]
        output_train, output_valid = materialize_dataset(
            loaded_train,
            loaded_valid,
            {"original-a": "Teacher A", "b": "Teacher B"},
            "teacher@revision",
        )
        assert [item["target"] for item in output_train] == [
            "Teacher A",
            "Teacher A",
            "Teacher B",
        ]
        assert output_train[0]["reference_target_sha256"] == text_sha256(
            "Reference A"
        )
        assert output_train[0]["target_source"] == (
            "marian-source-only-sequence-distillation"
        )
        assert output_valid == valid
        filtered_train, _ = materialize_dataset(
            loaded_train,
            loaded_valid,
            {"b": "Teacher B"},
            "teacher@revision",
            {"original-a"},
        )
        assert [item["id"] for item in filtered_train] == ["b"]
        inconsistent = [
            row("a", "Source A", "Reference A", "same"),
            row("b", "Different source", "Reference B", "same"),
        ]
        try:
            unique_training_sources(inconsistent)
        except SystemExit as error:
            assert "inconsistent source" in str(error)
        else:
            raise AssertionError("inconsistent repeated sources must fail")

    print("Source-only Marian sequence-target contracts passed.")


if __name__ == "__main__":
    main()
