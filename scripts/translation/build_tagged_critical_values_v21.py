#!/usr/bin/env python3
"""Build V21's unambiguous source-bound value-tag and plain-replay dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from typed_critical_token_policy import (
    EN_SCALES,
    EN_SMALL,
    EN_TENS,
    EN_TOKEN_RE,
    JA_ERA_RE,
    JA_NUMBER_RE,
    PROTECTED_RE,
    decimal_text,
    decimal_value,
    english_word_value,
    is_unambiguous_japanese_number,
    japanese_number_value,
    joins_number_phrase,
    normalize,
    typed_signature,
)


LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
TAG_COUNT = 32
TAGS = tuple(f"<v{index:02d}>" for index in range(TAG_COUNT))


@dataclass(frozen=True)
class ValueSpan:
    start: int
    end: int
    canonical: str
    surface: str
    kind: str


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and other_start < end for other_start, other_end in occupied)


def english_number_spans(value: str, occupied: list[tuple[int, int]]) -> list[ValueSpan]:
    matches = list(EN_TOKEN_RE.finditer(value))
    output: list[ValueSpan] = []
    index = 0
    while index < len(matches):
        match = matches[index]
        if overlaps(match.span(), occupied):
            index += 1
            continue
        token = match.group(0).lower()
        direct = decimal_value(token) if token[0].isdigit() else None
        if direct is not None:
            end = match.end()
            if (
                index + 1 < len(matches)
                and matches[index + 1].group(0).lower() in EN_SCALES
                and joins_number_phrase(value, match, matches[index + 1])
                and not overlaps(matches[index + 1].span(), occupied)
            ):
                direct *= EN_SCALES[matches[index + 1].group(0).lower()]
                end = matches[index + 1].end()
                index += 1
            output.append(
                ValueSpan(
                    start=match.start(),
                    end=end,
                    canonical=f"number:{decimal_text(direct)}",
                    surface=value[match.start() : end],
                    kind="number",
                )
            )
            index += 1
            continue
        if token[0].isdigit():
            index += 1
            continue
        components = token.split("-")
        if not all(
            component in EN_SMALL
            or component in EN_TENS
            or component in EN_SCALES
            or component == "and"
            for component in components
        ):
            index += 1
            continue
        phrase = list(components)
        cursor = index + 1
        end = match.end()
        while cursor < len(matches):
            if (
                overlaps(matches[cursor].span(), occupied)
                or not joins_number_phrase(value, matches[cursor - 1], matches[cursor])
            ):
                break
            following = matches[cursor].group(0).lower().split("-")
            if not all(
                component in EN_SMALL
                or component in EN_TENS
                or component in EN_SCALES
                or component == "and"
                for component in following
            ):
                break
            phrase.extend(following)
            end = matches[cursor].end()
            cursor += 1
        parsed = english_word_value(phrase)
        if parsed is not None:
            output.append(
                ValueSpan(
                    start=match.start(),
                    end=end,
                    canonical=f"number:{parsed}",
                    surface=value[match.start() : end],
                    kind="number",
                )
            )
            index = cursor
        else:
            index += 1
    return output


def japanese_number_spans(value: str, occupied: list[tuple[int, int]]) -> list[ValueSpan]:
    output: list[ValueSpan] = []
    era_occupied: list[tuple[int, int]] = []
    for match in JA_ERA_RE.finditer(value):
        if overlaps(match.span(), occupied):
            continue
        year = japanese_number_value(match.group(2))
        if year is None:
            continue
        era_base = {
            "明治": 1867,
            "大正": 1911,
            "昭和": 1925,
            "平成": 1988,
            "令和": 2018,
        }[match.group(1)]
        output.append(
            ValueSpan(
                start=match.start(),
                end=match.end(),
                canonical=f"number:{decimal_text(Decimal(era_base) + year)}",
                surface=match.group(0),
                kind="era-year",
            )
        )
        era_occupied.append(match.span())
    all_occupied = occupied + era_occupied
    for match in JA_NUMBER_RE.finditer(value):
        if overlaps(match.span(), all_occupied):
            continue
        token = match.group(0)
        if not is_unambiguous_japanese_number(token, value, match.start(), match.end()):
            continue
        parsed = japanese_number_value(token)
        if parsed is None:
            continue
        output.append(
            ValueSpan(
                start=match.start(),
                end=match.end(),
                canonical=f"number:{decimal_text(parsed)}",
                surface=token,
                kind="number",
            )
        )
    return output


def value_spans(value: str, language: str) -> list[ValueSpan]:
    normalized = normalize(value)
    protected = [
        ValueSpan(
            start=match.start(),
            end=match.end(),
            canonical=f"protected:{match.group(0)}",
            surface=match.group(0),
            kind="protected",
        )
        for match in PROTECTED_RE.finditer(normalized)
    ]
    occupied = [(span.start, span.end) for span in protected]
    numbers = (
        english_number_spans(normalized, occupied)
        if language == "en-US"
        else japanese_number_spans(normalized, occupied)
        if language == "ja-JP"
        else None
    )
    if numbers is None:
        raise ValueError(f"unsupported language: {language}")
    return sorted((*protected, *numbers), key=lambda span: (span.start, span.end))


def unique_by_canonical(spans: list[ValueSpan]) -> dict[str, ValueSpan] | None:
    counts = Counter(span.canonical for span in spans)
    if any(count != 1 for count in counts.values()):
        return None
    return {span.canonical: span for span in spans}


def replace_spans(value: str, replacements: list[tuple[ValueSpan, str]]) -> str:
    normalized = normalize(value)
    output = normalized
    for span, replacement in sorted(replacements, key=lambda item: item[0].start, reverse=True):
        output = output[: span.start] + replacement + output[span.end :]
    return output


def tagged_row(row: dict[str, Any], direction: str) -> dict[str, Any] | None:
    source_language, target_language = LANGUAGES[direction]
    source = str(row["source"])
    target = str(row["target"])
    source_signature = typed_signature(source, source_language)
    target_signature = typed_signature(target, target_language)
    if (
        source_signature != target_signature
        or source_signature.opaque_numbers
        or not (
            source_signature.protected
            or source_signature.numbers
            or source_signature.percentages
        )
    ):
        return None
    source_values = unique_by_canonical(value_spans(source, source_language))
    target_values = unique_by_canonical(value_spans(target, target_language))
    if (
        source_values is None
        or target_values is None
        or not source_values
        or set(source_values) != set(target_values)
        or len(source_values) > TAG_COUNT
    ):
        return None

    ordered_keys = [
        span.canonical
        for span in sorted(source_values.values(), key=lambda span: span.start)
    ]
    tag_by_key = {key: TAGS[index] for index, key in enumerate(ordered_keys)}
    source_replacements = [
        (span, tag_by_key[key]) for key, span in source_values.items()
    ]
    target_replacements = [
        (span, tag_by_key[key]) for key, span in target_values.items()
    ]
    sidecar = []
    for key in ordered_keys:
        source_span = source_values[key]
        target_span = target_values[key]
        sidecar.append(
            {
                "tag": tag_by_key[key],
                "canonical": key,
                "kind": source_span.kind,
                "source_surface": source_span.surface,
                "target_surface": target_span.surface,
                "source_has_ascii_digits": any(
                    character.isascii() and character.isdigit()
                    for character in source_span.surface
                ),
                "target_has_ascii_digits": any(
                    character.isascii() and character.isdigit()
                    for character in target_span.surface
                ),
            }
        )
    return {
        **row,
        "source": replace_spans(source, source_replacements),
        "target": replace_spans(target, target_replacements),
        "original_source": source,
        "original_target": target,
        "source_origin": row.get("source_origin", row.get("origin")),
        "origin": "tagged-critical-value-v21",
        "training_arm": "tagged-focus",
        "value_sidecar": sidecar,
        "tag_count": len(sidecar),
        "text_derivation": (
            "replace each unique source-target aligned protected value with the "
            "same atomic tag; preserve both original surfaces in value_sidecar"
        ),
    }


def deterministic_order(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{row['id']}".encode("utf-8")
        ).hexdigest(),
    )


def validate_parent(parent: Path, direction: str) -> tuple[dict, list[dict], list[dict]]:
    manifest_path = parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment") != "typed-numeric-preservation-v20-curriculum"
        or manifest.get("direction") != direction
        or manifest.get("promotion_eligible") is not False
        or manifest.get("does_not_authorize_app_integration") is not True
        or manifest.get("does_not_authorize_public_upload") is not True
    ):
        raise SystemExit("V20 parent manifest is invalid or authorizes promotion")
    rows = {}
    for split in ("train", "valid"):
        path = parent / f"{split}.jsonl"
        record = manifest.get("outputs", {}).get(split, {})
        if (
            record.get("sha256") != sha256(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise SystemExit(f"V20 parent manifest does not authenticate {split}")
        rows[split] = load_jsonl(path)
    return manifest, rows["train"], rows["valid"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--tagged-train-rows", type=int, default=4096)
    parser.add_argument("--plain-replay-rows", type=int, default=4096)
    parser.add_argument("--seed", default="mimi-v21-tagged-critical-values")
    args = parser.parse_args()

    if min(args.tagged_train_rows, args.plain_replay_rows) < 1:
        raise SystemExit("V21 train row counts must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    parent_manifest, parent_train, parent_valid = validate_parent(
        args.parent,
        args.direction,
    )
    eligible_train = [
        tagged
        for row in parent_train
        if row.get("origin") == "typed-numeric-target"
        and (tagged := tagged_row(row, args.direction)) is not None
    ]
    tagged_valid = [
        tagged
        for row in parent_valid
        if row.get("origin") == "typed-numeric-target"
        and (tagged := tagged_row(row, args.direction)) is not None
    ]
    replay_candidates = [
        {
            **row,
            "source_origin": row.get("source_origin", row.get("origin")),
            "origin": "plain-preservation-replay-v21",
            "training_arm": "plain-replay",
            "value_sidecar": [],
            "tag_count": 0,
        }
        for row in parent_train
        if row.get("origin") == "base-preservation-replay"
    ]
    if len(eligible_train) < args.tagged_train_rows:
        raise SystemExit(
            f"only {len(eligible_train)} unambiguous tagged train rows, "
            f"need {args.tagged_train_rows}"
        )
    if len(replay_candidates) < args.plain_replay_rows:
        raise SystemExit(
            f"only {len(replay_candidates)} plain replay rows, "
            f"need {args.plain_replay_rows}"
        )

    tagged_train = deterministic_order(eligible_train, args.seed)[
        : args.tagged_train_rows
    ]
    plain_replay = deterministic_order(replay_candidates, f"{args.seed}-replay")[
        : args.plain_replay_rows
    ]
    combined_train = deterministic_order(
        [*tagged_train, *plain_replay],
        f"{args.seed}-combined",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": (args.output / "train.jsonl", combined_train),
        "tagged_valid": (args.output / "tagged-valid.jsonl", tagged_valid),
        "plain_valid": (args.output / "plain-valid.jsonl", parent_valid),
    }
    for _, (path, rows) in outputs.items():
        write_jsonl(path, rows)

    tag_count_distribution = Counter(
        row["tag_count"] for row in [*tagged_train, *tagged_valid]
    )
    manifest = {
        "schema_version": 1,
        "experiment": "tagged-critical-values-v21-dataset",
        "direction": args.direction,
        "status": "feasibility-dataset-only",
        "seed": args.seed,
        "tags": list(TAGS),
        "tag_count": TAG_COUNT,
        "selection": {
            "alignment": (
                "typed signatures equal, no opaque numbers, each canonical value "
                "occurs exactly once in source and target, maximum 32 values"
            ),
            "eligible_tagged_train": len(eligible_train),
            "selected_tagged_train": len(tagged_train),
            "selected_plain_replay": len(plain_replay),
            "tagged_valid": len(tagged_valid),
            "plain_valid": len(parent_valid),
            "tag_count_distribution": {
                str(key): value
                for key, value in sorted(tag_count_distribution.items())
            },
        },
        "parent": {
            "directory": str(args.parent),
            "manifest": {
                "path": str(args.parent / "manifest.json"),
                "sha256": sha256(args.parent / "manifest.json"),
            },
            "contamination_screen": parent_manifest.get("contamination_screen"),
        },
        "licenses": parent_manifest.get("effective_licenses"),
        "target_source": "licensed human references from authenticated V20 parent",
        "synthetic_translations": 0,
        "teacher_outputs": 0,
        "private_reasoning_traces_used": False,
        "outputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows": len(rows),
            }
            for name, (path, rows) in outputs.items()
        },
        "does_not_authorize_training": True,
        "does_not_authorize_protected_evaluation": True,
        "does_not_authorize_app_integration": True,
        "does_not_authorize_public_upload": True,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "direction": args.direction,
                "eligibleTaggedTrain": len(eligible_train),
                "selectedTaggedTrain": len(tagged_train),
                "selectedPlainReplay": len(plain_replay),
                "taggedValid": len(tagged_valid),
                "manifestSHA256": sha256(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
