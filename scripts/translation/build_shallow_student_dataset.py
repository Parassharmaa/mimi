#!/usr/bin/env python3
"""Build a large, domain-balanced licensed dataset for shallow Marian recovery."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator


DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
CORPUS_METADATA = {
    "kftt": ("wikipedia", "human-kftt-replay"),
    "alt": ("human-translated-news", "human-alt-parallel"),
    "tatoeba": (
        "conversational",
        "human-tatoeba-bidirectional-agreement-filtered",
    ),
    "jlt": (
        "ministry-published-legal",
        "finalized-japanese-law-translation",
    ),
    "ui": ("mimi-product-ui", "mimi-shipped-ui-pair"),
}
ALLOWED_LICENSES = {
    "CC-BY-2.0-FR",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "PDL-1.0-compatible-CC-BY-4.0",
    "project-owned",
}
NOISY_MARKUP = re.compile(r"@[-,.]@|<[^>]+>|&(?:quot|amp|lt|gt);", re.I)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SystemExit(f"expected JSON object at {path}:{line_number}")
            yield value


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text)
    if not value:
        return set()
    if len(value) < size:
        return {value}
    return {value[index : index + size] for index in range(len(value) - size + 1)}


class NgramIndex:
    def __init__(self, texts: list[str]) -> None:
        self.values = [ngrams(text) for text in texts if text.strip()]
        postings: dict[str, list[int]] = defaultdict(list)
        for index, grams in enumerate(self.values):
            for gram in grams:
                postings[gram].append(index)
        self.postings = dict(postings)

    def matches(self, text: str, maximum: float) -> bool:
        candidate = ngrams(text)
        intersections: Counter[int] = Counter()
        for gram in candidate:
            intersections.update(self.postings.get(gram, ()))
        return any(
            overlap / max(1, len(candidate) + len(self.values[index]) - overlap)
            > maximum
            for index, overlap in intersections.items()
        )


def row_texts(row: dict) -> tuple[str, str]:
    if "messages" not in row:
        return str(row.get("source", "")), str(row.get("target", ""))
    messages = row.get("messages", [])
    if len(messages) != 3 or [message.get("role") for message in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise SystemExit("chat corpus row has an unexpected message shape")
    return str(messages[1].get("content", "")), str(messages[2].get("content", ""))


def raw_identity(row: dict) -> tuple[str, str]:
    if "messages" in row:
        metadata = row.get("metadata", {})
        return str(metadata.get("direction", "")), str(metadata.get("source_id", ""))
    languages = (row.get("source_language"), row.get("target_language"))
    direction = next(
        (name for name, expected in DIRECTIONS.items() if languages == expected),
        "",
    )
    return direction, str(row.get("source_id", ""))


def parse_parallel_row(row: dict, corpus: str) -> dict:
    if "messages" in row:
        metadata = row.get("metadata", {})
        direction = str(metadata.get("direction", ""))
        source, target = row_texts(row)
        source_id = str(metadata.get("source_id", ""))
        license_name = str(metadata.get("license", ""))
        attribution = str(metadata.get("attribution", ""))
        provenance = f"{metadata.get('source', corpus)} / {source_id} / {attribution}"
    else:
        languages = (row.get("source_language"), row.get("target_language"))
        direction = next(
            (name for name, expected in DIRECTIONS.items() if languages == expected),
            "",
        )
        source, target = row_texts(row)
        source_id = str(row.get("source_id", ""))
        license_name = str(row.get("source_license", ""))
        attribution = str(row.get("attribution", ""))
        provenance = str(row.get("source_provenance", "")) or attribution
    row_identity = str(row.get("id", "")) if "messages" not in row else ""
    row_identity = row_identity or source_id
    if direction not in DIRECTIONS:
        raise SystemExit(f"unsupported direction in {corpus}: {source_id}")
    if not source_id or not source.strip() or not target.strip():
        raise SystemExit(f"incomplete parallel row in {corpus}: {source_id}")
    if license_name not in ALLOWED_LICENSES:
        raise SystemExit(f"unapproved license in {corpus}: {source_id} / {license_name}")
    if not provenance.strip():
        raise SystemExit(f"missing attribution/provenance in {corpus}: {source_id}")
    if not attribution.strip():
        if license_name != "project-owned":
            raise SystemExit(f"missing attribution/provenance in {corpus}: {source_id}")
        attribution = f"Mimi project-owned shipped copy / {provenance.strip()}"
    domain, origin = CORPUS_METADATA[corpus]
    source_language, target_language = DIRECTIONS[direction]
    return {
        "id": f"shallow-training:{corpus}:{row_identity}:{direction}",
        "source_id": source_id,
        "source": source.strip(),
        "target": target.strip(),
        "source_language": source_language,
        "target_language": target_language,
        "source_license": license_name,
        "source_provenance": provenance.strip(),
        "attribution": attribution.strip(),
        "domain": domain,
        "origin": origin,
    }


def protected_texts(paths: list[Path]) -> list[str]:
    values: list[str] = []
    for path in paths:
        for row in iter_jsonl(path):
            values.append(str(row.get("source", "")))
            values.extend(str(reference) for reference in row.get("references", []))
    if not any(value.strip() for value in values):
        raise SystemExit("protected suites contain no text")
    return values


def deterministic_rank(seed: str, corpus: str, row_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{corpus}\0{row_id}".encode()).digest()
    return int.from_bytes(digest, "big")


def top_ranked(rows: Iterator[dict], count: int, seed: str, corpus: str) -> list[dict]:
    heap: list[tuple[int, str, dict]] = []
    for row in rows:
        rank = deterministic_rank(seed, corpus, row["id"])
        item = (-rank, row["id"], row)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]


def load_validation(path: Path, direction: str) -> tuple[list[dict], dict]:
    manifest_path = path / "manifest.json"
    valid_path = path / "valid.jsonl"
    if not manifest_path.is_file() or not valid_path.is_file():
        raise SystemExit("validation dataset requires manifest.json and valid.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("outputs", {}).get("valid", {}).get("sha256")
    if expected_hash != sha256(valid_path):
        raise SystemExit("validation manifest does not authenticate valid.jsonl")
    expected_languages = DIRECTIONS[direction]
    rows = list(iter_jsonl(valid_path))
    identifiers = [str(row.get("id", "")) for row in rows]
    if (
        not rows
        or not all(identifiers)
        or len(identifiers) != len(set(identifiers))
        or any(
            (row.get("source_language"), row.get("target_language"))
            != expected_languages
            for row in rows
        )
    ):
        raise SystemExit("validation dataset has invalid IDs or direction")
    return rows, {
        "directory": str(path),
        "manifest_sha256": sha256(manifest_path),
        "valid_sha256": sha256(valid_path),
        "valid_rows": len(rows),
    }


def repeat_rows(rows: list[dict], repeats: int, seed: str, corpus: str) -> list[dict]:
    output: list[dict] = []
    for repeat_index in range(repeats):
        ordered = sorted(
            rows,
            key=lambda row: deterministic_rank(
                f"{seed}:repeat:{repeat_index}", corpus, row["id"]
            ),
        )
        for row in ordered:
            identifier = row["id"]
            if repeat_index:
                identifier += f":repeat-{repeat_index}"
            output.append(
                {
                    **row,
                    "id": identifier,
                    "original_id": row["id"],
                    "training_repeat_index": repeat_index,
                }
            )
    return output


def interleave(groups: dict[str, list[dict]]) -> list[dict]:
    output: list[dict] = []
    maximum = max(len(rows) for rows in groups.values())
    for index in range(maximum):
        for corpus in CORPUS_METADATA:
            rows = groups.get(corpus, [])
            if index < len(rows):
                output.append(rows[index])
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def keyed_integers(values: list[str], allowed: set[str], label: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for value in values:
        key, separator, raw_count = value.partition("=")
        if not separator or key not in allowed or key in output:
            raise SystemExit(f"{label} requires unique NAME=COUNT values")
        try:
            count = int(raw_count)
        except ValueError as error:
            raise SystemExit(f"{label} count must be an integer: {value}") from error
        if count < 1:
            raise SystemExit(f"{label} count must be positive: {value}")
        output[key] = count
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dataset", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--direction", choices=tuple(DIRECTIONS), required=True)
    parser.add_argument("--corpus", action="append", required=True)
    parser.add_argument("--ui-dataset", type=Path, required=True)
    parser.add_argument("--protected-suite", action="append", type=Path, required=True)
    parser.add_argument("--cap", action="append", default=[])
    parser.add_argument("--repeat", action="append", default=[])
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    parser.add_argument("--seed", default="mimi-shallow-student-v1")
    args = parser.parse_args()
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    if not 0 <= args.maximum_jaccard <= 1:
        raise SystemExit("maximum-jaccard must be between zero and one")

    corpus_paths: dict[str, Path] = {}
    for value in args.corpus:
        name, separator, raw_path = value.partition("=")
        allowed_corpora = set(CORPUS_METADATA) - {"ui"}
        if not separator or name not in allowed_corpora or name in corpus_paths:
            raise SystemExit(
                "--corpus requires one unique kftt|alt|tatoeba|jlt=PATH value"
            )
        corpus_paths[name] = Path(raw_path)
    if not {"kftt", "alt", "tatoeba"} <= set(corpus_paths):
        raise SystemExit("kftt, alt, and tatoeba corpora are required; jlt is optional")
    caps = keyed_integers(args.cap, set(corpus_paths), "--cap")
    repeats = keyed_integers(args.repeat, set(CORPUS_METADATA), "--repeat")
    expected_repeats = set(corpus_paths) | {"ui"}
    if set(caps) != set(corpus_paths) or set(repeats) != expected_repeats:
        raise SystemExit("declare caps for all parallel corpora and repeats for all corpora")

    validation_rows, validation_record = load_validation(
        args.validation_dataset, args.direction
    )
    validation_index = NgramIndex(
        [text for row in validation_rows for text in (row["source"], row["target"])]
    )
    protected_index = NgramIndex(protected_texts(args.protected_suite))
    seen_sources: set[str] = set()
    selected: dict[str, list[dict]] = {}
    rejected: Counter[str] = Counter()
    input_records: dict[str, dict] = {}

    for corpus, path in corpus_paths.items():
        identities = Counter(
            identity
            for row in iter_jsonl(path)
            if (identity := raw_identity(row))[0] == args.direction
        )

        def eligible() -> Iterator[dict]:
            for raw in iter_jsonl(path):
                identity = raw_identity(raw)
                if identity[0] != args.direction:
                    continue
                if identities[identity] != 1:
                    rejected[f"{corpus}:ambiguous-source-id"] += 1
                    continue
                row = parse_parallel_row(raw, corpus)
                source_norm = normalized(row["source"])
                reason = None
                if source_norm in seen_sources:
                    reason = "duplicate-source"
                elif NOISY_MARKUP.search(row["source"]) or NOISY_MARKUP.search(row["target"]):
                    reason = "noisy-markup"
                elif validation_index.matches(row["source"], args.maximum_jaccard) or validation_index.matches(
                    row["target"], args.maximum_jaccard
                ):
                    reason = "validation-overlap"
                elif protected_index.matches(row["source"], args.maximum_jaccard) or protected_index.matches(
                    row["target"], args.maximum_jaccard
                ):
                    reason = "protected-overlap"
                if reason:
                    rejected[f"{corpus}:{reason}"] += 1
                    continue
                seen_sources.add(source_norm)
                yield row

        selected[corpus] = top_ranked(eligible(), caps[corpus], args.seed, corpus)
        if len(selected[corpus]) != caps[corpus]:
            raise SystemExit(
                f"{corpus} retained {len(selected[corpus])} eligible rows; need {caps[corpus]}"
            )
        manifest_path = path.parent / "manifest.json"
        parent_manifest = None
        if manifest_path.is_file():
            parent_value = json.loads(manifest_path.read_text(encoding="utf-8"))
            declared_hash = parent_value.get("outputs", {}).get("train", {}).get("sha256")
            if declared_hash and declared_hash != sha256(path):
                raise SystemExit(f"{corpus} manifest does not authenticate train input")
            if corpus == "jlt" and declared_hash != sha256(path):
                raise SystemExit("jlt requires an authenticated train output")
            parent_manifest = {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
                "train_output_authenticated": declared_hash == sha256(path),
            }
        input_records[corpus] = {
            "path": str(path),
            "sha256": sha256(path),
            "parent_manifest": parent_manifest,
            "selected_unique": len(selected[corpus]),
        }

    ui_train = args.ui_dataset / "train.jsonl"
    ui_manifest = args.ui_dataset / "manifest.json"
    if not ui_train.is_file() or not ui_manifest.is_file():
        raise SystemExit("UI dataset requires train.jsonl and manifest.json")
    ui_rows: list[dict] = []
    for raw in iter_jsonl(ui_train):
        if (raw.get("source_language"), raw.get("target_language")) != DIRECTIONS[
            args.direction
        ]:
            continue
        if raw.get("origin") != CORPUS_METADATA["ui"][1]:
            continue
        row = parse_parallel_row(raw, "ui")
        source_norm = normalized(row["source"])
        if (
            source_norm in seen_sources
            or validation_index.matches(row["source"], args.maximum_jaccard)
            or validation_index.matches(row["target"], args.maximum_jaccard)
            or protected_index.matches(row["source"], args.maximum_jaccard)
            or protected_index.matches(row["target"], args.maximum_jaccard)
        ):
            rejected["ui:overlap"] += 1
            continue
        seen_sources.add(source_norm)
        ui_rows.append(row)
    if not ui_rows:
        raise SystemExit("no eligible Mimi UI rows remain")
    selected["ui"] = ui_rows
    input_records["ui"] = {
        "path": str(ui_train),
        "sha256": sha256(ui_train),
        "parent_manifest": {"path": str(ui_manifest), "sha256": sha256(ui_manifest)},
        "selected_unique": len(ui_rows),
    }

    repeated = {
        corpus: repeat_rows(rows, repeats[corpus], args.seed, corpus)
        for corpus, rows in selected.items()
    }
    train_rows = interleave(repeated)
    identifiers = [row["id"] for row in train_rows]
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit("emitted training IDs are not unique")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    train_path = args.output_directory / "train.jsonl"
    valid_path = args.output_directory / "valid.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, validation_rows)
    manifest = {
        "schema_version": 1,
        "experiment": "high-data compact-model capacity recovery control",
        "direction": args.direction,
        "seed": args.seed,
        "target_source": "licensed-human-reference",
        "promotion_eligible": False,
        "private_reasoning_traces_used": False,
        "maximum_protected_five_gram_jaccard": args.maximum_jaccard,
        "inputs": {
            **input_records,
            "validation": validation_record,
            "protected_suites": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.protected_suite
            ],
        },
        "selection": {
            "policy": "deterministic SHA-256 rank after validation/protected screening",
            "unique_caps": caps,
            "repeat_factors": repeats,
            "unique_by_corpus": {
                corpus: len(rows) for corpus, rows in selected.items()
            },
            "emitted_by_corpus": {
                corpus: len(rows) for corpus, rows in repeated.items()
            },
            "rejected": dict(sorted(rejected.items())),
        },
        "effective_licenses": {
            "train": dict(sorted(Counter(row["source_license"] for row in train_rows).items())),
            "valid": dict(
                sorted(Counter(row["source_license"] for row in validation_rows).items())
            ),
        },
        "counts": {"train": len(train_rows), "valid": len(validation_rows)},
        "outputs": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "valid": {"path": str(valid_path), "sha256": sha256(valid_path)},
        },
    }
    manifest_path = args.output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
