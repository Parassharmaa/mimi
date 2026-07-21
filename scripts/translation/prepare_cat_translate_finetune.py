#!/usr/bin/env python3
"""Build contamination-screened CAT-Translate prompt/completion fine-tuning data."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path


LANGUAGE_NAMES = {"en-US": "English", "ja-JP": "Japanese"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text)
    if len(value) < size:
        return {value}
    return {value[index:index + size] for index in range(len(value) - size + 1)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProtectedIndex:
    def __init__(self, suites: list[Path]):
        self.entries: list[tuple[str, set[str]]] = []
        self.exact: dict[str, str] = {}
        self.by_gram: dict[str, set[int]] = defaultdict(set)
        for path in suites:
            for row in load_jsonl(path):
                texts = [row.get("source", ""), *row.get("references", [])]
                for text in texts:
                    value = str(text).strip()
                    if not value:
                        continue
                    label = str(row.get("id", path.name))
                    exact = normalized(value)
                    self.exact[exact] = label
                    grams = ngrams(value)
                    index = len(self.entries)
                    self.entries.append((label, grams))
                    for gram in grams:
                        self.by_gram[gram].add(index)

    def match(self, text: str, maximum_jaccard: float) -> tuple[str, float] | None:
        exact = self.exact.get(normalized(text))
        if exact is not None:
            return exact, 1.0
        candidate = ngrams(text)
        possible = set().union(*(self.by_gram[gram] for gram in candidate if gram in self.by_gram))
        for index in possible:
            label, protected = self.entries[index]
            similarity = len(candidate & protected) / max(1, len(candidate | protected))
            if similarity > maximum_jaccard:
                return label, similarity
        return None


def prompt(row: dict) -> str:
    try:
        source = LANGUAGE_NAMES[row["source_language"]]
        target = LANGUAGE_NAMES[row["target_language"]]
    except KeyError as error:
        raise SystemExit(f"unsupported language in {row.get('id')}: {error.args[0]}") from error
    return f"Translate the following {source} text into {target}.\n\n{row['source']}"


def prepared(row: dict) -> dict:
    return {
        "prompt": prompt(row),
        "completion": row["target"],
        "id": row["id"],
        "direction": f"{row['source_language']}>{row['target_language']}",
        "domain": row["domain"],
        "origin": row["origin"],
        "source_license": row["source_license"],
        "source_provenance": row["source_provenance"],
        "attribution": row.get("attribution", row["source_provenance"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--en-ja", type=Path, required=True)
    parser.add_argument("--ja-en", type=Path, required=True)
    parser.add_argument("--protected-suite", type=Path, action="append", default=[])
    parser.add_argument("--include-origin", action="append", default=[])
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 <= args.maximum_jaccard <= 1:
        raise SystemExit("maximum-jaccard must be between zero and one")

    protected = ProtectedIndex(args.protected_suite)
    inputs = [args.en_ja, args.ja_en]
    outputs: dict[str, list[dict]] = {"train": [], "valid": []}
    rejected: dict[str, int] = defaultdict(int)
    seen_ids: set[str] = set()
    seen_pairs: dict[tuple[str, str], str] = {}

    for split in outputs:
        for directory in inputs:
            for row in load_jsonl(directory / f"{split}.jsonl"):
                identifier = str(row.get("id", "")).strip()
                source = str(row.get("source", "")).strip()
                target = str(row.get("target", "")).strip()
                if not identifier or not source or not target:
                    raise SystemExit(f"invalid training row in {directory}/{split}.jsonl")
                if identifier in seen_ids:
                    raise SystemExit(f"duplicate training ID: {identifier}")
                seen_ids.add(identifier)
                if args.include_origin and row.get("origin") not in args.include_origin:
                    rejected[f"origin-excluded-{split}"] += 1
                    continue
                match = protected.match(source, args.maximum_jaccard) or protected.match(
                    target, args.maximum_jaccard
                )
                if match is not None:
                    rejected[f"protected-{split}"] += 1
                    continue
                pair = (normalized(source), normalized(target))
                previous_split = seen_pairs.get(pair)
                if previous_split is not None:
                    if previous_split != split:
                        raise SystemExit(f"cross-split duplicate pair: {identifier}")
                    rejected[f"duplicate-{split}"] += 1
                    continue
                seen_pairs[pair] = split
                outputs[split].append(prepared(row))

    args.output.mkdir(parents=True, exist_ok=True)
    output_records = {}
    for split, rows in outputs.items():
        path = args.output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        output_records[split] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "sha256": sha256(path),
        }

    manifest = {
        "schema_version": 1,
        "purpose": "CAT-Translate bidirectional QLoRA fine-tuning",
        "model_prompt_contract": "CyberAgent CAT-Translate user-only instruction",
        "loss_contract": "assistant completion only; invoke mlx_lm.lora with --mask-prompt",
        "inputs": [
            {
                "path": str(directory.resolve()),
                "train_sha256": sha256(directory / "train.jsonl"),
                "valid_sha256": sha256(directory / "valid.jsonl"),
            }
            for directory in inputs
        ],
        "protected_suites": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in args.protected_suite
        ],
        "maximum_five_gram_jaccard": args.maximum_jaccard,
        "included_origins": sorted(args.include_origin),
        "outputs": output_records,
        "rejected": dict(sorted(rejected.items())),
        "license_policy": (
            "retain per-row attribution; adapter distribution must satisfy the combined "
            "CC-BY-SA/CC-BY obligations recorded by the inputs"
        ),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
