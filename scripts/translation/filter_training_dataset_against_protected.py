#!/usr/bin/env python3
"""Filter an authenticated translation dataset against held-out JSONL suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing JSON input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int) -> frozenset[str]:
    value = normalized(text)
    if not value:
        return frozenset()
    if len(value) < size:
        return frozenset({value})
    return frozenset(
        value[index:index + size] for index in range(len(value) - size + 1)
    )


def text_fields(row: dict, *, protected: bool) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for field in ("source", "target"):
        value = str(row.get(field, "")).strip()
        if value:
            values.append((field, value))
    if protected:
        for field in ("references", "referenceSegments"):
            raw = row.get(field, [])
            if isinstance(raw, list):
                values.extend(
                    (field, str(value).strip())
                    for value in raw
                    if str(value).strip()
                )
        raw_segments = row.get("segments", [])
        if isinstance(raw_segments, list):
            for item in raw_segments:
                value = (
                    str(item.get("source", "")).strip()
                    if isinstance(item, dict)
                    else str(item).strip()
                )
                if value:
                    values.append(("segments", value))
    return values


class ProtectedIndex:
    def __init__(self, paths: list[Path], ngram_size: int) -> None:
        self.entries: list[dict] = []
        self.exact: dict[str, list[int]] = defaultdict(list)
        self.inverted: dict[str, set[int]] = defaultdict(set)
        for path in paths:
            for row in load_jsonl(path):
                identifier = str(row.get("id", "unknown"))
                for field, text in text_fields(row, protected=True):
                    value = normalized(text)
                    grams = ngrams(text, ngram_size)
                    if not value or not grams:
                        continue
                    index = len(self.entries)
                    self.entries.append({
                        "id": identifier,
                        "field": field,
                        "path": str(path.resolve()),
                        "ngrams": grams,
                    })
                    self.exact[value].append(index)
                    for gram in grams:
                        self.inverted[gram].add(index)
        if not self.entries:
            raise SystemExit("protected suites contain no source/reference text")

    def match(self, text: str, maximum_jaccard: float, ngram_size: int) -> dict | None:
        value = normalized(text)
        if not value:
            return None
        exact = self.exact.get(value)
        if exact:
            entry = self.entries[exact[0]]
            return {
                "kind": "exact",
                "jaccard": 1.0,
                "protectedID": entry["id"],
                "protectedField": entry["field"],
                "protectedPath": entry["path"],
            }
        candidate = ngrams(text, ngram_size)
        possible: set[int] = set()
        for gram in candidate:
            possible.update(self.inverted.get(gram, ()))
        best_index: int | None = None
        best_similarity = 0.0
        for index in possible:
            heldout = self.entries[index]["ngrams"]
            similarity = len(candidate & heldout) / max(1, len(candidate | heldout))
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index
        if best_index is None or best_similarity <= maximum_jaccard:
            return None
        entry = self.entries[best_index]
        return {
            "kind": "near",
            "jaccard": round(best_similarity, 8),
            "protectedID": entry["id"],
            "protectedField": entry["field"],
            "protectedPath": entry["path"],
        }


def authenticated_parent(dataset: Path) -> tuple[Path, Path, Path, dict]:
    manifest_path = dataset / "manifest.json"
    train_path, valid_path = dataset / "train.jsonl", dataset / "valid.jsonl"
    manifest = load_json(manifest_path)
    for split, path in (("train", train_path), ("valid", valid_path)):
        expected = manifest.get("outputs", {}).get(split, {}).get("sha256")
        if not expected or expected != sha256(path):
            raise SystemExit(f"parent {split} hash does not match its manifest")
    return manifest_path, train_path, valid_path, manifest


def effective_license(row: dict) -> str:
    return str(row.get("source_license") or row.get("license") or "unknown")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protected-suite", type=Path, action="append", required=True)
    parser.add_argument("--character-ngram-size", type=int, default=5)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.character_ngram_size < 1:
        raise SystemExit("character-ngram-size must be positive")
    if not 0 <= args.maximum_jaccard < 1:
        raise SystemExit("maximum-jaccard must be at least zero and below one")

    manifest_path, train_path, valid_path, parent = authenticated_parent(args.dataset)
    protected = ProtectedIndex(args.protected_suite, args.character_ngram_size)
    outputs: dict[str, list[dict]] = {}
    exclusions: list[dict] = []
    input_counts: dict[str, int] = {}
    for split, path in (("train", train_path), ("valid", valid_path)):
        source_rows = load_jsonl(path)
        input_counts[split] = len(source_rows)
        retained: list[dict] = []
        for row in source_rows:
            matches: list[dict] = []
            for field, text in text_fields(row, protected=False):
                match = protected.match(
                    text, args.maximum_jaccard, args.character_ngram_size
                )
                if match is not None:
                    matches.append({"datasetField": field, **match})
            if matches:
                exclusions.append({
                    "split": split,
                    "datasetID": str(row.get("id", "unknown")),
                    "matches": matches,
                })
            else:
                retained.append(row)
        if split == "valid" and not retained:
            raise SystemExit("filtering removed every validation row")
        outputs[split] = retained

    args.output.mkdir(parents=True, exist_ok=True)
    output_paths = {
        split: args.output / f"{split}.jsonl" for split in ("train", "valid")
    }
    for split, path in output_paths.items():
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in outputs[split]
            ),
            encoding="utf-8",
        )

    exclusion_kinds = Counter(
        match["kind"]
        for exclusion in exclusions
        for match in exclusion["matches"]
    )
    result = {
        "schema_version": 1,
        "experiment": "held-out contamination-screened translation dataset",
        "direction": parent.get("direction"),
        "promotion_eligible": False,
        "target_source": parent.get("target_source"),
        "filter": {
            "normalization": "Unicode NFKC, casefold, remove whitespace",
            "character_ngram_size": args.character_ngram_size,
            "maximum_jaccard_exclusive": args.maximum_jaccard,
            "fields": {
                "dataset": ["source", "target"],
                "protected": [
                    "source",
                    "target",
                    "references",
                    "segments",
                    "referenceSegments",
                ],
            },
        },
        "counts": {
            "input": input_counts,
            "output": {split: len(rows) for split, rows in outputs.items()},
            "excludedRows": len(exclusions),
            "matchesByKind": dict(sorted(exclusion_kinds.items())),
        },
        "zero_hits_at_threshold": not exclusions,
        "effective_licenses": {
            split: dict(sorted(Counter(effective_license(row) for row in rows).items()))
            for split, rows in outputs.items()
        },
        "origins": {
            split: dict(
                sorted(Counter(str(row.get("origin", "unknown")) for row in rows).items())
            )
            for split, rows in outputs.items()
        },
        "inputs": {
            "parent_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256(manifest_path),
            },
            "parent_train": {
                "path": str(train_path.resolve()),
                "sha256": sha256(train_path),
            },
            "parent_valid": {
                "path": str(valid_path.resolve()),
                "sha256": sha256(valid_path),
            },
            "protected_suites": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in args.protected_suite
            ],
        },
        "outputs": {
            split: {"path": str(path.resolve()), "sha256": sha256(path)}
            for split, path in output_paths.items()
        },
        "exclusions": exclusions,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
