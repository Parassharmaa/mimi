#!/usr/bin/env python3
"""Freeze a source-selected, law-group-held-out EN<->JA safety validation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator


DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
BUCKET_WEIGHTS = {
    "negation": 2,
    "critical-structure": 4,
    "legal-enumeration": 2,
    "long-form": 1,
    "general": 1,
}
CRITICAL_RE = re.compile(
    r"https?://[^\s]+|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>|%|\d"
)
EN_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|neither|nor|without|cannot|can't|shall not|must not|"
    r"is prohibited|are prohibited)\b",
    re.IGNORECASE,
)
JA_NEGATION_RE = re.compile(r"ない|ません|なかった|ませんでした|ぬ|ず|禁止|不可|除く")
ENUMERATION_RE = re.compile(
    r"^(?:\([ivxlcdm]+\)|Article\b|Chapter\b|Section\b|Part\b|Appended\b|"
    r"Table\b|\d+[.)])",
    re.IGNORECASE,
)
LAW_ID_RE = re.compile(r"^(law-[^:]+):tu-[^:]+$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"invalid JSON: {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise SystemExit(f"expected JSON object: {path}:{line_number}")
            yield row


def normalized(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def ngrams(value: str, size: int = 5) -> set[str]:
    text = normalized(value)
    if not text:
        return set()
    if len(text) < size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


class NgramIndex:
    def __init__(self, values: Iterable[str]) -> None:
        self.values: list[set[str]] = []
        postings: dict[str, list[int]] = defaultdict(list)
        for value in values:
            grams = ngrams(value)
            if not grams:
                continue
            index = len(self.values)
            self.values.append(grams)
            for gram in grams:
                postings[gram].append(index)
        self.postings = dict(postings)

    def matches(self, value: str, maximum: float) -> bool:
        candidate = ngrams(value)
        intersections: Counter[int] = Counter()
        for gram in candidate:
            intersections.update(self.postings.get(gram, ()))
        return any(
            overlap
            / max(1, len(candidate) + len(self.values[index]) - overlap)
            > maximum
            for index, overlap in intersections.items()
        )


def law_id(source_id: str) -> str:
    match = LAW_ID_RE.fullmatch(source_id)
    if match is None:
        raise SystemExit(f"invalid Japanese Law source_id: {source_id}")
    return match.group(1)


def direction(row: dict) -> str:
    languages = (row.get("source_language"), row.get("target_language"))
    for name, expected in DIRECTIONS.items():
        if languages == expected:
            return name
    raise SystemExit(f"invalid Japanese Law direction: {row.get('id')}")


def validate_jlt_row(row: dict) -> None:
    if (
        row.get("origin") != "finalized-japanese-law-translation"
        or row.get("translation_status") != "finalized"
        or row.get("source_license") != "PDL-1.0-compatible-CC-BY-4.0"
        or row.get("training_only") is not True
        or row.get("promotion_eligible") is not False
    ):
        raise SystemExit(f"unexpected Japanese Law row policy: {row.get('id')}")
    for field in (
        "source_id",
        "source",
        "target",
        "attribution",
        "source_provenance",
        "source_normalized_sha256",
        "target_normalized_sha256",
    ):
        if not row.get(field):
            raise SystemExit(f"Japanese Law row lacks {field}: {row.get('id')}")


def load_pairs(path: Path) -> dict[str, dict[str, dict]]:
    pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    ambiguous: set[str] = set()
    for row in iter_jsonl(path):
        validate_jlt_row(row)
        name = direction(row)
        source_id = str(row["source_id"])
        if name in pairs[source_id]:
            ambiguous.add(source_id)
        pairs[source_id][name] = row
    complete: dict[str, dict[str, dict]] = {}
    for source_id, rows in pairs.items():
        if source_id in ambiguous or set(rows) != set(DIRECTIONS):
            continue
        if (
            rows["en-ja"]["source"] != rows["ja-en"]["target"]
            or rows["en-ja"]["target"] != rows["ja-en"]["source"]
        ):
            raise SystemExit(f"paired Japanese Law text differs: {source_id}")
        complete[source_id] = rows
    return complete


def split_inventory(path: Path) -> tuple[set[str], set[str], int]:
    laws: set[str] = set()
    text_hashes: set[str] = set()
    rows = 0
    for row in iter_jsonl(path):
        validate_jlt_row(row)
        rows += 1
        laws.add(law_id(str(row["source_id"])))
        text_hashes.add(str(row["source_normalized_sha256"]))
        text_hashes.add(str(row["target_normalized_sha256"]))
    return laws, text_hashes, rows


def protected_texts(paths: list[Path]) -> tuple[list[str], list[dict]]:
    texts: list[str] = []
    records: list[dict] = []
    for path in paths:
        rows = 0
        for row in iter_jsonl(path):
            source = str(row.get("source", ""))
            references = row.get("references") or []
            if not source:
                raise SystemExit(f"protected suite row lacks source: {path}")
            texts.append(source)
            texts.extend(str(reference) for reference in references if str(reference))
            rows += 1
        records.append({"path": str(path), "sha256": sha256(path), "rows": rows})
    return texts, records


def buckets(en: str, ja: str) -> set[str]:
    labels: set[str] = set()
    if EN_NEGATION_RE.search(en) or JA_NEGATION_RE.search(ja):
        labels.add("negation")
    if CRITICAL_RE.search(en) or CRITICAL_RE.search(ja):
        labels.add("critical-structure")
    if ENUMERATION_RE.search(en):
        labels.add("legal-enumeration")
    if len(en) >= 200 or len(ja) >= 120:
        labels.add("long-form")
    return labels or {"general"}


def quotas(pairs: int) -> dict[str, int]:
    total_weight = sum(BUCKET_WEIGHTS.values())
    remaining_pairs = pairs - len(BUCKET_WEIGHTS)
    values = {
        name: 1 + remaining_pairs * weight // total_weight
        for name, weight in BUCKET_WEIGHTS.items()
    }
    remainder = pairs - sum(values.values())
    order = ("critical-structure", "negation", "legal-enumeration", "long-form", "general")
    for index in range(remainder):
        values[order[index % len(order)]] += 1
    return values


def rank(seed: str, source_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{source_id}".encode()).hexdigest()


def authenticate_manifest(directory: Path) -> tuple[dict, dict[str, Path]]:
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("document_grouped_split") is not True:
        raise SystemExit("Japanese Law manifest is not document-grouped")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise SystemExit("Japanese Law manifest lacks outputs")
    paths: dict[str, Path] = {}
    for split in ("train", "valid", "test"):
        record = outputs.get(split)
        if not isinstance(record, dict):
            raise SystemExit(f"Japanese Law manifest lacks {split} output")
        output = Path(str(record["path"]))
        if not output.is_absolute():
            output = (Path.cwd() / output).resolve()
        if sha256(output) != record.get("sha256"):
            raise SystemExit(f"Japanese Law {split} hash mismatch")
        rows = sum(1 for line in output.open(encoding="utf-8") if line.strip())
        if rows != record.get("rows"):
            raise SystemExit(f"Japanese Law {split} row count mismatch")
        paths[split] = output
    return manifest, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jlt_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs", type=int, default=400)
    parser.add_argument("--seed", default="mimi-legal-safety-validation-v1")
    parser.add_argument("--source-split", choices=("valid", "test"), default="valid")
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    parser.add_argument("--protected-suite", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.pairs < 5:
        raise SystemExit("--pairs must be at least five")
    if not 0.0 <= args.maximum_jaccard < 1.0:
        raise SystemExit("--maximum-jaccard must be in [0, 1)")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    manifest, paths = authenticate_manifest(args.jlt_directory.resolve())
    train_laws, train_hashes, train_rows = split_inventory(paths["train"])
    valid_laws_inventory, valid_hashes, valid_rows = split_inventory(paths["valid"])
    test_laws, test_hashes, test_rows = split_inventory(paths["test"])
    source_pairs = load_pairs(paths[args.source_split])
    source_laws = {law_id(source_id) for source_id in source_pairs}
    valid_laws = valid_laws_inventory
    if valid_laws & train_laws or valid_laws & test_laws or train_laws & test_laws:
        raise SystemExit("Japanese Law document-grouped splits overlap by law ID")
    expected_source_laws = valid_laws if args.source_split == "valid" else test_laws
    if source_laws != expected_source_laws:
        raise SystemExit("complete paired source laws differ from split inventory")

    protected_values, protected_records = protected_texts(args.protected_suite)
    protected_exact = {normalized(value) for value in protected_values}
    protected_index = NgramIndex(protected_values)
    other_split_hashes = (
        train_hashes | test_hashes
        if args.source_split == "valid"
        else train_hashes | valid_hashes
    )
    eligible: dict[str, list[str]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for source_id, rows in source_pairs.items():
        en_row = rows["en-ja"]
        en = str(en_row["source"])
        ja = str(en_row["target"])
        hashes = {
            str(en_row["source_normalized_sha256"]),
            str(en_row["target_normalized_sha256"]),
        }
        if hashes & other_split_hashes:
            other_names = "train-or-test" if args.source_split == "valid" else "train-or-valid"
            rejected[f"exact-text-in-{other_names}"] += 1
            continue
        if normalized(en) in protected_exact or normalized(ja) in protected_exact:
            rejected["exact-protected-overlap"] += 1
            continue
        if protected_index.matches(en, args.maximum_jaccard) or protected_index.matches(
            ja, args.maximum_jaccard
        ):
            rejected["near-protected-overlap"] += 1
            continue
        for name in buckets(en, ja):
            eligible[name].append(source_id)

    selected: list[tuple[str, str]] = []
    selected_ids: set[str] = set()
    selected_counts = quotas(args.pairs)
    selection_order = (
        "long-form",
        "general",
        "negation",
        "legal-enumeration",
        "critical-structure",
    )
    for name in selection_order:
        count = selected_counts[name]
        available = sorted(
            (value for value in eligible[name] if value not in selected_ids),
            key=lambda value: rank(args.seed, value),
        )
        if len(available) < count:
            raise SystemExit(f"need {count} {name} pairs, found {len(available)}")
        chosen = available[:count]
        selected.extend((source_id, name) for source_id in chosen)
        selected_ids.update(chosen)

    suite: list[dict] = []
    suite_name = (
        "legal-safety-validation-v1"
        if args.source_split == "valid"
        else "legal-safety-test-v1"
    )
    split_name = (
        "legal-safety-validation" if args.source_split == "valid" else "legal-safety-test"
    )
    for source_id, selection_bucket in selected:
        rows = source_pairs[source_id]
        for name, languages in DIRECTIONS.items():
            row = rows[name]
            suite.append(
                {
                    "id": f"{suite_name}:jlt:{source_id}:{name}",
                    "sourceLanguage": languages[0],
                    "targetLanguage": languages[1],
                    "domain": "ministry-published-legal",
                    "source": row["source"],
                    "references": [row["target"]],
                    "claimEligible": False,
                    "split": split_name,
                    "license": row["source_license"],
                    "provenance": row["attribution"],
                    "reviewStatus": "authenticated-human-reference-policy-blocked",
                    "sourceCorpus": f"jlt-{args.source_split}",
                    "sourceID": source_id,
                    "documentID": law_id(source_id),
                    "selectionBucket": selection_bucket,
                    "attribution": row["attribution"],
                }
            )
    suite.sort(key=lambda row: row["id"])
    if len(suite) != args.pairs * 2:
        raise SystemExit("legal safety suite has the wrong case count")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in suite),
        encoding="utf-8",
    )
    source_record = (
        {
            "manifest": {
                "path": str(args.jlt_directory.resolve() / "manifest.json"),
                "sha256": sha256(args.jlt_directory.resolve() / "manifest.json"),
            },
            "valid": {
                "path": str(paths["valid"]),
                "sha256": sha256(paths["valid"]),
                "available_complete_pairs": len(source_pairs),
                "law_groups": len(source_laws),
            },
            "document_grouped_split": manifest["document_grouped_split"],
            "split_seed": manifest.get("split_seed"),
            "train_rows_scanned": train_rows,
            "test_rows_scanned": test_rows,
            "train_law_groups": len(train_laws),
            "test_law_groups": len(test_laws),
        }
        if args.source_split == "valid"
        else {
            "manifest": {
                "path": str(args.jlt_directory.resolve() / "manifest.json"),
                "sha256": sha256(args.jlt_directory.resolve() / "manifest.json"),
            },
            "test": {
                "path": str(paths["test"]),
                "sha256": sha256(paths["test"]),
                "available_complete_pairs": len(source_pairs),
                "law_groups": len(source_laws),
            },
            "document_grouped_split": manifest["document_grouped_split"],
            "split_seed": manifest.get("split_seed"),
            "train_rows_scanned": train_rows,
            "valid_rows_scanned": valid_rows,
            "train_law_groups": len(train_laws),
            "valid_law_groups": len(valid_laws),
        }
    )
    output_manifest = {
        "schema_version": 1,
        "suite": suite_name,
        "purpose": (
            "checkpoint selection and structural validation; not a promotion claim suite"
            if args.source_split == "valid"
            else "independent post-selection structural validation; not a promotion claim suite"
        ),
        "seed": args.seed,
        "pairs": args.pairs,
        "cases": len(suite),
        "cases_per_direction": args.pairs,
        "selection_uses_model_outputs": False,
        "private_reasoning_traces_used": False,
        "claim_eligible": False,
        "does_not_authorize_model_promotion": True,
        "does_not_authorize_app_integration": True,
        "source": source_record,
        "selection": {
            "policy": "multi-label source structure; restrictive quota order; unique deterministic SHA-256 rank",
            "quota_order": list(selection_order),
            "bucket_weights": BUCKET_WEIGHTS,
            "available_by_bucket": {
                name: len(eligible[name]) for name in BUCKET_WEIGHTS
            },
            "selected_by_bucket": selected_counts,
            "rejected": dict(sorted(rejected.items())),
        },
        "contamination_controls": {
            "law_ids_disjoint_across_jlt_train_valid_test": True,
            (
                "exact_normalized_text_disjoint_from_jlt_train_and_test"
                if args.source_split == "valid"
                else "exact_normalized_text_disjoint_from_jlt_train_and_valid"
            ): True,
            "maximum_protected_five_gram_jaccard": args.maximum_jaccard,
            "protected_suites": protected_records,
        },
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
