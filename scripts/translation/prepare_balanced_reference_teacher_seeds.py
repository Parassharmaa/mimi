#!/usr/bin/env python3
"""Mine a balanced, licensed, reference-hidden EN<->JA teacher seed pool.

The exact incumbent MLX pack scores a deterministic candidate pool from each
licensed corpus. Selection balances directions and domains, stratifies by
student sequence uncertainty, and maximizes encoder-embedding diversity. Human
references remain local and are never passed to a teacher generation prompt.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import sacrebleu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dqrd_selection import hybrid_select, selection_summary  # noqa: E402


DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
CORPORA = {
    "kftt": "professional-wikipedia-hard",
    "alt": "human-translated-news-hard",
    "tatoeba": "everyday-conversation-hard",
}
ALLOWED_LICENSES = {
    "CC-BY-2.0-FR",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "project-owned",
}
NOISY_MARKUP_RE = re.compile(r"@[-,.]@|<[^>]+>|&(?:quot|amp|lt|gt);", re.I)


def load_rows(path: Path) -> list[dict]:
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


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text)
    if not value:
        return set()
    if len(value) < size:
        return {value}
    return {value[index:index + size] for index in range(len(value) - size + 1)}


class NgramIndex:
    """Exact Jaccard screening using postings instead of a full corpus scan."""

    def __init__(self, values: list[set[str]]) -> None:
        self.values = values
        postings: dict[str, list[int]] = defaultdict(list)
        for index, grams in enumerate(values):
            for gram in grams:
                postings[gram].append(index)
        self.postings = dict(postings)

    def matches(self, text: str, maximum: float) -> bool:
        candidate = ngrams(text)
        if not candidate:
            return False
        intersections: Counter[int] = Counter()
        for gram in candidate:
            intersections.update(self.postings.get(gram, ()))
        return any(
            overlap / max(1, len(candidate) + len(self.values[index]) - overlap) > maximum
            for index, overlap in intersections.items()
        )


def near(text: str, protected: NgramIndex, maximum: float) -> bool:
    return protected.matches(text, maximum)


def deterministic_rank(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


def dataset_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = [path / name for name in ("train.jsonl", "valid.jsonl")]
        if not all(item.is_file() for item in files):
            raise SystemExit(f"excluded dataset needs train.jsonl and valid.jsonl: {path}")
        return files
    raise SystemExit(f"missing excluded dataset: {path}")


def row_texts(row: dict) -> tuple[str, str]:
    if "messages" in row:
        messages = row.get("messages", [])
        if len(messages) != 3 or [message.get("role") for message in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise SystemExit("excluded chat dataset has an unexpected message shape")
        return str(messages[1].get("content", "")), str(messages[2].get("content", ""))
    return str(row.get("source", "")), str(row.get("target", ""))


def parse_parallel_row(row: dict, corpus: str) -> dict:
    if corpus not in CORPORA:
        raise SystemExit(f"unsupported corpus: {corpus}")
    if "messages" in row:
        metadata = row.get("metadata", {})
        direction = str(metadata.get("direction", ""))
        source, reference = row_texts(row)
        identifier = str(metadata.get("source_id", ""))
        license_name = str(metadata.get("license", ""))
        attribution = str(metadata.get("attribution", ""))
        provenance = f"{metadata.get('source', corpus)} / {identifier} / {attribution}"
    else:
        languages = (row.get("source_language"), row.get("target_language"))
        direction = next(
            (name for name, expected in DIRECTIONS.items() if languages == expected),
            "",
        )
        source = str(row.get("source", ""))
        reference = str(row.get("target", ""))
        identifier = str(row.get("source_id", ""))
        license_name = str(row.get("source_license", ""))
        attribution = str(row.get("attribution", ""))
        provenance = str(row.get("source_provenance", "")) or attribution
    if direction not in DIRECTIONS:
        raise SystemExit(f"unsupported direction in {corpus}: {identifier}")
    if not identifier or not source.strip() or not reference.strip():
        raise SystemExit(f"incomplete parallel row in {corpus}: {identifier}")
    if license_name not in ALLOWED_LICENSES or not provenance.strip() or not attribution.strip():
        raise SystemExit(f"missing distributable license/provenance in {corpus}: {identifier}")
    return {
        "raw_source_id": identifier,
        "direction": direction,
        "source": source.strip(),
        "reference": reference.strip(),
        "license": license_name,
        "provenance": provenance.strip(),
        "attribution": attribution.strip(),
        "domain": CORPORA[corpus],
        "corpus": corpus,
    }


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


def protected_grams(paths: list[Path]) -> list[set[str]]:
    output: list[set[str]] = []
    for path in paths:
        for row in load_rows(path):
            for text in (row.get("source", ""), *row.get("references", [])):
                if str(text).strip():
                    output.append(ngrams(str(text)))
    return output


def exclusion_policy(
    datasets: list[Path],
    extra_jsonl: list[Path],
) -> tuple[set[str], set[str], list[set[str]], list[Path]]:
    files = [item for path in datasets for item in dataset_files(path)]
    excluded_sources: set[str] = set()
    validation_texts: set[str] = set()
    for path in [*files, *extra_jsonl]:
        for row in load_rows(path):
            if "references" in row:
                source = str(row.get("source", ""))
                targets = [str(value) for value in row.get("references", [])]
            else:
                source, target = row_texts(row)
                targets = [target]
            if source.strip():
                excluded_sources.add(normalized(source))
            if path.name == "valid.jsonl":
                validation_texts.update(
                    normalized(text)
                    for text in [source, *targets]
                    if text.strip()
                )
    validation_grams = [ngrams(text) for text in validation_texts if text]
    return excluded_sources, validation_texts, validation_grams, files


def eligible_by_corpus_direction(
    corpus_paths: dict[str, Path],
    excluded_sources: set[str],
    validation_texts: set[str],
    validation_grams: list[set[str]],
    heldout_grams: list[set[str]],
    maximum_jaccard: float,
) -> tuple[dict[tuple[str, str], list[dict]], Counter[str]]:
    validation_index = NgramIndex(validation_grams)
    heldout_index = NgramIndex(heldout_grams)
    output: dict[tuple[str, str], list[dict]] = {
        (corpus, direction): []
        for corpus in CORPORA
        for direction in DIRECTIONS
    }
    rejected: Counter[str] = Counter()
    global_sources: set[tuple[str, str]] = set()
    for corpus, path in corpus_paths.items():
        raw_rows = load_rows(path)
        identity_counts = Counter(raw_identity(row) for row in raw_rows)
        for raw in raw_rows:
            row = parse_parallel_row(raw, corpus)
            identity = (row["direction"], row["raw_source_id"])
            if identity_counts[identity] != 1:
                rejected[f"{corpus}:ambiguous-source-id"] += 1
                continue
            source_norm = normalized(row["source"])
            reference_norm = normalized(row["reference"])
            source_key = (DIRECTIONS[row["direction"]][0], source_norm)
            reason = None
            if source_norm in excluded_sources:
                reason = "existing-student-or-prior-teacher-source"
            elif source_norm in validation_texts or reference_norm in validation_texts:
                reason = "student-validation-overlap"
            elif near(row["source"], validation_index, maximum_jaccard) or near(
                row["reference"], validation_index, maximum_jaccard
            ):
                reason = "near-student-validation"
            elif near(row["source"], heldout_index, maximum_jaccard) or near(
                row["reference"], heldout_index, maximum_jaccard
            ):
                reason = "near-protected-evaluation"
            elif source_key in global_sources:
                reason = "duplicate-normalized-source"
            elif NOISY_MARKUP_RE.search(row["source"]) or NOISY_MARKUP_RE.search(row["reference"]):
                reason = "noisy-markup"
            if reason:
                rejected[f"{corpus}:{reason}"] += 1
                continue
            global_sources.add(source_key)
            row["id"] = f"teacher-balanced:{corpus}:{row['direction']}:{row['raw_source_id']}"
            output[(corpus, row["direction"])].append(row)
    return output, rejected


def validate_file_table(root: Path, files: dict, label: str) -> None:
    if not isinstance(files, dict) or not files:
        raise SystemExit(f"{label} has no authenticated file table")
    for relative, expected in files.items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != expected.get("bytes")
            or sha256(path) != expected.get("sha256")
        ):
            raise SystemExit(f"{label} integrity failure: {relative}")


def validate_model_pack(path: Path) -> tuple[dict, str]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"model pack lacks manifest: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "mimi-mlx-marian-pair-v1":
        raise SystemExit("balanced mining requires an authenticated Marian pair pack")
    validate_file_table(path, manifest.get("files"), "model pack")
    for direction in DIRECTIONS:
        child_path = path / direction / "manifest.json"
        child = json.loads(child_path.read_text(encoding="utf-8"))
        validate_file_table(path / direction, child.get("files"), f"model {direction}")
    return manifest, sha256(manifest_path)


def inventory_report(
    eligible: dict[tuple[str, str], list[dict]],
    rejected: Counter[str],
) -> dict:
    return {
        "eligible": {
            corpus: {
                direction: len(eligible[(corpus, direction)])
                for direction in DIRECTIONS
            }
            for corpus in CORPORA
        },
        "rejected": dict(sorted(rejected.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_pack", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corpus", action="append", required=True)
    parser.add_argument("--protected-suite", type=Path, action="append", required=True)
    parser.add_argument("--exclude-dataset", type=Path, action="append", required=True)
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--pool-per-domain-direction", type=int, default=600)
    parser.add_argument("--select-per-domain-direction", type=int, default=400)
    parser.add_argument("--minimum-student-chrf", type=float, default=5.0)
    parser.add_argument("--maximum-tokens", type=int, default=192)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    parser.add_argument("--seed", default="mimi-balanced-reference-teacher-v1")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if not 0 < args.select_per_domain_direction <= args.pool_per_domain_direction:
        raise SystemExit("selection count must be positive and no larger than the pool")
    if args.maximum_tokens < 1 or not 0 <= args.maximum_jaccard <= 1:
        raise SystemExit("invalid token or contamination threshold")

    corpus_paths: dict[str, Path] = {}
    for value in args.corpus:
        name, separator, raw_path = value.partition("=")
        if not separator or name not in CORPORA or name in corpus_paths:
            raise SystemExit(f"--corpus must be one unique kftt|alt|tatoeba=PATH value: {value}")
        corpus_paths[name] = Path(raw_path)
    if set(corpus_paths) != set(CORPORA):
        raise SystemExit("exactly one kftt, alt, and tatoeba corpus is required")

    pack_manifest, pack_manifest_sha = validate_model_pack(args.model_pack)
    excluded_sources, validation_texts, validation_grams, exclusion_files = exclusion_policy(
        args.exclude_dataset,
        args.exclude_jsonl,
    )
    heldout_grams = protected_grams(args.protected_suite)
    eligible, rejected = eligible_by_corpus_direction(
        corpus_paths,
        excluded_sources,
        validation_texts,
        validation_grams,
        heldout_grams,
        args.maximum_jaccard,
    )
    inventory = inventory_report(eligible, rejected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.inventory_only:
        report = {
            "schema_version": 1,
            "purpose": "licensed balanced teacher-pool inventory; no training rows emitted",
            "model_pack_manifest_sha256": pack_manifest_sha,
            "maximum_five_gram_jaccard": args.maximum_jaccard,
            **inventory,
        }
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    for key, rows in eligible.items():
        if len(rows) < args.pool_per_domain_direction:
            raise SystemExit(
                f"{key[0]} {key[1]} has {len(rows)} eligible rows; "
                f"need pool {args.pool_per_domain_direction}"
            )

    import mlx.core as mx
    from transformers import PreTrainedTokenizerFast

    from marian_mlx import load_model

    selected_output: list[dict] = []
    selection_reports: dict[str, dict] = {}
    for direction in DIRECTIONS:
        model_path = args.model_pack / direction
        child_manifest = json.loads((model_path / "manifest.json").read_text(encoding="utf-8"))
        model = load_model(
            model_path / "model.safetensors",
            quantization_bits=int(child_manifest["bits"]),
            quantization_group_size=int(child_manifest["group_size"]),
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(model_path / "tokenizer.json"),
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )
        source_prefix = (child_manifest.get("source_prefixes") or {}).get(direction, "")
        for corpus in CORPORA:
            ranked = sorted(
                eligible[(corpus, direction)],
                key=lambda row: deterministic_rank(args.seed, row["id"]),
            )[:args.pool_per_domain_direction]
            scored: list[dict] = []
            for index, row in enumerate(ranked, start=1):
                encoded = tokenizer.encode(source_prefix + row["source"])
                output_ids, diagnostics = model.generate_with_diagnostics(
                    encoded,
                    args.maximum_tokens,
                )
                mx.synchronize()
                hypothesis = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                student_chrf = sacrebleu.sentence_chrf(
                    hypothesis,
                    [row["reference"]],
                    word_order=2,
                ).score
                sequence_nll = float(diagnostics["student_sequence_nll"])
                if not hypothesis or not math.isfinite(sequence_nll):
                    raise SystemExit(f"invalid student result: {row['id']}")
                scored.append({
                    **row,
                    "student_hypothesis": hypothesis,
                    "student_chrf_pp": student_chrf,
                    "student_sequence_nll": sequence_nll,
                    "_selection_embedding": diagnostics["encoder_embedding"],
                })
                if index % 100 == 0:
                    print(
                        f"scored {direction} {corpus}: {index}/{len(ranked)}",
                        file=sys.stderr,
                        flush=True,
                    )
            usable = [
                row for row in scored
                if row["student_chrf_pp"] >= args.minimum_student_chrf
            ]
            if len(usable) < args.select_per_domain_direction:
                raise SystemExit(
                    f"{direction} {corpus} has only {len(usable)} rows at chrF++ >= "
                    f"{args.minimum_student_chrf}; need {args.select_per_domain_direction}"
                )
            selected = hybrid_select(
                usable,
                args.select_per_domain_direction,
                f"{args.seed}:{corpus}:{direction}",
            )
            selection_reports[f"{corpus}:{direction}"] = {
                "eligible": len(eligible[(corpus, direction)]),
                "pool": len(ranked),
                "usable": len(usable),
                "selected": len(selected),
                "minimum_student_chrf_pp": args.minimum_student_chrf,
                **selection_summary(selected),
            }
            source_language, target_language = DIRECTIONS[direction]
            for row in selected:
                selected_output.append({
                    "id": row["id"],
                    "split": "train",
                    "source_language": source_language,
                    "target_language": target_language,
                    "domain": row["domain"],
                    "source": row["source"],
                    "license": row["license"],
                    "provenance": row["provenance"],
                    "reference_translation": row["reference"],
                    "reference_provenance": row["provenance"],
                    "student_hypothesis": row["student_hypothesis"],
                    "student_chrf_pp": row["student_chrf_pp"],
                    "student_sequence_nll": row["student_sequence_nll"],
                    "selection_uncertainty_stratum": row["selection_uncertainty_stratum"],
                    "selection_diversity_distance": row["selection_diversity_distance"],
                    "selection_rank": row["selection_rank"],
                    "selection": "incumbent uncertainty thirds plus encoder cosine k-center",
                    "reference_exposed_to_teacher": False,
                })
        del model, tokenizer
        gc.collect()
        mx.clear_cache()

    selected_output.sort(key=lambda row: row["id"])
    expected = len(CORPORA) * len(DIRECTIONS) * args.select_per_domain_direction
    if len(selected_output) != expected:
        raise SystemExit(f"expected {expected} selected seeds, found {len(selected_output)}")
    identifiers = [row["id"] for row in selected_output]
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit("selected seed IDs are not unique")
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected_output
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "purpose": "balanced reference-hidden local-teacher seeds; training only",
        "promotion_eligible": False,
        "reference_exposed_to_teacher": False,
        "reasoning_trace_requested_or_stored": False,
        "seed": args.seed,
        "selection": selection_reports,
        "counts": {
            "selected": len(selected_output),
            "by_domain_direction": dict(sorted(Counter(
                f"{row['domain']}:{row['source_language']}>{row['target_language']}"
                for row in selected_output
            ).items())),
            **inventory,
        },
        "licenses": dict(sorted(Counter(row["license"] for row in selected_output).items())),
        "model": {
            "path": str(args.model_pack.resolve()),
            "root_manifest_sha256": pack_manifest_sha,
            "source_revisions": pack_manifest.get("source_revisions"),
        },
        "inputs": {
            "corpora": {
                name: {"path": str(path.resolve()), "sha256": sha256(path)}
                for name, path in sorted(corpus_paths.items())
            },
            "protected_suites": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in args.protected_suite
            ],
            "excluded_datasets": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in exclusion_files
            ],
            "excluded_jsonl": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in args.exclude_jsonl
            ],
        },
        "output": {"path": str(args.output.resolve()), "sha256": sha256(args.output)},
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
