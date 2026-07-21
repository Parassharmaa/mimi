#!/usr/bin/env python3
"""Prepare a conservative conversational Tatoeba parallel corpus.

The ManyThings/Tatoeba rows are human-contributed and commercially usable with
per-row attribution, but their quality is variable. This gate requires an exact
reciprocal pair, rejects ambiguous one-to-many mappings and protected-suite near
matches, then retains only pairs on which both pinned directional students have
moderate agreement with the human references. Student agreement is a noise
filter, not independent quality evidence: ElanMT documents Tatoeba in its own
training data, so this output must never be used as evaluation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
LATIN_RE = re.compile(r"[A-Za-z]")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
NOISY_RE = re.compile(r"https?://|www\.|<[^>]+>|&(?:quot|amp|lt|gt);", re.I)


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text).replace(" ", "")
    return {value[index:index + size] for index in range(max(1, len(value) - size + 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def protected_ngrams(path: Path) -> list[set[str]]:
    protected: list[set[str]] = []
    for row in load_jsonl(path):
        for text in (row["source"], *row.get("references", [])):
            protected.append(ngrams(str(text)))
    return protected


def chat_pair(row: dict) -> tuple[str, str, str, str, str]:
    metadata = row.get("metadata", {})
    messages = row.get("messages", [])
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise SystemExit(f"invalid Tatoeba chat row: {metadata.get('source_id')}")
    return (
        str(metadata.get("source_id", "")).strip(),
        str(metadata.get("direction", "")).strip(),
        str(messages[1].get("content", "")).strip(),
        str(messages[2].get("content", "")).strip(),
        str(metadata.get("attribution", "")).strip(),
    )


def reciprocal_pairs(directory: Path) -> tuple[list[dict], Counter]:
    raw: list[dict] = []
    rejected: Counter = Counter()
    for split in ("train", "valid", "test"):
        rows = load_jsonl(directory / f"{split}.jsonl")
        if len(rows) % 2:
            raise SystemExit(f"Tatoeba {split} split has an unpaired final row")
        # prepare_tatoeba.py writes the two directions of each linked sentence
        # pair consecutively. The English sentence ID is not globally unique:
        # Tatoeba can link it to several Japanese translations. Preserve those
        # linked pairs here so the source-level ambiguity gate below can reject
        # every conflicting mapping instead of silently picking one.
        for index in range(0, len(rows), 2):
            linked: dict[str, tuple[str, str, str, dict]] = {}
            for row in rows[index:index + 2]:
                source_id, direction, source, target, attribution = chat_pair(row)
                metadata = row["metadata"]
                if not source_id or direction not in LANGUAGES or not source or not target:
                    raise SystemExit(f"incomplete Tatoeba row in {split}: {source_id}")
                if direction in linked:
                    raise SystemExit(f"duplicate linked direction in {split}: {source_id}/{direction}")
                if metadata.get("license") != "CC-BY-2.0-FR" or not attribution:
                    raise SystemExit(f"missing Tatoeba license/attribution: {source_id}")
                linked[direction] = (source, target, attribution, metadata)
            if set(linked) != set(LANGUAGES):
                rejected["missing-reciprocal-direction"] += 1
                continue
            en_source, ja_target, en_attribution, en_metadata = linked["en-ja"]
            ja_source, en_target, ja_attribution, ja_metadata = linked["ja-en"]
            if en_metadata["source_id"] != ja_metadata["source_id"]:
                rejected["source-id-mismatch"] += 1
                continue
            source_id = str(en_metadata["source_id"])
            if normalized(en_source) != normalized(en_target) or normalized(ja_source) != normalized(ja_target):
                rejected["non-reciprocal-text"] += 1
                continue
            if en_attribution != ja_attribution:
                rejected["attribution-mismatch"] += 1
                continue
            raw.append({
                "source_id": source_id,
                "split": split,
                "english": en_source,
                "japanese": ja_source,
                "attribution": en_attribution,
            })

    # Remove every member of a one-to-many source mapping in either direction.
    target_maps: dict[str, dict[str, set[str]]] = {
        "en-ja": defaultdict(set),
        "ja-en": defaultdict(set),
    }
    for row in raw:
        target_maps["en-ja"][normalized(row["english"])].add(normalized(row["japanese"]))
        target_maps["ja-en"][normalized(row["japanese"])].add(normalized(row["english"]))
    clean: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for row in raw:
        english = normalized(row["english"])
        japanese = normalized(row["japanese"])
        if len(target_maps["en-ja"][english]) != 1 or len(target_maps["ja-en"][japanese]) != 1:
            rejected["ambiguous-source"] += 1
            continue
        pair = (english, japanese)
        if pair in seen_pairs:
            rejected["duplicate-pair"] += 1
            continue
        seen_pairs.add(pair)
        clean.append(row)
    return clean, rejected


def eligible_text(row: dict, protected: list[set[str]], maximum_jaccard: float) -> str | None:
    english = row["english"]
    japanese = row["japanese"]
    if not (3 <= len(english) <= 120 and 2 <= len(japanese) <= 80):
        return "length"
    if len(LATIN_RE.findall(english)) < 2 or len(JAPANESE_RE.findall(japanese)) < 2:
        return "language"
    if NOISY_RE.search(english) or NOISY_RE.search(japanese):
        return "markup"
    if any(
        jaccard(ngrams(text), heldout) > maximum_jaccard
        for text in (english, japanese)
        for heldout in protected
    ):
        return "contamination"
    return None


def stable_sample(rows: list[dict], count: int, seed: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}\0{row['source_id']}".encode()).digest(),
    )[: min(count, len(rows))]


def score_direction(rows: list[dict], model_path: Path, direction: str, maximum_tokens: int) -> dict[str, dict]:
    import mlx.core as mx
    import sacrebleu
    from transformers import PreTrainedTokenizerFast

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from marian_mlx import load_model

    manifest_path = model_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("direction") != direction:
        raise SystemExit(f"model direction mismatch: {model_path}")
    model = load_model(
        model_path / "model.safetensors",
        quantization_bits=int(manifest["bits"]),
        quantization_group_size=int(manifest["group_size"]),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_path / "tokenizer.json"),
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    source_key, target_key = ("english", "japanese") if direction == "en-ja" else ("japanese", "english")
    scored: dict[str, dict] = {}
    for index, row in enumerate(rows, start=1):
        output_ids = model.generate(tokenizer.encode(row[source_key]), maximum_tokens)
        mx.synchronize()
        hypothesis = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        score = sacrebleu.sentence_chrf(
            hypothesis,
            [row[target_key]],
            word_order=2,
        ).score
        scored[row["source_id"]] = {"hypothesis": hypothesis, "chrf_pp": score}
        if index % 100 == 0:
            print(f"scored {direction}: {index}/{len(rows)}", file=sys.stderr, flush=True)
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return scored


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_tatoeba", type=Path)
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--en-ja-model", type=Path, required=True)
    parser.add_argument("--ja-en-model", type=Path, required=True)
    parser.add_argument("--maximum-train-pairs", type=int, default=4_000)
    parser.add_argument("--maximum-valid-pairs", type=int, default=500)
    parser.add_argument("--maximum-test-pairs", type=int, default=500)
    parser.add_argument("--minimum-en-ja-chrf", type=float, default=35.0)
    parser.add_argument("--minimum-ja-en-chrf", type=float, default=50.0)
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    parser.add_argument("--maximum-tokens", type=int, default=128)
    parser.add_argument("--sampling-seed", default="mimi-tatoeba-agreement-v1")
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    if min(args.maximum_train_pairs, args.maximum_valid_pairs, args.maximum_test_pairs) < 0:
        raise SystemExit("maximum pair counts must be non-negative")
    if not 0 <= args.maximum_jaccard <= 1:
        raise SystemExit("maximum-jaccard must be between zero and one")

    pairs, rejected = reciprocal_pairs(args.prepared_tatoeba)
    protected = protected_ngrams(args.protected_benchmark)
    clean: list[dict] = []
    for row in pairs:
        reason = eligible_text(row, protected, args.maximum_jaccard)
        if reason:
            rejected[reason] += 1
        else:
            clean.append(row)
    limits = {
        "train": args.maximum_train_pairs,
        "valid": args.maximum_valid_pairs,
        "test": args.maximum_test_pairs,
    }
    sampled = [
        row
        for split, limit in limits.items()
        for row in stable_sample(
            [candidate for candidate in clean if candidate["split"] == split],
            limit,
            f"{args.sampling_seed}:{split}",
        )
    ]
    if not sampled:
        raise SystemExit("no eligible Tatoeba pairs")

    scores = {
        "en-ja": score_direction(sampled, args.en_ja_model, "en-ja", args.maximum_tokens),
        "ja-en": score_direction(sampled, args.ja_en_model, "ja-en", args.maximum_tokens),
    }
    outputs: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    accepted_pairs = 0
    for row in sampled:
        en_score = scores["en-ja"][row["source_id"]]
        ja_score = scores["ja-en"][row["source_id"]]
        if en_score["chrf_pp"] < args.minimum_en_ja_chrf:
            rejected["low-en-ja-model-agreement"] += 1
            continue
        if ja_score["chrf_pp"] < args.minimum_ja_en_chrf:
            rejected["low-ja-en-model-agreement"] += 1
            continue
        accepted_pairs += 1
        for direction, source_key, target_key in (
            ("en-ja", "english", "japanese"),
            ("ja-en", "japanese", "english"),
        ):
            source_language, target_language = LANGUAGES[direction]
            outputs[row["split"]].append({
                "id": f"tatoeba:{row['source_id']}:{direction}",
                "source_id": row["source_id"],
                "source_language": source_language,
                "target_language": target_language,
                "source": row[source_key],
                "target": row[target_key],
                "domain": "conversational",
                "origin": "human-tatoeba-bidirectional-agreement-filtered",
                "source_license": "CC-BY-2.0-FR",
                "source_provenance": f"Tatoeba via ManyThings / {row['source_id']}",
                "attribution": row["attribution"],
                "quality_control": {
                    "en_ja_student_chrf_pp": en_score["chrf_pp"],
                    "ja_en_student_chrf_pp": ja_score["chrf_pp"],
                    "pretrained_overlap_warning": "ElanMT documents Tatoeba training; use only as training-noise filter, never evaluation",
                },
            })

    if not outputs["train"] or not outputs["valid"]:
        raise SystemExit("agreement thresholds left an empty train or validation split")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        write_jsonl(args.output_directory / f"{split}.jsonl", rows)
    manifest = {
        "schema_version": 1,
        "source": "Tatoeba via ManyThings",
        "license": "CC-BY-2.0-FR per row",
        "purpose": "conversational training ablation only; never evaluation",
        "quality_policy": "reciprocal human pair, unambiguous source, protected-screened, bidirectional pinned-student agreement",
        "pretrained_overlap_warning": "ElanMT documents Tatoeba in pretraining; agreement is not independent quality evidence",
        "input": {
            "directory": str(args.prepared_tatoeba),
            "manifest_sha256": sha256(args.prepared_tatoeba / "manifest.json"),
        },
        "protected_benchmark": {
            "path": str(args.protected_benchmark),
            "sha256": sha256(args.protected_benchmark),
            "maximum_jaccard": args.maximum_jaccard,
        },
        "models": {
            "en-ja": {"path": str(args.en_ja_model), "manifest_sha256": sha256(args.en_ja_model / "manifest.json")},
            "ja-en": {"path": str(args.ja_en_model), "manifest_sha256": sha256(args.ja_en_model / "manifest.json")},
        },
        "thresholds": {
            "minimum_en_ja_chrf_pp": args.minimum_en_ja_chrf,
            "minimum_ja_en_chrf_pp": args.minimum_ja_en_chrf,
        },
        "sampling": {"seed": args.sampling_seed, "maximum_pairs": limits},
        "counts": {
            "reciprocal_unambiguous_eligible": len(clean),
            "scored": len(sampled),
            "accepted_pairs": accepted_pairs,
            "outputs": {split: len(rows) for split, rows in outputs.items()},
            "rejected": dict(sorted(rejected.items())),
        },
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
