#!/usr/bin/env python3
"""Mine hard, licensed teacher seeds using the exact quantized MLX student.

The teacher receives only source text. Existing professional/shipping
translations and weak-student hypotheses stay local for provenance and human
review; they are never embedded in the Batch request by
prepare_synthetic_batch.py.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import mlx.core as mx
import sacrebleu
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marian_mlx import load_model  # noqa: E402
from dqrd_selection import hybrid_select, selection_summary  # noqa: E402


LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
NOISY_MARKUP_RE = re.compile(r"@[-,.]@|https?://|www\.|<[^>]+>|&(?:quot|amp|lt|gt);", re.I)
LIST_PREFIX_RE = re.compile(r"^(?:[-－]|\d+[.)、．])")


def load_rows(path: Path) -> list[dict]:
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


def near_protected(text: str, protected: list[set[str]], threshold: float) -> bool:
    candidate = ngrams(text)
    return any(
        len(candidate & heldout) / max(1, len(candidate | heldout)) > threshold
        for heldout in protected
    )


def deterministic_rank(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


def split_bucket(source_id: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{source_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def kftt_pair(row: dict, direction: str) -> dict | None:
    metadata = row.get("metadata", {})
    if metadata.get("direction") != direction:
        return None
    messages = row.get("messages", [])
    if len(messages) != 3 or [message.get("role") for message in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise SystemExit(f"unexpected KFTT row shape: {metadata.get('source_id')}")
    source = str(messages[1]["content"]).strip()
    reference = str(messages[2]["content"]).strip()
    if NOISY_MARKUP_RE.search(source) or NOISY_MARKUP_RE.search(reference):
        return None
    if LIST_PREFIX_RE.search(source) or source.startswith(("(", "（")):
        return None
    if direction == "en-ja" and not (12 <= len(source) <= 180 and 3 <= len(reference) <= 80):
        return None
    if direction == "ja-en" and not (3 <= len(source) <= 80 and 12 <= len(reference) <= 180):
        return None
    terminators = {
        "en": set(".?!:;)]}\"'"),
        "ja": set("。！？：；）］】」』.?!\"'"),
    }
    source_kind, reference_kind = ("en", "ja") if direction == "en-ja" else ("ja", "en")
    if source[-1] not in terminators[source_kind] or reference[-1] not in terminators[reference_kind]:
        return None
    return {
        "source_id": str(metadata["source_id"]),
        "source": source,
        "reference": reference,
        "license": str(metadata["license"]),
        "provenance": f"{metadata['source']} / {metadata['source_id']} / {metadata['attribution']}",
    }


def mine_direction(
    rows: list[dict],
    model_path: Path,
    direction: str,
    pool_size: int,
    selected_count: int,
    seed: str,
    maximum_tokens: int,
    protected: list[set[str]],
    maximum_jaccard: float,
    minimum_student_chrf: float,
    selection_strategy: str,
) -> tuple[list[dict], dict]:
    candidates = [
        pair
        for row in rows
        if (pair := kftt_pair(row, direction)) is not None
        and not near_protected(pair["source"], protected, maximum_jaccard)
        and not near_protected(pair["reference"], protected, maximum_jaccard)
    ]
    candidates.sort(key=lambda row: deterministic_rank(seed, f"{direction}:{row['source_id']}"))
    pool = candidates[:pool_size]
    if len(pool) < selected_count:
        raise SystemExit(
            f"{direction} has only {len(pool)} eligible pool rows for {selected_count} selections"
        )

    manifest = json.loads((model_path / "manifest.json").read_text(encoding="utf-8"))
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
    scored: list[dict] = []
    for index, row in enumerate(pool, start=1):
        token_ids = tokenizer.encode(row["source"])
        output_ids, diagnostics = model.generate_with_diagnostics(token_ids, maximum_tokens)
        mx.synchronize()
        hypothesis = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        score = sacrebleu.sentence_chrf(
            hypothesis,
            [row["reference"]],
            word_order=2,
        ).score
        scored.append(
            {
                **row,
                "id": row["source_id"],
                "student_hypothesis": hypothesis,
                "student_chrf_pp": score,
                "student_sequence_nll": diagnostics["student_sequence_nll"],
                "_selection_embedding": diagnostics["encoder_embedding"],
            }
        )
        if index % 100 == 0:
            print(f"mined {direction}: {index}/{len(pool)}", file=sys.stderr, flush=True)

    # Low-but-nonzero agreement identifies a weak student without letting
    # fragmentary alignment failures dominate the teacher/reviewer budget.
    eligible_scored = [
        row for row in scored if row["student_chrf_pp"] >= minimum_student_chrf
    ]
    if len(eligible_scored) < selected_count:
        raise SystemExit(
            f"{direction} has only {len(eligible_scored)} scored rows at or above "
            f"chrF++ {minimum_student_chrf}; need {selected_count}"
        )
    if selection_strategy == "uncertainty-diversity":
        selected = hybrid_select(
            eligible_scored,
            selected_count,
            f"{seed}:{direction}",
        )
        selection = selection_summary(selected)
    else:
        eligible_scored.sort(
            key=lambda row: (
                row["student_chrf_pp"],
                deterministic_rank(seed, f"tie:{direction}:{row['source_id']}"),
            )
        )
        selected = [
            {key: value for key, value in row.items() if key != "_selection_embedding"}
            for row in eligible_scored[:selected_count]
        ]
        selection = {"algorithm": "lowest-reference-chrf-v1"}
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    scores = sorted(row["student_chrf_pp"] for row in selected)
    summary = {
        "eligible": len(candidates),
        "pool": len(pool),
        "selected": len(selected),
        "minimum_student_chrf_pp": minimum_student_chrf,
        "selected_chrf_pp": {
            "minimum": scores[0],
            "median": scores[len(scores) // 2],
            "maximum": scores[-1],
        },
        "selection": selection,
        "model_revision": manifest["source_revision"],
        "model_sha256": manifest["files"]["model.safetensors"]["sha256"],
    }
    return selected, summary


def ui_seeds(path: Path, protected: list[set[str]], maximum_jaccard: float) -> list[dict]:
    output: list[dict] = []
    for row in load_rows(path):
        if near_protected(row["source"], protected, maximum_jaccard) or near_protected(
            row["target"], protected, maximum_jaccard
        ):
            continue
        direction = (
            "en-ja"
            if (row.get("source_language"), row.get("target_language")) == LANGUAGES["en-ja"]
            else "ja-en"
        )
        if (row.get("source_language"), row.get("target_language")) != LANGUAGES[direction]:
            raise SystemExit(f"unsupported Mimi UI direction: {row.get('id')}")
        output.append(
            {
                "id": f"teacher-ui:{direction}:{row['id']}",
                "split": "train",
                "source_language": row["source_language"],
                "target_language": row["target_language"],
                "domain": "mimi-product-ui",
                "source": row["source"],
                "license": row["source_license"],
                "provenance": row["source_provenance"],
                "reference_translation": row["target"],
                "reference_provenance": row["source_provenance"],
                "selection": "Mimi shipping UI pair",
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kftt_train", type=Path)
    parser.add_argument("mimi_ui_train", type=Path)
    parser.add_argument("model_pack", type=Path)
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pool-per-direction", type=int, default=1_500)
    parser.add_argument("--hard-kftt-per-direction", type=int, default=900)
    parser.add_argument("--maximum-tokens", type=int, default=192)
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    parser.add_argument("--minimum-student-chrf", type=float, default=10.0)
    parser.add_argument(
        "--selection-strategy",
        choices=("uncertainty-diversity", "hardest-reference"),
        default="uncertainty-diversity",
    )
    parser.add_argument("--seed", default="mimi-dqrd-distillation-v1")
    parser.add_argument("--split-seed", default="mimi-distillation-v1")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-synthetic-train", type=int, default=500)
    parser.add_argument("--minimum-synthetic-validation", type=int, default=50)
    args = parser.parse_args()

    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if not 0 < args.hard_kftt_per_direction <= args.pool_per_direction:
        raise SystemExit("hard selection must be positive and no larger than the pool")
    if not 0 < args.validation_fraction < 0.5:
        raise SystemExit("validation-fraction must be between 0 and 0.5")
    if min(args.minimum_synthetic_train, args.minimum_synthetic_validation) < 1:
        raise SystemExit("minimum synthetic train and validation counts must be positive")

    kftt = load_rows(args.kftt_train)
    protected = [
        ngrams(text)
        for row in load_rows(args.protected_benchmark)
        for text in (row["source"], *row.get("references", []))
    ]
    output = ui_seeds(args.mimi_ui_train, protected, args.maximum_jaccard)
    summaries: dict[str, dict] = {}
    for direction in LANGUAGES:
        selected, summary = mine_direction(
            kftt,
            args.model_pack / direction,
            direction,
            args.pool_per_direction,
            args.hard_kftt_per_direction,
            args.seed,
            args.maximum_tokens,
            protected,
            args.maximum_jaccard,
            args.minimum_student_chrf,
            args.selection_strategy,
        )
        source_language, target_language = LANGUAGES[direction]
        for row in selected:
            output.append(
                {
                    "id": f"teacher-kftt:{direction}:{row['source_id']}",
                    "split": "train",
                    "source_language": source_language,
                    "target_language": target_language,
                    "domain": "professional-wikipedia-hard",
                    "source": row["source"],
                    "license": row["license"],
                    "provenance": row["provenance"],
                    "reference_translation": row["reference"],
                    "reference_provenance": row["provenance"],
                    "student_hypothesis": row["student_hypothesis"],
                    "student_chrf_pp": row["student_chrf_pp"],
                    "student_sequence_nll": row["student_sequence_nll"],
                    "selection_uncertainty_stratum": row.get(
                        "selection_uncertainty_stratum"
                    ),
                    "selection_diversity_distance": row.get(
                        "selection_diversity_distance"
                    ),
                    "selection_rank": row.get("selection_rank"),
                    "selection": (
                        "uncertainty-plus-encoder-diversity with exact 4-bit MLX student"
                        if args.selection_strategy == "uncertainty-diversity"
                        else "hard example mined with exact 4-bit MLX student"
                    ),
                }
            )
        summaries[direction] = summary

    identifiers = [row["id"] for row in output]
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit("seed output contains duplicate IDs")
    source_keys = [
        (
            row["source_language"],
            row["target_language"],
            normalized(row["source"]),
        )
        for row in output
    ]
    if len(source_keys) != len(set(source_keys)):
        raise SystemExit("seed output contains a duplicate normalized source in one direction")
    output.sort(key=lambda row: row["id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    counts = {
        direction: sum(
            1
            for row in output
            if (row["source_language"], row["target_language"]) == LANGUAGES[direction]
        )
        for direction in LANGUAGES
    }
    approval_capacity: dict[str, dict] = {}
    for direction, languages in LANGUAGES.items():
        direction_rows = [
            row
            for row in output
            if (row["source_language"], row["target_language"]) == languages
        ]
        validation_count = sum(
            split_bucket(row["id"], args.split_seed) < args.validation_fraction
            for row in direction_rows
        )
        training_count = len(direction_rows) - validation_count
        if not training_count or not validation_count:
            raise SystemExit(f"deterministic split left an empty partition: {direction}")
        required_fraction = max(
            args.minimum_synthetic_train / training_count,
            args.minimum_synthetic_validation / validation_count,
        )
        approval_capacity[direction] = {
            "pre_review_train": training_count,
            "pre_review_validation": validation_count,
            "minimum_train": args.minimum_synthetic_train,
            "minimum_validation": args.minimum_synthetic_validation,
            "minimum_uniform_approval_fraction": required_fraction,
            "maximum_uniform_rejection_fraction": 1 - required_fraction,
        }
    manifest = {
        "schema_version": 1,
        "provisional": True,
        "provisional_reason": (
            "The final adjudicated held-out suite is not frozen; rerun all contamination "
            "checks and regenerate this batch before submission."
        ),
        "seed": args.seed,
        "selection_strategy": args.selection_strategy,
        "counts": counts,
        "approval_capacity": approval_capacity,
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "hard_mining": summaries,
        "teacher_visibility": "source text only; references and student hypotheses remain local",
        "maximum_protected_jaccard": args.maximum_jaccard,
        "inputs": {
            "kftt_train": {"path": str(args.kftt_train), "sha256": sha256(args.kftt_train)},
            "mimi_ui_train": {
                "path": str(args.mimi_ui_train),
                "sha256": sha256(args.mimi_ui_train),
            },
            "model_pack_manifest": {
                "path": str(args.model_pack / "manifest.json"),
                "sha256": sha256(args.model_pack / "manifest.json"),
            },
            "protected_benchmark": {
                "path": str(args.protected_benchmark),
                "sha256": sha256(args.protected_benchmark),
            },
        },
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
