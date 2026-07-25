#!/usr/bin/env python3
"""Evaluate v10 full-precision checkpoints on its frozen development split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from transformers import MarianMTModel, MarianTokenizer

from audit_translation_structures import critical_tokens, tokens
from typed_critical_token_policy import typed_preserves


EXPERIMENT = "canonical-sequence-v10-ja-en-error-stratified"
TEACHER_ORIGIN = "gpt56-claude5-approved-canonical-sequence"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def repeated_token_loop(ids: list[int]) -> bool:
    for width in range(3, min(16, len(ids) // 3) + 1):
        for start in range(0, len(ids) - width * 3 + 1):
            phrase = ids[start : start + width]
            if (
                ids[start + width : start + width * 2] == phrase
                and ids[start + width * 2 : start + width * 3] == phrase
            ):
                return True
    return False


def trim_generated(
    values: list[int],
    *,
    eos_token_id: int,
    pad_token_id: int,
    decoder_start_token_id: int | None,
) -> tuple[list[int], bool]:
    output = list(values)
    if output and decoder_start_token_id is not None and (
        output[0] == decoder_start_token_id
    ):
        output = output[1:]
    terminated = False
    trimmed = []
    for value in output:
        if value == eos_token_id:
            terminated = True
            break
        if value == pad_token_id:
            break
        trimmed.append(value)
    return trimmed, terminated


def generation_rows(
    checkpoint: Path,
    tokenizer: MarianTokenizer,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    maximum_source_tokens: int,
    maximum_target_tokens: int,
) -> list[dict[str, Any]]:
    model = MarianMTModel.from_pretrained(checkpoint).to(device)
    model.eval()
    output = []
    eos = int(tokenizer.eos_token_id)
    pad = int(tokenizer.pad_token_id)
    start = model.config.decoder_start_token_id
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch_rows = rows[offset : offset + batch_size]
            batch = tokenizer(
                [str(row["source"]) for row in batch_rows],
                padding=True,
                truncation=True,
                max_length=maximum_source_tokens,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            generated = model.generate(
                **batch,
                do_sample=False,
                num_beams=1,
                max_new_tokens=maximum_target_tokens,
            )
            for row, raw in zip(batch_rows, generated.tolist()):
                ids, terminated = trim_generated(
                    raw,
                    eos_token_id=eos,
                    pad_token_id=pad,
                    decoder_start_token_id=start,
                )
                output.append(
                    {
                        "id": row["id"],
                        "hypothesis": tokenizer.decode(
                            ids, skip_special_tokens=True
                        ),
                        "output_token_ids": ids,
                        "terminated": terminated,
                        "reached_generation_limit": (
                            not terminated
                            and len(ids) >= maximum_target_tokens
                        ),
                        "repeated_token_loop": repeated_token_loop(ids),
                    }
                )
    if device.type == "mps":
        torch.mps.synchronize()
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return output


def sentence_score(hypothesis: str, reference: str) -> float:
    return float(
        sacrebleu.sentence_chrf(
            hypothesis,
            [reference],
            word_order=2,
        ).score
    )


def structure_failures(
    rows: list[dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    result = {
        "exact": set(),
        "typed": set(),
        "negation": set(),
        "generation": set(),
    }
    for row in rows:
        case_id = str(row["id"])
        generated = outputs[case_id]
        source = str(row["source"])
        hypothesis = str(generated["hypothesis"])
        if critical_tokens(source) != critical_tokens(hypothesis):
            result["exact"].add(case_id)
        if not typed_preserves(
            source,
            hypothesis,
            "ja-JP",
            "en-US",
        ):
            result["typed"].add(case_id)
        if tokens(source)["negative"] != tokens(hypothesis)["negative"]:
            result["negation"].add(case_id)
        if (
            generated["reached_generation_limit"]
            or generated["repeated_token_loop"]
            or not hypothesis.strip()
        ):
            result["generation"].add(case_id)
    return result


def aggregate(
    rows: list[dict[str, Any]],
    generated: list[dict[str, Any]],
) -> dict[str, Any]:
    indexed = {str(row["id"]): row for row in generated}
    if set(indexed) != {str(row["id"]) for row in rows}:
        raise SystemExit("generation does not cover the exact v10 validation set")
    hypotheses = [str(indexed[str(row["id"])]["hypothesis"]) for row in rows]
    references = [str(row["target"]) for row in rows]
    scores = {
        str(row["id"]): sentence_score(hypothesis, reference)
        for row, hypothesis, reference in zip(rows, hypotheses, references)
    }
    strata: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        stratum = str(
            row.get("v10_stratum")
            or (
                "teacher"
                if row.get("origin") == TEACHER_ORIGIN
                else row.get("sequence_role", "unknown")
            )
        )
        strata[stratum].append(str(row["id"]))
    failures = structure_failures(rows, indexed)
    return {
        "corpus_chrf_pp": float(
            sacrebleu.corpus_chrf(
                hypotheses,
                [references],
                word_order=2,
            ).score
        ),
        "corpus_bleu": float(
            sacrebleu.corpus_bleu(
                hypotheses,
                [references],
                tokenize="13a",
            ).score
        ),
        "mean_sentence_chrf_pp": sum(scores.values()) / len(scores),
        "sentence_chrf_pp": scores,
        "strata": {
            stratum: {
                "cases": len(case_ids),
                "mean_sentence_chrf_pp": sum(scores[value] for value in case_ids)
                / len(case_ids),
                "corpus_chrf_pp": float(
                    sacrebleu.corpus_chrf(
                        [indexed[value]["hypothesis"] for value in case_ids],
                        [[
                            str(
                                next(
                                    row["target"]
                                    for row in rows
                                    if str(row["id"]) == value
                                )
                            )
                            for value in case_ids
                        ]],
                        word_order=2,
                    ).score
                ),
            }
            for stratum, case_ids in sorted(strata.items())
        },
        "failures": {
            name: sorted(values) for name, values in failures.items()
        },
        "outputs": indexed,
    }


def bootstrap(
    values: list[float],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.90,
) -> dict[str, Any]:
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": sum(values) / len(values),
        "lower": means[max(0, math.floor(samples * tail))],
        "upper": means[
            min(samples - 1, math.ceil(samples * (1.0 - tail)) - 1)
        ],
        "samples": samples,
        "seed": seed,
        "confidence": confidence,
    }


def validate_candidate_manifest(
    manifest: dict[str, Any],
    contract: dict[str, Any],
    *,
    step: int,
) -> None:
    training = contract["training"]
    if (
        manifest.get("direction") != "ja-en"
        or manifest.get("student_repository")
        != contract["parent"]["repository"]
        or manifest.get("student_revision")
        != contract["parent"]["revision"]
        or manifest.get("checkpoint_step") != step
        or manifest.get("initial_checkpoint", {}).get("model_sha256")
        != contract["parent"]["model"]["sha256"]
        or manifest.get("preservation_checkpoint", {}).get("model_sha256")
        != contract["parent"]["model"]["sha256"]
        or manifest.get("dataset_manifest", {}).get("sha256")
        != contract["dataset"]["manifest"]["sha256"]
    ):
        raise SystemExit(f"candidate lineage differs at step {step}")
    actual = manifest.get("hyperparameters", {})
    expected = {
        "seed": training["seed"],
        "batch_size": training["batch_size"],
        "gradient_accumulation": training["gradient_accumulation"],
        "max_steps": training["max_steps"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "warmup_steps": training["warmup_steps"],
        "evaluation_steps": training["evaluation_steps"],
        "max_source_tokens": training["max_source_tokens"],
        "max_target_tokens": training["max_target_tokens"],
        "gradient_checkpointing": training["gradient_checkpointing"],
        "frozen_base_kl_weight": training["frozen_parent_kl_weight"],
        "l2_to_base_weight": training["l2_to_parent_weight"],
        "domain_loss_weight_start": training[
            "canonical_sequence_loss_weight_start"
        ],
        "domain_loss_weight_end": training[
            "canonical_sequence_loss_weight_end"
        ],
        "curriculum_ramp_steps": training["curriculum_ramp_steps"],
        "preservation_origins": training["preservation_origins"],
    }
    if any(actual.get(key) != value for key, value in expected.items()):
        raise SystemExit(f"candidate hyperparameters differ at step {step}")


def candidate_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    requirements: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deltas = {
        "corpus_chrf_pp": candidate["corpus_chrf_pp"]
        - baseline["corpus_chrf_pp"],
        "mean_sentence_chrf_pp": candidate["mean_sentence_chrf_pp"]
        - baseline["mean_sentence_chrf_pp"],
        "teacher_slice_chrf_pp": candidate["strata"]["teacher"][
            "mean_sentence_chrf_pp"
        ]
        - baseline["strata"]["teacher"]["mean_sentence_chrf_pp"],
        "long_legal_chrf_pp": candidate["strata"]["legal:long"][
            "mean_sentence_chrf_pp"
        ]
        - baseline["strata"]["legal:long"]["mean_sentence_chrf_pp"],
        "by_stratum": {
            stratum: candidate["strata"][stratum]["mean_sentence_chrf_pp"]
            - baseline["strata"][stratum]["mean_sentence_chrf_pp"]
            for stratum in baseline["strata"]
        },
    }
    deltas["worst_stratum_chrf_pp"] = min(
        deltas["by_stratum"].values()
    )
    new_failures = {
        name: sorted(
            set(candidate["failures"][name])
            - set(baseline["failures"][name])
        )
        for name in ("exact", "typed", "negation", "generation")
    }
    gates = [
        {
            "name": "corpus-chrf++",
            "passed": deltas["corpus_chrf_pp"]
            >= requirements["corpus_chrf_pp_delta_minimum"],
            "actual": deltas["corpus_chrf_pp"],
            "minimum": requirements["corpus_chrf_pp_delta_minimum"],
        },
        {
            "name": "mean-sentence-chrf++",
            "passed": deltas["mean_sentence_chrf_pp"]
            >= requirements["mean_sentence_chrf_pp_delta_minimum"],
            "actual": deltas["mean_sentence_chrf_pp"],
            "minimum": requirements[
                "mean_sentence_chrf_pp_delta_minimum"
            ],
        },
        {
            "name": "teacher-slice-chrf++",
            "passed": deltas["teacher_slice_chrf_pp"]
            >= requirements["teacher_slice_chrf_pp_delta_minimum"],
            "actual": deltas["teacher_slice_chrf_pp"],
            "minimum": requirements[
                "teacher_slice_chrf_pp_delta_minimum"
            ],
        },
        {
            "name": "long-legal-chrf++",
            "passed": deltas["long_legal_chrf_pp"]
            >= requirements["long_legal_chrf_pp_delta_minimum"],
            "actual": deltas["long_legal_chrf_pp"],
            "minimum": requirements[
                "long_legal_chrf_pp_delta_minimum"
            ],
        },
        {
            "name": "worst-stratum-chrf++",
            "passed": deltas["worst_stratum_chrf_pp"]
            >= requirements["worst_stratum_chrf_pp_delta_minimum"],
            "actual": deltas["worst_stratum_chrf_pp"],
            "minimum": requirements[
                "worst_stratum_chrf_pp_delta_minimum"
            ],
        },
        *[
            {
                "name": f"new-{name}-failures",
                "passed": len(new_failures[name])
                <= requirements[requirement],
                "actual": len(new_failures[name]),
                "maximum": requirements[requirement],
                "case_ids": new_failures[name],
            }
            for name, requirement in (
                ("exact", "new_exact_critical_failures_maximum"),
                ("typed", "new_typed_critical_failures_maximum"),
                ("negation", "new_negation_failures_maximum"),
                (
                    "generation",
                    "new_repetition_or_generation_limit_failures_maximum",
                ),
            )
        ],
    ]
    return gates, {
        "deltas": deltas,
        "new_failures": new_failures,
    }


def public_aggregate(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"sentence_chrf_pp", "outputs"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("STEP", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if args.batch_size < 1 or args.bootstrap_samples < 100:
        raise SystemExit("batch size and bootstrap samples must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    contract = load_json(args.contract)
    if (
        contract.get("experiment") != EXPERIMENT
        or contract.get("status")
        != "preregistered-ready-for-one-arm-training"
        or contract.get("internal_selection", {}).get(
            "selection_uses_protected_outputs"
        )
        is not False
        or contract.get("app_change_authorized") is not False
    ):
        raise SystemExit("v10 contract is invalid")
    candidate_paths = {
        int(step): Path(path) for step, path in args.candidate
    }
    if sorted(candidate_paths) != contract["internal_selection"][
        "candidate_steps"
    ]:
        raise SystemExit("candidate checkpoint steps differ from the contract")

    valid_path = Path(contract["dataset"]["valid"]["path"])
    if sha256(valid_path) != contract["dataset"]["valid"]["sha256"]:
        raise SystemExit("v10 validation data hash differs")
    rows = load_jsonl(valid_path)
    if len(rows) != 1_024:
        raise SystemExit("v10 validation row count differs")
    parent = Path(contract["parent"]["path"])
    if (
        sha256(parent / "model.safetensors")
        != contract["parent"]["model"]["sha256"]
    ):
        raise SystemExit("v10 parent hash differs")
    tokenizer = MarianTokenizer.from_pretrained(parent)
    device = torch.device(args.device)
    maximum_source_tokens = contract["training"]["max_source_tokens"]
    maximum_target_tokens = contract["training"]["max_target_tokens"]

    baseline_generated = generation_rows(
        parent,
        tokenizer,
        rows,
        device=device,
        batch_size=args.batch_size,
        maximum_source_tokens=maximum_source_tokens,
        maximum_target_tokens=maximum_target_tokens,
    )
    baseline = aggregate(rows, baseline_generated)
    candidates = []
    for step in sorted(candidate_paths):
        path = candidate_paths[step]
        manifest_path = path / "mimi_training_manifest.json"
        manifest = load_json(manifest_path)
        validate_candidate_manifest(
            manifest,
            contract,
            step=step,
        )
        generated = generation_rows(
            path,
            tokenizer,
            rows,
            device=device,
            batch_size=args.batch_size,
            maximum_source_tokens=maximum_source_tokens,
            maximum_target_tokens=maximum_target_tokens,
        )
        metrics = aggregate(rows, generated)
        gates, decision = candidate_decision(
            baseline,
            metrics,
            contract["internal_selection"]["requirements"],
        )
        deltas = [
            metrics["sentence_chrf_pp"][str(row["id"])]
            - baseline["sentence_chrf_pp"][str(row["id"])]
            for row in rows
        ]
        ranked = sorted(
            (
                (
                    metrics["sentence_chrf_pp"][str(row["id"])]
                    - baseline["sentence_chrf_pp"][str(row["id"])],
                    str(row["id"]),
                )
                for row in rows
            )
        )
        samples = {
            "largest_regressions": [
                {
                    "delta": delta,
                    "id": case_id,
                    "source": next(
                        row["source"]
                        for row in rows
                        if str(row["id"]) == case_id
                    ),
                    "reference": next(
                        row["target"]
                        for row in rows
                        if str(row["id"]) == case_id
                    ),
                    "baseline": baseline["outputs"][case_id]["hypothesis"],
                    "candidate": metrics["outputs"][case_id]["hypothesis"],
                }
                for delta, case_id in ranked[:10]
            ],
            "largest_improvements": [
                {
                    "delta": delta,
                    "id": case_id,
                    "source": next(
                        row["source"]
                        for row in rows
                        if str(row["id"]) == case_id
                    ),
                    "reference": next(
                        row["target"]
                        for row in rows
                        if str(row["id"]) == case_id
                    ),
                    "baseline": baseline["outputs"][case_id]["hypothesis"],
                    "candidate": metrics["outputs"][case_id]["hypothesis"],
                }
                for delta, case_id in reversed(ranked[-10:])
            ],
        }
        candidates.append(
            {
                "step": step,
                "checkpoint": {
                    "path": str(path),
                    "model": record(path / "model.safetensors"),
                    "training_manifest": record(manifest_path),
                },
                "training_checkpoint_metrics": manifest[
                    "checkpoint_metrics"
                ],
                "metrics": public_aggregate(metrics),
                **decision,
                "paired_sentence_chrf_pp_bootstrap": bootstrap(
                    deltas,
                    samples=args.bootstrap_samples,
                    seed=20260802 + step,
                ),
                "gates": gates,
                "eligible": all(gate["passed"] for gate in gates),
                "samples": samples,
            }
        )

    eligible = [item for item in candidates if item["eligible"]]
    eligible.sort(
        key=lambda item: (
            item["deltas"]["mean_sentence_chrf_pp"],
            item["deltas"]["long_legal_chrf_pp"],
            item["deltas"]["teacher_slice_chrf_pp"],
            -float(item["training_checkpoint_metrics"]["loss"]),
        ),
        reverse=True,
    )
    selected = eligible[0] if eligible else None
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": (
            "internal-gate-passed"
            if selected is not None
            else "internal-gate-rejected"
        ),
        "decision": (
            "Authorize exact-q4 conversion and protected evaluation only."
            if selected is not None
            else (
                "Stop v10 before quantization, protected evaluation, app work, "
                "or public upload."
            )
        ),
        "contract": record(args.contract),
        "evaluation_implementation": record(Path(__file__).resolve()),
        "device": str(device),
        "batch_size": args.batch_size,
        "baseline": {
            "checkpoint": contract["parent"],
            "metrics": public_aggregate(baseline),
        },
        "candidates": candidates,
        "selected_step": selected["step"] if selected is not None else None,
        "selected_checkpoint": (
            selected["checkpoint"] if selected is not None else None
        ),
        "exact_q4_conversion_authorized": selected is not None,
        "protected_evaluation_authorized_after_exact_q4": selected
        is not None,
        "app_change_authorized": False,
        "bundle_creation_authorized": False,
        "promotion_authorized": False,
        "public_upload_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "status": result["status"],
                "selected_step": result["selected_step"],
                "candidates": [
                    {
                        "step": item["step"],
                        "eligible": item["eligible"],
                        "deltas": item["deltas"],
                        "failed_gates": [
                            gate["name"]
                            for gate in item["gates"]
                            if not gate["passed"]
                        ],
                    }
                    for item in candidates
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
