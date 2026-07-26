#!/usr/bin/env python3
"""Evaluate every V18 checkpoint before semantic audit or q4 conversion."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from transformers import MarianMTModel, MarianTokenizer

from audit_translation_structures import critical_tokens, tokens
from evaluate_canonical_sequence_v10_internal import repeated_token_loop, trim_generated
from typed_critical_token_policy import typed_preserves


EXPERIMENT = "shared-bidirectional-v18-wide-dense-phase1"
EVALUATION_EXPERIMENT = "shared-bidirectional-v18-full-precision-internal-eval"
DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
SOURCE_PREFIXES = {"en-ja": "<2ja> ", "ja-en": "<2en> "}
SUITE_ORDER = (
    "selector",
    "canary",
    "development-accuracy-v1",
    "legal-safety-validation-v1",
    "legal-safety-test-v1",
)
TOKENIZER_ASSETS = (
    "source.spm",
    "special_tokens_map.json",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSONL input {path}: {error}") from error
    if not values or not all(isinstance(value, dict) for value in values):
        raise SystemExit(f"expected non-empty JSON-object lines: {path}")
    identifiers = [str(value.get("id", "")).strip() for value in values]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise SystemExit(f"empty or duplicate suite ID: {path}")
    return values


def display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": display_path(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def hardware_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "numpy": importlib.metadata.version("numpy"),
        "sentencepiece": importlib.metadata.version("sentencepiece"),
        "sacremoses": importlib.metadata.version("sacremoses"),
        "sacrebleu": importlib.metadata.version("sacrebleu"),
    }


def direction(row: dict[str, Any]) -> str:
    pair = (row.get("sourceLanguage"), row.get("targetLanguage"))
    if pair == (None, None):
        value = str(row.get("direction", ""))
        if value in DIRECTIONS:
            return value
    for name, expected in DIRECTIONS.items():
        if pair == expected:
            return name
    raise SystemExit(f"unsupported translation direction for {row.get('id')}: {pair}")


def references(row: dict[str, Any]) -> list[str]:
    values = row.get("references")
    if isinstance(values, list) and values and all(
        isinstance(value, str) and value.strip() for value in values
    ):
        return values
    target = row.get("target")
    if isinstance(target, str) and target.strip():
        return [target]
    raise SystemExit(f"suite row lacks a reference: {row.get('id')}")


def source_segments(row: dict[str, Any]) -> list[str]:
    values = row.get("segments")
    if isinstance(values, list) and values:
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise SystemExit(f"invalid source segments: {row.get('id')}")
        return values
    source = row.get("source")
    if not isinstance(source, str) or not source.strip():
        raise SystemExit(f"suite row lacks source text: {row.get('id')}")
    return [source]


def normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    name = direction(row)
    source_language, target_language = DIRECTIONS[name]
    return {
        **row,
        "direction": name,
        "sourceLanguage": source_language,
        "targetLanguage": target_language,
        "source": str(row["source"]),
        "references": references(row),
        "segments": source_segments(row),
    }


def validate_suites(
    evaluation_contract: dict[str, Any],
    root: Path,
) -> dict[str, dict[str, Any]]:
    suites = {}
    for name in SUITE_ORDER:
        item = evaluation_contract.get("suites", {}).get(name, {})
        path = root / str(item.get("path", ""))
        if (
            not path.is_file()
            or item.get("sha256") != sha256(path)
            or int(item.get("cases", -1)) <= 0
        ):
            raise SystemExit(f"evaluation suite differs: {name}")
        rows = [normalized_row(row) for row in load_jsonl(path)]
        if len(rows) != int(item["cases"]):
            raise SystemExit(f"evaluation suite row count differs: {name}")
        if name == "selector":
            counts = {
                direction_name: sum(
                    row["direction"] == direction_name for row in rows
                )
                for direction_name in DIRECTIONS
            }
            if counts != {"en-ja": 256, "ja-en": 256}:
                raise SystemExit("V18 selector is not exactly direction-balanced")
        suites[name] = {"path": path, "rows": rows}
    return suites


def authenticate_checkpoint(
    checkpoint: Path,
    *,
    step: int,
    contract_sha256: str,
) -> dict[str, Any]:
    manifest_path = checkpoint / "mimi_checkpoint_manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment") != EXPERIMENT
        or int(manifest.get("step", -1)) != step
        or manifest.get("contract_sha256") != contract_sha256
        or manifest.get("immutable_scheduled_checkpoint") is not True
        or manifest.get("inference_config_use_cache") is not True
    ):
        raise SystemExit(f"checkpoint lineage differs at step {step}")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise SystemExit(f"checkpoint has no file inventory at step {step}")
    actual = {
        str(path.relative_to(checkpoint)): path
        for path in checkpoint.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(actual) != set(declared):
        raise SystemExit(f"checkpoint file inventory differs at step {step}")
    for name, item in actual.items():
        file_record = declared[name]
        if (
            not isinstance(file_record, dict)
            or item.stat().st_size != int(file_record.get("bytes", -1))
            or sha256(item) != file_record.get("sha256")
        ):
            raise SystemExit(f"checkpoint file differs at step {step}: {name}")
    return manifest


def validate_teacher(
    direction_name: str,
    training_contract: dict[str, Any],
    evaluation_contract: dict[str, Any],
    root: Path,
) -> Path:
    teacher = root / training_contract["teachers"][direction_name]["path"]
    declared = evaluation_contract.get("teachers", {}).get(direction_name, {})
    files = declared.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit(f"{direction_name} teacher inventory is missing")
    actual = {
        path.name: path
        for path in teacher.iterdir()
        if path.is_file()
    }
    if set(actual) != set(files):
        raise SystemExit(f"{direction_name} teacher inventory differs")
    for name, path in actual.items():
        item = files[name]
        if (
            not isinstance(item, dict)
            or path.stat().st_size != int(item.get("bytes", -1))
            or sha256(path) != item.get("sha256")
        ):
            raise SystemExit(f"{direction_name} teacher asset differs: {name}")
    lineage = training_contract["teachers"][direction_name]["lineage_manifest"]
    if (
        lineage["sha256"]
        != files[Path(str(lineage["path"])).name]["sha256"]
        or training_contract["teachers"][direction_name]["weights_sha256"]
        != files["model.safetensors"]["sha256"]
    ):
        raise SystemExit(f"{direction_name} teacher lineage differs")
    return teacher


def load_candidate_tokenizer(
    checkpoint: Path,
    evaluation_contract: dict[str, Any],
) -> MarianTokenizer:
    expected = evaluation_contract.get("candidate_tokenizer_assets", {})
    for name in TOKENIZER_ASSETS:
        path = checkpoint / name
        if not path.is_file() or sha256(path) != expected.get(name):
            raise SystemExit(f"candidate tokenizer asset differs: {name}")
    tokenizer = MarianTokenizer.from_pretrained(checkpoint)
    actual_prefix_ids = {
        direction_name: tokenizer.encode(prefix, add_special_tokens=False)
        for direction_name, prefix in SOURCE_PREFIXES.items()
    }
    if actual_prefix_ids != evaluation_contract.get("prefix_token_ids"):
        raise SystemExit("candidate direction-prefix token IDs differ")
    return tokenizer


def validate_training_summary(
    summary_path: Path,
    training_contract: dict[str, Any],
    evaluation_contract: dict[str, Any],
    checkpoints: dict[int, Path],
    root: Path,
) -> dict[str, Any]:
    summary = load_json(summary_path)
    contract_sha256 = evaluation_contract["training_contract"]["sha256"]
    phase = training_contract["phase1_training"]
    expected_hyperparameters = phase["hyperparameters"]
    actual_hyperparameters = summary.get("hyperparameters", {})
    for key, value in expected_hyperparameters.items():
        if actual_hyperparameters.get(key) != value:
            raise SystemExit(f"completed run hyperparameter differs: {key}")
    expected_checkpoint_records = {
        int(item.get("step", -1)): item
        for item in summary.get("checkpoints", [])
        if isinstance(item, dict)
    }
    if (
        summary.get("completed") is not True
        or summary.get("contract", {}).get("sha256") != contract_sha256
        or summary.get("source_prefixes") != SOURCE_PREFIXES
        or summary.get("runtime") != evaluation_contract["runtime"]["packages"]
        or summary.get("device") != evaluation_contract["runtime"]["device"]
        or summary.get("hardware") != evaluation_contract["runtime"]["hardware"]
        or summary.get("operating_system")
        != evaluation_contract["runtime"]["operating_system"]
        or sorted(expected_checkpoint_records) != sorted(checkpoints)
        or [int(item.get("step", -1)) for item in summary.get("history", [])]
        != [0, *sorted(checkpoints)]
        or summary.get("dataset", {}).get("manifest_sha256")
        != training_contract["dataset"]["manifest"]["sha256"]
        or summary.get("dataset", {}).get("train_sha256")
        != training_contract["dataset"]["train_sha256"]
        or summary.get("dataset", {}).get("valid_sha256")
        != training_contract["dataset"]["valid_sha256"]
        or summary.get("dataset", {}).get("selection_sha256")
        != training_contract["phase1_training"]["selection_artifact"]["sha256"]
    ):
        raise SystemExit("completed V18 training summary is not the authorized run")
    for direction_name in DIRECTIONS:
        if (
            summary.get("teachers", {}).get(direction_name, {}).get(
                "model_sha256"
            )
            != training_contract["teachers"][direction_name]["weights_sha256"]
        ):
            raise SystemExit(f"completed run teacher differs: {direction_name}")
    for step, checkpoint in checkpoints.items():
        checkpoint_record = expected_checkpoint_records[step]
        if (
            Path(str(checkpoint_record.get("path", ""))).resolve()
            != checkpoint.resolve()
            or checkpoint_record.get("manifest_sha256")
            != sha256(checkpoint / "mimi_checkpoint_manifest.json")
        ):
            raise SystemExit(f"completed run checkpoint binding differs: {step}")
    expected_summary = checkpoints[250].parents[1] / "mimi_training_manifest.json"
    if summary_path.resolve() != expected_summary.resolve():
        raise SystemExit("training summary is outside the checkpoint run directory")
    return summary


def validate_inputs(
    training_contract_path: Path,
    evaluation_contract_path: Path,
    training_summary_path: Path,
    checkpoints: dict[int, Path],
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    training_contract = load_json(training_contract_path)
    evaluation_contract = load_json(evaluation_contract_path)
    training_contract_sha256 = sha256(training_contract_path)
    if (
        training_contract.get("experiment") != EXPERIMENT
        or training_contract.get("promotion_authorized") is not False
        or training_contract.get("app_change_authorized") is not False
        or evaluation_contract.get("schema_version") != 1
        or evaluation_contract.get("experiment") != EVALUATION_EXPERIMENT
        or evaluation_contract.get("status")
        != "frozen-full-precision-evaluation-only"
        or evaluation_contract.get("training_contract", {}).get("sha256")
        != training_contract_sha256
        or evaluation_contract.get("implementation", {}).get("sha256")
        != sha256(Path(__file__).resolve())
        or evaluation_contract.get("semantic_audit_required_before_q4") is not True
        or evaluation_contract.get("q4_conversion_authorized") is not False
        or evaluation_contract.get("app_change_authorized") is not False
        or training_contract.get("architecture", {}).get("source_prefixes")
        != SOURCE_PREFIXES
        or evaluation_contract.get("runtime", {}).get("batch_size") != 4
        or evaluation_contract.get("runtime", {}).get("greedy_num_beams") != 1
        or evaluation_contract.get("runtime", {}).get("use_cache") is not True
        or evaluation_contract.get("runtime", {}).get("packages")
        != training_contract.get("runtime", {}).get("packages")
        or evaluation_contract.get("runtime", {}).get("device") != "mps"
        or evaluation_contract.get("selection_after_mechanical_gate")
        != "highest-selector-macro-then-lowest-independent-selector-loss-then-lowest-step"
        or evaluation_contract.get("semantic_failure_behavior")
        != "stop-experiment-without-falling-through-to-another-checkpoint"
    ):
        raise SystemExit("V18 evaluation contract is invalid")
    expected_steps = [
        int(value) for value in evaluation_contract.get("checkpoint_steps", [])
    ]
    if sorted(checkpoints) != expected_steps:
        raise SystemExit("supplied checkpoint steps differ from evaluation contract")
    observed = evaluation_contract.get("observed_before_evaluation_freeze", {})
    if (
        observed.get("maximum_observed_step") != 250
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(observed.get("step_250_checkpoint_manifest_sha256", "")),
        )
        or 250 not in checkpoints
        or sha256(checkpoints[250] / "mimi_checkpoint_manifest.json")
        != observed.get("step_250_checkpoint_manifest_sha256")
    ):
        raise SystemExit("evaluation freeze disclosure is invalid")
    if (
        load_json(
            checkpoints[250] / "mimi_checkpoint_manifest.json"
        ).get("metrics")
        != observed.get("step_250_selector_metrics")
    ):
        raise SystemExit("disclosed step-250 metrics differ")
    suites = validate_suites(evaluation_contract, root)
    for step, checkpoint in checkpoints.items():
        authenticate_checkpoint(
            checkpoint,
            step=step,
            contract_sha256=training_contract_sha256,
        )
    summary = validate_training_summary(
        training_summary_path,
        training_contract,
        evaluation_contract,
        checkpoints,
        root,
    )
    return training_contract, evaluation_contract, summary, suites


def tokenizer_source_truncated(
    tokenizer: MarianTokenizer,
    text: str,
    maximum_source_tokens: int,
) -> bool:
    return len(tokenizer.encode(text, add_special_tokens=True)) > maximum_source_tokens


def generate(
    checkpoint: Path,
    tokenizer: MarianTokenizer,
    suites: dict[str, dict[str, Any]],
    *,
    active_directions: set[str],
    source_prefixes: dict[str, str],
    device: torch.device,
    batch_size: int,
    maximum_source_tokens: int,
    maximum_target_tokens: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    model = MarianMTModel.from_pretrained(checkpoint).to(device).eval()
    eos = int(tokenizer.eos_token_id)
    pad = int(tokenizer.pad_token_id)
    start = model.config.decoder_start_token_id
    generated_by_suite: dict[str, dict[str, dict[str, Any]]] = {}
    with torch.inference_mode():
        for suite_name, suite in suites.items():
            rows = [
                row
                for row in suite["rows"]
                if row["direction"] in active_directions
            ]
            units = []
            for row in rows:
                prefix = source_prefixes.get(row["direction"], "")
                for index, segment in enumerate(row["segments"]):
                    text = prefix + segment
                    units.append(
                        {
                            "case_id": str(row["id"]),
                            "segment_index": index,
                            "text": text,
                            "input_token_ids": tokenizer.encode(
                                text,
                                add_special_tokens=True,
                            )[:maximum_source_tokens],
                            "source_truncated": tokenizer_source_truncated(
                                tokenizer,
                                text,
                                maximum_source_tokens,
                            ),
                        }
                    )
            unit_outputs: dict[tuple[str, int], dict[str, Any]] = {}
            for offset in range(0, len(units), batch_size):
                batch_units = units[offset : offset + batch_size]
                encoded = tokenizer(
                    [unit["text"] for unit in batch_units],
                    padding=True,
                    truncation=True,
                    max_length=maximum_source_tokens,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                raw_outputs = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=maximum_target_tokens,
                    use_cache=True,
                )
                for unit, raw in zip(batch_units, raw_outputs.tolist()):
                    token_ids, terminated = trim_generated(
                        raw,
                        eos_token_id=eos,
                        pad_token_id=pad,
                        decoder_start_token_id=start,
                    )
                    unit_outputs[(unit["case_id"], unit["segment_index"])] = {
                        "hypothesis": tokenizer.decode(
                            token_ids,
                            skip_special_tokens=True,
                        ).strip(),
                        "output_token_ids": token_ids,
                        "input_token_ids": unit["input_token_ids"],
                        "terminated": terminated,
                        "source_truncated": unit["source_truncated"],
                        "reached_generation_limit": (
                            not terminated
                            and len(token_ids) >= maximum_target_tokens
                        ),
                        "repeated_token_loop": repeated_token_loop(token_ids),
                    }
            cases = {}
            for row in rows:
                case_id = str(row["id"])
                segments = [
                    unit_outputs[(case_id, index)]
                    for index in range(len(row["segments"]))
                ]
                hypotheses = [segment["hypothesis"] for segment in segments]
                cases[case_id] = {
                    "id": case_id,
                    "direction": row["direction"],
                    "sourceLanguage": row["sourceLanguage"],
                    "targetLanguage": row["targetLanguage"],
                    "source": row["source"],
                    "source_segments": row["segments"],
                    "references": row["references"],
                    "reference_segments": row.get("referenceSegments"),
                    "hypothesis": "\n".join(hypotheses),
                    "segment_hypotheses": hypotheses,
                    "segment_input_token_ids": [
                        segment["input_token_ids"] for segment in segments
                    ],
                    "segment_output_token_ids": [
                        segment["output_token_ids"] for segment in segments
                    ],
                    "source_truncated": any(
                        segment["source_truncated"] for segment in segments
                    ),
                    "reached_generation_limit": any(
                        segment["reached_generation_limit"] for segment in segments
                    ),
                    "repeated_token_loop": any(
                        segment["repeated_token_loop"] for segment in segments
                    ),
                    "empty_segment": any(
                        not hypothesis.strip() for hypothesis in hypotheses
                    ),
                    "adjacent_duplicate_segment": any(
                        left.strip()
                        and left.strip() == right.strip()
                        for left, right in zip(hypotheses, hypotheses[1:])
                    ),
                }
            generated_by_suite[suite_name] = cases
    if device.type == "mps":
        torch.mps.synchronize()
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return generated_by_suite


def evaluate_selector_loss(
    checkpoint: Path,
    tokenizer: MarianTokenizer,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    maximum_source_tokens: int,
    maximum_target_tokens: int,
) -> float:
    model = MarianMTModel.from_pretrained(checkpoint).to(device).eval()
    weighted_losses = []
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch_rows = rows[offset : offset + batch_size]
            batch = tokenizer(
                [
                    SOURCE_PREFIXES[row["direction"]] + row["source"]
                    for row in batch_rows
                ],
                text_target=[str(row["target"]) for row in batch_rows],
                padding=True,
                truncation=True,
                max_length=maximum_source_tokens,
                return_tensors="pt",
            )
            labels = batch["labels"]
            if labels.shape[1] > maximum_target_tokens:
                labels = labels[:, :maximum_target_tokens]
                labels[:, -1] = tokenizer.eos_token_id
            labels[labels == tokenizer.pad_token_id] = -100
            batch["labels"] = labels
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = float(model(**batch, use_cache=False).loss)
            weighted_losses.append((loss, len(batch_rows)))
    if device.type == "mps":
        torch.mps.synchronize()
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return sum(loss * count for loss, count in weighted_losses) / sum(
        count for _, count in weighted_losses
    )


def multi_references(rows: list[dict[str, Any]]) -> list[list[str]]:
    maximum = max(len(row["references"]) for row in rows)
    return [
        [
            row["references"][min(index, len(row["references"]) - 1)]
            for row in rows
        ]
        for index in range(maximum)
    ]


def computed_failures(
    rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    failures: dict[str, set[str]] = {
        "exact": set(),
        "typed": set(),
        "negation": set(),
        "generation": set(),
    }
    for row in rows:
        case_id = str(row["id"])
        source_segments = row.get("source_segments")
        hypothesis_segments = row.get("segment_hypotheses")
        if (
            not isinstance(source_segments, list)
            or not isinstance(hypothesis_segments, list)
            or len(source_segments) != len(hypothesis_segments)
        ):
            raise SystemExit(f"segment evidence differs: {case_id}")
        for source_value, hypothesis_value in zip(
            source_segments,
            hypothesis_segments,
        ):
            source = str(source_value)
            hypothesis = str(hypothesis_value)
            if critical_tokens(source) != critical_tokens(hypothesis):
                failures["exact"].add(case_id)
            if not typed_preserves(
                source,
                hypothesis,
                str(row["sourceLanguage"]),
                str(row["targetLanguage"]),
            ):
                failures["typed"].add(case_id)
            if tokens(source)["negative"] != tokens(hypothesis)["negative"]:
                failures["negation"].add(case_id)
        hypothesis = str(row["hypothesis"])
        if (
            row["source_truncated"]
            or row["reached_generation_limit"]
            or row["repeated_token_loop"]
            or row["empty_segment"]
            or row["adjacent_duplicate_segment"]
            or not hypothesis.strip()
        ):
            failures["generation"].add(case_id)
    return {name: sorted(values) for name, values in failures.items()}


def aggregate(
    suite_rows: list[dict[str, Any]],
    generated: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = {str(row["id"]) for row in suite_rows}
    if set(generated) != expected:
        raise SystemExit("generation does not cover an exact direction suite")
    rows = [generated[str(row["id"])] for row in suite_rows]
    hypotheses = [str(row["hypothesis"]) for row in rows]
    sentence_scores = {
        str(row["id"]): float(
            sacrebleu.sentence_chrf(
                str(row["hypothesis"]),
                row["references"],
                word_order=2,
            ).score
        )
        for row in rows
    }
    return {
        "cases": len(rows),
        "chrFPlusPlus": float(
            sacrebleu.corpus_chrf(
                hypotheses,
                multi_references(rows),
                word_order=2,
            ).score
        ),
        "BLEU": float(
            sacrebleu.corpus_bleu(
                hypotheses,
                multi_references(rows),
                tokenize="intl",
            ).score
        ),
        "meanSentenceChrFPlusPlus": (
            sum(sentence_scores.values()) / len(sentence_scores)
        ),
        "sentenceChrFPlusPlus": sentence_scores,
        "failures": computed_failures(rows),
        "outputs": generated,
    }


def aggregate_all(
    suites: dict[str, dict[str, Any]],
    generated: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    output = {}
    for suite_name, suite in suites.items():
        output[suite_name] = {}
        for direction_name in DIRECTIONS:
            rows = [
                row
                for row in suite["rows"]
                if row["direction"] == direction_name
            ]
            cases = {
                case_id: row
                for case_id, row in generated[suite_name].items()
                if row["direction"] == direction_name
            }
            output[suite_name][direction_name] = aggregate(rows, cases)
    return output


def public_metrics(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"sentenceChrFPlusPlus", "outputs"}
    }


def gate(
    name: str,
    actual: float | int,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, Any]:
    if (minimum is None) == (maximum is None):
        raise ValueError("gate requires exactly one bound")
    passed = actual >= minimum if minimum is not None else actual <= maximum
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "minimum" if minimum is not None else "maximum": (
            minimum if minimum is not None else maximum
        ),
    }


def new_failures(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, list[str]]:
    return {
        name: sorted(
            set(candidate["failures"][name])
            - set(baseline["failures"][name])
        )
        for name in ("exact", "typed", "negation", "generation")
    }


def checkpoint_decision(
    candidate: dict[str, dict[str, Any]],
    teachers: dict[str, dict[str, Any]],
    requirements: dict[str, Any],
    regression_suites: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selector = candidate["selector"]
    selector_macro = sum(
        selector[direction_name]["chrFPlusPlus"]
        for direction_name in DIRECTIONS
    ) / len(DIRECTIONS)
    gates = [
        gate(
            f"selector-{direction_name}-minimum",
            selector[direction_name]["chrFPlusPlus"],
            minimum=float(requirements["minimum_chrf_pp"][direction_name]),
        )
        for direction_name in DIRECTIONS
    ]
    gates.append(
        gate(
            "selector-macro-minimum",
            selector_macro,
            minimum=float(requirements["minimum_macro_direction_chrf_pp"]),
        )
    )
    maximum_regression = float(
        requirements["maximum_directional_chrf_pp_regression_from_teacher"]
    )
    failures_by_suite = {}
    for suite_name in regression_suites:
        failures_by_suite[suite_name] = {}
        for direction_name in DIRECTIONS:
            baseline = teachers[suite_name][direction_name]
            current = candidate[suite_name][direction_name]
            gates.append(
                gate(
                    f"{suite_name}-{direction_name}-teacher-regression",
                    current["chrFPlusPlus"] - baseline["chrFPlusPlus"],
                    minimum=-maximum_regression,
                )
            )
            difference = new_failures(baseline, current)
            failures_by_suite[suite_name][direction_name] = difference
            for failure_type in ("exact", "typed", "negation", "generation"):
                gates.append(
                    gate(
                        f"{suite_name}-{direction_name}-new-{failure_type}",
                        len(difference[failure_type]),
                        maximum=0,
                    )
                )
    return gates, {
        "selector_macro_direction_chrf_pp": selector_macro,
        "new_failures": failures_by_suite,
    }


def semantic_queue(
    suites: dict[str, dict[str, Any]],
    teachers: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    queue = []
    for suite_name in SUITE_ORDER:
        rows = {str(row["id"]): row for row in suites[suite_name]["rows"]}
        for direction_name in DIRECTIONS:
            teacher_outputs = teachers[suite_name][direction_name]["outputs"]
            candidate_outputs = candidate[suite_name][direction_name]["outputs"]
            for case_id in sorted(candidate_outputs):
                row = rows[case_id]
                queue.append(
                    {
                        "suite": suite_name,
                        "case_id": case_id,
                        "direction": direction_name,
                        "domain": row.get("domain"),
                        "source": row["source"],
                        "source_segments": row["segments"],
                        "references": row["references"],
                        "reference_segments": row.get("referenceSegments"),
                        "teacher": teacher_outputs[case_id]["hypothesis"],
                        "teacher_segments": teacher_outputs[case_id][
                            "segment_hypotheses"
                        ],
                        "teacher_segment_input_token_ids": teacher_outputs[
                            case_id
                        ]["segment_input_token_ids"],
                        "teacher_segment_output_token_ids": teacher_outputs[
                            case_id
                        ]["segment_output_token_ids"],
                        "candidate": candidate_outputs[case_id]["hypothesis"],
                        "candidate_segments": candidate_outputs[case_id][
                            "segment_hypotheses"
                        ],
                        "candidate_segment_input_token_ids": candidate_outputs[
                            case_id
                        ]["segment_input_token_ids"],
                        "candidate_segment_output_token_ids": candidate_outputs[
                            case_id
                        ]["segment_output_token_ids"],
                    }
                )
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_contract", type=Path)
    parser.add_argument("evaluation_contract", type=Path)
    parser.add_argument("training_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("STEP", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if args.batch_size < 1:
        raise SystemExit("batch size must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    root = Path(__file__).resolve().parents[2]
    checkpoints = {int(step): Path(path) for step, path in args.candidate}
    if len(checkpoints) != len(args.candidate):
        raise SystemExit("duplicate checkpoint step")
    training_contract, evaluation_contract, training_summary, suites = validate_inputs(
        args.training_contract,
        args.evaluation_contract,
        args.training_summary,
        checkpoints,
        root,
    )
    if args.batch_size != int(evaluation_contract["runtime"]["batch_size"]):
        raise SystemExit("batch size differs from the frozen evaluation contract")
    if (
        args.device != evaluation_contract["runtime"]["device"]
        or runtime_versions() != evaluation_contract["runtime"]["packages"]
        or hardware_name() != evaluation_contract["runtime"]["hardware"]
        or platform.platform()
        != evaluation_contract["runtime"]["operating_system"]
    ):
        raise SystemExit("evaluation runtime differs from its frozen contract")
    maximum_source_tokens = int(
        training_contract["phase1_training"]["hyperparameters"][
            "max_source_tokens"
        ]
    )
    maximum_target_tokens = int(
        training_contract["phase1_training"]["hyperparameters"][
            "max_target_tokens"
        ]
    )
    device = torch.device(args.device)

    teacher_generated: dict[str, dict[str, dict[str, Any]]] = {
        suite_name: {} for suite_name in SUITE_ORDER
    }
    for direction_name in DIRECTIONS:
        teacher_path = validate_teacher(
            direction_name,
            training_contract,
            evaluation_contract,
            root,
        )
        teacher_tokenizer = MarianTokenizer.from_pretrained(teacher_path)
        generated = generate(
            teacher_path,
            teacher_tokenizer,
            suites,
            active_directions={direction_name},
            source_prefixes={},
            device=device,
            batch_size=args.batch_size,
            maximum_source_tokens=maximum_source_tokens,
            maximum_target_tokens=maximum_target_tokens,
        )
        for suite_name in SUITE_ORDER:
            teacher_generated[suite_name].update(generated[suite_name])
    teachers = aggregate_all(suites, teacher_generated)

    candidates = []
    candidate_full_metrics: dict[int, dict[str, dict[str, Any]]] = {}
    requirements = training_contract["evaluation_gates"]["full_precision_internal"]
    regression_suites = list(requirements["required_regression_suites"])
    if regression_suites != list(
        evaluation_contract["required_regression_suites"]
    ):
        raise SystemExit("regression-suite gate scope differs")
    for step in sorted(checkpoints):
        checkpoint = checkpoints[step]
        checkpoint_manifest = authenticate_checkpoint(
            checkpoint,
            step=step,
            contract_sha256=sha256(args.training_contract),
        )
        candidate_tokenizer = load_candidate_tokenizer(
            checkpoint,
            evaluation_contract,
        )
        generated = generate(
            checkpoint,
            candidate_tokenizer,
            suites,
            active_directions=set(DIRECTIONS),
            source_prefixes=SOURCE_PREFIXES,
            device=device,
            batch_size=args.batch_size,
            maximum_source_tokens=maximum_source_tokens,
            maximum_target_tokens=maximum_target_tokens,
        )
        metrics = aggregate_all(suites, generated)
        independent_loss = evaluate_selector_loss(
            checkpoint,
            candidate_tokenizer,
            suites["selector"]["rows"],
            device=device,
            batch_size=args.batch_size,
            maximum_source_tokens=maximum_source_tokens,
            maximum_target_tokens=maximum_target_tokens,
        )
        gates, decision = checkpoint_decision(
            metrics,
            teachers,
            requirements,
            regression_suites,
        )
        for direction_name in DIRECTIONS:
            gates.append(
                gate(
                    f"selector-{direction_name}-training-evaluation-parity",
                    abs(
                        metrics["selector"][direction_name]["chrFPlusPlus"]
                        - float(
                            checkpoint_manifest["metrics"]["directions"][
                                direction_name
                            ]["chrf_pp"]
                        )
                    ),
                    maximum=1e-9,
                )
            )
        gates.append(
            gate(
                "selector-macro-training-evaluation-parity",
                abs(
                    decision["selector_macro_direction_chrf_pp"]
                    - float(
                        checkpoint_manifest["metrics"][
                            "macro_direction_chrf_pp"
                        ]
                    )
                ),
                maximum=1e-9,
            )
        )
        gates.append(
            gate(
                "selector-loss-training-evaluation-parity",
                abs(
                    independent_loss
                    - float(checkpoint_manifest["metrics"]["loss"])
                ),
                maximum=1e-9,
            )
        )
        candidate_full_metrics[step] = metrics
        candidates.append(
            {
                "step": step,
                "checkpoint": {
                    "path": display_path(checkpoint, root),
                    "model": record(checkpoint / "model.safetensors", root),
                    "manifest": record(
                        checkpoint / "mimi_checkpoint_manifest.json",
                        root,
                    ),
                },
                "training_selector_metrics": checkpoint_manifest["metrics"],
                "metrics": {
                    suite_name: {
                        direction_name: public_metrics(
                            metrics[suite_name][direction_name]
                        )
                        for direction_name in DIRECTIONS
                    }
                    for suite_name in SUITE_ORDER
                },
                **decision,
                "independent_selector_loss": independent_loss,
                "mechanical_gates": gates,
                "mechanical_gate_passed": all(item["passed"] for item in gates),
                "eligible_for_q4": False,
            }
        )

    mechanical = [item for item in candidates if item["mechanical_gate_passed"]]
    mechanical.sort(
        key=lambda item: (
            item["selector_macro_direction_chrf_pp"],
            -float(item["independent_selector_loss"]),
            -int(item["step"]),
        ),
        reverse=True,
    )
    selected = mechanical[0] if mechanical else None
    selected_semantic_queue = (
        semantic_queue(
            suites,
            teachers,
            candidate_full_metrics[int(selected["step"])],
        )
        if selected is not None
        else []
    )
    result = {
        "schema_version": 1,
        "experiment": EVALUATION_EXPERIMENT,
        "status": (
            "semantic-audit-required"
            if selected is not None
            else "mechanical-gate-rejected"
        ),
        "training_contract": record(args.training_contract, root),
        "evaluation_contract": record(args.evaluation_contract, root),
        "training_summary": record(args.training_summary, root),
        "evaluation_implementation": record(Path(__file__).resolve(), root),
        "device": str(device),
        "batch_size": args.batch_size,
        "maximum_source_tokens": maximum_source_tokens,
        "maximum_target_tokens": maximum_target_tokens,
        "teacher_baselines": {
            suite_name: {
                direction_name: public_metrics(
                    teachers[suite_name][direction_name]
                )
                for direction_name in DIRECTIONS
            }
            for suite_name in SUITE_ORDER
        },
        "candidates": candidates,
        "selected_for_semantic_audit_step": (
            selected["step"] if selected is not None else None
        ),
        "selected_checkpoint": (
            selected["checkpoint"] if selected is not None else None
        ),
        "selected_semantic_audit_queue": selected_semantic_queue,
        "selected_semantic_audit_cases": len(selected_semantic_queue),
        "semantic_audit_complete": False,
        "critical_meaning_gate_passed": False,
        "semantic_failure_behavior": evaluation_contract[
            "semantic_failure_behavior"
        ],
        "q4_conversion_authorized": False,
        "app_change_authorized": False,
        "bundle_replacement_authorized": False,
        "public_upload_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": display_path(args.output, root),
                "sha256": sha256(args.output),
                "status": result["status"],
                "selected_for_semantic_audit_step": result[
                    "selected_for_semantic_audit_step"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
