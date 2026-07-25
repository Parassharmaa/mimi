#!/usr/bin/env python3
"""Evaluate and select a parent-specialist Marian weight interpolation."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import sacrebleu
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer

from audit_translation_structures import critical_tokens, tokens
from evaluate_gpt56_student_continuation import repeated_token_loop
from train_marian_automated_preference_full import save_candidate
from train_marian_claude5_preference import load_claude5_preferences
from train_marian_claude5_preference_replay import (
    ReplayCollator,
    ReplayRows,
    load_replay,
    move,
    replay_delta,
)
from train_marian_dqo import PreferenceCollator, PreferenceRows, evaluate, hardware_name
from training_manifest_provenance import sha256
from typed_critical_token_policy import typed_preserves


EXPERIMENT = "parent-specialist-interpolation-v9-ja-en"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def validate_contract(
    path: Path,
    preference_directory: Path,
    replay_directory: Path,
    parent: Path,
    specialist: Path,
) -> dict:
    contract = load_json(path)
    if (
        contract.get("schema_version") != 1
        or contract.get("experiment") != EXPERIMENT
        or contract.get("status") != "preregistered-ready-for-interpolation"
        or contract.get("direction") != "ja-en"
        or contract.get("promotion_authorized") is not False
        or contract.get("app_change_authorized") is not False
        or contract.get("public_upload_authorized") is not False
        or contract.get("protected_evaluation_authorized") is not False
    ):
        raise SystemExit("v9 interpolation contract is invalid")
    for name, directory, filenames in (
        (
            "preferences",
            preference_directory,
            ("manifest.json", "train.jsonl", "valid.jsonl"),
        ),
        (
            "replay",
            replay_directory,
            ("manifest.json", "replay-train.jsonl", "replay-valid.jsonl"),
        ),
    ):
        binding = contract["datasets"][name]
        if binding.get("directory") != str(directory):
            raise SystemExit(f"v9 contract binds a different {name} directory")
        for filename in filenames:
            if binding["files"][filename]["sha256"] != sha256(directory / filename):
                raise SystemExit(f"v9 contract {name} hash differs: {filename}")
    if (
        contract.get("parent", {}).get("path") != str(parent)
        or contract.get("parent", {}).get("model_sha256")
        != sha256(parent / "model.safetensors")
        or contract.get("specialist", {}).get("path") != str(specialist)
        or contract.get("specialist", {}).get("model_sha256")
        != sha256(specialist / "model.safetensors")
        or contract.get("implementation", {}).get("runner", {}).get("sha256")
        != sha256(Path(__file__).resolve())
    ):
        raise SystemExit("v9 contract model or runner binding differs")
    return contract


def trim_generated(ids: list[int], eos_token_id: int, pad_token_id: int) -> tuple[list[int], bool]:
    if eos_token_id in ids:
        return ids[: ids.index(eos_token_id) + 1], True
    while ids and ids[-1] == pad_token_id:
        ids.pop()
    return ids, False


def safety_sets(
    rows: list[dict],
    hypotheses: list[str],
    token_ids: list[list[int]],
    terminated: list[bool],
) -> dict:
    exact = set()
    typed = set()
    negation = set()
    generation = set()
    for row, hypothesis, output_ids, did_terminate in zip(
        rows, hypotheses, token_ids, terminated
    ):
        identifier = str(row["id"])
        source = str(row["source"])
        if critical_tokens(source) != critical_tokens(hypothesis):
            exact.add(identifier)
        if not typed_preserves(source, hypothesis, "ja-JP", "en-US"):
            typed.add(identifier)
        if tokens(source)["negative"] != tokens(hypothesis)["negative"]:
            negation.add(identifier)
        if (
            not hypothesis.strip()
            or not did_terminate
            or repeated_token_loop(output_ids)
        ):
            generation.add(identifier)
    return {
        "exact": exact,
        "typed": typed,
        "negation": negation,
        "generation": generation,
    }


@torch.inference_mode()
def evaluate_replay(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    loader: DataLoader,
    rows: list[dict],
    device: torch.device,
) -> dict:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    hypotheses: list[str] = []
    all_token_ids: list[list[int]] = []
    all_terminated: list[bool] = []
    for packed in loader:
        batch = move(packed["model"], device)
        outputs = model(**batch)
        labels = batch["labels"]
        loss_sum += float(
            F.cross_entropy(
                outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ).cpu()
        )
        token_count += int(labels.ne(-100).sum().cpu())
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=False,
            num_beams=1,
            max_new_tokens=192,
        )
        for raw_ids in generated.cpu().tolist():
            output_ids, terminated = trim_generated(
                raw_ids, tokenizer.eos_token_id, tokenizer.pad_token_id
            )
            all_token_ids.append(output_ids)
            all_terminated.append(terminated)
            hypotheses.append(
                tokenizer.decode(output_ids, skip_special_tokens=True)
            )
    if device.type == "mps":
        torch.mps.synchronize()
    references = [str(row["target"]) for row in rows]
    overall = sacrebleu.corpus_chrf(
        hypotheses, [references], word_order=2
    ).score
    slices = {}
    for name, predicate in (
        ("legal", lambda row: row["origin"] == "finalized-japanese-law-translation"),
        ("nonlegal", lambda row: row["origin"] != "finalized-japanese-law-translation"),
    ):
        indices = [index for index, row in enumerate(rows) if predicate(row)]
        slices[name] = {
            "cases": len(indices),
            "chrFPlusPlus": sacrebleu.corpus_chrf(
                [hypotheses[index] for index in indices],
                [[references[index] for index in indices]],
                word_order=2,
            ).score,
        }
    model.train(was_training)
    return {
        "tokenNLL": loss_sum / token_count,
        "chrFPlusPlus": overall,
        "slices": slices,
        "safety": safety_sets(
            rows, hypotheses, all_token_ids, all_terminated
        ),
    }


def gate(preference: dict, replay: dict, contract: dict, alpha: float) -> dict:
    required = contract["internal_selection_gate"]
    checks = [
        {
            "name": "selectable-alpha",
            "passed": alpha in contract["interpolation"]["selectable_alphas"],
            "actual": alpha,
            "required": contract["interpolation"]["selectable_alphas"],
        },
        {
            "name": "relative-pair-accuracy",
            "passed": preference["relative_pair_accuracy"]
            >= required["relative_pair_accuracy_minimum"],
            "actual": preference["relative_pair_accuracy"],
            "required": required["relative_pair_accuracy_minimum"],
        },
        {
            "name": "relative-margin",
            "passed": preference["relative_margin"]
            > required["relative_margin_minimum_exclusive"],
            "actual": preference["relative_margin"],
            "required": f"> {required['relative_margin_minimum_exclusive']}",
        },
        {
            "name": "replay-token-nll-delta",
            "passed": replay["tokenNLLDelta"]
            <= required["replay_token_nll_delta_maximum"],
            "actual": replay["tokenNLLDelta"],
            "required": f"<= {required['replay_token_nll_delta_maximum']}",
        },
        {
            "name": "replay-chrf++-delta",
            "passed": replay["chrFPlusPlusDelta"]
            >= required["replay_chrf_pp_delta_minimum"],
            "actual": replay["chrFPlusPlusDelta"],
            "required": f">= {required['replay_chrf_pp_delta_minimum']}",
        },
        {
            "name": "legal-replay-chrf++-delta",
            "passed": replay["legalChrFPlusPlusDelta"]
            >= required["legal_replay_chrf_pp_delta_minimum"],
            "actual": replay["legalChrFPlusPlusDelta"],
            "required": f">= {required['legal_replay_chrf_pp_delta_minimum']}",
        },
        *[
            {
                "name": f"replay-{field}",
                "passed": not replay[field],
                "actual": replay[field],
                "required": 0,
            }
            for field in ("newExact", "newTyped", "newNegation", "newGeneration")
        ],
    ]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def interpolate(
    model: MarianMTModel,
    reference: MarianMTModel,
    specialist_parameters: dict[str, torch.Tensor],
    alpha: float,
) -> None:
    reference_parameters = dict(reference.named_parameters())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parent = reference_parameters[name]
            specialist = specialist_parameters[name].to(
                device=parameter.device, dtype=parameter.dtype
            )
            parameter.copy_(parent + alpha * (specialist - parent))


def serializable_replay(metrics: dict) -> dict:
    return {
        **metrics,
        "safety": {
            name: sorted(values) for name, values in metrics["safety"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preference_directory", type=Path)
    parser.add_argument("replay_directory", type=Path)
    parser.add_argument("parent_checkpoint", type=Path)
    parser.add_argument("specialist_checkpoint", type=Path)
    parser.add_argument("experiment_contract", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    args = parser.parse_args()
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    contract = validate_contract(
        args.experiment_contract,
        args.preference_directory,
        args.replay_directory,
        args.parent_checkpoint,
        args.specialist_checkpoint,
    )
    _, preference_valid, preference_manifest = load_claude5_preferences(
        args.preference_directory
    )
    _, replay_valid, replay_manifest = load_replay(args.replay_directory)
    device = torch.device(args.device)
    tokenizer = MarianTokenizer.from_pretrained(args.parent_checkpoint)
    reference = MarianMTModel.from_pretrained(args.parent_checkpoint).to(device)
    reference.eval()
    reference.requires_grad_(False)
    model = MarianMTModel.from_pretrained(args.specialist_checkpoint).to(device)
    specialist_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    preference_loader = DataLoader(
        PreferenceRows(preference_valid),
        batch_size=4,
        shuffle=False,
        collate_fn=PreferenceCollator(tokenizer, 192, 192),
    )
    replay_loader = DataLoader(
        ReplayRows(replay_valid),
        batch_size=8,
        shuffle=False,
        collate_fn=ReplayCollator(tokenizer, 192, 192),
    )
    parent_replay = evaluate_replay(
        reference, tokenizer, replay_loader, replay_valid, device
    )
    reference.eval()
    history = []
    eligible = []
    alphas = [
        *contract["interpolation"]["diagnostic_anchor_alphas"],
        *contract["interpolation"]["selectable_alphas"],
    ]
    for alpha in sorted(set(float(value) for value in alphas)):
        interpolate(model, reference, specialist_parameters, alpha)
        preference = evaluate(
            model, reference, preference_loader, device, 0.10
        )
        replay = replay_delta(
            evaluate_replay(model, tokenizer, replay_loader, replay_valid, device),
            parent_replay,
        )
        reference.eval()
        current = {
            "alpha": alpha,
            "preference": preference,
            "replay": replay,
            "internal_gate": gate(preference, replay, contract, alpha),
        }
        history.append(current)
        if current["internal_gate"]["passed"]:
            eligible.append(current)
        print(json.dumps(current, ensure_ascii=False), flush=True)

    def selection_key(item: dict) -> tuple:
        return (
            item["preference"]["relative_pair_accuracy"],
            item["preference"]["relative_margin"],
            item["replay"]["legalChrFPlusPlusDelta"],
            -item["alpha"],
        )

    best_eligible = max(eligible, key=selection_key) if eligible else None
    selectable = [
        item
        for item in history
        if item["alpha"] in contract["interpolation"]["selectable_alphas"]
    ]
    best_diagnostic = max(selectable, key=selection_key)
    selected = best_eligible or best_diagnostic
    interpolate(model, reference, specialist_parameters, selected["alpha"])
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": EXPERIMENT,
        "operation": "training-free linear parent-specialist weight interpolation",
        "status": (
            "internal-interpolation-gate-passed"
            if best_eligible is not None
            else "internal-interpolation-gate-rejected"
        ),
        "direction": "ja-en",
        "student_repository": contract["parent"]["repository"],
        "student_revision": contract["parent"]["revision"],
        "license": contract["parent"]["license"],
        "private_reasoning_traces_used": False,
        "human_reviewers_used": False,
        "training_steps": 0,
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "protected_evaluation_authorized": best_eligible is not None,
        "parent": contract["parent"],
        "specialist": contract["specialist"],
        "experiment_contract": {
            "path": str(args.experiment_contract),
            "sha256": sha256(args.experiment_contract),
        },
        "datasets": {
            "preferences": {
                "directory": str(args.preference_directory),
                "manifest_sha256": sha256(
                    args.preference_directory / "manifest.json"
                ),
                "valid_pairs": len(preference_valid),
                "effective_licenses": preference_manifest["effective_licenses"],
            },
            "replay": {
                "directory": str(args.replay_directory),
                "manifest_sha256": sha256(args.replay_directory / "manifest.json"),
                "valid_rows": len(replay_valid),
                "effective_licenses": replay_manifest["effective_licenses"],
            },
        },
        "interpolation": contract["interpolation"],
        "parent_replay_validation": serializable_replay(parent_replay),
        "best": selected,
        "best_eligible": best_eligible,
        "history": history,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
    }
    save_candidate(model, tokenizer, args.output_directory, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output_directory),
                "status": manifest["status"],
                "best": selected,
                "model_sha256": sha256(
                    args.output_directory / "model.safetensors"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
