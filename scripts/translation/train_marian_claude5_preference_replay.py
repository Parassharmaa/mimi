#!/usr/bin/env python3
"""Train v8 Claude preferences with licensed preservation replay."""

from __future__ import annotations

import argparse
import json
import platform
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import MarianMTModel, MarianTokenizer, get_linear_schedule_with_warmup

from audit_translation_structures import critical_tokens, tokens
from evaluate_gpt56_student_continuation import repeated_token_loop
from train_marian_automated_preference_full import capture_state, save_candidate
from train_marian_claude5_preference import load_claude5_preferences
from train_marian_distillation import frozen_base_kl, l2_to_frozen_base
from train_marian_dqo import (
    PreferenceCollator,
    PreferenceRows,
    dqo_loss,
    evaluate,
    hardware_name,
    pair_logps,
)
from training_manifest_provenance import sha256
from typed_critical_token_policy import typed_preserves


EXPERIMENT = "canonical-pairwise-preference-replay-v8-ja-en"
REPLAY_EXPERIMENT = "canonical-pairwise-v8-ja-en-licensed-replay"


class ReplayRows(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class ReplayCollator:
    def __init__(
        self,
        tokenizer: MarianTokenizer,
        max_source_tokens: int,
        max_target_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_source_tokens = max_source_tokens
        self.max_target_tokens = max_target_tokens

    def __call__(self, rows: list[dict]) -> dict:
        batch = self.tokenizer(
            [str(row["source"]) for row in rows],
            text_target=[str(row["target"]) for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_source_tokens,
            return_tensors="pt",
        )
        labels = batch["labels"]
        if labels.shape[1] > self.max_target_tokens:
            labels = labels[:, : self.max_target_tokens]
            labels[:, -1] = self.tokenizer.eos_token_id
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return {
            "model": batch,
            "ids": [str(row["id"]) for row in rows],
        }


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_replay(directory: Path) -> tuple[list[dict], list[dict], dict]:
    manifest_path = directory / "manifest.json"
    paths = {
        "train": directory / "replay-train.jsonl",
        "valid": directory / "replay-valid.jsonl",
    }
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment") != REPLAY_EXPERIMENT
        or manifest.get("status") != "frozen-protected-screened-replay"
        or manifest.get("direction") != "ja-en"
        or manifest.get("private_reasoning_traces_used") is not False
        or manifest.get("promotion_eligible") is not False
        or manifest.get("decontamination", {}).get("train_valid_source_overlap")
        is not False
        or manifest.get("decontamination", {}).get(
            "preference_replay_source_overlap"
        )
        is not False
    ):
        raise SystemExit("v8 replay manifest is invalid")
    rows = {}
    for split, path in paths.items():
        expected = manifest.get("outputs", {}).get(split, {})
        if (
            expected.get("sha256") != sha256(path)
            or expected.get("rows") != manifest.get("counts", {}).get(split)
        ):
            raise SystemExit(f"v8 replay {split} hash or count differs")
        rows[split] = load_jsonl(path)
        if len(rows[split]) != expected["rows"]:
            raise SystemExit(f"v8 replay {split} row count differs")
        if any(
            (row.get("source_language"), row.get("target_language"))
            != ("ja-JP", "en-US")
            or not str(row.get("source", "")).strip()
            or not str(row.get("target", "")).strip()
            or row.get("replay_role") != "licensed-parent-preservation"
            or row.get("replay_split") != split
            for row in rows[split]
        ):
            raise SystemExit(f"v8 replay {split} row contract failed")
        licenses = dict(
            sorted(Counter(str(row["source_license"]) for row in rows[split]).items())
        )
        if licenses != manifest.get("effective_licenses", {}).get(split):
            raise SystemExit(f"v8 replay {split} licenses differ")
    if {row["id"] for row in rows["train"]} & {
        row["id"] for row in rows["valid"]
    }:
        raise SystemExit("v8 replay IDs overlap")
    return rows["train"], rows["valid"], manifest


def validate_contract(
    path: Path,
    preference_directory: Path,
    replay_directory: Path,
    parent: Path,
) -> dict:
    contract = load_json(path)
    if (
        contract.get("schema_version") != 1
        or contract.get("experiment") != EXPERIMENT
        or contract.get("status") != "preregistered-ready-for-training"
        or contract.get("direction") != "ja-en"
        or contract.get("promotion_authorized") is not False
        or contract.get("app_change_authorized") is not False
        or contract.get("public_upload_authorized") is not False
        or contract.get("protected_evaluation_authorized") is not False
    ):
        raise SystemExit("v8 experiment contract is invalid")
    bindings = contract.get("datasets", {})
    for name, directory, files in (
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
        if bindings.get(name, {}).get("directory") != str(directory):
            raise SystemExit(f"v8 contract binds a different {name} directory")
        for filename in files:
            if bindings[name]["files"][filename]["sha256"] != sha256(
                directory / filename
            ):
                raise SystemExit(f"v8 contract {name} hash differs: {filename}")
    if (
        contract.get("parent", {}).get("path") != str(parent)
        or contract.get("parent", {}).get("model_sha256")
        != sha256(parent / "model.safetensors")
        or contract.get("implementation", {}).get("trainer", {}).get("sha256")
        != sha256(Path(__file__).resolve())
    ):
        raise SystemExit("v8 contract parent or trainer binding differs")
    return contract


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def safety_sets(rows: list[dict], hypotheses: list[str], token_ids: list[list[int]]) -> dict:
    exact = set()
    typed = set()
    negation = set()
    generation = set()
    for row, hypothesis, output_ids in zip(rows, hypotheses, token_ids):
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
            or len(output_ids) >= 192
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
    hypotheses = []
    all_token_ids = []
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
        generated_ids = generated.cpu().tolist()
        all_token_ids.extend(generated_ids)
        hypotheses.extend(
            tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
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
        "safety": safety_sets(rows, hypotheses, all_token_ids),
    }


def replay_delta(current: dict, parent: dict) -> dict:
    safety = {
        f"new{name[0].upper()}{name[1:]}": sorted(
            current["safety"][name] - parent["safety"][name]
        )
        for name in ("exact", "typed", "negation", "generation")
    }
    return {
        "tokenNLLDelta": current["tokenNLL"] - parent["tokenNLL"],
        "chrFPlusPlusDelta": current["chrFPlusPlus"] - parent["chrFPlusPlus"],
        "legalChrFPlusPlusDelta": (
            current["slices"]["legal"]["chrFPlusPlus"]
            - parent["slices"]["legal"]["chrFPlusPlus"]
        ),
        "nonlegalChrFPlusPlusDelta": (
            current["slices"]["nonlegal"]["chrFPlusPlus"]
            - parent["slices"]["nonlegal"]["chrFPlusPlus"]
        ),
        **safety,
    }


def internal_gate(preference: dict, replay: dict, contract: dict, step: int) -> dict:
    required = contract["internal_selection_gate"]
    checks = [
        {
            "name": "trained-checkpoint",
            "passed": step > 0,
            "actual": step,
            "required": "> 0",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preference_directory", type=Path)
    parser.add_argument("replay_directory", type=Path)
    parser.add_argument("parent_checkpoint", type=Path)
    parser.add_argument("experiment_contract", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--replay-batch-size", type=int, default=4)
    parser.add_argument("--replay-evaluation-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--evaluation-steps", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--chosen-sft-weight", type=float, default=0.02)
    parser.add_argument("--replay-sft-weight", type=float, default=0.25)
    parser.add_argument("--replay-kl-weight", type=float, default=0.10)
    parser.add_argument("--l2-to-parent-weight", type=float, default=0.10)
    parser.add_argument("--max-source-tokens", type=int, default=192)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    contract = validate_contract(
        args.experiment_contract,
        args.preference_directory,
        args.replay_directory,
        args.parent_checkpoint,
    )
    actual_hyperparameters = {
        name: getattr(args, name)
        for name in (
            "seed",
            "batch_size",
            "replay_batch_size",
            "replay_evaluation_batch_size",
            "gradient_accumulation",
            "max_steps",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "evaluation_steps",
            "beta",
            "chosen_sft_weight",
            "replay_sft_weight",
            "replay_kl_weight",
            "l2_to_parent_weight",
            "max_source_tokens",
            "max_target_tokens",
        )
    }
    if contract.get("training") != actual_hyperparameters:
        raise SystemExit("arguments differ from the preregistered v8 recipe")
    if min(
        args.batch_size,
        args.replay_batch_size,
        args.replay_evaluation_batch_size,
        args.gradient_accumulation,
        args.max_steps,
        args.evaluation_steps,
    ) < 1 or min(
        args.learning_rate,
        args.beta,
        args.replay_sft_weight,
        args.replay_kl_weight,
    ) <= 0:
        raise SystemExit("invalid v8 hyperparameters")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")

    preference_train, preference_valid, preference_manifest = (
        load_claude5_preferences(args.preference_directory)
    )
    replay_train, replay_valid, replay_manifest = load_replay(args.replay_directory)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = MarianTokenizer.from_pretrained(args.parent_checkpoint)
    model = MarianMTModel.from_pretrained(args.parent_checkpoint).to(device)
    reference = MarianMTModel.from_pretrained(args.parent_checkpoint).to(device)
    reference.eval()
    reference.requires_grad_(False)
    base_parameters = {
        name: parameter.detach() for name, parameter in reference.named_parameters()
    }
    preference_collator = PreferenceCollator(
        tokenizer, args.max_source_tokens, args.max_target_tokens
    )
    replay_collator = ReplayCollator(
        tokenizer, args.max_source_tokens, args.max_target_tokens
    )
    generator = torch.Generator().manual_seed(args.seed)
    preference_loader = DataLoader(
        PreferenceRows(preference_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=preference_collator,
    )
    preference_valid_loader = DataLoader(
        PreferenceRows(preference_valid),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=preference_collator,
    )
    replay_loader = DataLoader(
        ReplayRows(replay_train),
        batch_size=args.replay_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
        collate_fn=replay_collator,
    )
    replay_valid_loader = DataLoader(
        ReplayRows(replay_valid),
        batch_size=args.replay_evaluation_batch_size,
        shuffle=False,
        collate_fn=replay_collator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    parent_replay = evaluate_replay(
        reference, tokenizer, replay_valid_loader, replay_valid, device
    )
    reference.eval()
    base_preference = evaluate(
        model, reference, preference_valid_loader, device, args.beta
    )
    base_replay_delta = replay_delta(parent_replay, parent_replay)
    history = [
        {
            "step": 0,
            "preference": base_preference,
            "replay": base_replay_delta,
            "internal_gate": internal_gate(
                base_preference, base_replay_delta, contract, 0
            ),
        }
    ]
    best_eligible = None
    best_eligible_state = None
    best_diagnostic = history[0]
    best_diagnostic_state = capture_state(model)
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    replay_iterator = iter(replay_loader)
    model.train()
    while update_step < args.max_steps:
        for pair in preference_loader:
            try:
                packed_replay = next(replay_iterator)
            except StopIteration:
                replay_iterator = iter(replay_loader)
                packed_replay = next(replay_iterator)
            chosen = move(pair["chosen"], device)
            rejected = move(pair["rejected"], device)
            replay_batch = move(packed_replay["model"], device)
            policy_chosen, policy_rejected = pair_logps(model, chosen, rejected)
            with torch.inference_mode():
                reference_chosen, reference_rejected = pair_logps(
                    reference, chosen, rejected
                )
            preference_loss, relative_margin = dqo_loss(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                args.beta,
            )
            chosen_sft_loss = -policy_chosen.mean()
            replay_outputs = model(**replay_batch)
            with torch.inference_mode():
                parent_replay_outputs = reference(**replay_batch)
            preservation_mask = torch.ones(
                replay_batch["labels"].shape[0],
                dtype=torch.bool,
                device=device,
            )
            replay_kl = frozen_base_kl(
                replay_outputs.logits,
                parent_replay_outputs.logits,
                replay_batch["labels"],
                preservation_mask,
            )
            l2_loss = l2_to_frozen_base(model, base_parameters)
            combined = (
                preference_loss
                + args.chosen_sft_weight * chosen_sft_loss
                + args.replay_sft_weight * replay_outputs.loss
                + args.replay_kl_weight * replay_kl
                + args.l2_to_parent_weight * l2_loss
            )
            (combined / args.gradient_accumulation).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            if (
                update_step % args.evaluation_steps == 0
                or update_step == args.max_steps
            ):
                preference_metrics = evaluate(
                    model,
                    reference,
                    preference_valid_loader,
                    device,
                    args.beta,
                )
                current_replay = evaluate_replay(
                    model,
                    tokenizer,
                    replay_valid_loader,
                    replay_valid,
                    device,
                )
                reference.eval()
                current_replay_delta = replay_delta(current_replay, parent_replay)
                gate = internal_gate(
                    preference_metrics,
                    current_replay_delta,
                    contract,
                    update_step,
                )
                current = {
                    "step": update_step,
                    "preference": preference_metrics,
                    "replay": current_replay_delta,
                    "internal_gate": gate,
                    "last_train_objective": {
                        "preference_loss": float(preference_loss.detach().cpu()),
                        "chosen_sft_loss": float(chosen_sft_loss.detach().cpu()),
                        "replay_sft_loss": float(replay_outputs.loss.detach().cpu()),
                        "replay_kl": float(replay_kl.detach().cpu()),
                        "l2_to_parent": float(l2_loss.detach().cpu()),
                        "relative_margin": float(relative_margin.mean().detach().cpu()),
                    },
                }
                history.append(current)
                state = capture_state(model)
                diagnostic_key = (
                    preference_metrics["relative_pair_accuracy"],
                    preference_metrics["relative_margin"],
                    current_replay_delta["legalChrFPlusPlusDelta"],
                )
                prior_diagnostic_key = (
                    best_diagnostic["preference"]["relative_pair_accuracy"],
                    best_diagnostic["preference"]["relative_margin"],
                    best_diagnostic["replay"]["legalChrFPlusPlusDelta"],
                )
                if diagnostic_key > prior_diagnostic_key:
                    best_diagnostic = current
                    best_diagnostic_state = state
                if gate["passed"]:
                    eligible_key = (
                        preference_metrics["relative_pair_accuracy"],
                        preference_metrics["relative_margin"],
                        current_replay_delta["legalChrFPlusPlusDelta"],
                        -preference_metrics["loss"],
                    )
                    if best_eligible is None or eligible_key > (
                        best_eligible["preference"]["relative_pair_accuracy"],
                        best_eligible["preference"]["relative_margin"],
                        best_eligible["replay"]["legalChrFPlusPlusDelta"],
                        -best_eligible["preference"]["loss"],
                    ):
                        best_eligible = current
                        best_eligible_state = state
                print(
                    json.dumps(
                        {
                            "current": current,
                            "bestEligible": best_eligible,
                            "bestDiagnostic": best_diagnostic,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if update_step >= args.max_steps:
                break

    selected = best_eligible or best_diagnostic
    selected_state = best_eligible_state or best_diagnostic_state
    model.load_state_dict(selected_state, strict=True)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": EXPERIMENT,
        "operation": "Claude-5 preference update with one-to-one licensed replay",
        "status": (
            "internal-preservation-gate-passed"
            if best_eligible is not None
            else "internal-preservation-gate-rejected"
        ),
        "direction": "ja-en",
        "student_repository": contract["parent"]["repository"],
        "student_revision": contract["parent"]["revision"],
        "license": contract["parent"]["license"],
        "private_reasoning_traces_used": False,
        "human_reviewers_used": False,
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "protected_evaluation_authorized": best_eligible is not None,
        "parent": contract["parent"],
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
                "train_pairs": len(preference_train),
                "valid_pairs": len(preference_valid),
                "effective_licenses": preference_manifest["effective_licenses"],
            },
            "replay": {
                "directory": str(args.replay_directory),
                "manifest_sha256": sha256(args.replay_directory / "manifest.json"),
                "train_rows": len(replay_train),
                "valid_rows": len(replay_valid),
                "effective_licenses": replay_manifest["effective_licenses"],
            },
        },
        "dataset_manifest": contract["dataset_manifest"],
        "objective": {
            "loss": (
                "parent-relative DPO plus chosen SFT, licensed replay SFT, "
                "frozen-parent replay KL, and L2-to-parent"
            ),
            "selection": (
                "eligible checkpoints only: preference accuracy/margin plus replay "
                "NLL, chrF++, legal chrF++, and zero-new-safety gates"
            ),
        },
        "hyperparameters": actual_hyperparameters,
        "parent_replay_validation": {
            **parent_replay,
            "safety": {
                name: sorted(values)
                for name, values in parent_replay["safety"].items()
            },
        },
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "best": selected,
        "best_eligible": best_eligible,
        "history": history,
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
