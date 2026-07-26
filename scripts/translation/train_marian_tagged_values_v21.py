#!/usr/bin/env python3
"""Train V21's bounded JA→EN source-bound critical-value tag experiment."""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
import transformers
from audit_translation_structures import tokens as structure_tokens
from build_tagged_critical_values_v21 import TAGS
from torch.utils.data import DataLoader
from train_marian_distillation import (
    TranslationRows,
    checkpoint_identity,
    checkpoint_lineage_manifests,
    hardware_name,
    load_rows,
    move,
    sha256,
    synchronize,
)
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    get_linear_schedule_with_warmup,
)
from typed_critical_token_policy import (
    single_percentage_preserves,
    strict_tokens,
    typed_preserves,
)


EXPERIMENT = "tagged-critical-values-v21-ja-en"
TAG_PATTERN = re.compile(r"<v\d+>")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def list_collator(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def next_batch(iterator: Any, loader: DataLoader) -> tuple[list[dict[str, Any]], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def preserves_strict_tokens(source: str, output: str) -> bool:
    return strict_tokens(source) == strict_tokens(
        output
    ) or single_percentage_preserves(source, output)


def validate_contract(contract_path: Path, *, root: Path) -> dict[str, Any]:
    contract = load_json(contract_path)
    if (
        contract.get("experiment") != EXPERIMENT
        or contract.get("status") != "preregistered-ready-for-one-arm-training"
        or contract.get("direction") != "ja-en"
        or contract.get("training_authorized") is not True
        or contract.get("q4_conversion_authorized") is not False
        or contract.get("protected_evaluation_authorized") is not False
        or contract.get("app_change_authorized") is not False
        or contract.get("public_upload_authorized") is not False
    ):
        raise SystemExit("V21 contract safety state differs")
    implementation = contract.get("implementation", {}).get("trainer", {})
    if (
        implementation.get("path")
        != "scripts/translation/train_marian_tagged_values_v21.py"
        or sha256(root / implementation["path"]) != implementation.get("sha256")
    ):
        raise SystemExit("V21 trainer differs from the frozen contract")
    for section, field in (
        ("dataset", "manifest"),
        ("dataset", "train"),
        ("dataset", "tagged_valid"),
        ("dataset", "plain_valid"),
        ("initial_checkpoint", "model"),
        ("initial_checkpoint", "training_manifest"),
        ("preservation_checkpoint", "model"),
        ("preservation_checkpoint", "training_manifest"),
    ):
        record = contract[section][field]
        if sha256(root / record["path"]) != record["sha256"]:
            raise SystemExit(f"V21 bound input differs: {section}.{field}")
    manifest = load_json(root / contract["dataset"]["manifest"]["path"])
    if (
        manifest.get("experiment") != "tagged-critical-values-v21-dataset"
        or manifest.get("direction") != "ja-en"
        or manifest.get("does_not_authorize_training") is not True
        or manifest.get("does_not_authorize_app_integration") is not True
        or manifest.get("does_not_authorize_public_upload") is not True
    ):
        raise SystemExit("V21 dataset manifest safety state differs")
    return contract


def validate_rows(
    train_rows: list[dict[str, Any]],
    tagged_valid: list[dict[str, Any]],
    plain_valid: list[dict[str, Any]],
    contract: dict[str, Any],
) -> None:
    counts = contract["dataset"]["counts"]
    tagged_train = [
        row for row in train_rows if row.get("training_arm") == "tagged-focus"
    ]
    plain_train = [
        row for row in train_rows if row.get("training_arm") == "plain-replay"
    ]
    if (
        len(train_rows) != counts["train"]
        or len(tagged_train) != counts["tagged_train"]
        or len(plain_train) != counts["plain_train"]
        or len(tagged_valid) != counts["tagged_valid"]
        or len(plain_valid) != counts["plain_valid"]
    ):
        raise SystemExit("V21 bound row counts differ")
    allowed = set(TAGS)
    for row in [*tagged_train, *tagged_valid]:
        expected = [str(item["tag"]) for item in row.get("value_sidecar", [])]
        if (
            not expected
            or len(expected) != row.get("tag_count")
            or len(expected) != len(set(expected))
            or not set(expected) <= allowed
            or Counter(TAG_PATTERN.findall(str(row["source"]))) != Counter(expected)
            or Counter(TAG_PATTERN.findall(str(row["target"]))) != Counter(expected)
            or not row.get("original_source")
            or not row.get("original_target")
        ):
            raise SystemExit(f"V21 tagged row is malformed: {row.get('id')}")
    if any(
        TAG_PATTERN.search(str(row["source"]))
        or TAG_PATTERN.search(str(row["target"]))
        or row.get("tag_count") != 0
        or row.get("value_sidecar") != []
        for row in plain_train
    ):
        raise SystemExit("V21 plain replay contains tag state")


class TagCollator:
    def __init__(
        self,
        tokenizer: MarianTokenizer,
        *,
        maximum_source_tokens: int,
        maximum_target_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.maximum_source_tokens = maximum_source_tokens
        self.maximum_target_tokens = maximum_target_tokens

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            [str(row["source"]) for row in rows],
            text_target=[str(row["target"]) for row in rows],
            padding=True,
            truncation=True,
            max_length=self.maximum_source_tokens,
            return_tensors="pt",
        )
        labels = batch["labels"]
        if labels.shape[1] > self.maximum_target_tokens:
            labels = labels[:, : self.maximum_target_tokens]
            labels[:, -1] = self.tokenizer.eos_token_id
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        batch["plain_mask"] = torch.tensor(
            [row.get("training_arm") == "plain-replay" for row in rows],
            dtype=torch.bool,
        )
        return batch


def tagged_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tag_ids: torch.Tensor,
    *,
    correct_tag_weight: float,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    valid = labels.ne(-100)
    tag_positions = (
        labels[..., None].eq(tag_ids[None, None, :]).any(dim=-1) & valid
    )
    weights = torch.ones_like(losses)
    weights = torch.where(
        tag_positions,
        torch.full_like(weights, correct_tag_weight),
        weights,
    )
    weights *= valid.to(weights.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def extra_tag_unlikelihood(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tag_ids: torch.Tensor,
) -> torch.Tensor:
    probabilities = F.softmax(logits.float(), dim=-1)
    tag_probabilities = probabilities.index_select(-1, tag_ids)
    total_tag_probability = tag_probabilities.sum(dim=-1)
    valid = labels.ne(-100)
    tag_positions = (
        labels[..., None].eq(tag_ids[None, None, :]).any(dim=-1) & valid
    )
    safe_labels = labels.clamp_min(0)
    correct_probability = probabilities.gather(
        -1,
        safe_labels[..., None],
    ).squeeze(-1)
    forbidden_probability = torch.where(
        tag_positions,
        total_tag_probability - correct_probability,
        total_tag_probability,
    ).clamp(min=0.0, max=1.0 - 1e-6)
    losses = -torch.log1p(-forbidden_probability)
    return losses[valid].mean() if bool(valid.any()) else losses.new_zeros(())


def frozen_parent_kl_on_plain(
    student_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    labels: torch.Tensor,
    plain_mask: torch.Tensor,
    *,
    base_vocabulary_size: int,
) -> torch.Tensor:
    token_mask = labels.ne(-100) & plain_mask[:, None]
    if not bool(token_mask.any()):
        return student_logits.new_zeros((), dtype=torch.float32)
    student_old = student_logits[..., :base_vocabulary_size].float()
    divergences = F.kl_div(
        F.log_softmax(student_old, dim=-1),
        F.softmax(parent_logits.float(), dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return divergences[token_mask].mean()


def l2_to_parent_prefix(
    model: MarianMTModel,
    parent_parameters: dict[str, torch.Tensor],
) -> torch.Tensor:
    squared = None
    for name, parameter in model.named_parameters():
        reference = parent_parameters.get(name)
        if reference is None:
            raise RuntimeError(f"V21 parent lacks parameter: {name}")
        candidate = parameter
        if candidate.shape != reference.shape:
            if (
                candidate.ndim >= 1
                and candidate.shape[0] > reference.shape[0]
                and candidate.shape[1:] == reference.shape[1:]
            ):
                candidate = candidate[: reference.shape[0]]
            else:
                raise RuntimeError(
                    f"V21 parameter shape differs unexpectedly: {name}: "
                    f"{tuple(candidate.shape)} vs {tuple(reference.shape)}"
                )
        value = (candidate.float() - reference.float()).square().sum()
        squared = value if squared is None else squared + value
    if squared is None:
        raise RuntimeError("V21 model has no trainable parameters")
    return squared


def restore_tagged_output(
    output: str,
    sidecar: list[dict[str, Any]],
) -> str | None:
    expected = [str(item["tag"]) for item in sidecar]
    if Counter(TAG_PATTERN.findall(output)) != Counter(expected):
        return None
    restored = output
    for item in sidecar:
        surface = (
            item["source_surface"]
            if item["kind"] == "protected" or item["source_has_ascii_digits"]
            else item["target_surface"]
        )
        restored = restored.replace(str(item["tag"]), str(surface))
    return None if TAG_PATTERN.search(restored) else restored


def has_repeated_token_loop(token_ids: list[int]) -> bool:
    for width in range(3, min(16, len(token_ids) // 3) + 1):
        if token_ids[-width:] == token_ids[-2 * width : -width] == token_ids[
            -3 * width : -2 * width
        ]:
            return True
    return False


@torch.inference_mode()
def generate_records(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    maximum_source_tokens: int,
    maximum_target_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    was_training = model.training
    model.eval()
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        subset = rows[start : start + batch_size]
        inputs = tokenizer(
            [str(row["source"]) for row in subset],
            padding=True,
            truncation=True,
            max_length=maximum_source_tokens,
            return_tensors="pt",
        )
        inputs = move(inputs, device)
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=maximum_target_tokens,
        )
        hypotheses = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, hypothesis, token_row in zip(
            subset,
            hypotheses,
            generated.detach().cpu().tolist(),
        ):
            retained = [
                token
                for token in token_row
                if token not in {tokenizer.pad_token_id, tokenizer.eos_token_id}
            ]
            records.append(
                {
                    "id": str(row["id"]),
                    "hypothesis": hypothesis.strip(),
                    "token_ids": retained,
                    "reached_generation_limit": len(retained)
                    >= maximum_target_tokens,
                    "repeated_token_loop": has_repeated_token_loop(retained),
                }
            )
    synchronize(device)
    if was_training:
        model.train()
    return records


def chrf_by_domain(
    rows: list[dict[str, Any]],
    hypotheses: list[str],
    references: list[str],
) -> dict[str, dict[str, float | int]]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row.get("domain", "unknown"))].append(index)
    return {
        domain: {
            "cases": len(indices),
            "chrf_pp": sacrebleu.corpus_chrf(
                [hypotheses[index] for index in indices],
                [[references[index] for index in indices]],
                word_order=2,
            ).score,
        }
        for domain, indices in sorted(grouped.items())
    }


def evaluate_plain(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    maximum_source_tokens: int,
    maximum_target_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    generated = generate_records(
        model,
        tokenizer,
        rows,
        batch_size=batch_size,
        maximum_source_tokens=maximum_source_tokens,
        maximum_target_tokens=maximum_target_tokens,
        device=device,
    )
    hypotheses = [str(item["hypothesis"]) for item in generated]
    references = [str(row["target"]) for row in rows]
    failures = {
        "exact": [],
        "typed": [],
        "negation": [],
        "generation": [],
        "extra_tag": [],
    }
    for row, item in zip(rows, generated):
        identifier = str(row["id"])
        source = str(row["source"])
        hypothesis = str(item["hypothesis"])
        if not preserves_strict_tokens(source, hypothesis):
            failures["exact"].append(identifier)
        if not typed_preserves(source, hypothesis, "ja-JP", "en-US"):
            failures["typed"].append(identifier)
        if structure_tokens(source)["negative"] != structure_tokens(hypothesis)[
            "negative"
        ]:
            failures["negation"].append(identifier)
        if (
            not hypothesis
            or item["reached_generation_limit"]
            or item["repeated_token_loop"]
        ):
            failures["generation"].append(identifier)
        if TAG_PATTERN.search(hypothesis):
            failures["extra_tag"].append(identifier)
    return {
        "cases": len(rows),
        "chrf_pp": sacrebleu.corpus_chrf(
            hypotheses,
            [references],
            word_order=2,
        ).score,
        "domains": chrf_by_domain(rows, hypotheses, references),
        "failures": failures,
    }


def evaluate_tagged(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    maximum_source_tokens: int,
    maximum_target_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    generated = generate_records(
        model,
        tokenizer,
        rows,
        batch_size=batch_size,
        maximum_source_tokens=maximum_source_tokens,
        maximum_target_tokens=maximum_target_tokens,
        device=device,
    )
    restored_outputs: list[str] = []
    exact_tag_ids: list[str] = []
    exact_tag_order: list[str] = []
    strict_failures: list[str] = []
    typed_failures: list[str] = []
    generation_failures: list[str] = []
    for row, item in zip(rows, generated):
        identifier = str(row["id"])
        expected = [str(value["tag"]) for value in row["value_sidecar"]]
        observed = TAG_PATTERN.findall(str(item["hypothesis"]))
        if Counter(observed) == Counter(expected):
            exact_tag_ids.append(identifier)
        if observed == expected:
            exact_tag_order.append(identifier)
        restored = restore_tagged_output(
            str(item["hypothesis"]),
            list(row["value_sidecar"]),
        )
        restored_outputs.append(restored or "")
        if (
            not item["hypothesis"]
            or item["reached_generation_limit"]
            or item["repeated_token_loop"]
        ):
            generation_failures.append(identifier)
        if restored is None or not preserves_strict_tokens(
            str(row["original_source"]),
            restored,
        ):
            strict_failures.append(identifier)
        if restored is None or not typed_preserves(
            str(row["original_source"]),
            restored,
            "ja-JP",
            "en-US",
        ):
            typed_failures.append(identifier)
    references = [str(row["original_target"]) for row in rows]
    return {
        "cases": len(rows),
        "exact_tag_identity_and_multiplicity": len(exact_tag_ids) / len(rows),
        "exact_tag_order": len(exact_tag_order) / len(rows),
        "restored_chrf_pp": sacrebleu.corpus_chrf(
            restored_outputs,
            [references],
            word_order=2,
        ).score,
        "strict_failure_count": len(strict_failures),
        "typed_failure_count": len(typed_failures),
        "generation_failure_count": len(generation_failures),
        "strict_failure_ids": strict_failures,
        "typed_failure_ids": typed_failures,
        "generation_failure_ids": generation_failures,
        "domains": chrf_by_domain(rows, restored_outputs, references),
    }


def selection_gates(
    tagged: dict[str, Any],
    plain: dict[str, Any],
    *,
    tagged_step_zero: dict[str, Any],
    plain_parent: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    gates = contract["selection"]
    baseline_strict = int(tagged_step_zero["strict_failure_count"])
    maximum_strict = int(
        baseline_strict * (1.0 - gates["minimum_strict_failure_reduction_fraction"])
    )
    domain_deltas = {
        domain: float(values["chrf_pp"])
        - float(plain_parent["domains"][domain]["chrf_pp"])
        for domain, values in plain["domains"].items()
    }
    new_failures = {}
    for name, identifiers in plain["failures"].items():
        new_failures[name] = sorted(
            set(identifiers) - set(plain_parent["failures"][name])
        )
    checks = {
        "tag_identity": tagged["exact_tag_identity_and_multiplicity"]
        >= gates["minimum_exact_tag_identity_and_multiplicity"],
        "strict_failure_reduction": tagged["strict_failure_count"]
        <= maximum_strict,
        "full_validation": plain["chrf_pp"]
        >= plain_parent["chrf_pp"] + gates["minimum_full_validation_chrf_pp_delta"],
        "domains": min(domain_deltas.values())
        >= gates["minimum_domain_chrf_pp_delta"],
        "new_structural_failures": all(
            len(new_failures[name]) <= gates["maximum_new_failures"][name]
            for name in new_failures
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "maximum_strict_failure_count": maximum_strict,
        "domain_chrf_pp_deltas": domain_deltas,
        "new_failures": new_failures,
    }


def save_checkpoint(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    output: Path,
    manifest: dict[str, Any],
) -> None:
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty checkpoint: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "mimi_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("checkpoint_directory", type=Path)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    args = parser.parse_args()
    if args.checkpoint_directory.exists() and any(args.checkpoint_directory.iterdir()):
        raise SystemExit(
            "refusing to overwrite non-empty checkpoint directory: "
            f"{args.checkpoint_directory}"
        )
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    root = Path(__file__).resolve().parents[2]
    contract = validate_contract(args.contract, root=root)
    training = contract["training"]
    if training["checkpoint_steps"] != [
        training["evaluation_steps"],
        training["max_steps"],
    ]:
        raise SystemExit("V21 checkpoint schedule differs")
    if training["batch_size"] % 2:
        raise SystemExit("V21 batch size must divide evenly across both arms")

    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    train_rows = load_rows(root / contract["dataset"]["train"]["path"], "ja-en")
    tagged_valid = load_rows(
        root / contract["dataset"]["tagged_valid"]["path"],
        "ja-en",
    )
    plain_valid = load_rows(
        root / contract["dataset"]["plain_valid"]["path"],
        "ja-en",
    )
    validate_rows(train_rows, tagged_valid, plain_valid, contract)
    tagged_train = [
        row for row in train_rows if row["training_arm"] == "tagged-focus"
    ]
    plain_train = [
        row for row in train_rows if row["training_arm"] == "plain-replay"
    ]

    initial_checkpoint = root / contract["initial_checkpoint"]["path"]
    preservation_checkpoint = root / contract["preservation_checkpoint"]["path"]
    identity = (
        "ja-en",
        contract["initial_checkpoint"]["repository"],
        contract["initial_checkpoint"]["revision"],
    )
    if checkpoint_identity(initial_checkpoint) != identity:
        raise SystemExit("V21 initial checkpoint identity differs")
    if checkpoint_identity(preservation_checkpoint) != identity:
        raise SystemExit("V21 preservation checkpoint identity differs")

    tokenizer = MarianTokenizer.from_pretrained(initial_checkpoint)
    base_vocabulary_size = len(tokenizer)
    if base_vocabulary_size != contract["vocabulary"]["base_size"]:
        raise SystemExit("V21 base vocabulary size differs")
    if tokenizer.add_tokens(list(TAGS)) != len(TAGS):
        raise SystemExit("V21 tokenizer did not add every declared tag")
    tag_ids_list = [int(tokenizer.convert_tokens_to_ids(tag)) for tag in TAGS]
    if tag_ids_list != list(
        range(base_vocabulary_size, base_vocabulary_size + len(TAGS))
    ):
        raise SystemExit("V21 tag IDs are not the declared contiguous suffix")
    for tag, tag_id in zip(TAGS, tag_ids_list):
        if tokenizer.encode(tag, add_special_tokens=False) != [tag_id]:
            raise SystemExit(f"V21 tag is not atomic: {tag}")

    model = MarianMTModel.from_pretrained(initial_checkpoint).to(device)
    parent = MarianMTModel.from_pretrained(preservation_checkpoint).to(device)
    parent.eval()
    parent.requires_grad_(False)
    parent_parameters = {
        name: parameter.detach() for name, parameter in parent.named_parameters()
    }
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    if (
        model.config.vocab_size != contract["vocabulary"]["expanded_size"]
        or model.model.shared.weight.data_ptr() != model.lm_head.weight.data_ptr()
    ):
        raise SystemExit("V21 resized vocabulary or tied output projection differs")
    tag_ids = torch.tensor(tag_ids_list, dtype=torch.long, device=device)

    half_batch = training["batch_size"] // 2
    tagged_loader = DataLoader(
        TranslationRows(tagged_train),
        batch_size=half_batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=list_collator,
    )
    plain_loader = DataLoader(
        TranslationRows(plain_train),
        batch_size=half_batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
        collate_fn=list_collator,
    )
    collator = TagCollator(
        tokenizer,
        maximum_source_tokens=training["max_source_tokens"],
        maximum_target_tokens=training["max_target_tokens"],
    )
    tagged_iterator = iter(tagged_loader)
    plain_iterator = iter(plain_loader)

    shared_embedding = model.model.shared.weight
    other_parameters = [
        parameter for parameter in model.parameters() if parameter is not shared_embedding
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": other_parameters,
                "lr": training["base_learning_rate"],
                "weight_decay": training["weight_decay"],
            },
            {
                "params": [shared_embedding],
                "lr": training["new_tag_embedding_learning_rate"],
                "weight_decay": 0.0,
            },
        ]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training["warmup_steps"],
        num_training_steps=training["max_steps"],
    )

    plain_parent = evaluate_plain(
        parent,
        MarianTokenizer.from_pretrained(preservation_checkpoint),
        plain_valid,
        batch_size=training["evaluation_batch_size"],
        maximum_source_tokens=training["max_source_tokens"],
        maximum_target_tokens=training["max_target_tokens"],
        device=device,
    )
    tagged_step_zero = evaluate_tagged(
        model,
        tokenizer,
        tagged_valid,
        batch_size=training["evaluation_batch_size"],
        maximum_source_tokens=training["max_source_tokens"],
        maximum_target_tokens=training["max_target_tokens"],
        device=device,
    )
    plain_step_zero = evaluate_plain(
        model,
        tokenizer,
        plain_valid,
        batch_size=training["evaluation_batch_size"],
        maximum_source_tokens=training["max_source_tokens"],
        maximum_target_tokens=training["max_target_tokens"],
        device=device,
    )
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "tagged_validation": tagged_step_zero,
            "plain_validation": plain_step_zero,
            "plain_parent": plain_parent,
        }
    ]
    objective_history: list[dict[str, Any]] = []
    common = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": EXPERIMENT,
        "direction": "ja-en",
        "student_repository": identity[1],
        "student_revision": identity[2],
        "license": "CC-BY-SA-4.0",
        "contract": {
            "path": str(args.contract),
            "sha256": sha256(args.contract),
        },
        "implementation": {
            "path": "scripts/translation/train_marian_tagged_values_v21.py",
            "sha256": sha256(Path(__file__)),
        },
        "dataset": contract["dataset"],
        "initial_checkpoint": {
            **contract["initial_checkpoint"],
            "lineage_manifests": checkpoint_lineage_manifests(initial_checkpoint),
        },
        "preservation_checkpoint": {
            **contract["preservation_checkpoint"],
            "lineage_manifests": checkpoint_lineage_manifests(
                preservation_checkpoint
            ),
        },
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "hyperparameters": training,
        "vocabulary": {
            **contract["vocabulary"],
            "tags": list(TAGS),
            "tag_ids": tag_ids_list,
            "tokenizer_atomicity_verified": True,
            "shared_embedding_and_output_projection_tied": True,
            "old_embedding_rows_frozen": True,
        },
        "objective": {
            "sequence": "licensed human target token cross entropy",
            "correct_tag_weight": training["correct_tag_loss_weight"],
            "extra_tag_unlikelihood": (
                "penalize every tag at ordinary positions and every wrong tag "
                "at a declared tag position"
            ),
            "plain_replay_retention": (
                "frozen-parent KL over the original vocabulary plus parameter L2"
            ),
            "balanced_arms": True,
        },
        "synthetic_translations": 0,
        "teacher_outputs": 0,
        "private_reasoning_traces_used": False,
        "promotion_eligible": False,
        "q4_conversion_authorized": False,
        "protected_evaluation_authorized": False,
        "app_change_authorized": False,
        "bundle_replacement_authorized": False,
        "public_upload_authorized": False,
    }

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    update_step = 0
    while update_step < training["max_steps"]:
        tagged_batch, tagged_iterator = next_batch(tagged_iterator, tagged_loader)
        plain_batch, plain_iterator = next_batch(plain_iterator, plain_loader)
        batch = move(collator([*tagged_batch, *plain_batch]), device)
        plain_mask = batch.pop("plain_mask")
        labels = batch["labels"]
        student = model(**batch)
        sequence_loss = tagged_cross_entropy(
            student.logits,
            labels,
            tag_ids,
            correct_tag_weight=training["correct_tag_loss_weight"],
        )
        unlikelihood = extra_tag_unlikelihood(student.logits, labels, tag_ids)
        plain_indices = plain_mask.nonzero(as_tuple=False).squeeze(-1)
        parent_batch = {
            "input_ids": batch["input_ids"].index_select(0, plain_indices),
            "attention_mask": batch["attention_mask"].index_select(0, plain_indices),
            "labels": labels.index_select(0, plain_indices),
        }
        with torch.inference_mode():
            parent_logits = parent(**parent_batch).logits
        kl = frozen_parent_kl_on_plain(
            student.logits.index_select(0, plain_indices),
            parent_logits,
            labels.index_select(0, plain_indices),
            torch.ones(len(plain_indices), dtype=torch.bool, device=device),
            base_vocabulary_size=base_vocabulary_size,
        )
        l2 = l2_to_parent_prefix(model, parent_parameters)
        combined = (
            sequence_loss
            + training["extra_tag_unlikelihood_weight"] * unlikelihood
            + training["frozen_parent_kl_weight"] * kl
            + training["l2_to_parent_weight"] * l2
        )
        (combined / training["gradient_accumulation"]).backward()
        micro_step += 1
        if micro_step % training["gradient_accumulation"]:
            continue
        if shared_embedding.grad is None:
            raise RuntimeError("V21 shared embedding received no gradient")
        shared_embedding.grad[:base_vocabulary_size].zero_()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            training["maximum_gradient_norm"],
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1
        objective_history.append(
            {
                "step": update_step,
                "sequence_loss": float(sequence_loss.detach().cpu()),
                "extra_tag_unlikelihood": float(unlikelihood.detach().cpu()),
                "frozen_parent_kl": float(kl.detach().cpu()),
                "l2_to_parent": float(l2.detach().cpu()),
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "base_learning_rate": optimizer.param_groups[0]["lr"],
                "new_tag_embedding_learning_rate": optimizer.param_groups[1]["lr"],
            }
        )
        if update_step in training["checkpoint_steps"]:
            tagged_metrics = evaluate_tagged(
                model,
                tokenizer,
                tagged_valid,
                batch_size=training["evaluation_batch_size"],
                maximum_source_tokens=training["max_source_tokens"],
                maximum_target_tokens=training["max_target_tokens"],
                device=device,
            )
            plain_metrics = evaluate_plain(
                model,
                tokenizer,
                plain_valid,
                batch_size=training["evaluation_batch_size"],
                maximum_source_tokens=training["max_source_tokens"],
                maximum_target_tokens=training["max_target_tokens"],
                device=device,
            )
            gates = selection_gates(
                tagged_metrics,
                plain_metrics,
                tagged_step_zero=tagged_step_zero,
                plain_parent=plain_parent,
                contract=contract,
            )
            history.append(
                {
                    "step": update_step,
                    "tagged_validation": tagged_metrics,
                    "plain_validation": plain_metrics,
                    "selection_gates": gates,
                }
            )
            checkpoint = args.checkpoint_directory / f"step-{update_step:07d}"
            save_checkpoint(
                model,
                tokenizer,
                checkpoint,
                {
                    **common,
                    "checkpoint_step": update_step,
                    "history": history,
                    "objective_history": objective_history,
                },
            )
            print(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "step": update_step,
                        "tagged_validation": tagged_metrics,
                        "plain_validation": plain_metrics,
                        "selection_gates": gates,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
