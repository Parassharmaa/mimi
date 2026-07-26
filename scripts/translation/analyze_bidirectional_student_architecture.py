#!/usr/bin/env python3
"""Reproducible parameter, q4-size, and compute analysis for a shared Marian student."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

VOCABULARY = 32_001
DIMENSIONS = 512
GROUP_SIZE = 64
QUANTIZATION_BITS = 4
FLOAT_BYTES = 2

# The authenticated preferred-v3 EN->JA child contains one tokenizer.json,
# tokenizer_config.json, and manifest beside model.safetensors.
SINGLE_MODEL_ASSET_BYTES = 2_402_797

# Calibrated only for projecting safetensors metadata, not tensor payload.
HEADER_BYTES_PER_TENSOR = 108
HEADER_FIXED_BYTES = 1_024


@dataclass(frozen=True)
class Architecture:
    name: str
    encoder_layers: int
    decoder_layers: int
    encoder_ffn: int
    decoder_ffn: int
    initialization: str
    role: str


ARCHITECTURES = (
    Architecture(
        name="incumbent-shape-6e6d-ffn2048",
        encoder_layers=6,
        decoder_layers=6,
        encoder_ffn=2_048,
        decoder_ffn=2_048,
        initialization="authenticated directional Marian checkpoint",
        role="calibration-only",
    ),
    Architecture(
        name="shared-wide-6e6d-ffn4608",
        encoder_layers=6,
        decoder_layers=6,
        encoder_ffn=4_608,
        decoder_ffn=4_608,
        initialization=(
            "output-preserving FFN widening from one directional parent; "
            "dual-teacher distillation is still required for the other direction"
        ),
        role="recommended first shared-capacity control",
    ),
    Architecture(
        name="shared-deep-18e4d-ffn2048",
        encoder_layers=18,
        decoder_layers=4,
        encoder_ffn=2_048,
        decoder_ffn=2_048,
        initialization=(
            "purpose-trained depth expansion; existing Marian post-norm blocks "
            "cannot be appended as exact identity layers"
        ),
        role="capacity-matched deep-encoder ablation",
    ),
    Architecture(
        name="shared-deep-24e4d-ffn2048",
        encoder_layers=24,
        decoder_layers=4,
        encoder_ffn=2_048,
        decoder_ffn=2_048,
        initialization="purpose-trained encoder-aware sequence distillation",
        role="conditional larger dense control",
    ),
    Architecture(
        name="shared-deep-30e4d-ffn2048",
        encoder_layers=30,
        decoder_layers=4,
        encoder_ffn=2_048,
        decoder_ffn=2_048,
        initialization="purpose-trained encoder-aware sequence distillation",
        role="upper end of the registered 90-130M capacity range",
    ),
)

PROFILES = (
    ("live-short", 16, 16),
    ("sentence", 64, 32),
    ("long-segment", 192, 64),
)


def architecture_counts(architecture: Architecture) -> dict[str, int]:
    d = DIMENSIONS
    encoder_matrices = (
        4 * d * d + 2 * d * architecture.encoder_ffn
    ) * architecture.encoder_layers
    decoder_matrices = (
        8 * d * d + 2 * d * architecture.decoder_ffn
    ) * architecture.decoder_layers
    matrix_elements = VOCABULARY * d + encoder_matrices + decoder_matrices

    encoder_vectors = (
        architecture.encoder_ffn + 9 * d
    ) * architecture.encoder_layers
    decoder_vectors = (
        architecture.decoder_ffn + 15 * d
    ) * architecture.decoder_layers
    vector_elements = VOCABULARY + encoder_vectors + decoder_vectors

    dense_tensor_count = (
        2
        + 16 * architecture.encoder_layers
        + 26 * architecture.decoder_layers
    )
    quantized_matrix_count = (
        1
        + 6 * architecture.encoder_layers
        + 10 * architecture.decoder_layers
    )
    q4_tensor_count = dense_tensor_count + 2 * quantized_matrix_count
    return {
        "matrix_elements": matrix_elements,
        "vector_elements": vector_elements,
        "parameters": matrix_elements + vector_elements,
        "q4_tensor_count": q4_tensor_count,
    }


def projected_q4_model_bytes(counts: dict[str, int]) -> int:
    # MLX affine q4/group-64 stores packed 4-bit weights plus float16 scale and
    # bias for every group. Biases, layer norms, and final_logits_bias remain
    # float16. The tiny safetensors-header term is calibrated independently.
    matrix_payload = counts["matrix_elements"] * (
        QUANTIZATION_BITS / 8 + 2 * FLOAT_BYTES / GROUP_SIZE
    )
    vector_payload = counts["vector_elements"] * FLOAT_BYTES
    header = (
        counts["q4_tensor_count"] * HEADER_BYTES_PER_TENSOR
        + HEADER_FIXED_BYTES
    )
    return round(matrix_payload + vector_payload + header)


def approximate_macs(
    architecture: Architecture,
    *,
    source_tokens: int,
    target_tokens: int,
) -> int:
    """Count leading matrix/attention multiply-accumulates for cached greedy MT."""
    d = DIMENSIONS
    encoder_layer = (
        source_tokens * (4 * d * d + 2 * d * architecture.encoder_ffn)
        + 2 * source_tokens * source_tokens * d
    )
    decoder_layer = (
        target_tokens * (6 * d * d + 2 * d * architecture.decoder_ffn)
        + 2 * source_tokens * d * d
        + target_tokens * (target_tokens + 1) * d
        + 2 * source_tokens * target_tokens * d
    )
    return (
        architecture.encoder_layers * encoder_layer
        + architecture.decoder_layers * decoder_layer
    )


def analyze() -> dict[str, Any]:
    baseline = ARCHITECTURES[0]
    baseline_macs = {
        name: approximate_macs(
            baseline,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
        )
        for name, source_tokens, target_tokens in PROFILES
    }
    rows = []
    for architecture in ARCHITECTURES:
        counts = architecture_counts(architecture)
        model_bytes = projected_q4_model_bytes(counts)
        compute = {}
        for name, source_tokens, target_tokens in PROFILES:
            macs = approximate_macs(
                architecture,
                source_tokens=source_tokens,
                target_tokens=target_tokens,
            )
            compute[name] = {
                "source_tokens": source_tokens,
                "target_tokens": target_tokens,
                "approximate_macs": macs,
                "ratio_to_incumbent_shape": macs / baseline_macs[name],
            }
        rows.append(
            {
                **asdict(architecture),
                **counts,
                "projected_q4_model_bytes": model_bytes,
                "projected_single_model_pack_bytes": (
                    model_bytes + SINGLE_MODEL_ASSET_BYTES
                ),
                "compute_profiles": compute,
            }
        )
    return {
        "schema_version": 1,
        "analysis": "single-physical-bidirectional-marian-capacity",
        "constants": {
            "vocabulary": VOCABULARY,
            "dimensions": DIMENSIONS,
            "attention_heads": 8,
            "quantization": {
                "bits": QUANTIZATION_BITS,
                "group_size": GROUP_SIZE,
                "scale_and_bias_dtype": "float16",
            },
            "single_model_asset_bytes": SINGLE_MODEL_ASSET_BYTES,
            "preferred_model_target_bytes": 150_000_000,
            "hard_model_ceiling_bytes": 500_000_000,
        },
        "architectures": rows,
        "decision": {
            "recommended_first_control": "shared-wide-6e6d-ffn4608",
            "reason": (
                "It is the smallest fully shared 90M-class option, preserves one "
                "directional parent's exact function at initialization, keeps the "
                "already-supported 512-wide/6e6d topology, and has a flatter "
                "compute increase across long inputs than the deep alternatives."
            ),
            "training_authorized": False,
            "app_change_authorized": False,
        },
        "limitations": [
            "Projected bytes are architecture estimates, not a converted bundle claim.",
            "MAC ratios are shape-based compute estimates, not measured Apple-Silicon latency.",
            "Output-preserving widening protects only the initialization direction; the shared model must still pass independent retention gates in both directions.",
            "The current Swift Marian loader hard-codes six layers and FFN width 2048; no Swift change is justified before a trained candidate passes Python/MLX quality gates.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(),
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
