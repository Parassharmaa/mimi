#!/usr/bin/env python3
"""Convert a pinned ElanMT safetensors checkpoint to a quantized MLX pack."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from sentencepiece import sentencepiece_model_pb2
from tokenizers import SentencePieceUnigramTokenizer
from tokenizers.processors import TemplateProcessing

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marian_mlx import load_model  # noqa: E402


COPY_FILES = (
    "config.json",
    "generation_config.json",
    "source.spm",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
)

OPTIONAL_COPY_FILES = ("special_tokens_map.json",)

KFTT_NOTICE = (
    "The data used in this service contains English contents which is translated "
    "by the National Institute of Information and Communications Technology "
    "(NICT) from Japanese sentences on Wikipedia. Our use of this data is licensed "
    "by the Creative Commons Attribution-Share-Alike License 3.0. Please refer to "
    "http://creativecommons.org/licenses/by-sa/3.0/ or "
    "http://alaginrc.nict.go.jp/WikiCorpus/ for details."
)

ATTRIBUTIONS_BY_LICENSE = {
    "CC-BY-SA-3.0": {
        "corpus": "Kyoto Free Translation Task (KFTT) 1.0",
        "license": "CC-BY-SA-3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "source_url": "https://www.phontron.com/kftt/",
        "required_notice": KFTT_NOTICE,
        "release_condition": (
            "Blocked until derivative-weight share-alike policy, attribution "
            "placement, and source offer are reviewed and documented."
        ),
    },
    "CC-BY-2.0-FR": {
        "corpus": "Tatoeba sentence pairs via ManyThings",
        "license": "CC-BY-2.0-FR",
        "license_url": "https://creativecommons.org/licenses/by/2.0/fr/",
        "source_url": "https://www.manythings.org/anki/",
        "required_notice": (
            "Preserve per-row Tatoeba sentence IDs, contributor names, source "
            "links, and license references from the authenticated training data."
        ),
        "release_condition": (
            "Blocked until a release attribution sidecar includes every retained "
            "row's contributor metadata."
        ),
    },
    "CC-BY-4.0": {
        "corpus": "NICT openly licensed research corpora",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_url": "https://alaginrc.nict.go.jp/resources/alt-gt/",
        "required_notice": (
            "Preserve the exact corpus-level NICT attribution and citations from "
            "the authenticated dataset manifest and release sidecar."
        ),
        "release_condition": "Attribution review required before distribution.",
    },
    "PDL-1.0-compatible-CC-BY-4.0": {
        "corpus": "Japanese Law Translation Database System",
        "license": "Japan Public Data License 1.0 compatible terms",
        "license_url": "https://www.digital.go.jp/en/resources/open_data/public_data_license_v1.0",
        "source_url": "https://www.japaneselawtranslation.go.jp/en/",
        "required_notice": (
            "Contains source and translation data from the Japanese Law Translation "
            "Database System. The English translations are not official texts. "
            "Preserve the database source URL, access date, terms URL, and Japan "
            "Public Data License 1.0 notice in the release attribution sidecar."
        ),
        "release_condition": (
            "Include the database disclaimer, source link, access date, terms, and "
            "Japan Public Data License 1.0 notice before distribution."
        ),
    },
    "project-owned": {
        "corpus": "Mimi paired shipping copy",
        "license": "project-owned",
        "source_url": "https://github.com/paras/Mimi",
        "required_notice": "Preserve source repository revision and source lines.",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_fast_tokenizer(source_spm: Path, output: Path) -> None:
    # tokenizers imports this generated binding by its historical top-level name.
    sys.modules["sentencepiece_model_pb2"] = sentencepiece_model_pb2
    tokenizer = SentencePieceUnigramTokenizer.from_spm(str(source_spm))
    tokenizer.add_special_tokens(["</s>", "<unk>", "<pad>"])
    tokenizer.post_processor = TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", 0)]
    )
    tokenizer.save(str(output))


def write_swift_tokenizer_config(path: Path) -> None:
    configuration = json.loads(path.read_text(encoding="utf-8"))
    # swift-transformers uses the tokenizer class to select its Unigram model
    # implementation. All preprocessing/decoding details remain in tokenizer.json.
    configuration["tokenizer_class"] = "T5Tokenizer"
    path.write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def training_data_provenance(training_manifest: dict) -> dict | None:
    dataset_manifest = training_manifest.get("dataset_manifest")
    if not dataset_manifest:
        dataset_manifest = training_manifest.get("dataset", {}).get("manifest")
    if not dataset_manifest:
        return None

    dataset_payload = {}
    declared_path = dataset_manifest.get("path")
    declared_sha256 = dataset_manifest.get("sha256")
    if declared_path and declared_sha256:
        manifest_path = Path(declared_path)
        if not manifest_path.is_file():
            raise SystemExit(f"missing declared dataset manifest: {manifest_path}")
        actual_sha256 = digest(manifest_path)
        if actual_sha256 != declared_sha256:
            raise SystemExit(
                "dataset manifest digest mismatch: "
                f"declared {declared_sha256}, found {actual_sha256}"
            )
        dataset_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    effective_licenses = dataset_manifest.get("effective_licenses") or {}
    license_names = sorted(
        {
            license_name
            for split in effective_licenses.values()
            if isinstance(split, dict)
            for license_name in split
        }
    )
    required_attributions = [
        ATTRIBUTIONS_BY_LICENSE[license_name]
        for license_name in license_names
        if license_name in ATTRIBUTIONS_BY_LICENSE
    ]
    return {
        "training_manifest_sha256": training_manifest.get("manifest_sha256"),
        "dataset_manifest": dataset_manifest,
        "target_source": dataset_manifest.get("target_source"),
        "effective_licenses": effective_licenses,
        "teacher_models": (
            dataset_manifest.get("teacher_models")
            or dataset_payload.get("teacher_models")
            or {}
        ),
        "required_attributions": required_attributions,
        "distribution_status": (
            "blocked-pending-share-alike-and-attribution-review"
            if {"CC-BY-SA-3.0", "CC-BY-2.0-FR"}.intersection(license_names)
            else "provenance-incomplete-not-approved-for-distribution"
            if not effective_licenses
            else "research-candidate-not-approved-for-distribution"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--direction", choices=("en-ja", "ja-en", "bidirectional"), required=True
    )
    parser.add_argument("--bits", type=int, choices=(4, 6, 8), default=4)
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    args = parser.parse_args()

    source_weights = args.source / "model.safetensors"
    if not source_weights.is_file():
        raise SystemExit(f"missing source weights: {source_weights}")
    missing = [name for name in COPY_FILES if not (args.source / name).is_file()]
    if missing:
        raise SystemExit(f"source checkpoint is missing: {', '.join(missing)}")

    model = load_model(source_weights)
    encoder_ffn_dimensions = int(model.encoder.layers[0].fc1.weight.shape[0])
    decoder_ffn_dimensions = int(model.decoder.layers[0].fc1.weight.shape[0])
    # Full-parameter training checkpoints are normally saved in float32, while
    # the pinned ElanMT sources are float16. Normalize before quantization so
    # scales, biases, layer norms, and the final bias do not silently add ~4 MB
    # per direction or change the shipping compute type.
    model.set_dtype(mx.float16)
    nn.quantize(model, group_size=args.group_size, bits=args.bits)
    mx.eval(model.parameters())

    args.output.mkdir(parents=True, exist_ok=True)
    output_weights = args.output / "model.safetensors"
    model.save_weights(str(output_weights))
    for name in (*COPY_FILES, *OPTIONAL_COPY_FILES):
        if not (args.source / name).is_file():
            continue
        shutil.copy2(args.source / name, args.output / name)
    write_fast_tokenizer(args.source / "source.spm", args.output / "tokenizer.json")
    write_swift_tokenizer_config(args.output / "tokenizer_config.json")

    training_manifest_path = args.source / "mimi_training_manifest.json"
    training_manifest = (
        json.loads(training_manifest_path.read_text(encoding="utf-8"))
        if training_manifest_path.is_file()
        else {}
    )
    if training_manifest:
        training_manifest["manifest_sha256"] = digest(training_manifest_path)
    training_data = training_data_provenance(training_manifest)
    converter_path = Path(__file__).resolve()
    workspace = Path.cwd().resolve()
    try:
        converter_record_path = converter_path.relative_to(workspace).as_posix()
    except ValueError:
        converter_record_path = str(converter_path)

    manifest = {
        "format": "mimi-mlx-marian-v1",
        "architecture": "Marian encoder-decoder",
        "encoder_layers": len(model.encoder.layers),
        "decoder_layers": len(model.decoder.layers),
        "encoder_ffn_dimensions": encoder_ffn_dimensions,
        "decoder_ffn_dimensions": decoder_ffn_dimensions,
        "direction": args.direction,
        "source_repository": args.repository,
        "source_revision": args.revision,
        "source_weights_sha256": digest(source_weights),
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "ElanMT by ELAN MITSUA Project / Abstract Engine",
        "bits": args.bits,
        "group_size": args.group_size,
        "compute_dtype": "float16",
        "conversion": {
            "schemaVersion": 1,
            "operation": "float16-normalize-then-mlx-affine-quantize",
            "sourceWeightsSha256": digest(source_weights),
            "outputWeightsSha256": digest(output_weights),
            "quantization": {
                "bits": args.bits,
                "groupSize": args.group_size,
                "computeDtype": "float16",
            },
            "tool": {
                "path": converter_record_path,
                "sha256": digest(converter_path),
            },
            "runtime": {
                "python": platform.python_version(),
                "mlx": importlib.metadata.version("mlx"),
                "tokenizers": importlib.metadata.version("tokenizers"),
                "sentencepiece": importlib.metadata.version("sentencepiece"),
            },
        },
        "source_prefixes": training_manifest.get("source_prefixes"),
        "training_data": training_data,
        "files": {
            item.name: {"bytes": item.stat().st_size, "sha256": digest(item)}
            for item in sorted(args.output.iterdir())
            if item.is_file() and item.name != "manifest.json"
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": directory_bytes(args.output),
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
