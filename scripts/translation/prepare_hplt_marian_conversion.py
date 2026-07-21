#!/usr/bin/env python3
"""Stage a pinned HPLT Marian checkpoint for Transformers' audited converter.

HPLT publishes one native Marian NPZ, a joint SentencePiece model, and the
SentencePiece score vocabulary. Transformers' Marian converter expects the
same NPZ plus source/target SentencePiece files, an ID-valued ``vocab.yml``,
and ``decoder.yml``. This script creates those deterministic compatibility
files without duplicating the large checkpoint on the same filesystem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import sentencepiece as spm


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            hasher.update(chunk)
    return hasher.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--direction", choices=("en-ja", "ja-en"), required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", default="CC-BY-4.0")
    parser.add_argument("--beam-size", type=int, default=1)
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    if args.beam_size < 1:
        raise SystemExit("beam-size must be positive")

    model = args.source_directory / "model.npz.best-chrf.npz"
    sentencepiece = args.source_directory / f"model.{args.direction}.spm"
    score_vocabulary = args.source_directory / f"model.{args.direction}.vocab"
    for path in (model, sentencepiece, score_vocabulary):
        if not path.is_file():
            raise SystemExit(f"missing HPLT input: {path}")

    processor = spm.SentencePieceProcessor(model_file=str(sentencepiece))
    pieces = [processor.id_to_piece(index) for index in range(processor.vocab_size())]
    if len(pieces) != len(set(pieces)):
        raise SystemExit("SentencePiece model contains duplicate pieces")
    score_lines = score_vocabulary.read_text(encoding="utf-8").splitlines()
    if len(score_lines) != len(pieces):
        raise SystemExit("SentencePiece model and score vocabulary sizes differ")
    score_pieces = [line.rsplit("\t", 1)[0] for line in score_lines]
    if score_pieces != pieces:
        raise SystemExit("SentencePiece model and score vocabulary piece order differs")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    staged_model = args.output_directory / "model.npz"
    link_or_copy(model, staged_model)
    for name in ("source.spm", "target.spm"):
        shutil.copy2(sentencepiece, args.output_directory / name)
    vocabulary = "".join(
        f"{json.dumps(piece, ensure_ascii=False)}: {index}\n"
        for index, piece in enumerate(pieces)
    )
    (args.output_directory / "vocab.yml").write_text(vocabulary, encoding="utf-8")
    (args.output_directory / "decoder.yml").write_text(
        f"beam-size: {args.beam_size}\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "operation": "stage-native-hplt-marian-for-transformers-conversion",
        "direction": args.direction,
        "repository": args.repository,
        "revision": args.revision,
        "license": args.license,
        "beam_size": args.beam_size,
        "vocabulary_size_before_pad": len(pieces),
        "inputs": {
            "model": {"path": str(model), "sha256": sha256(model)},
            "sentencepiece": {
                "path": str(sentencepiece),
                "sha256": sha256(sentencepiece),
            },
            "score_vocabulary": {
                "path": str(score_vocabulary),
                "sha256": sha256(score_vocabulary),
            },
        },
    }
    (args.output_directory / "mimi_hplt_conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
