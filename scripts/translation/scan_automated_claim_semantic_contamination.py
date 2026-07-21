#!/usr/bin/env python3
"""Exhaustively scan frozen claim text against unique controlled exposure text."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import unicodedata
from pathlib import Path
from typing import Iterator

from validate_automated_benchmark_suite import sha256, validate_exposure_manifest
from validate_benchmark_suite import iter_training_text, normalized


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def rows(path: Path) -> list[dict]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values or not all(isinstance(value, dict) for value in values):
        raise SystemExit(f"expected non-empty JSONL objects: {path}")
    return values


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def language_bucket(text: str) -> str:
    japanese = 0
    latin = 0
    for character in unicodedata.normalize("NFKC", text):
        codepoint = ord(character)
        if (
            0x3040 <= codepoint <= 0x30FF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            japanese += 1
        elif "LATIN" in unicodedata.name(character, ""):
            latin += 1
    return "ja" if japanese > max(1, latin // 4) else "en"


def language_from_tag(value: object) -> str:
    tag = str(value or "").lower()
    if tag.startswith("ja"):
        return "ja"
    if tag.startswith("en"):
        return "en"
    raise SystemExit(f"unsupported benchmark language tag: {value}")


def exposure_text(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"invalid exposure JSONL row: {path}:{row_number}")
            for field, text in iter_training_text(row):
                yield {
                    "text": text,
                    "language": language_bucket(text),
                    "path": str(path),
                    "row": row_number,
                    "field": field,
                }


def query_records(suite: list[dict]) -> tuple[list[dict], bool]:
    output: list[dict] = []
    claim_ready = True
    for row in suite:
        case_id = str(row.get("id", "")).strip()
        source = str(row.get("source", "")).strip()
        references = [str(value).strip() for value in row.get("references", [])]
        if not case_id or not source or any(not value for value in references):
            raise SystemExit(f"invalid benchmark text: {case_id}")
        output.append(
            {
                "caseID": case_id,
                "role": "source",
                "text": source,
                "language": language_from_tag(row.get("sourceLanguage")),
            }
        )
        for index, reference in enumerate(references):
            output.append(
                {
                    "caseID": case_id,
                    "role": f"reference[{index}]",
                    "text": reference,
                    "language": language_from_tag(row.get("targetLanguage")),
                }
            )
        if row.get("claimEligible") is not True or len(references) != 2:
            claim_ready = False
    return output, claim_ready


def mean_pool(model_output, attention_mask, torch):
    token_embeddings = model_output[0]
    expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * expanded, 1) / torch.clamp(expanded.sum(1), min=1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("claim_manifest", type=Path)
    parser.add_argument("exposure_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.batch_size < 1:
        raise SystemExit("batch size must be positive")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    claim = load(args.claim_manifest)
    suite = rows(args.suite)
    extraction_paths, _, exposure = validate_exposure_manifest(
        args.exposure_manifest, claim
    )
    queries, claim_ready = query_records(suite)
    threshold = float(claim["contaminationPolicy"]["maximumSemanticSimilarity"])
    if not 0 < threshold < 1:
        raise SystemExit("invalid semantic similarity threshold")

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    cache = str(args.cache_directory.resolve()) if args.cache_directory is not None else None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=cache,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=cache,
        trust_remote_code=False,
    ).to(device)
    model.eval()

    def encode(texts: list[str]):
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
            pooled = mean_pool(output, encoded["attention_mask"], torch)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    query_embeddings = encode([record["text"] for record in queries])
    query_languages = [record["language"] for record in queries]
    best_scores = [-math.inf] * len(queries)
    best_records: list[dict | None] = [None] * len(queries)
    seen: set[str] = set()
    unique_by_language = {"en": 0, "ja": 0}
    raw_texts = 0
    batch: list[dict] = []

    def consume(values: list[dict]) -> None:
        if not values:
            return
        embeddings = encode([value["text"] for value in values])
        similarities = embeddings @ query_embeddings.T
        allowed = torch.tensor(
            [
                [value["language"] == query_language for query_language in query_languages]
                for value in values
            ],
            dtype=torch.bool,
            device=device,
        )
        similarities = similarities.masked_fill(~allowed, -torch.inf)
        maxima, indices = torch.max(similarities, dim=0)
        maxima_values = maxima.detach().cpu().numpy()
        index_values = indices.detach().cpu().numpy()
        for query_index, score in enumerate(maxima_values.tolist()):
            if math.isfinite(score) and score > best_scores[query_index]:
                selected = values[int(index_values[query_index])]
                best_scores[query_index] = float(score)
                best_records[query_index] = selected

    for extraction_path in extraction_paths:
        for record in exposure_text(extraction_path):
            raw_texts += 1
            key = normalized(record["text"])
            if key in seen:
                continue
            seen.add(key)
            unique_by_language[record["language"]] += 1
            batch.append(record)
            if len(batch) >= args.batch_size:
                consume(batch)
                batch = []
    consume(batch)

    by_case: dict[str, dict] = {}
    for query, score, nearest in zip(queries, best_scores, best_records):
        if nearest is None or not math.isfinite(score):
            raise SystemExit(f"no same-language exposure text for query: {query['caseID']}")
        evidence = {
            "queryRole": query["role"],
            "querySHA256": text_sha256(query["text"]),
            "maximumSimilarity": float(np.float32(score)),
            "nearestExposureSHA256": text_sha256(nearest["text"]),
            "nearestExposurePath": nearest["path"],
            "nearestExposureRow": nearest["row"],
            "nearestExposureField": nearest["field"],
        }
        current = by_case.get(query["caseID"])
        if current is None or evidence["maximumSimilarity"] > current["maximumSimilarity"]:
            by_case[query["caseID"]] = {"caseID": query["caseID"], **evidence}

    results = [by_case[str(row["id"])] for row in suite]
    failures = [value for value in results if value["maximumSimilarity"] > threshold]
    if claim_ready:
        status = "passed" if not failures else "failed"
    else:
        status = (
            "passed-source-only-not-claim-eligible"
            if not failures
            else "failed-source-only-not-claim-eligible"
        )
    output = {
        "schemaVersion": 1,
        "status": status,
        "claimEligible": claim_ready and not failures,
        "suiteSHA256": sha256(args.suite),
        "exposureManifestSHA256": sha256(args.exposure_manifest),
        "threshold": threshold,
        "embedderModel": args.model,
        "embedderRevision": args.revision,
        "embedderLicense": "Apache-2.0",
        "embeddingDimension": int(query_embeddings.shape[1]),
        "maximumSequenceTokens": 128,
        "device": device,
        "batchSize": args.batch_size,
        "exhaustiveUniqueExposureTextScan": True,
        "candidatePrefilterUsed": False,
        "sourceAndReferencesScanned": claim_ready,
        "queryTextCount": len(queries),
        "comparisonLanguagePolicy": "same-declared-language-via-deterministic-script-bucket",
        "exposureTextCounts": {
            "raw": raw_texts,
            "unique": len(seen),
            "uniqueByLanguage": unique_by_language,
            "manifestDeclared": sum(int(asset.get("textCount", 0)) for asset in exposure["assets"]),
        },
        "failureCount": len(failures),
        "packages": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "failureCount": len(failures),
                "rawExposureTexts": raw_texts,
                "uniqueExposureTexts": len(seen),
                "queryTexts": len(queries),
                "output": str(args.output),
                "outputSHA256": sha256(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
