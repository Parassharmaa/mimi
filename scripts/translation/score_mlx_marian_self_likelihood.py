#!/usr/bin/env python3
"""Measure exact cached-greedy self-likelihood for routed Marian candidates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from transformers import PreTrainedTokenizerFast

from marian_mlx import EOS_TOKEN_ID, PAD_TOKEN_ID, load_model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_report(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise SystemExit(f"invalid translation report: {path}")
    return value


def indexed(report: dict, label: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in report["results"]:
        case_id = str(row.get("caseID", "")).strip()
        if not case_id or case_id in output:
            raise SystemExit(f"{label} has an empty or duplicate case ID")
        output[case_id] = row
    if not output:
        raise SystemExit(f"{label} has no cases")
    return output


def validate_model(path: Path) -> tuple[dict, str]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"model manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit(f"model manifest has no integrity table: {manifest_path}")
    for relative, expected in files.items():
        candidate = path / relative
        if not candidate.is_file():
            raise SystemExit(f"model file is missing: {candidate}")
        if (
            candidate.stat().st_size != expected.get("bytes")
            or sha256(candidate) != expected.get("sha256")
        ):
            raise SystemExit(f"model integrity failure: {candidate}")
    return manifest, sha256(manifest_path)


def direction(row: dict) -> str:
    pair = (row.get("sourceLanguage"), row.get("targetLanguage"))
    if pair == ("en-US", "ja-JP"):
        return "en-ja"
    if pair == ("ja-JP", "en-US"):
        return "ja-en"
    raise SystemExit(f"unsupported direction for {row.get('caseID')}: {pair}")


def assert_same_case(left: dict, right: dict, case_id: str) -> None:
    for field in (
        "caseID",
        "sourceLanguage",
        "targetLanguage",
        "domain",
        "source",
        "references",
        "claimEligible",
    ):
        if left.get(field) != right.get(field):
            raise SystemExit(f"candidate reports disagree on {field}: {case_id}")


def generate_cached_with_nll(
    model,
    input_ids: list[int],
    maximum_tokens: int,
) -> tuple[list[int], float, int]:
    """Run the shipping-shaped cached greedy path and retain chosen-token NLL."""
    encoder_states = model.encode(mx.array([input_ids], dtype=mx.int32))
    decoder_id = PAD_TOKEN_ID
    caches = None
    output: list[int] = []
    token_nll: list[float] = []
    for position_offset in range(maximum_tokens):
        logits, caches = model.decode_step(
            decoder_id,
            encoder_states,
            caches,
            position_offset,
            None,
        )
        next_logits = logits[0, -1]
        next_logits[PAD_TOKEN_ID] = -1e9
        token = int(mx.argmax(next_logits).item())
        token_nll.append(float((mx.logsumexp(next_logits) - next_logits[token]).item()))
        if token == EOS_TOKEN_ID:
            break
        output.append(token)
        decoder_id = token
    mx.synchronize()
    return output, sum(token_nll) / max(1, len(token_nll)), len(token_nll)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generalist_report", type=Path)
    parser.add_argument("en_ja_expert_report", type=Path)
    parser.add_argument("ja_en_expert_report", type=Path)
    parser.add_argument("routed_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--generalist-en-ja-model", type=Path, required=True)
    parser.add_argument("--generalist-ja-en-model", type=Path, required=True)
    parser.add_argument("--expert-en-ja-model", type=Path, required=True)
    parser.add_argument("--expert-ja-en-model", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.max_tokens < 1:
        raise SystemExit("max tokens must be positive")

    report_paths = {
        "generalistReport": args.generalist_report,
        "enJAExpertReport": args.en_ja_expert_report,
        "jaENExpertReport": args.ja_en_expert_report,
        "routedReport": args.routed_report,
    }
    reports = {name: load_report(path) for name, path in report_paths.items()}
    rows = {name: indexed(report, name) for name, report in reports.items()}
    case_ids = set(rows["routedReport"])
    if any(set(values) != case_ids for values in rows.values()):
        raise SystemExit("candidate reports do not have exact common coverage")
    declared_inputs = reports["routedReport"].get("inputs", {})
    for key in ("generalistReport", "enJAExpertReport", "jaENExpertReport"):
        if declared_inputs.get(key, {}).get("sha256") != sha256(report_paths[key]):
            raise SystemExit(f"routed report is not bound to {key}")

    model_paths = {
        "generalist-en-ja": args.generalist_en_ja_model,
        "generalist-ja-en": args.generalist_ja_en_model,
        "expert-en-ja": args.expert_en_ja_model,
        "expert-ja-en": args.expert_ja_en_model,
    }
    manifests: dict[str, dict] = {}
    model_manifest_hashes: dict[str, str] = {}
    for label, path in model_paths.items():
        manifests[label], model_manifest_hashes[label] = validate_model(path)

    routed_expert_ids = {
        case_id
        for case_id, row in rows["routedReport"].items()
        if row.get("selectedEngine") == "expert"
    }
    if not routed_expert_ids:
        raise SystemExit("routed report has no expert-selected cases")

    evidence: dict[str, dict] = {
        case_id: {
            "caseID": case_id,
            "direction": direction(rows["routedReport"][case_id]),
        }
        for case_id in sorted(routed_expert_ids)
    }
    scoring_seconds = 0.0
    for current_direction in ("en-ja", "ja-en"):
        selected_ids = [
            case_id
            for case_id in sorted(routed_expert_ids)
            if evidence[case_id]["direction"] == current_direction
        ]
        for kind in ("generalist", "expert"):
            model_path = model_paths[f"{kind}-{current_direction}"]
            manifest = manifests[f"{kind}-{current_direction}"]
            quantization = (int(manifest["bits"]), int(manifest["group_size"]))
            model = load_model(
                model_path / "model.safetensors",
                quantization_bits=quantization[0],
                quantization_group_size=quantization[1],
            )
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(model_path / "tokenizer.json"),
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>",
            )
            prefix = (manifest.get("source_prefixes") or {}).get(current_direction, "")
            candidate_label = (
                "generalistReport"
                if kind == "generalist"
                else ("enJAExpertReport" if current_direction == "en-ja" else "jaENExpertReport")
            )
            for case_id in selected_ids:
                routed = rows["routedReport"][case_id]
                candidate = rows[candidate_label][case_id]
                assert_same_case(routed, candidate, case_id)
                if kind == "expert" and routed.get("hypothesis") != candidate.get("hypothesis"):
                    raise SystemExit(f"routed output is not expert output: {case_id}")
                encoded = tokenizer.encode(prefix + routed["source"])
                started = time.perf_counter()
                output_ids, mean_nll, scored_tokens = generate_cached_with_nll(
                    model,
                    encoded,
                    args.max_tokens,
                )
                scoring_seconds += time.perf_counter() - started
                expected_ids = candidate.get("outputTokenIDs")
                if output_ids != expected_ids:
                    raise SystemExit(
                        f"cached greedy token mismatch for {kind} {case_id}: "
                        f"expected {expected_ids}, got {output_ids}"
                    )
                evidence[case_id][kind] = {
                    "meanChosenTokenNLL": mean_nll,
                    "scoredTokenCountIncludingEOS": scored_tokens,
                    "outputTokenIDsSHA256": hashlib.sha256(
                        json.dumps(output_ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
            del model, tokenizer
            gc.collect()
            mx.clear_cache()

    model_revision = hashlib.sha256(
        json.dumps(model_manifest_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "public-development self-likelihood expert-veto evidence",
        "claimEligible": False,
        "algorithm": "exact cached-greedy mean chosen-token NLL including EOS",
        "modelRevision": f"four-engine-manifests-sha256:{model_revision}",
        "hardware": platform.machine(),
        "peakResidentBytesDuringSequentialScoring": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scoringSeconds": scoring_seconds,
        "maximumGeneratedTokens": args.max_tokens,
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in report_paths.items()
        },
        "models": {
            name: {
                "path": str(path),
                "manifestSha256": model_manifest_hashes[name],
            }
            for name, path in model_paths.items()
        },
        "runtime": {
            "scriptSha256": sha256(Path(__file__).resolve()),
            "marianRuntimeSha256": sha256(Path(__file__).resolve().parent / "marian_mlx.py"),
        },
        "results": [evidence[case_id] for case_id in sorted(evidence)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote self-likelihood evidence for {len(evidence)} cases to {args.output}")


if __name__ == "__main__":
    main()
