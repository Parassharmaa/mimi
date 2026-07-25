#!/usr/bin/env python3
"""Estimate Mimi Batch API token cost without submitting any requests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_synthetic_batch import public_contract, request_contract  # noqa: E402


DEFAULT_INPUT_PRICE = 2.50
DEFAULT_OUTPUT_PRICE = 15.00
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--encoding", default="o200k_base")
    parser.add_argument("--planning-output-tokens", type=int, default=220)
    parser.add_argument("--input-price-per-million", type=float, default=DEFAULT_INPUT_PRICE)
    parser.add_argument("--output-price-per-million", type=float, default=DEFAULT_OUTPUT_PRICE)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.planning_output_tokens < 1:
        raise SystemExit("planning-output-tokens must be positive")
    try:
        import tiktoken
    except ImportError as error:
        raise SystemExit(
            "tiktoken is required; run with `uv run --with tiktoken`"
        ) from error

    contract = request_contract(args.requests)
    encoding = tiktoken.get_encoding(args.encoding)
    rows = read_jsonl(args.requests)
    input_tokens = 0
    maximum_output_tokens = 0
    for row in rows:
        body = row["body"]
        # Encoding the complete compact body intentionally includes the schema,
        # metadata, and message framing. It is a planning estimate, not an API
        # invoice prediction.
        encoded_body = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        input_tokens += len(encoding.encode(encoded_body))
        maximum_output_tokens += int(body["max_output_tokens"])
    planned_output_tokens = len(rows) * args.planning_output_tokens

    input_cost = input_tokens * args.input_price_per_million / 1_000_000
    planned_output_cost = (
        planned_output_tokens * args.output_price_per_million / 1_000_000
    )
    maximum_output_cost = (
        maximum_output_tokens * args.output_price_per_million / 1_000_000
    )
    report = {
        "schema_version": 1,
        "submitted": False,
        "contract": public_contract(contract),
        "estimation": {
            "encoding": args.encoding,
            "tiktoken_version": importlib.metadata.version("tiktoken"),
            "method": (
                "encode each complete compact request body, including structured-output "
                "schema; actual billed input and output tokens may differ"
            ),
            "estimated_input_tokens": input_tokens,
            "planning_output_tokens_per_request": args.planning_output_tokens,
            "planning_output_tokens": planned_output_tokens,
            "maximum_output_tokens": maximum_output_tokens,
        },
        "pricing_usd_per_million_tokens": {
            "input": args.input_price_per_million,
            "output": args.output_price_per_million,
            "source": PRICING_SOURCE,
            "operator_must_refresh_before_submission": True,
        },
        "estimated_cost_usd": {
            "input": round(input_cost, 4),
            "planning_output": round(planned_output_cost, 4),
            "planning_total": round(input_cost + planned_output_cost, 4),
            "maximum_output": round(maximum_output_cost, 4),
            "input_plus_maximum_output": round(input_cost + maximum_output_cost, 4),
        },
        "retention_warning": (
            "store:false applies to nested Responses, but Batch API application "
            "state is retained until deleted and is not Zero Data Retention eligible"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report"] = {
        "path": str(args.output),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
