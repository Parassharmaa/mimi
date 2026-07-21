#!/usr/bin/env python3
"""Restore labeled URL/number placeholders in a Marian benchmark report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RESERVED_RE = re.compile(r"\[(?:NUM|URL)\d+\]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mapping", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    mapping_payload = json.loads(args.mapping.read_text(encoding="utf-8"))
    mappings = {str(row["caseID"]): row for row in mapping_payload["mappings"]}
    report = json.loads(args.report.read_text(encoding="utf-8"))
    failures = []
    restored_rows = []
    for row in report["results"]:
        case_id = str(row["caseID"])
        mapping = mappings[case_id]
        hypothesis = str(row["hypothesis"])
        reasons = []
        for placeholder, value in mapping["replacements"].items():
            count = hypothesis.count(placeholder)
            if count != 1:
                reasons.append(f"{placeholder}:count={count}")
            else:
                hypothesis = hypothesis.replace(placeholder, value)
        unknown = RESERVED_RE.findall(hypothesis)
        if unknown:
            reasons.append(f"unknown={','.join(sorted(set(unknown)))}")
        restored = {
            **row,
            "source": mapping["originalSource"],
            "hypothesis": hypothesis,
            "outputTokenIDs": None,
            "protectedTokenRestoration": "passed" if not reasons else "failed",
        }
        restored_rows.append(restored)
        if reasons:
            failures.append({"caseID": case_id, "reasons": reasons})
    if set(mappings) != {str(row["caseID"]) for row in report["results"]}:
        raise SystemExit("mapping and report coverage differ")
    output = {
        **{key: value for key, value in report.items() if key != "results"},
        "engine": f"{report['engine']}:protected-token-ablation",
        "protectedTokenRestoration": {
            "cases": len(restored_rows),
            "failures": failures,
            "passed": len(restored_rows) - len(failures),
        },
        "doesNotAuthorizeAppIntegration": True,
        "results": restored_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(restored_rows), "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
