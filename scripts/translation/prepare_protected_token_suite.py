#!/usr/bin/env python3
"""Replace atomic URLs and numbers with labeled Marian-safe placeholders."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


TOKEN_RE = re.compile(
    r"(?P<url>https?://[^\s]+)|"
    r"(?P<number>(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d))"
)
RESERVED_RE = re.compile(r"\[(?:NUM|URL)\d+\]")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output_suite", type=Path)
    parser.add_argument("mapping", type=Path)
    args = parser.parse_args()
    output_rows = []
    mappings = []
    for line_number, line in enumerate(args.suite.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        source = unicodedata.normalize("NFKC", str(row["source"]))
        if RESERVED_RE.search(source):
            raise SystemExit(f"source contains reserved placeholder at line {line_number}")
        values: dict[str, str] = {}
        counters = {"url": 0, "number": 0}

        def replace(match: re.Match[str]) -> str:
            kind = "url" if match.lastgroup == "url" else "number"
            prefix = "URL" if kind == "url" else "NUM"
            placeholder = f"[{prefix}{counters[kind]}]"
            counters[kind] += 1
            values[placeholder] = match.group(0)
            return placeholder

        protected = TOKEN_RE.sub(replace, source)
        output_rows.append({**row, "source": protected})
        mappings.append(
            {
                "caseID": row["id"],
                "originalSource": row["source"],
                "protectedSource": protected,
                "replacements": values,
            }
        )
    args.output_suite.parent.mkdir(parents=True, exist_ok=True)
    args.output_suite.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    payload = {
        "schemaVersion": 1,
        "purpose": "development-only reversible URL/number placeholder ablation",
        "inputSuite": {"path": str(args.suite), "sha256": sha256(args.suite)},
        "outputSuite": {"path": str(args.output_suite), "sha256": sha256(args.output_suite)},
        "cases": len(output_rows),
        "casesWithReplacements": sum(bool(row["replacements"]) for row in mappings),
        "mappings": mappings,
    }
    args.mapping.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in ("cases", "casesWithReplacements")}, indent=2))


if __name__ == "__main__":
    main()
