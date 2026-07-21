#!/usr/bin/env python3
"""Export pinned source-router scores for cross-runtime parity verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from source_expert_router import SourceExpertRouter


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("en_ja_router", type=Path)
    parser.add_argument("ja_en_router", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    routers = {
        "en-ja": SourceExpertRouter.load(args.en_ja_router),
        "ja-en": SourceExpertRouter.load(args.ja_en_router),
    }
    if any(router.direction != direction for direction, router in routers.items()):
        raise SystemExit("router direction does not match its command-line role")
    direction_for_source = {"en-US": "en-ja", "ja-JP": "ja-en"}
    results = []
    seen_ids = set()
    for line_number, line in enumerate(args.suite.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row["id"])
        if case_id in seen_ids:
            raise SystemExit(f"duplicate case ID at line {line_number}: {case_id}")
        seen_ids.add(case_id)
        source_language = str(row["sourceLanguage"])
        if source_language not in direction_for_source:
            raise SystemExit(f"unsupported source language at line {line_number}")
        direction = direction_for_source[source_language]
        source = str(row["source"])
        router = routers[direction]
        results.append(
            {
                "caseID": case_id,
                "direction": direction,
                "source": source,
                "score": router.score(source),
                "routesToExpert": router.routes_to_expert(source),
            }
        )
    if not results:
        raise SystemExit("suite is empty")
    payload = {
        "schemaVersion": 1,
        "purpose": "Python-to-Swift source-only expert-router parity",
        "suite": {"path": str(args.suite), "sha256": sha256(args.suite)},
        "routers": {
            "en-ja": {"path": str(args.en_ja_router), "sha256": sha256(args.en_ja_router)},
            "ja-en": {"path": str(args.ja_en_router), "sha256": sha256(args.ja_en_router)},
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(results)} router predictions to {args.output}")


if __name__ == "__main__":
    main()
