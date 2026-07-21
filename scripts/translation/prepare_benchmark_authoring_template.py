#!/usr/bin/env python3
"""Create an incomplete, manifest-sized human benchmark authoring template."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def apportioned(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {domain: total * weight for domain, weight in weights.items()}
    counts = {domain: math.floor(value) for domain, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(weights, key=lambda domain: (-(raw[domain] - counts[domain]), domain))
    for domain in order[:remaining]:
        counts[domain] += 1
    return counts


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases_per_direction = int(manifest["exactCasesPerDirection"])
    minimum_references = int(manifest["referencePolicy"]["minimumReferencesPerCase"])
    domains = apportioned(cases_per_direction, manifest["domains"])
    rows: list[dict] = []
    for direction in manifest["directions"]:
        source_language, target_language = direction.split(">", 1)
        prefix = "en" if source_language == "en-US" else "ja"
        index = 0
        for domain, count in domains.items():
            for _ in range(count):
                index += 1
                rows.append(
                    {
                        "id": f"heldout-{prefix}-{index:03d}",
                        "documentID": "",
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language,
                        "domain": domain,
                        "source": "",
                        "references": ["" for _ in range(minimum_references)],
                        "sourceAuthorID": "",
                        "referenceAuthorIDs": ["" for _ in range(minimum_references)],
                        "split": "heldout-draft",
                        "license": "",
                        "provenance": "",
                        "reviewStatus": "authoring-template-incomplete",
                        "claimEligible": False,
                        "sourceGeneratedByAI": None,
                        "referenceGeneratedByAI": None,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    template_manifest = {
        "schemaVersion": 1,
        "status": "incomplete-human-authoring-template-not-reviewable",
        "suiteID": manifest["suiteID"],
        "cases": len(rows),
        "casesPerDirection": cases_per_direction,
        "domainsPerDirection": domains,
        "requiredManualFields": [
            "documentID",
            "source",
            "references",
            "sourceAuthorID",
            "referenceAuthorIDs",
            "license",
            "provenance",
            "sourceGeneratedByAI=false",
            "referenceGeneratedByAI=false",
        ],
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(template_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(template_manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
