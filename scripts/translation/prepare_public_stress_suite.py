#!/usr/bin/env python3
"""Build a large non-claimable EN↔JA stress suite from licensed human corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CORPORA = {
    "kftt": {"pairs": 200, "domain": "professional-wikipedia"},
    "alt": {"pairs": 100, "domain": "human-translated-news"},
    "tatoeba": {"pairs": 100, "domain": "everyday-conversation"},
    "jlt": {"pairs": 200, "domain": "ministry-published-legal"},
}
DIRECTION_LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path, corpus: str) -> list[dict]:
    output: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" in row:
            metadata = row["metadata"]
            output.append(
                {
                    "source_id": str(metadata["source_id"]),
                    "direction": metadata["direction"],
                    "source": row["messages"][1]["content"],
                    "target": row["messages"][2]["content"],
                    "license": metadata["license"],
                    "attribution": metadata["attribution"],
                }
            )
        else:
            direction = {
                ("en-US", "ja-JP"): "en-ja",
                ("ja-JP", "en-US"): "ja-en",
            }.get((row.get("source_language"), row.get("target_language")))
            if direction is None:
                raise SystemExit(f"invalid direction in {corpus}: {row.get('id')}")
            output.append(
                {
                    "source_id": str(row["source_id"]),
                    "direction": direction,
                    "source": row["source"],
                    "target": row["target"],
                    "license": row["source_license"],
                    "attribution": row["attribution"],
                }
            )
    return output


def paired(rows: list[dict], corpus: str) -> dict[str, dict[str, dict]]:
    pairs: dict[str, dict[str, dict]] = {}
    ambiguous: set[str] = set()
    for row in rows:
        current = pairs.setdefault(row["source_id"], {})
        if row["direction"] in current:
            ambiguous.add(row["source_id"])
            continue
        current[row["direction"]] = row
    return {
        source_id: directions
        for source_id, directions in pairs.items()
        if source_id not in ambiguous and set(directions) == set(DIRECTION_LANGUAGES)
    }


def rank(seed: str, corpus: str, source_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{corpus}\0{source_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kftt_test", type=Path)
    parser.add_argument("alt_test", type=Path)
    parser.add_argument("tatoeba_test", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", default="mimi-public-stress-v1")
    parser.add_argument("--suite-name", default="public-stress-v1")
    parser.add_argument("--id-prefix", default="public-stress")
    parser.add_argument("--kftt-pairs", type=int, default=CORPORA["kftt"]["pairs"])
    parser.add_argument("--alt-pairs", type=int, default=CORPORA["alt"]["pairs"])
    parser.add_argument("--tatoeba-pairs", type=int, default=CORPORA["tatoeba"]["pairs"])
    parser.add_argument("--jlt-test", type=Path)
    parser.add_argument("--jlt-pairs", type=int, default=CORPORA["jlt"]["pairs"])
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    pair_counts = {
        "kftt": args.kftt_pairs,
        "alt": args.alt_pairs,
        "tatoeba": args.tatoeba_pairs,
    }
    if args.jlt_test is not None:
        pair_counts["jlt"] = args.jlt_pairs
    if any(value < 1 for value in pair_counts.values()):
        raise SystemExit("all public-stress pair counts must be positive")
    paths = {
        "kftt": args.kftt_test,
        "alt": args.alt_test,
        "tatoeba": args.tatoeba_test,
    }
    if args.jlt_test is not None:
        paths["jlt"] = args.jlt_test
    suite: list[dict] = []
    inputs: dict[str, dict] = {}
    for corpus in paths:
        policy = CORPORA[corpus]
        rows = load_rows(paths[corpus], corpus)
        pairs = paired(rows, corpus)
        selected_ids = sorted(
            pairs,
            key=lambda source_id: rank(args.seed, corpus, source_id),
        )[: pair_counts[corpus]]
        if len(selected_ids) != pair_counts[corpus]:
            raise SystemExit(
                f"need {pair_counts[corpus]} complete {corpus} pairs, found {len(selected_ids)}"
            )
        inputs[corpus] = {
            "path": str(paths[corpus]),
            "sha256": sha256(paths[corpus]),
            "available_pairs": len(pairs),
            "selected_pairs": len(selected_ids),
            "domain": policy["domain"],
        }
        for source_id in selected_ids:
            for direction, languages in DIRECTION_LANGUAGES.items():
                row = pairs[source_id][direction]
                suite.append(
                    {
                        "id": f"{args.id_prefix}:{corpus}:{source_id}:{direction}",
                        "sourceLanguage": languages[0],
                        "targetLanguage": languages[1],
                        "domain": policy["domain"],
                        "source": row["source"],
                        "references": [row["target"]],
                        "claimEligible": False,
                        "split": "public-stress",
                        "license": row["license"],
                        "provenance": row["attribution"],
                        "reviewStatus": "bootstrap-unreviewed",
                        "sourceCorpus": corpus,
                        "sourceID": source_id,
                        "attribution": row["attribution"],
                    }
                )
    suite.sort(key=lambda row: row["id"])
    cases_per_direction = sum(pair_counts.values())
    expected_cases = cases_per_direction * len(DIRECTION_LANGUAGES)
    if len(suite) != expected_cases:
        raise SystemExit(f"expected {expected_cases} stress cases, found {len(suite)}")
    for direction in DIRECTION_LANGUAGES.values():
        count = sum(
            (row["sourceLanguage"], row["targetLanguage"]) == direction
            for row in suite
        )
        if count != cases_per_direction:
            raise SystemExit(
                f"expected {cases_per_direction} cases for {direction}, found {count}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in suite),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "suite": args.suite_name,
        "seed": args.seed,
        "cases": len(suite),
        "cases_per_direction": cases_per_direction,
        "inputs": inputs,
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
        "claim_eligible": False,
        "limitations": [
            "one human reference per case rather than two independently reviewed references",
            "public corpora may overlap ElanMT pretraining or model selection",
            "Japanese Law Translation cases are formal legal text, not live-speech data",
            "corpus domains do not satisfy Mimi's product-domain promotion quotas",
            "this stress suite cannot replace the sealed 800-case human-authored benchmark",
        ],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
