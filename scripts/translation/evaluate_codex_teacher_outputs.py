#!/usr/bin/env python3
"""Score source-only Codex teacher candidates against licensed train references.

These metrics diagnose teacher-data quality only. They are not held-out model
promotion results, and the per-row oracle is explicitly reference-leaking.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

from filter_synthetic_batch import normalized, response_payload


STYLES = (
    "natural-spoken",
    "concise-caption",
    "meaning-conservative",
)


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing report: {path}") from error


def score_group(group: list[dict]) -> dict:
    chrf = CHRF(word_order=2)
    bleu = BLEU(tokenize="intl", effective_order=True)
    references = [row["reference"] for row in group]
    style_scores: dict[str, dict] = {}
    for style in STYLES:
        hypotheses = [row["candidates"][style] for row in group]
        style_scores[style] = {
            "chrf_pp": chrf.corpus_score(hypotheses, [references]).score,
            "bleu_intl": bleu.corpus_score(hypotheses, [references]).score,
            "normalized_exact_matches": sum(
                normalized(hypothesis) == normalized(reference)
                for hypothesis, reference in zip(hypotheses, references)
            ),
        }

    oracle_hypotheses: list[str] = []
    oracle_styles: Counter[str] = Counter()
    for row in group:
        sentence_scores = {
            style: chrf.sentence_score(
                row["candidates"][style],
                [row["reference"]],
            ).score
            for style in STYLES
        }
        best_style = max(STYLES, key=lambda style: sentence_scores[style])
        oracle_styles[best_style] += 1
        oracle_hypotheses.append(row["candidates"][best_style])
    return {
        "rows": len(group),
        "styles": style_scores,
        "reference_leaking_oracle_upper_bound": {
            "chrf_pp": chrf.corpus_score(oracle_hypotheses, [references]).score,
            "bleu_intl": bleu.corpus_score(
                oracle_hypotheses,
                [references],
            ).score,
            "selected_style_counts": dict(sorted(oracle_styles.items())),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", type=Path)
    parser.add_argument("teacher_output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--review-queue",
        type=Path,
        help="optional deterministic-filter queue for admitted-source counts",
    )
    args = parser.parse_args()

    seed_by_id = {str(row["id"]): row for row in rows(args.seeds)}
    evaluated: list[dict] = []
    seen: set[str] = set()
    risk_tags: Counter[str] = Counter()
    for result in rows(args.teacher_output):
        custom_id = str(result["custom_id"])
        if custom_id in seen:
            raise SystemExit(f"duplicate teacher result: {custom_id}")
        seen.add(custom_id)
        seed = seed_by_id.get(custom_id)
        if seed is None:
            raise SystemExit(f"unknown teacher result: {custom_id}")
        reference = str(seed.get("reference_translation") or "").strip()
        if not reference:
            continue
        payload, _ = response_payload(result)
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 3:
            raise SystemExit(f"{custom_id}: expected exactly three candidates")
        candidate_by_style: dict[str, str] = {}
        for candidate in candidates:
            style = str(candidate.get("style"))
            if style not in STYLES or style in candidate_by_style:
                raise SystemExit(f"{custom_id}: invalid or duplicate style {style!r}")
            translation = str(candidate.get("translation") or "").strip()
            if not translation:
                raise SystemExit(f"{custom_id}: empty {style} candidate")
            candidate_by_style[style] = translation
            risk_tags.update(str(tag) for tag in candidate.get("risk_tags", []))
        if set(candidate_by_style) != set(STYLES):
            raise SystemExit(f"{custom_id}: incomplete candidate styles")
        direction = (
            "en-ja" if seed["source_language"] == "en-US" else "ja-en"
        )
        evaluated.append(
            {
                "custom_id": custom_id,
                "direction": direction,
                "domain": seed.get("domain", "unknown"),
                "reference": reference,
                "candidates": candidate_by_style,
            }
        )

    groups: dict[str, list[dict]] = {
        "all": evaluated,
        "en-ja": [row for row in evaluated if row["direction"] == "en-ja"],
        "ja-en": [row for row in evaluated if row["direction"] == "ja-en"],
    }
    scored = {
        name: score_group(group)
        for name, group in groups.items()
        if group
    }
    admission = None
    if args.review_queue:
        queue = rows(args.review_queue)
        admitted_ids = {str(row["source_id"]) for row in queue}
        admission = {
            "admitted_reference_sources": len(admitted_ids & seen),
            "teacher_result_sources": len(seen),
            "admission_rate": (
                len(admitted_ids & seen) / len(seen) if seen else 0.0
            ),
            "queue_rows": len(queue),
        }

    report = {
        "report_type": "codex-teacher-training-reference-diagnostic",
        "promotion_evaluation": False,
        "reference_leakage": {
            "teacher_input": False,
            "metric_computation": True,
            "oracle_selection": True,
        },
        "teacher_result_sources": len(seen),
        "licensed_reference_rows": len(evaluated),
        "directions": scored,
        "risk_tags": dict(sorted(risk_tags.items())),
        "deterministic_filter": admission,
        "metrics": {
            "chrf": "sacreBLEU chrF with word_order=2 (chrF++)",
            "bleu": "sacreBLEU BLEU with intl tokenization and effective order",
        },
    }
    exclusive_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
