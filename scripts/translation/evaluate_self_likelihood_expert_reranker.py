#!/usr/bin/env python3
"""Calibrate and evaluate a self-likelihood veto for source-routed experts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import sacrebleu

from typed_critical_token_policy import typed_preserves


JA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
EN_RE = re.compile(r"[A-Za-z]")
SPLIT_SALT = "mimi-self-likelihood-expert-reranker-v1"
MARGINS = tuple(value / 20 for value in range(-40, 41))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_report(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise SystemExit(f"invalid report: {path}")
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


def direction(row: dict) -> str:
    pair = (row.get("sourceLanguage"), row.get("targetLanguage"))
    if pair == ("en-US", "ja-JP"):
        return "en-ja"
    if pair == ("ja-JP", "en-US"):
        return "ja-en"
    raise SystemExit(f"unsupported direction for {row.get('caseID')}: {pair}")


def split(case_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}\0{case_id}".encode()).digest()
    return "calibration" if digest[0] < 128 else "test"


def sentence_chrf(hypothesis: str, references: list[str]) -> float:
    return sacrebleu.sentence_chrf(hypothesis, references, word_order=2).score


def valid_forward(row: dict, hypothesis: str) -> bool:
    normalized_source = "".join(unicodedata.normalize("NFKC", row["source"]).casefold().split())
    normalized_hypothesis = "".join(
        unicodedata.normalize("NFKC", hypothesis).casefold().split()
    )
    if not normalized_hypothesis or normalized_hypothesis == normalized_source:
        return False
    if row["targetLanguage"] == "ja-JP" and not JA_RE.search(hypothesis):
        return False
    if row["targetLanguage"] == "en-US" and not EN_RE.search(hypothesis):
        return False
    ratio = len(normalized_hypothesis) / max(1, len(normalized_source))
    return 0.12 <= ratio <= 8.0 and typed_preserves(
        row["source"],
        hypothesis,
        row["sourceLanguage"],
        row["targetLanguage"],
    )


def choose(
    generalist_valid: bool,
    expert_valid: bool,
    generalist_nll: float,
    expert_nll: float,
    margin: float,
) -> str:
    if expert_valid and not generalist_valid:
        return "expert"
    if generalist_valid and not expert_valid:
        return "generalist"
    if not generalist_valid and not expert_valid:
        return "generalist"
    advantage = generalist_nll - expert_nll
    return "expert" if advantage >= margin else "generalist"


def latency(row: dict) -> float:
    values = row.get("warmLatencySeconds") or [row.get("latencySeconds")]
    if not values or values[0] is None:
        raise SystemExit(f"missing latency evidence: {row.get('caseID')}")
    return float(values[0])


def mean(values: list[float]) -> float:
    if not values:
        raise SystemExit("empty calibration slice")
    return sum(values) / len(values)


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


def write_report(path: Path, value: dict) -> None:
    if path.exists() and path.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generalist_report", type=Path)
    parser.add_argument("en_ja_expert_report", type=Path)
    parser.add_argument("ja_en_expert_report", type=Path)
    parser.add_argument("routed_report", type=Path)
    parser.add_argument("self_likelihood_report", type=Path)
    parser.add_argument("output_full", type=Path)
    parser.add_argument("output_calibration", type=Path)
    parser.add_argument("output_test", type=Path)
    parser.add_argument("selection_report", type=Path)
    args = parser.parse_args()
    outputs = (
        args.output_full,
        args.output_calibration,
        args.output_test,
        args.selection_report,
    )
    for output in outputs:
        if output.exists() and output.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {output}")

    paths = {
        "generalistReport": args.generalist_report,
        "enJAExpertReport": args.en_ja_expert_report,
        "jaENExpertReport": args.ja_en_expert_report,
        "routedReport": args.routed_report,
        "selfLikelihoodReport": args.self_likelihood_report,
    }
    reports = {name: load_report(path) for name, path in paths.items()}
    rows = {name: indexed(report, name) for name, report in reports.items()}
    base_ids = set(rows["routedReport"])
    if any(
        set(rows[name]) != base_ids
        for name in ("generalistReport", "enJAExpertReport", "jaENExpertReport")
    ):
        raise SystemExit("forward reports do not have exact common coverage")

    likelihood_inputs = reports["selfLikelihoodReport"].get("inputs", {})
    for name in ("generalistReport", "enJAExpertReport", "jaENExpertReport", "routedReport"):
        if likelihood_inputs.get(name, {}).get("sha256") != sha256(paths[name]):
            raise SystemExit(f"self-likelihood report is not bound to {name}")

    routed_expert_ids = {
        case_id
        for case_id, row in rows["routedReport"].items()
        if row.get("selectedEngine") == "expert"
    }
    if set(rows["selfLikelihoodReport"]) != routed_expert_ids:
        raise SystemExit("self-likelihood report does not cover the exact routed expert population")

    evidence: dict[str, dict] = {}
    for case_id in sorted(routed_expert_ids):
        routed = rows["routedReport"][case_id]
        current_direction = direction(routed)
        generalist = rows["generalistReport"][case_id]
        expert_label = "enJAExpertReport" if current_direction == "en-ja" else "jaENExpertReport"
        expert = rows[expert_label][case_id]
        assert_same_case(routed, generalist, case_id)
        assert_same_case(routed, expert, case_id)
        if routed.get("hypothesis") != expert.get("hypothesis"):
            raise SystemExit(f"routed output does not equal expert output: {case_id}")
        diagnostic = rows["selfLikelihoodReport"][case_id]
        if diagnostic.get("direction") != current_direction:
            raise SystemExit(f"self-likelihood direction mismatch: {case_id}")
        try:
            generalist_nll = float(diagnostic["generalist"]["meanChosenTokenNLL"])
            expert_nll = float(diagnostic["expert"]["meanChosenTokenNLL"])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"invalid self-likelihood evidence: {case_id}") from error
        if not (0 <= generalist_nll < 100 and 0 <= expert_nll < 100):
            raise SystemExit(f"out-of-range self-likelihood evidence: {case_id}")
        evidence[case_id] = {
            "direction": current_direction,
            "split": split(case_id),
            "generalist": generalist,
            "expert": expert,
            "generalistValid": valid_forward(routed, str(generalist["hypothesis"])),
            "expertValid": valid_forward(routed, str(expert["hypothesis"])),
            "generalistNLL": generalist_nll,
            "expertNLL": expert_nll,
            "compositeWarmLatencySeconds": latency(generalist) + latency(expert),
        }

    margins: dict[str, float] = {}
    calibration_curves: dict[str, list[dict]] = {}
    for current_direction in ("en-ja", "ja-en"):
        cases = [
            value
            for value in evidence.values()
            if value["direction"] == current_direction and value["split"] == "calibration"
        ]
        curve: list[dict] = []
        for margin in MARGINS:
            scores: list[float] = []
            expert_count = 0
            for value in cases:
                selected = choose(
                    value["generalistValid"],
                    value["expertValid"],
                    value["generalistNLL"],
                    value["expertNLL"],
                    margin,
                )
                expert_count += selected == "expert"
                scores.append(
                    sentence_chrf(
                        str(value[selected]["hypothesis"]),
                        list(value[selected]["references"]),
                    )
                )
            curve.append(
                {
                    "margin": margin,
                    "meanSentenceChrFPlusPlus": mean(scores),
                    "selectedExperts": expert_count,
                    "cases": len(cases),
                }
            )
        selected = max(
            curve,
            key=lambda value: (value["meanSentenceChrFPlusPlus"], value["margin"]),
        )
        margins[current_direction] = float(selected["margin"])
        calibration_curves[current_direction] = curve

    output_rows: list[dict] = []
    selection_counts: Counter[str] = Counter()
    for routed in reports["routedReport"]["results"]:
        case_id = routed["caseID"]
        row = copy.deepcopy(routed)
        row["selfLikelihoodRerankingSplit"] = split(case_id)
        if case_id in evidence:
            value = evidence[case_id]
            selected = choose(
                value["generalistValid"],
                value["expertValid"],
                value["generalistNLL"],
                value["expertNLL"],
                margins[value["direction"]],
            )
            candidate = value[selected]
            row["hypothesis"] = candidate["hypothesis"]
            row["outputTokenIDs"] = candidate.get("outputTokenIDs")
            row["selectedEngine"] = f"{selected}-self-likelihood-reranker"
            row["selfLikelihoodReranking"] = {
                "generalistMeanChosenTokenNLL": value["generalistNLL"],
                "expertMeanChosenTokenNLL": value["expertNLL"],
                "expertNLLAdvantage": value["generalistNLL"] - value["expertNLL"],
                "generalistTypedValid": value["generalistValid"],
                "expertTypedValid": value["expertValid"],
                "expertMargin": margins[value["direction"]],
            }
            composite = value["compositeWarmLatencySeconds"]
            row["latencySeconds"] = composite
            row["warmLatencySeconds"] = [composite]
            selection_counts[f"{value['direction']}:{selected}"] += 1
        else:
            selection_counts[f"{direction(row)}:source-router-{row.get('selectedEngine')}"] += 1
        output_rows.append(row)

    input_hashes = {name: sha256(path) for name, path in paths.items()}
    revision = hashlib.sha256(
        json.dumps(input_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    base_report = {
        "schemaVersion": 1,
        "createdAt": reports["routedReport"].get("createdAt"),
        "engine": "simulated:source-router-plus-self-likelihood-expert-veto-v1",
        "modelRevision": f"self-likelihood-evidence-sha256:{revision}",
        "hardware": reports["routedReport"].get("hardware"),
        "operatingSystem": reports["routedReport"].get("operatingSystem"),
        "preparationSeconds": 0.0,
        "peakResidentBytes": None,
        "modelBytes": reports["routedReport"].get("modelBytes"),
        "physicalModelCount": reports["routedReport"].get("physicalModelCount"),
        "doesNotAuthorizeAppIntegration": True,
        "benchmarkConfiguration": {
            "claimEligible": False,
            "splitSalt": SPLIT_SALT,
            "calibrationRule": "SHA-256 first byte below 128",
            "candidatePopulation": "source-router expert cases only",
            "directionMargins": margins,
            "latencyAccounting": "sequential generalist forward plus expert forward; generation NLL retained from chosen logits",
            "residentMemoryMeasurement": "not measured end-to-end for this simulated selector",
        },
        "runtimeImplementation": {
            "selectorSHA256": sha256(Path(__file__).resolve()),
            "likelihoodRuntime": reports["selfLikelihoodReport"].get("runtime"),
            "forwardRuntime": reports["routedReport"].get("runtimeImplementation"),
        },
        "inputs": {
            name: {"path": str(path), "sha256": digest}
            for name, path, digest in (
                (name, paths[name], input_hashes[name]) for name in paths
            )
        },
        "results": output_rows,
    }
    calibration_report = {
        **base_report,
        "results": [
            row for row in output_rows if row["selfLikelihoodRerankingSplit"] == "calibration"
        ],
    }
    test_report = {
        **base_report,
        "results": [
            row for row in output_rows if row["selfLikelihoodRerankingSplit"] == "test"
        ],
    }
    write_report(args.output_full, base_report)
    write_report(args.output_calibration, calibration_report)
    write_report(args.output_test, test_report)

    selection = {
        "schemaVersion": 1,
        "status": "public-development-ablation-complete",
        "claimEligible": False,
        "policy": "source router first; exact typed safety; relative mean chosen-token NLL expert veto",
        "splitSalt": SPLIT_SALT,
        "margins": margins,
        "selectionCounts": dict(sorted(selection_counts.items())),
        "calibrationCurves": calibration_curves,
        "inputs": base_report["inputs"],
        "outputs": {
            "full": {"path": str(args.output_full), "sha256": sha256(args.output_full)},
            "calibration": {
                "path": str(args.output_calibration),
                "sha256": sha256(args.output_calibration),
            },
            "test": {"path": str(args.output_test), "sha256": sha256(args.output_test)},
        },
    }
    write_report(args.selection_report, selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
