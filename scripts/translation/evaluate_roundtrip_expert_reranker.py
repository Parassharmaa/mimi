#!/usr/bin/env python3
"""Calibrate and evaluate a reverse-consistency veto for routed experts."""

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
SPLIT_SALT = "mimi-roundtrip-expert-reranker-v1"
MARGINS = tuple(value / 2 for value in range(0, 61))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise SystemExit(f"invalid translation report: {path}")
    return value


def indexed(report: dict, label: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in report["results"]:
        identifier = str(row.get("caseID", "")).strip()
        if not identifier or identifier in output:
            raise SystemExit(f"{label} has an empty or duplicate case ID")
        output[identifier] = row
    if not output:
        raise SystemExit(f"{label} has no cases")
    return output


def direction(row: dict) -> str:
    return "en-ja" if row["sourceLanguage"] == "en-US" else "ja-en"


def split(case_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}\0{case_id}".encode()).digest()
    return "calibration" if digest[0] < 128 else "test"


def original_case_id(roundtrip_case_id: str, kind: str) -> str:
    prefix = f"roundtrip-{kind}:"
    if not roundtrip_case_id.startswith(prefix):
        raise SystemExit(f"unexpected {kind} roundtrip ID: {roundtrip_case_id}")
    return roundtrip_case_id[len(prefix) :]


def sentence_chrf(hypothesis: str, references: list[str]) -> float:
    return sacrebleu.sentence_chrf(hypothesis, references, word_order=2).score


def valid_forward(row: dict, hypothesis: str) -> bool:
    normalized_source = "".join(unicodedata.normalize("NFKC", row["source"]).casefold().split())
    normalized_hypothesis = "".join(
        unicodedata.normalize("NFKC", hypothesis).casefold().split()
    )
    if not normalized_hypothesis or normalized_hypothesis == normalized_source:
        return False
    target_language = row["targetLanguage"]
    if target_language == "ja-JP" and not JA_RE.search(hypothesis):
        return False
    if target_language == "en-US" and not EN_RE.search(hypothesis):
        return False
    ratio = len(normalized_hypothesis) / max(1, len(normalized_source))
    return 0.12 <= ratio <= 8.0 and typed_preserves(
        row["source"],
        hypothesis,
        row["sourceLanguage"],
        target_language,
    )


def choose(
    generalist_valid: bool,
    expert_valid: bool,
    generalist_cycle: float,
    expert_cycle: float,
    margin: float,
) -> str:
    if expert_valid and not generalist_valid:
        return "expert"
    if generalist_valid and not expert_valid:
        return "generalist"
    if not generalist_valid and not expert_valid:
        return "generalist"
    return "expert" if expert_cycle - generalist_cycle >= margin else "generalist"


def latency(row: dict) -> float:
    values = row.get("warmLatencySeconds") or [row.get("latencySeconds")]
    if not values or values[0] is None:
        raise SystemExit(f"missing latency evidence: {row.get('caseID')}")
    return float(values[0])


def mean(values: list[float]) -> float:
    if not values:
        raise SystemExit("empty calibration or evaluation slice")
    return sum(values) / len(values)


def write_report(path: Path, report: dict) -> None:
    if path.exists() and path.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generalist_report", type=Path)
    parser.add_argument("en_ja_expert_report", type=Path)
    parser.add_argument("ja_en_expert_report", type=Path)
    parser.add_argument("routed_report", type=Path)
    parser.add_argument("generalist_roundtrip_report", type=Path)
    parser.add_argument("expert_roundtrip_report", type=Path)
    parser.add_argument("output_full", type=Path)
    parser.add_argument("output_calibration", type=Path)
    parser.add_argument("output_test", type=Path)
    parser.add_argument("selection_report", type=Path)
    args = parser.parse_args()
    for output in (
        args.output_full,
        args.output_calibration,
        args.output_test,
        args.selection_report,
    ):
        if output.exists() and output.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {output}")

    paths = {
        "generalist": args.generalist_report,
        "en-ja-expert": args.en_ja_expert_report,
        "ja-en-expert": args.ja_en_expert_report,
        "routed": args.routed_report,
        "generalist-roundtrip": args.generalist_roundtrip_report,
        "expert-roundtrip": args.expert_roundtrip_report,
    }
    reports = {name: load(path) for name, path in paths.items()}
    rows_by_report = {name: indexed(report, name) for name, report in reports.items()}
    base_ids = set(rows_by_report["routed"])
    if any(
        set(rows_by_report[name]) != base_ids
        for name in ("generalist", "en-ja-expert", "ja-en-expert")
    ):
        raise SystemExit("forward reports do not have exact common coverage")

    roundtrips: dict[str, dict[str, dict]] = {"generalist": {}, "expert": {}}
    for kind in roundtrips:
        for row in reports[f"{kind}-roundtrip"]["results"]:
            case_id = original_case_id(str(row.get("caseID", "")), kind)
            if case_id in roundtrips[kind]:
                raise SystemExit(f"duplicate {kind} roundtrip case: {case_id}")
            roundtrips[kind][case_id] = row
    routed_expert_ids = {
        case_id
        for case_id, row in rows_by_report["routed"].items()
        if row.get("selectedEngine") == "expert"
    }
    if set(roundtrips["generalist"]) != routed_expert_ids or set(
        roundtrips["expert"]
    ) != routed_expert_ids:
        raise SystemExit("roundtrip reports do not cover the exact routed expert population")

    evidence: dict[str, dict] = {}
    for case_id in sorted(routed_expert_ids):
        routed = rows_by_report["routed"][case_id]
        generalist = rows_by_report["generalist"][case_id]
        expert = rows_by_report[f"{direction(routed)}-expert"][case_id]
        for candidate in (generalist, expert):
            for field in (
                "sourceLanguage",
                "targetLanguage",
                "domain",
                "source",
                "references",
                "claimEligible",
            ):
                if candidate.get(field) != routed.get(field):
                    raise SystemExit(f"forward candidate mismatch on {field}: {case_id}")
        if routed.get("hypothesis") != expert.get("hypothesis"):
            raise SystemExit(f"routed output does not equal expert output: {case_id}")
        generalist_back = roundtrips["generalist"][case_id]
        expert_back = roundtrips["expert"][case_id]
        if (
            generalist_back.get("source") != generalist.get("hypothesis")
            or expert_back.get("source") != expert.get("hypothesis")
            or generalist_back.get("references") != [routed["source"]]
            or expert_back.get("references") != [routed["source"]]
        ):
            raise SystemExit(f"roundtrip report is not bound to forward candidates: {case_id}")
        evidence[case_id] = {
            "direction": direction(routed),
            "split": split(case_id),
            "generalist": generalist,
            "expert": expert,
            "generalistValid": valid_forward(routed, str(generalist["hypothesis"])),
            "expertValid": valid_forward(routed, str(expert["hypothesis"])),
            "generalistCycle": sentence_chrf(
                str(generalist_back["hypothesis"]), [str(routed["source"])]
            ),
            "expertCycle": sentence_chrf(
                str(expert_back["hypothesis"]), [str(routed["source"])]
            ),
            "compositeWarmLatencySeconds": sum(
                latency(value)
                for value in (generalist, expert, generalist_back, expert_back)
            ),
        }

    margins: dict[str, float] = {}
    calibration_curves: dict[str, list[dict]] = {}
    for current_direction in ("en-ja", "ja-en"):
        cases = [
            (case_id, value)
            for case_id, value in evidence.items()
            if value["direction"] == current_direction and value["split"] == "calibration"
        ]
        curve: list[dict] = []
        for margin in MARGINS:
            scores = []
            expert_count = 0
            for _, value in cases:
                selection = choose(
                    value["generalistValid"],
                    value["expertValid"],
                    value["generalistCycle"],
                    value["expertCycle"],
                    margin,
                )
                expert_count += selection == "expert"
                scores.append(
                    sentence_chrf(
                        str(value[selection]["hypothesis"]),
                        list(value[selection]["references"]),
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
    for routed in reports["routed"]["results"]:
        case_id = routed["caseID"]
        row = copy.deepcopy(routed)
        row["roundtripRerankingSplit"] = split(case_id)
        if case_id in evidence:
            value = evidence[case_id]
            selected = choose(
                value["generalistValid"],
                value["expertValid"],
                value["generalistCycle"],
                value["expertCycle"],
                margins[value["direction"]],
            )
            candidate = value[selected]
            row["hypothesis"] = candidate["hypothesis"]
            row["outputTokenIDs"] = candidate.get("outputTokenIDs")
            row["selectedEngine"] = f"{selected}-roundtrip-reranker"
            row["roundtripReranking"] = {
                "generalistCycleChrFPlusPlus": value["generalistCycle"],
                "expertCycleChrFPlusPlus": value["expertCycle"],
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
        "createdAt": reports["routed"].get("createdAt"),
        "engine": "simulated:source-router-plus-roundtrip-expert-veto-v1",
        "modelRevision": f"roundtrip-evidence-sha256:{revision}",
        "hardware": reports["routed"].get("hardware"),
        "operatingSystem": reports["routed"].get("operatingSystem"),
        "preparationSeconds": 0.0,
        # Individual reports do not measure the live three-model residency that
        # this selector would require. Do not mislabel one-report RSS as an
        # end-to-end product measurement.
        "peakResidentBytes": None,
        "modelBytes": reports["routed"].get("modelBytes"),
        "physicalModelCount": reports["routed"].get("physicalModelCount"),
        "doesNotAuthorizeAppIntegration": True,
        "benchmarkConfiguration": {
            "claimEligible": False,
            "splitSalt": SPLIT_SALT,
            "calibrationRule": "SHA-256 first byte below 128",
            "candidatePopulation": "source-router expert cases only",
            "directionMargins": margins,
            "latencyAccounting": "sequential generalist forward plus expert forward plus two generalist reverse passes",
            "residentMemoryMeasurement": "not measured end-to-end for this simulated selector",
        },
        "runtimeImplementation": {
            "selectorSHA256": sha256(Path(__file__).resolve()),
            "forwardRuntime": reports["routed"].get("runtimeImplementation"),
            "generalistRoundtripRuntime": reports["generalist-roundtrip"].get(
                "runtimeImplementation"
            ),
            "expertRoundtripRuntime": reports["expert-roundtrip"].get("runtimeImplementation"),
        },
        "inputs": {
            name: {"path": str(path), "sha256": digest}
            for name, path, digest in (
                (name, paths[name], input_hashes[name]) for name in paths
            )
        },
        "results": output_rows,
    }
    calibration_report = {**base_report, "results": [
        row for row in output_rows if row["roundtripRerankingSplit"] == "calibration"
    ]}
    test_report = {**base_report, "results": [
        row for row in output_rows if row["roundtripRerankingSplit"] == "test"
    ]}
    write_report(args.output_full, base_report)
    write_report(args.output_calibration, calibration_report)
    write_report(args.output_test, test_report)

    selection = {
        "schemaVersion": 1,
        "status": "public-development-ablation-complete",
        "claimEligible": False,
        "policy": "source router first; exact typed safety; reverse chrF++ margin veto",
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
