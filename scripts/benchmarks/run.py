#!/usr/bin/env python3
"""Run a frozen benchmark matrix serially, retaining every attempt and timeout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from metrics import derive_metrics, edit_counts, units
from prepare_suite import sha256
from integrity import seal_run

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = {"apple": {"en-US", "ja-JP"}, "mimi-whisper": {"en-US", "ja-JP"}, "phonon": {"en-US"}, "parakeet": {"ja-JP"}}


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def fingerprint(root: Path | None) -> dict | None:
    if root is None or not root.is_dir(): return None
    files = {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(root.rglob("*")) if path.is_file()}
    if not files: return None
    return {"sha256": json_hash(files), "bytes": sum(item["bytes"] for item in files.values()), "files": files}


def read_json_output(command: list[str], timeout: float = 30) -> dict:
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    for line in result.stdout.splitlines():
        if line.startswith("{"):
            try: return json.loads(line)
            except json.JSONDecodeError: continue
    raise ValueError("Command produced no JSON object")


def load_suite(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["caseID"] for row in rows}) != len(rows): raise ValueError("Suite must have unique case IDs")
    for row in rows:
        if row.get("claimEligible") is not False: raise ValueError("This runner accepts non-claimable development fixtures only")
        if row["language"] not in {"en-US", "ja-JP"}: raise ValueError("Unsupported fixture language")
        if not row.get("groupID") or not row.get("condition") or not row.get("fixtureClass"):
            raise ValueError("Fixture grouping and provenance are required")
        if not isinstance(row["source"], str) or not isinstance(row["references"], list): raise ValueError("Invalid reference fields")
        if any(not isinstance(reference, str) or not reference.strip() for reference in row["references"]): raise ValueError("Invalid translation reference")
        if row["condition"] != "silence" and not row["source"]: raise ValueError("Non-silence cases need source references")
        if "translation" in row.get("lanes", []) and not row["references"]: raise ValueError("Text quality cases need translation references")
        if row.get("audio"):
            audio = (path.parent / row["audio"]).resolve()
            if not audio.is_relative_to(path.parent.resolve()): raise ValueError("Audio must stay inside the frozen suite directory")
            if sha256(audio) != row["audioSha256"]: raise ValueError("Audio hash mismatch: " + row["caseID"])
            duration = row.get("audioDurationSeconds")
            if not isinstance(duration, (int, float)) or not 0 < duration <= 7200: raise ValueError("Invalid audio duration")
            for key in ("speechStartSeconds", "speechEndSeconds"):
                value = row.get(key)
                if value is not None and not 0 <= value <= duration: raise ValueError("Speech annotation is outside the audio")
            if row.get("speechStartSeconds") is not None and row.get("speechEndSeconds") is not None and row["speechEndSeconds"] < row["speechStartSeconds"]:
                raise ValueError("Speech end precedes speech start")
    return rows


def validate_protocol(protocol: dict) -> None:
    integer_bounds = {"repetitions": (1, 20), "warmups": (0, 5), "tailMinimumIndependentUnits": (20, 10000), "bootstrapReplicates": (100, 100000)}
    for key, (low, high) in integer_bounds.items():
        value = protocol.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError("Invalid protocol field: " + key)
    for key, low, high in (("chunkSeconds", .01, 1), ("translationDrainSeconds", .1, 60), ("attemptTimeoutSeconds", .1, 1800), ("confidenceLevel", .5, .999)):
        value = protocol.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Invalid protocol field: " + key)
    if protocol.get("confidenceLevel") != 0.95:
        raise ValueError("Protocol v1 reports 95% confidence intervals")
    if protocol.get("schemaVersion") != 1 or protocol.get("parallelism") != 1:
        raise ValueError("Only schema v1 with serial execution is supported")
    if protocol.get("normalization") != "NFKC-casefold-punctuation-symbols-whitespace-v1" or protocol.get("quantileMethod") != "linear-interpolation-type-7":
        raise ValueError("Unsupported metric definition")
    if protocol.get("primaryLatency") != {"speech": "audioStartToFirstTextMs", "translation": "referenceSubmitToFinalTranslationMs", "pipeline": "audioStartToFirstTranslationMs"}:
        raise ValueError("Unsupported primary latency endpoints")
    if not isinstance(protocol.get("seed"), int): raise ValueError("Seed must be an integer")


def phases(protocol: dict) -> list[str]:
    return ["cold"] + ["warmup"] * protocol["warmups"] + ["warm"] * protocol["repetitions"]


def execute_case(app: Path, request: dict, directory: Path, timeout: float) -> tuple[list[dict], dict]:
    directory.mkdir()
    request_path = directory / "request.json"
    trace_path = directory / "trace.jsonl"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    environment = os.environ.copy()
    if request["engine"] == "mimi-whisper" and request["speechModelPath"]:
        environment["MIMI_WHISPER_MLX_MODEL_DIR"] = request["speechModelPath"]
    started = time.monotonic()
    records = []
    offset = 0
    pending = b""
    active_attempt = None
    deadline = started + timeout
    timed_out = None
    trace_error = None
    def read_new():
        nonlocal offset, pending, active_attempt, deadline
        if not trace_path.exists(): return
        with trace_path.open("rb") as source:
            source.seek(offset)
            pending += source.read()
            offset = source.tell()
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            if not line: continue
            record = json.loads(line)
            if not isinstance(record, dict): raise ValueError("Trace records must be JSON objects")
            records.append(record)
            if record.get("kind") == "attempt_started":
                active_attempt = record["attemptIndex"]
                deadline = time.monotonic() + timeout
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        process = subprocess.Popen([str(app), "--benchmark-case", str(request_path), "--trace-output", str(trace_path)],
                                   stdout=stdout, stderr=stderr, env=environment, start_new_session=True)
        try:
            while process.poll() is None:
                read_new()
                if time.monotonic() >= deadline:
                    timed_out = active_attempt if active_attempt is not None else 0
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                time.sleep(0.05)
        except (ValueError, UnicodeError) as error:
            trace_error = str(error)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if trace_error is None:
            try: read_new()
            except (ValueError, UnicodeError) as error: trace_error = str(error)
    return records, {"returnCode": process.returncode, "timedOutAttempt": timed_out,
                     "processWallMs": (time.monotonic() - started) * 1000,
                     "truncatedTraceTail": bool(pending), "traceError": trace_error, "deadlineSeconds": timeout}


def attempt_rows(records: list[dict], process: dict, case: dict, lane: str, engine: str, protocol: dict) -> list[dict]:
    events = defaultdict(list)
    results = {}
    for record in records:
        index = record.get("attemptIndex")
        if record.get("schemaVersion") != 1 or not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(phases(protocol)):
            raise ValueError("Invalid trace schema or unplanned attempt index")
        if record.get("record") == "event": events[index].append(record)
        elif record.get("record") == "result":
            if index in results: raise ValueError("Duplicate result for attempt")
            results[index] = record
    rows = []
    for index, phase in enumerate(phases(protocol)):
        result = results.get(index)
        if result is not None:
            if result["caseID"] != case["caseID"] or result["phase"] != phase or result["lane"] != lane or result["engine"] != engine:
                raise ValueError("Native result does not match the planned attempt")
            relevant = [result["speechStatus"]] if lane == "speech" else [result["translationStatus"]]
            if lane == "pipeline": relevant.append(result["speechStatus"])
            outcome = "success" if all(status == "success" for status in relevant) else ("timeout" if "timeout" in relevant else "error")
        else:
            launched = index == 0 and process.get("returnCode") is not None
            outcome = "timeout" if process.get("timedOutAttempt") == index else ("error" if events[index] or launched else "not_run")
            result = {"speechStatus": outcome if lane != "translation" else "not_applicable",
                      "translationStatus": outcome if lane != "speech" else "not_applicable",
                      "transcript": "", "translation": "", "wallMs": None, "lane": lane}
        derived = derive_metrics(events[index], result, case)
        speech_quality = None
        if lane != "translation" and result["speechStatus"] in {"success", "empty_output", "hallucination"}:
            speech_quality = edit_counts(units(case["source"], case["language"]), units(result["transcript"], case["language"]))
        row = {"schemaVersion": 1, "caseID": case["caseID"], "groupID": case["groupID"], "language": case["language"],
               "condition": case["condition"], "fixtureClass": case["fixtureClass"], "domain": case["domain"],
               "caseSha256": json_hash(case), "lane": lane, "engine": engine, "phase": phase, "attemptIndex": index,
               "outcome": outcome, "observedAttempt": bool(events[index]) or (index == 0 and process.get("returnCode") is not None), "deadlineSeconds": protocol["attemptTimeoutSeconds"],
               "source": case["source"], "references": case["references"], "speechQuality": speech_quality,
               "result": result, **derived}
        row["deadlineCappedAttemptWallMs"] = result["wallMs"] if outcome == "success" else (
            protocol["attemptTimeoutSeconds"] * 1000 if events[index] else None)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--app", type=Path, default=ROOT / ".build/release/Mimi")
    parser.add_argument("--build-receipt", type=Path, default=ROOT / ".build/benchmark-build-receipt.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/benchmarks/protocol-v1.json")
    parser.add_argument("--translation-model", type=Path, default=ROOT / "App/Resources/TranslationModels")
    parser.add_argument("--whisper-model", type=Path)
    parser.add_argument("--phonon-model", type=Path)
    parser.add_argument("--parakeet-model", type=Path)
    parser.add_argument("--engines", nargs="+", choices=sorted(LANGUAGES), default=list(LANGUAGES))
    parser.add_argument("--lanes", nargs="+", choices=["speech", "translation", "pipeline"], default=["speech", "translation", "pipeline"])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--allow-debug", action="store_true")
    args = parser.parse_args()
    if len(set(args.engines)) != len(args.engines) or len(set(args.lanes)) != len(args.lanes):
        parser.error("Duplicate engines or lanes would duplicate planned attempts")
    protocol = json.loads(args.protocol.read_text())
    for key in ("repetitions", "warmups"):
        if getattr(args, key) is not None: protocol[key] = getattr(args, key)
    try: validate_protocol(protocol)
    except ValueError as error: parser.error(str(error))
    app = args.app.resolve(strict=True)
    environment = read_json_output([str(app), "--benchmark-environment"])
    sys.path.insert(0, str(ROOT / "scripts/speech"))
    from evidence_identity import adaptive_segmentation_implementation_sha256
    source_identity = adaptive_segmentation_implementation_sha256()
    source_matches = source_identity == environment["sourceIdentity"]
    if not source_matches and not args.allow_debug:
        parser.error("Executable source identity is stale; stamp and rebuild before recording a baseline")
    if environment["buildConfiguration"] != "release" and not args.allow_debug:
        parser.error("Use a release build for measurement; --allow-debug is for harness diagnostics only")
    build_receipt = None
    if not args.allow_debug:
        build_receipt = json.loads(args.build_receipt.read_text())
        if build_receipt["appSha256"] != sha256(app) or build_receipt["environment"]["sourceIdentity"] != environment["sourceIdentity"]:
            parser.error("Build receipt does not match the executable")
        if build_receipt["shaderSha256"] != sha256(app.parent / "mlx.metallib"):
            parser.error("Metal shader differs from the recorded build")
    suite = load_suite(args.suite)
    if args.case:
        unknown = set(args.case) - {case["caseID"] for case in suite}
        if unknown: parser.error("Unknown case IDs: " + str(sorted(unknown)))
        suite = [case for case in suite if case["caseID"] in args.case]
    if args.condition:
        suite = [case for case in suite if case["condition"] in args.condition]
    model_paths = {"mimi-whisper": args.whisper_model, "phonon": args.phonon_model, "parakeet": args.parakeet_model}
    fingerprints = {engine: fingerprint(path) for engine, path in model_paths.items()}
    fingerprints["translation"] = fingerprint(args.translation_model)
    plan = []
    unsupported = []
    for case in suite:
        for lane in args.lanes:
            if lane not in case["lanes"]: continue
            for engine in (["mimi-translation"] if lane == "translation" else args.engines):
                if engine != "mimi-translation" and case["language"] not in LANGUAGES[engine]:
                    unsupported.append({"caseID": case["caseID"], "lane": lane, "engine": engine, "reason": "unsupported_language"})
                else: plan.append((case, lane, engine))
    if not plan: parser.error("No cases match the requested lanes")
    random.Random(protocol["seed"]).shuffle(plan)
    used_engines = {engine for _, lane, engine in plan if lane != "translation"}
    needs_translation = any(lane != "speech" for _, lane, _ in plan)
    fingerprints = {engine: value for engine, value in fingerprints.items()
                    if engine in used_engines or (engine == "translation" and needs_translation)}
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.suite, args.output / "suite.jsonl")
    metadata = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                "protocolSha256": json_hash(protocol), "suiteSha256": sha256(args.suite), "appSha256": sha256(app),
                "appEnvironment": environment, "buildReceipt": build_receipt, "python": platform.python_version(), "unicodeDataVersion": unicodedata.unidata_version, "models": fingerprints,
                "packageResolvedSha256": sha256(ROOT / "Package.resolved"),
                "systemSpeechAssetRevision": None,
                "gitCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "workingTreePatchSha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
                "sourceIdentity": environment["sourceIdentity"],
                "evaluatorFiles": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
                "performanceEligible": environment["buildConfiguration"] == "release" and source_matches and not args.allow_debug, "sourceMatchesCheckout": source_matches, "claimEligible": False,
                "unsupportedCombinations": unsupported,
                "plan": [{"caseID": case["caseID"], "caseSha256": json_hash(case), "lane": lane, "engine": engine} for case, lane, engine in plan]}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    with (args.output / "attempts.jsonl").open("w") as output:
        for number, (case, lane, engine) in enumerate(plan):
            directory = args.output / f"case-{number:04d}"
            missing_model = (engine in fingerprints and fingerprints[engine] is None) or (lane != "speech" and fingerprints["translation"] is None)
            if missing_model:
                records, process = [], {"returnCode": None, "timedOutAttempt": None, "reason": "model_unavailable"}
                directory.mkdir()
            else:
                speech_root = model_paths.get(engine)
                request = {"caseID": case["caseID"], "lane": lane, "engine": engine, "language": case["language"], "source": case["source"],
                           "audioPath": str((args.suite.parent / case["audio"]).resolve()) if case.get("audio") else None,
                           "speechModelPath": str(speech_root.resolve()) if speech_root else None,
                           "translationModelPath": str(args.translation_model.resolve()) if lane != "speech" else None,
                           **{key: protocol[key] for key in ("repetitions", "warmups", "chunkSeconds", "translationDrainSeconds")}}
                records, process = execute_case(app, request, directory, protocol["attemptTimeoutSeconds"])
            try:
                if process.get("traceError"): raise ValueError(process["traceError"])
                rows = attempt_rows(records, process, case, lane, engine, protocol)
            except (ValueError, KeyError, TypeError) as error:
                process["reportError"] = str(error)
                rows = attempt_rows([], {"returnCode": 1}, case, lane, engine, protocol)
                rows[0]["outcome"] = "invalid_report"
            (directory / "process.json").write_text(json.dumps(process, indent=2) + "\n")
            for row in rows:
                row["artifactDirectory"] = directory.name
                row["unavailableReason"] = process.get("reason")
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            print(f"[{number + 1}/{len(plan)}] {case['caseID']} {lane}/{engine}: " + ", ".join(row["outcome"] for row in rows), flush=True)
    seal_run(args.output)
    print(args.output)


if __name__ == "__main__": main()
