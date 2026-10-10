#!/usr/bin/env python3
"""Fast tests for measurement boundaries, cohort validity, and failure accounting."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from compare import compare_rows, validate_cohort
from metrics import derive_metrics, edit_counts, quantile, removed_characters, units, validate_events
from report import statistics
from run import attempt_rows, execute_case, load_suite, validate_protocol
from integrity import seal_run, verify_run
from quality import paired_quality

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads((ROOT / "configs/benchmarks/protocol-v1.json").read_text())
PROTOCOL.update(repetitions=1, warmups=0, bootstrapReplicates=100)
CASE = {"caseID": "case", "groupID": "case", "language": "en-US", "source": "hello world",
        "references": ["こんにちは"], "condition": "clean", "domain": "test", "fixtureClass": "owned-synthetic"}


def event(identifier, kind, time, **fields):
    return {"schemaVersion": 1, "record": "event", "attemptIndex": 0, "phase": "cold",
            "eventID": identifier, "kind": kind, "atMs": time, **fields}


def result(**fields):
    return {"schemaVersion": 1, "record": "result", "attemptIndex": 0, "phase": "cold", "caseID": "case",
            "lane": "pipeline", "engine": "apple", "language": "en-US", "speechStatus": "success",
            "translationStatus": "success", "transcript": "hello world", "translation": "こんにちは",
            "wallMs": 500, "audioDurationSeconds": 0.2, **fields}


class MetricsTests(unittest.TestCase):
    def test_normalization_and_error_operations(self):
        self.assertEqual(units("HELLO， ４２!", "en-US"), ["hello", "42"])
        self.assertEqual(units("東京、４２。", "ja-JP"), list("東京42"))
        self.assertEqual(edit_counts(["a", "b", "c"], ["a", "x"]),
                         {"edits": 2, "substitutions": 1, "deletions": 1, "insertions": 0, "referenceUnits": 3, "rate": 2 / 3})
        self.assertIsNone(edit_counts([], ["noise"])["rate"])
        self.assertEqual(edit_counts([], ["noise"])["insertions"], 1)

    def test_appends_are_not_revision_churn(self):
        self.assertEqual(removed_characters("hello", "hello world"), 0)
        self.assertEqual(removed_characters("hello", "help"), 2)
        self.assertEqual(removed_characters("日本語", "日本"), 1)

    def test_one_clock_separates_prepare_queue_service_and_publication(self):
        events = [event(1, "speech_prepare_started", 0), event(2, "speech_ready", 100),
                  event(3, "audio_started", 110), event(4, "asr_partial", 150, text="hello", boundaryID=None),
                  event(5, "translation_started", 160, requestID="r1", requestSourceEventID=4, isPartial=True),
                  event(6, "translation_succeeded", 200, requestID="r1"),
                  event(7, "translation_published", 205, sourceEventID=4, isPartial=True, text="こんにちは"),
                  event(8, "asr_final", 290, text="hello world", boundaryID=None),
                  event(9, "audio_ended", 310), event(10, "speech_stopped", 330),
                  event(11, "translation_published", 340, sourceEventID=8, isPartial=False, text="こんにちは世界")]
        derived = derive_metrics(events, result(), {**CASE, "speechStartSeconds": 0.02})
        metrics = derived["metrics"]
        self.assertEqual(metrics["speechPreparationMs"], 100)
        self.assertEqual(metrics["audioStartToFirstTextMs"], 40)
        self.assertEqual(metrics["annotatedSpeechStartToFirstTextMs"], 20)
        self.assertEqual(metrics["audioStartToFirstTranslationMs"], 95)
        self.assertEqual(metrics["asrDrainMs"], 20)
        self.assertEqual(metrics["translationTailAfterAudioEndMs"], 30)
        self.assertEqual(derived["translationRequests"][0]["queueMs"], 10)
        self.assertEqual(derived["translationRequests"][0]["serviceMs"], 40)
        self.assertEqual(derived["publications"][0]["sourceToPublicationMs"], 55)
        self.assertIsNone(metrics["runtimeServiceRTF"])

    def test_missing_events_are_null_not_fast_zero(self):
        derived = derive_metrics([], result(), CASE)
        for key in ("audioStartToFirstTextMs", "audioStartToFirstTranslationMs", "translationPreparationMs", "droppedAudioSeconds", "translationRequestCount"):
            self.assertIsNone(derived["metrics"][key])

    def test_malformed_time_and_correlation_fail_closed(self):
        for events in ([event(1, "x", -1)], [event(1, "x", float("nan"))],
                       [event(1, "x", 10), event(2, "x", 9)],
                       [event(1, "x", 1), event(1, "x", 2)],
                       [event(1, "translation_started", 1, sourceEventID=99)]):
            with self.assertRaises(ValueError): validate_events(events)

    def test_repeats_do_not_create_independent_tail_samples(self):
        rows = [{"groupID": "one", "metrics": {"latency": n}} for n in range(100)]
        summary = statistics(rows, "latency", PROTOCOL)
        self.assertEqual(summary["observationCount"], 100)
        self.assertEqual(summary["independentUnitCount"], 1)
        self.assertIsNone(summary["p95"])
        self.assertIsNone(summary["p50ClusterBootstrapCI95"])
        self.assertAlmostEqual(quantile([0, 10, 20, 30], .95), 28.5)

    def test_failure_is_not_removed_from_planned_attempts(self):
        records = [event(1, "attempt_started", 0)]
        rows = attempt_rows(records, {"timedOutAttempt": 0, "returnCode": -15}, CASE, "pipeline", "apple", PROTOCOL)
        self.assertEqual([row["outcome"] for row in rows], ["timeout", "not_run"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["deadlineCappedAttemptWallMs"], 120000)
        self.assertIsNone(rows[0]["metrics"]["audioStartToFirstTextMs"])

    def test_crash_before_first_event_is_counted(self):
        rows = attempt_rows([], {"timedOutAttempt": None, "returnCode": 7}, CASE, "pipeline", "apple", PROTOCOL)
        self.assertEqual(rows[0]["outcome"], "error")
        self.assertTrue(rows[0]["observedAttempt"])
        self.assertEqual(rows[1]["outcome"], "not_run")

    def test_native_result_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            attempt_rows([result(caseID="other")], {"returnCode": 0}, CASE, "pipeline", "apple", PROTOCOL)

    def test_timeout_preserves_partial_utf8_trace_and_kills_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = root / "fake-app"
            app.write_text("#!" + sys.executable + "\n" + '''import json,sys,time
from pathlib import Path
trace=Path(sys.argv[sys.argv.index("--trace-output")+1])
with trace.open("wb") as output:
    output.write((json.dumps({"schemaVersion":1,"record":"event","attemptIndex":0,"phase":"cold","eventID":1,"kind":"attempt_started","atMs":0})+"\\n").encode()); output.flush()
    output.write(b'{"partial":"'+"日".encode()[:1]); output.flush()
    time.sleep(10)
''')
            app.chmod(0o755)
            records, process = execute_case(app, {"engine": "none", "speechModelPath": None}, root / "case", 1.0)
            self.assertEqual(process["timedOutAttempt"], 0)
            self.assertTrue(process["truncatedTraceTail"], (root / "case/stderr.log").read_text())
            self.assertEqual(len(records), 1)
            self.assertLess(process["processWallMs"], 3000)

    def test_invalid_protocol_cannot_mislabel_the_algorithm(self):
        validate_protocol(PROTOCOL)
        for key, value in (("repetitions", 0), ("chunkSeconds", float("nan")), ("parallelism", 2), ("quantileMethod", "nearest"), ("confidenceLevel", 1)):
            changed = copy.deepcopy(PROTOCOL); changed[key] = value
            with self.assertRaises(ValueError): validate_protocol(changed)

    def test_audio_corruption_is_rejected_before_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "audio.wav").write_bytes(b"changed")
            suite = root / "manifest.jsonl"
            suite.write_text(json.dumps({**CASE, "claimEligible": False, "audio": "audio.wav", "audioSha256": "wrong", "audioDurationSeconds": 1}))
            with self.assertRaises(ValueError): load_suite(suite)

    def test_receipt_detects_edited_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata.json").write_text("{}")
            (root / "attempts.jsonl").write_text("{}\n")
            seal_run(root)
            verify_run(root)
            (root / "attempts.jsonl").write_text('{"outcome":"success"}\n')
            with self.assertRaises(ValueError): verify_run(root)

    def test_quality_uses_paired_scored_rows_and_keeps_coverage(self):
        def row(case, edits):
            return {"caseID": case, "attemptIndex": 1, "lane": "speech", "speechQuality":
                    {"edits": edits, "referenceUnits": 10} if edits is not None else None}
        quality = paired_quality([row("a", 2), row("b", 6)], [row("a", 1), row("b", None)])
        self.assertEqual(quality["plannedPairedAttempts"], 2)
        self.assertEqual(quality["speech"]["pairedAttempts"], 1)
        self.assertAlmostEqual(quality["speech"]["delta"], -0.1)

    def test_unplanned_trace_attempt_is_rejected(self):
        with self.assertRaises(ValueError):
            attempt_rows([event(1, "attempt_started", 0, attemptIndex=99)], {"returnCode": 0}, CASE, "speech", "apple", PROTOCOL)

    def test_comparison_rejects_changed_cohorts(self):
        metadata = {"protocolSha256": "p", "suiteSha256": "s", "evaluatorFiles": {"metrics.py": "m"},
                    "appEnvironment": {"hardware": "Mac", "buildConfiguration": "release"}, "performanceEligible": True,
                    "plan": [{"caseID": "c", "caseSha256": "c", "lane": "speech", "engine": "apple"}], "models": {"translation": "m"}}
        validate_cohort(metadata, copy.deepcopy(metadata))
        for key in ("protocolSha256", "suiteSha256", "models", "evaluatorFiles"):
            other = copy.deepcopy(metadata); other[key] = "changed"
            with self.assertRaises(ValueError): validate_cohort(metadata, other)
        other = copy.deepcopy(metadata); other["appEnvironment"]["hardware"] = "other"
        with self.assertRaises(ValueError): validate_cohort(metadata, other)

    def test_paired_failure_cannot_look_like_speed_improvement(self):
        def row(outcome, latency, repetition):
            return {"groupID": "unit", "caseID": "case", "lane": "speech", "engine": "apple", "language": "en-US",
                    "condition": "clean", "phase": "warm", "attemptIndex": repetition, "outcome": outcome,
                    "metrics": {"audioStartToFirstTextMs": latency, "droppedAudioSeconds": 0}}
        compared = compare_rows([row("success", 100, 1), row("success", 110, 2)],
                                [row("error", 1, 1), row("success", 10, 2)], PROTOCOL)[0]
        self.assertEqual(compared["pairedOutcomeUnits"]["baselineOnly"], 1)
        self.assertTrue(compared["reliabilityRegression"])
        self.assertIsNone(compared["meanUnitMedianDelta"])


if __name__ == "__main__": unittest.main()
