"""Versioned scoring and event-derived latency. No model execution or file mutation."""
from __future__ import annotations

import math
import unicodedata
from collections import Counter

NORMALIZATION = "NFKC-casefold-punctuation-symbols-whitespace-v1"
LATENCY_METRICS = (
    "attemptStartToFirstTextMs", "attemptStartToFirstTranslationMs",
    "speechPreparationMs", "translationPreparationMs", "audioStartToFirstTextMs",
    "audioStartToFirstFinalMs", "audioStartToFirstTranslationMs", "audioStartToFinalTranslationMs",
    "annotatedSpeechEndToFinalTextMs", "annotatedSpeechEndToFinalTranslationMs", "acousticActivityEndToFinalTextMs",
    "annotatedSpeechStartToFirstTextMs", "annotatedSpeechStartToFirstTranslationMs",
    "acousticActivityStartToFirstTextMs", "asrDrainMs", "translationTailAfterAudioEndMs",
    "referenceSubmitToFinalTranslationMs", "maximumInputLatenessMs", "processPeakResidentBytes",
    "maximumQueueSeconds", "droppedAudioSeconds", "runtimeServiceRTF", "pacedWallRTF",
    "asrRemovedCharacters", "asrRevisionChurn", "translationRemovedCharacters", "translationRevisionChurn",
    "translationRequestCount", "translationFailureCount", "translationReuseCount",
)


def units(text: str, language: str, *, characters: bool = False) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c)[0] in {"P", "S", "Z", "C"} else c for c in text)
    return list("".join(text.split())) if characters or language == "ja-JP" else text.split()


def edit_counts(reference: list[str], hypothesis: list[str]) -> dict:
    # Each DP cell stores (distance, substitutions, deletions, insertions).
    previous = [(i, 0, 0, i) for i in range(len(hypothesis) + 1)]
    for i, expected in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, actual in enumerate(hypothesis, 1):
            diag = previous[j - 1]
            substitution = int(expected != actual)
            candidates = [(diag[0] + substitution, diag[1] + substitution, diag[2], diag[3]),
                          (previous[j][0] + 1, previous[j][1], previous[j][2] + 1, previous[j][3]),
                          (current[j - 1][0] + 1, current[j - 1][1], current[j - 1][2], current[j - 1][3] + 1)]
            current.append(min(candidates, key=lambda item: item[0]))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return {"edits": distance, "substitutions": substitutions, "deletions": deletions,
            "insertions": insertions, "referenceUnits": len(reference),
            "rate": distance / len(reference) if reference else None}


def quantile(values: list[float], fraction: float) -> float | None:
    if not values: return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def removed_characters(previous: str, current: str) -> int:
    prefix = 0
    for left, right in zip(previous, current):
        if left != right: break
        prefix += 1
    return len(previous) - prefix


def validate_events(events: list[dict]) -> None:
    previous_time = -1.0
    identifiers = set()
    for event in events:
        time = event.get("atMs")
        if not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0 or time < previous_time:
            raise ValueError("Trace timestamps must be finite, nonnegative, and monotonic")
        if not isinstance(event.get("eventID"), int) or event["eventID"] in identifiers:
            raise ValueError("Event IDs must be unique integers within an attempt")
        previous_time = time
        identifiers.add(event["eventID"])
    by_id = {event["eventID"]: event for event in events}
    for event in events:
        for key in ("sourceEventID", "requestSourceEventID"):
            target = event.get(key)
            if target is not None and (target not in by_id or by_id[target]["atMs"] > event["atMs"]):
                raise ValueError("Translation correlation must refer to an earlier source event")


def derive_metrics(events: list[dict], result: dict, case: dict) -> dict:
    validate_events(events)
    metrics = dict.fromkeys(LATENCY_METRICS)
    by_id = {event["eventID"]: event for event in events}
    def items(*kinds): return [event for event in events if event["kind"] in kinds]
    def first(kind):
        found = items(kind)
        return found[0]["atMs"] if found else None
    def difference(end, start): return end - start if end is not None and start is not None else None
    audio_start, audio_end = first("audio_started"), first("audio_ended")
    asr_events = items("asr_partial", "asr_final", "app_final")
    asr = [event for event in asr_events if event.get("text", "").strip()]
    finals = [event for event in items("asr_final", "app_final") if event.get("text", "").strip()]
    publications = [event for event in items("translation_published", "translation_reused") if event.get("text", "").strip()]
    final_publications = [event for event in publications if not event["isPartial"]]
    first_text = asr[0]["atMs"] if asr else None
    first_translation = publications[0]["atMs"] if publications else None
    final_translation = final_publications[-1]["atMs"] if final_publications else None
    metrics.update(
        attemptStartToFirstTextMs=difference(first_text, first("attempt_started")),
        attemptStartToFirstTranslationMs=difference(first_translation, first("attempt_started")),
        speechPreparationMs=difference(first("speech_ready"), first("speech_prepare_started")),
        translationPreparationMs=difference(first("translation_ready"), first("translation_prepare_started")),
        audioStartToFirstTextMs=difference(first_text, audio_start),
        audioStartToFirstFinalMs=difference(finals[0]["atMs"] if finals else None, audio_start),
        audioStartToFirstTranslationMs=difference(first_translation, audio_start),
        audioStartToFinalTranslationMs=difference(final_translation, audio_start),
        asrDrainMs=difference(first("speech_stopped"), audio_end),
        translationTailAfterAudioEndMs=max(0, final_translation - audio_end) if final_translation is not None and audio_end is not None else None,
        referenceSubmitToFinalTranslationMs=difference(final_translation, first("text_submitted")),
        maximumInputLatenessMs=result.get("maximumInputLatenessMs") if audio_start is not None else None,
        processPeakResidentBytes=result.get("processPeakResidentBytes"),
        translationRequestCount=len(items("translation_started")) if events and result.get("lane") != "speech" else None,
        translationFailureCount=len(items("translation_failed")) if events and result.get("lane") != "speech" else None,
        translationReuseCount=len(items("translation_reused")) if events and result.get("lane") != "speech" else None,
    )
    onset = case.get("speechStartSeconds")
    if onset is not None and audio_start is not None:
        metrics["annotatedSpeechStartToFirstTextMs"] = difference(first_text, audio_start + onset * 1000)
        metrics["annotatedSpeechStartToFirstTranslationMs"] = difference(first_translation, audio_start + onset * 1000)
    end = case.get("speechEndSeconds")
    if end is not None and audio_start is not None:
        metrics["annotatedSpeechEndToFinalTextMs"] = difference(finals[-1]["atMs"] if finals else None, audio_start + end * 1000)
        metrics["annotatedSpeechEndToFinalTranslationMs"] = difference(final_translation, audio_start + end * 1000)
    activity_end = case.get("acousticActivityEstimate", {}).get("endSeconds")
    if activity_end is not None and audio_start is not None:
        metrics["acousticActivityEndToFinalTextMs"] = difference(finals[-1]["atMs"] if finals else None, audio_start + activity_end * 1000)
    estimate = case.get("acousticActivityEstimate", {}).get("startSeconds")
    if estimate is not None and audio_start is not None:
        metrics["acousticActivityStartToFirstTextMs"] = difference(first_text, audio_start + estimate * 1000)
    duration = result.get("audioDurationSeconds")
    queue = result.get("queue", {})
    rate = queue.get("sampleRate")
    if rate:
        metrics["maximumQueueSeconds"] = queue["maximumQueuedSamples"] / rate if queue.get("maximumQueuedSamples") is not None else None
        metrics["droppedAudioSeconds"] = queue["droppedSamples"] / rate if queue.get("droppedSamples") is not None else None
    if duration:
        service = queue.get("runtimeServiceSeconds")
        metrics["runtimeServiceRTF"] = service / duration if service is not None else None
        stopped = first("speech_stopped")
        metrics["pacedWallRTF"] = (stopped - audio_start) / (duration * 1000) if stopped is not None and audio_start is not None else None

    # Appends cost zero. Resets at finalized boundaries do not create fake churn.
    previous = ""
    removed = 0
    for event in asr_events:
        removed += removed_characters(previous, event["text"])
        previous = "" if event["kind"] in {"asr_final", "app_final"} else event["text"]
    metrics["asrRemovedCharacters"] = removed if asr else None
    metrics["asrRevisionChurn"] = removed / len(result["transcript"]) if result.get("transcript") else None
    previous_by_boundary = {}
    removed = 0
    for event in publications:
        source = by_id.get(event.get("sourceEventID"), {})
        boundary = source.get("boundaryID")
        previous = previous_by_boundary.get(boundary, "")
        removed += removed_characters(previous, event["text"])
        previous_by_boundary[boundary] = event["text"] if event["isPartial"] else ""
    metrics["translationRemovedCharacters"] = removed if publications else None
    metrics["translationRevisionChurn"] = removed / len(result["translation"]) if result.get("translation") else None

    starts = {event["requestID"]: event for event in items("translation_started")}
    routes = {event["requestID"]: event for event in items("translation_route")}
    requests = []
    for event in items("translation_succeeded", "translation_failed"):
        started = starts.get(event.get("requestID"))
        if started is None: raise ValueError("Translation completion has no matching start")
        source = by_id.get(started.get("requestSourceEventID"))
        requests.append({"requestID": event["requestID"], "outcome": event["kind"].removeprefix("translation_"),
                         "queueMs": started["atMs"] - source["atMs"] if source else None,
                         "serviceMs": event["atMs"] - started["atMs"],
                         "isPartial": started["isPartial"], "error": event.get("error"), "outputTokens": routes.get(event["requestID"], {}).get("outputTokens"),
                         "route": routes.get(event["requestID"], {}).get("route")})
    publication_lags = []
    for event in publications:
        source = by_id.get(event.get("sourceEventID"))
        publication_lags.append({"eventID": event["eventID"], "sourceEventID": event.get("sourceEventID"),
                                 "isPartial": event["isPartial"], "reused": event["kind"] == "translation_reused",
                                 "sourceToPublicationMs": event["atMs"] - source["atMs"] if source else None})
    return {"metrics": metrics, "translationRequests": requests, "publications": publication_lags,
            "routeCounts": dict(Counter(event.get("route", "unknown") for event in items("translation_route"))),
            "speechWarnings": [event.get("error") for event in items("speech_warning")],
            "unmeasured": ["microphone capture latency", "SwiftUI layout latency", "compositor paint latency", "GPU kernel execution time"]}
