import Foundation
import MimiCore

@MainActor
func verifyBenchmarkTraceContract() async throws -> [String: Bool] {
    let url = FileManager.default.temporaryDirectory.appending(path: "mimi-benchmark-trace-\(UUID().uuidString).jsonl")
    defer { try? FileManager.default.removeItem(at: url) }
    let trace = try BenchmarkTrace(url: url)
    trace.begin(index: 0, phase: "cold")
    let scope = UUID()
    let worker = LocalTranslationStream { text, _ in
        try await Task.sleep(for: .milliseconds(15))
        if text == "failure" { throw CocoaError(.coderInvalidValue) }
        return "translated " + text
    }
    worker.onWork = { trace.observe($0) }
    trace.source("hello", partial: true, segmentID: nil, boundaryID: nil)
    worker.update(.init(segments: [], liveText: "hello", language: .english, scopeID: scope))
    try await waitForBenchmarkWorker { !worker.liveTranslation.isEmpty }
    let final = TranscriptSegment(text: "hello", language: .english)
    trace.source(final.text, partial: false, segmentID: final.id, boundaryID: nil)
    worker.update(.init(segments: [final], liveText: "", language: .english, scopeID: scope))
    let reusedWithoutInference = worker.translations[final.id] == "translated hello"
        && trace.events.filter { $0["kind"] as? String == "translation_started" }.count == 1
        && trace.events.contains { $0["kind"] as? String == "translation_reused" }

    trace.source("failure", partial: true, segmentID: nil, boundaryID: final.id)
    worker.update(.init(segments: [final], liveText: "failure", language: .english, scopeID: scope))
    try await waitForBenchmarkWorker { trace.events.contains { $0["kind"] as? String == "translation_failed" } }
    let starts = trace.events.filter { $0["kind"] as? String == "translation_started" }
    let terminals = trace.events.filter { ["translation_succeeded", "translation_failed"].contains($0["kind"] as? String ?? "") }
    let requestIDs = Set(starts.compactMap { $0["requestID"] as? String })
    let terminalIDs = Set(terminals.compactMap { $0["requestID"] as? String })
    let sourceIDs = Set(trace.events.filter { ($0["kind"] as? String)?.hasPrefix("asr_") == true }.compactMap { $0["eventID"] as? Int })
    let correlationComplete = starts.allSatisfy { event in
        guard let source = event["requestSourceEventID"] as? Int else { return false }
        return sourceIDs.contains(source)
    }
    let times = trace.events.compactMap { $0["atMs"] as? Double }
    let monotonic = zip(times, times.dropFirst()).allSatisfy { $0 <= $1 }
    let eventsBeforeClose = trace.events.count
    try trace.finish(["status": "fixture"])
    trace.emit("late_callback")
    let rejectsLateEvents = trace.events.count == eventsBeforeClose
    worker.reset()

    let next = try BenchmarkTrace(url: url, append: true)
    next.begin(index: 1, phase: "warm")
    try next.finish(["status": "fixture"])
    let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n")
    let records = try lines.map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
    let results = records.filter { $0["record"] as? String == "result" }
    return ["reuse_without_duplicate_inference": reusedWithoutInference,
            "request_start_finish_identity": requestIDs.count == 2 && requestIDs == terminalIDs,
            "source_correlation": correlationComplete, "monotonic_clock": monotonic,
            "failures_retained": terminals.contains { $0["kind"] as? String == "translation_failed" },
            "late_callbacks_cannot_pollute_next_attempt": rejectsLateEvents,
            "append_preserves_both_attempts": results.count == 2 && results.last?["attemptIndex"] as? Int == 1]
}

@MainActor
private func waitForBenchmarkWorker(_ ready: () -> Bool) async throws {
    for _ in 0..<200 {
        if ready() { return }
        try await Task.sleep(for: .milliseconds(5))
    }
    throw CocoaError(.coderInvalidValue)
}
