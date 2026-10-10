import Foundation
import MimiCore

/// Optional observations of production translation work, including reused finals.
struct TranslationWorkObservation {
    enum Kind: String { case started, succeeded, failed, published, reused }
    let kind: Kind
    let requestID: UUID?
    let source: String
    let language: SpeechLanguage
    let isPartial: Bool
    let segmentID: UUID?
    let boundaryID: UUID?
    var output: String? = nil
    var error: String? = nil
}

struct SpeechBenchmarkQueueMetrics: Codable {
    var sampleRate: Double? = nil
    var capacitySamples: Int? = nil
    var maximumQueuedSamples: Int? = nil
    var droppedSamples: Int? = nil
    /// Time awaiting the runtime actor, including finalization; not GPU kernel time.
    var runtimeServiceSeconds: Double? = nil
}

struct BenchmarkCaseRequest: Decodable {
    let caseID: String
    let lane: String
    let engine: String
    let language: SpeechLanguage
    let source: String
    let audioPath: String?
    let translationModelPath: String?
    let speechModelPath: String?
    let repetitions: Int
    let warmups: Int
    let chunkSeconds: Double
    let translationDrainSeconds: Double
}

/// One clock and an append-only trace survive a failed or externally timed-out run.
@MainActor
final class BenchmarkTrace {
    private let handle: FileHandle
    private var origin = ContinuousClock.now
    private var sequence = 0
    private var attemptIndex = 0
    private var phase = "cold"
    private var sourceEvents: [String: Int] = [:]
    private var requests: [UUID: Int] = [:]
    private(set) var events: [[String: Any]] = []
    private(set) var writeError: Error?
    private var deliveredSeconds = 0.0
    private var finished = false
    private(set) var activeRequestID: UUID?

    init(url: URL, append: Bool = false) throws {
        if append {
            guard FileManager.default.fileExists(atPath: url.path) else { throw CocoaError(.fileNoSuchFile) }
        } else {
            guard !FileManager.default.fileExists(atPath: url.path),
                  FileManager.default.createFile(atPath: url.path, contents: nil) else { throw CocoaError(.fileWriteUnknown) }
        }
        handle = try FileHandle(forWritingTo: url)
        if append { try handle.seekToEnd() }
    }

    deinit { try? handle.close() }

    func begin(index: Int, phase: String) {
        origin = .now
        sequence = 0
        attemptIndex = index
        self.phase = phase
        sourceEvents = [:]
        requests = [:]
        activeRequestID = nil
        events = []
        deliveredSeconds = 0
        emit("attempt_started", fields: ["thermalState": ProcessInfo.processInfo.thermalState.rawValue, "lowPowerMode": ProcessInfo.processInfo.isLowPowerModeEnabled])
    }

    var elapsedMS: Double { origin.duration(to: .now).seconds * 1_000 }

    func delivered(_ seconds: Double) { deliveredSeconds = seconds }

    @discardableResult
    func emit(_ kind: String, fields: [String: Any] = [:]) -> Int {
        guard !finished else { return 0 }
        sequence += 1
        var event: [String: Any] = [
            "schemaVersion": 1, "record": "event", "attemptIndex": attemptIndex,
            "phase": phase, "eventID": sequence, "kind": kind,
            "atMs": elapsedMS, "audioDeliveredSeconds": deliveredSeconds
        ]
        event.merge(fields) { _, new in new }
        events.append(event)
        write(event)
        return sequence
    }

    func source(_ text: String, partial: Bool, segmentID: UUID?, boundaryID: UUID?, origin: String = "asr") {
        let id = emit(origin + (partial ? "_partial" : "_final"), fields: [
            "text": text, "segmentID": segmentID?.uuidString as Any? ?? NSNull(),
            "boundaryID": boundaryID?.uuidString as Any? ?? NSNull()
        ])
        sourceEvents[key(text, partial: partial, segmentID: segmentID, boundaryID: boundaryID)] = id
    }

    func observe(_ event: TranslationWorkObservation) {
        let sourceID = sourceEvents[key(event.source, partial: event.isPartial, segmentID: event.segmentID, boundaryID: event.boundaryID)]
        if event.kind == .started { activeRequestID = event.requestID }
        if event.kind == .started, let requestID = event.requestID, let sourceID {
            requests[requestID] = sourceID
        }
        var fields: [String: Any] = [
            "source": event.source, "isPartial": event.isPartial,
            "segmentID": event.segmentID?.uuidString as Any? ?? NSNull(),
            "boundaryID": event.boundaryID?.uuidString as Any? ?? NSNull(),
            "requestID": event.requestID?.uuidString as Any? ?? NSNull(),
            "sourceEventID": sourceID as Any? ?? NSNull(),
            "requestSourceEventID": event.requestID.flatMap { requests[$0] } as Any? ?? NSNull()
        ]
        if let output = event.output { fields["text"] = output }
        if let error = event.error { fields["error"] = error }
        emit("translation_" + event.kind.rawValue, fields: fields)
    }

    func route(requestID: UUID?, memory: Bool, expert: Bool, outputTokens: Int?) {
        emit("translation_route", fields: [
            "requestID": requestID?.uuidString as Any? ?? NSNull(),
            "route": memory ? "memory" : (expert ? "expert" : "base"),
            "outputTokens": outputTokens as Any? ?? NSNull()
        ])
    }

    func finish(_ result: [String: Any]) throws {
        if let writeError { throw writeError }
        var value = result
        value["schemaVersion"] = 1
        value["record"] = "result"
        value["attemptIndex"] = attemptIndex
        value["phase"] = phase
        value["wallMs"] = elapsedMS
        write(value)
        if let writeError { throw writeError }
        try handle.synchronize()
        finished = true
    }

    private func key(_ text: String, partial: Bool, segmentID: UUID?, boundaryID: UUID?) -> String {
        "\(partial ? "partial" : "final"):\((partial ? boundaryID : segmentID)?.uuidString ?? "root"):\(text)"
    }

    private func write(_ record: [String: Any]) {
        guard writeError == nil else { return }
        do {
            var data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys, .withoutEscapingSlashes])
            data.append(0x0A)
            try handle.write(contentsOf: data)
        } catch { writeError = error }
    }
}
