@preconcurrency import AVFoundation
import Foundation
import MLXAudioCore
import MLXAudioSTT
import MimiCore

struct RealtimeBenchmarkReport: Codable, Sendable {
    let engine: String
    let mode: String
    let language: String
    let audioDurationSeconds: Double
    let wallSeconds: Double
    let modelLoadSeconds: Double?
    let firstTextAtSeconds: Double?
    let firstFinalAtSeconds: Double?
    let updateCount: Int
    let meanDecodeSeconds: Double?
    let maxDecodeSeconds: Double?
    let realTimeFactor: Double?
    let hypothesisChurn: Double
    let finalText: String
    let firstUpdates: [String]
    var referenceText: String? = nil
    var errorMetric: String? = nil
    var errorRate: Double? = nil

    func printJSON() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(self)
        guard let text = String(data: data, encoding: .utf8) else {
            throw RealtimeBenchmarkError.couldNotEncodeReport
        }
        print(text)
    }

    static func hypothesisChurn(_ updates: [String]) -> Double {
        guard updates.count > 1 else { return 0 }
        var total = 0.0
        for (previous, next) in zip(updates, updates.dropFirst()) {
            let denominator = max(previous.count, next.count, 1)
            total += Double(editDistance(Array(previous), Array(next))) / Double(denominator)
        }
        return total / Double(updates.count - 1)
    }

    mutating func score(against reference: String, language: SpeechLanguage) {
        referenceText = reference
        switch language {
        case .english:
            errorMetric = "WER"
            let expected = Self.englishWords(reference)
            let actual = Self.englishWords(finalText)
            errorRate = Self.normalizedEditDistance(actual, expected: expected)
        case .japanese:
            errorMetric = "CER"
            let expected = Self.japaneseCharacters(reference)
            let actual = Self.japaneseCharacters(finalText)
            errorRate = Self.normalizedEditDistance(actual, expected: expected)
        }
    }

    private static func normalizedEditDistance<T: Equatable>(
        _ actual: [T],
        expected: [T]
    ) -> Double {
        guard !expected.isEmpty else { return actual.isEmpty ? 0 : 1 }
        return Double(editDistance(actual, expected)) / Double(expected.count)
    }

    private static func englishWords(_ text: String) -> [String] {
        text.lowercased().split { character in
            !character.isLetter && !character.isNumber
        }.map(String.init)
    }

    private static func japaneseCharacters(_ text: String) -> [Character] {
        text.filter { $0.isLetter || $0.isNumber }.map { $0 }
    }

    private static func editDistance<T: Equatable>(_ left: [T], _ right: [T]) -> Int {
        var previous = Array(0...right.count)
        for (leftIndex, leftElement) in left.enumerated() {
            var current = Array(repeating: 0, count: right.count + 1)
            current[0] = leftIndex + 1
            for (rightIndex, rightElement) in right.enumerated() {
                current[rightIndex + 1] = min(
                    current[rightIndex] + 1,
                    previous[rightIndex + 1] + 1,
                    previous[rightIndex] + (leftElement == rightElement ? 0 : 1)
                )
            }
            previous = current
        }
        return previous[right.count]
    }
}

extension Duration {
    var seconds: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }
}

@MainActor
enum RealtimeBenchmarkRunner {
    static func runQwen(
        recordingAt url: URL,
        language: SpeechLanguage,
        simulateRealtime: Bool
    ) async throws -> RealtimeBenchmarkReport {
        let loadStartedAt = ContinuousClock.now
        let model = try await Qwen3ASRModel.fromPretrained(
            "mlx-community/Qwen3-ASR-0.6B-4bit"
        )
        let modelLoadSeconds = loadStartedAt.duration(to: .now).seconds
        let (sampleRate, mlxAudio) = try loadAudioArray(from: url, sampleRate: model.sampleRate)
        let samples = mlxAudio.asArray(Float.self)
        let audioDuration = Double(samples.count) / Double(sampleRate)
        let config = StreamingConfig(
            decodeIntervalSeconds: 0.5,
            boundaryDecodeIntervalSeconds: 0.2,
            boundaryBoostSeconds: 1.0,
            encoderWindowOverlapSeconds: 1.0,
            maxCachedWindows: 4,
            delayPreset: .realtime,
            language: language.displayName,
            temperature: 0,
            maxTokensPerPass: 256,
            minAgreementPasses: 2,
            boundaryMinAgreementPasses: 3,
            maxDecodeWindows: 2,
            finalizeCompletedWindows: true
        )
        let session = StreamingInferenceSession(model: model, config: config)
        let startedAt = ContinuousClock.now
        let eventTask = Task {
            await collectQwenEvents(session.events, startedAt: startedAt)
        }

        let chunkSampleCount = max(1, Int(Double(sampleRate) * 0.2))
        var offset = 0
        while offset < samples.count {
            try Task.checkCancellation()
            let end = min(samples.count, offset + chunkSampleCount)
            session.feedAudio(samples: Array(samples[offset..<end]))
            if simulateRealtime {
                try await Task.sleep(for: .seconds(Double(end - offset) / Double(sampleRate)))
            }
            offset = end
        }
        session.stop()
        let events = await eventTask.value
        let wallSeconds = startedAt.duration(to: .now).seconds
        return RealtimeBenchmarkReport(
            engine: "qwen3-asr-0.6b-4bit-mlx",
            mode: "dual-pass-streaming",
            language: language.rawValue,
            audioDurationSeconds: audioDuration,
            wallSeconds: wallSeconds,
            modelLoadSeconds: modelLoadSeconds,
            firstTextAtSeconds: events.firstTextAtSeconds,
            firstFinalAtSeconds: events.firstConfirmedAtSeconds,
            updateCount: events.updates.count,
            meanDecodeSeconds: nil,
            maxDecodeSeconds: nil,
            realTimeFactor: events.latestStats?.realTimeFactor,
            hypothesisChurn: RealtimeBenchmarkReport.hypothesisChurn(events.updates),
            finalText: events.finalText,
            firstUpdates: Array(events.updates.prefix(8))
        )
    }

    @available(macOS 26.0, *)
    static func runApple(
        recordingAt url: URL,
        language: SpeechLanguage,
        mode: AppleSpeechEngine.ResultMode,
        simulateRealtime: Bool
    ) async throws -> RealtimeBenchmarkReport {
        let audioFile = try AVAudioFile(forReading: url)
        let audioDuration = Double(audioFile.length) / audioFile.fileFormat.sampleRate
        let inputFormat = audioFile.processingFormat
        let frameCapacity: AVAudioFrameCount = 1_024
        let engine = AppleSpeechEngine(resultMode: mode)
        let startedAt = ContinuousClock.now
        var firstTextAt: Double?
        var firstFinalAt: Double?
        var updates: [String] = []
        var finalSegments: [String] = []

        try await engine.start(language: language, inputFormat: inputFormat) { event in
            let elapsed = startedAt.duration(to: .now).seconds
            switch event {
            case let .partial(text):
                let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !normalized.isEmpty else { return }
                if firstTextAt == nil { firstTextAt = elapsed }
                updates.append(normalized)
            case let .final(text):
                let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !normalized.isEmpty else { return }
                if firstTextAt == nil { firstTextAt = elapsed }
                if firstFinalAt == nil { firstFinalAt = elapsed }
                updates.append(normalized)
                finalSegments.append(normalized)
            }
        }

        while audioFile.framePosition < audioFile.length {
            let remaining = AVAudioFrameCount(audioFile.length - audioFile.framePosition)
            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: inputFormat,
                frameCapacity: min(frameCapacity, remaining)
            ) else {
                throw RealtimeBenchmarkError.couldNotAllocateAudioBuffer
            }
            try audioFile.read(into: buffer, frameCount: min(frameCapacity, remaining))
            engine.consume(buffer)
            if simulateRealtime {
                let duration = Double(buffer.frameLength) / inputFormat.sampleRate
                try await Task.sleep(for: .seconds(duration))
            }
        }

        await engine.stop()
        let wallSeconds = startedAt.duration(to: .now).seconds
        return RealtimeBenchmarkReport(
            engine: "apple-speech",
            mode: mode.rawValue,
            language: language.rawValue,
            audioDurationSeconds: audioDuration,
            wallSeconds: wallSeconds,
            modelLoadSeconds: nil,
            firstTextAtSeconds: firstTextAt,
            firstFinalAtSeconds: firstFinalAt,
            updateCount: updates.count,
            meanDecodeSeconds: nil,
            maxDecodeSeconds: nil,
            realTimeFactor: audioDuration > 0 ? wallSeconds / audioDuration : nil,
            hypothesisChurn: RealtimeBenchmarkReport.hypothesisChurn(updates),
            finalText: finalSegments.isEmpty ? (updates.last ?? "") : finalSegments.joined(separator: " "),
            firstUpdates: Array(updates.prefix(8))
        )
    }

    private static func collectQwenEvents(
        _ events: AsyncStream<MLXAudioSTT.TranscriptionEvent>,
        startedAt: ContinuousClock.Instant
    ) async -> QwenEventSummary {
        var summary = QwenEventSummary()
        for await event in events {
            let elapsed = startedAt.duration(to: .now).seconds
            switch event {
            case .provisional:
                // The session also emits a coalesced display update. Counting
                // token-level provisional events would overstate UI churn.
                break
            case let .confirmed(text):
                if summary.firstConfirmedAtSeconds == nil, !text.isEmpty {
                    summary.firstConfirmedAtSeconds = elapsed
                }
            case let .displayUpdate(confirmedText, provisionalText):
                summary.record(
                    text: [confirmedText, provisionalText]
                        .filter { !$0.isEmpty }
                        .joined(separator: " "),
                    elapsed: elapsed
                )
            case let .stats(stats):
                summary.latestStats = stats
            case let .ended(fullText):
                summary.finalText = fullText.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        if summary.finalText.isEmpty {
            summary.finalText = summary.updates.last ?? ""
        }
        return summary
    }
}

private struct QwenEventSummary: Sendable {
    var firstTextAtSeconds: Double?
    var firstConfirmedAtSeconds: Double?
    var updates: [String] = []
    var latestStats: StreamingStats?
    var finalText = ""

    mutating func record(text: String, elapsed: Double) {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty, updates.last != normalized else { return }
        if firstTextAtSeconds == nil {
            firstTextAtSeconds = elapsed
        }
        updates.append(normalized)
    }
}

private enum RealtimeBenchmarkError: LocalizedError {
    case couldNotAllocateAudioBuffer
    case couldNotEncodeReport

    var errorDescription: String? {
        switch self {
        case .couldNotAllocateAudioBuffer:
            "Mimi could not allocate a benchmark audio buffer."
        case .couldNotEncodeReport:
            "Mimi could not encode the benchmark report."
        }
    }
}
