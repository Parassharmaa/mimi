@preconcurrency import AVFoundation
import Foundation
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
        _ = (url, language, simulateRealtime)
        throw RealtimeBenchmarkError.removedModel
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

}

private enum RealtimeBenchmarkError: LocalizedError {
    case couldNotAllocateAudioBuffer
    case couldNotEncodeReport
    case removedModel

    var errorDescription: String? {
        switch self {
        case .couldNotAllocateAudioBuffer:
            "Mimi could not allocate a benchmark audio buffer."
        case .couldNotEncodeReport:
            "Mimi could not encode the benchmark report."
        case .removedModel:
            "This benchmark model is no longer included in Mimi."
        }
    }
}
