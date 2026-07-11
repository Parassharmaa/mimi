import Foundation

public enum SpeechLanguage: String, CaseIterable, Codable, Sendable, Identifiable {
    case english = "en-US"
    case japanese = "ja-JP"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .english: "English"
        case .japanese: "Japanese"
        }
    }

    public var nativeName: String {
        switch self {
        case .english: "English"
        case .japanese: "日本語"
        }
    }

    public var whisperLanguageCode: String {
        switch self {
        case .english: "en"
        case .japanese: "ja"
        }
    }

    public var translationTarget: SpeechLanguage {
        switch self {
        case .english: .japanese
        case .japanese: .english
        }
    }
}

public enum AudioSource: String, CaseIterable, Codable, Sendable, Identifiable {
    case microphone
    case outputAudio
    case applicationAudio
    case systemAudio

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .microphone: "Microphone"
        case .outputAudio: "Selected Audio Output"
        case .applicationAudio: "Selected App Audio"
        case .systemAudio: "Selected Display Audio"
        }
    }

    public var details: String {
        switch self {
        case .microphone: "Your selected microphone input"
        case .outputAudio: "Everything playing through the selected output device"
        case .applicationAudio: "Zoom, Chrome, or another selected app"
        case .systemAudio: "Audio associated with a display you choose"
        }
    }
}

public enum TranscriptionEngineID: String, CaseIterable, Codable, Sendable, Identifiable {
    case appleSpeechAnalyzer
    case whisperKitLargeV3Turbo
    case nemotronStreamingExperimental
    case qwen3StreamingExperimental

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .appleSpeechAnalyzer: "Apple Speech"
        case .whisperKitLargeV3Turbo: "Whisper Large-v3 (626 MB)"
        case .nemotronStreamingExperimental: "Nemotron 3.5 MLX (756 MB)"
        case .qwen3StreamingExperimental: "Qwen3-ASR 0.6B MLX (713 MB)"
        }
    }

    public var summary: String {
        switch self {
        case .appleSpeechAnalyzer:
            "Native, on-device live transcription. Available on macOS 26 and later."
        case .whisperKitLargeV3Turbo:
            "Downloadable Core ML accuracy model for English and Japanese."
        case .nemotronStreamingExperimental:
            "Experimental on-device MLX live transcription for English and Japanese. Uses bounded local windows for predictable memory."
        case .qwen3StreamingExperimental:
            "Experimental on-device MLX dual-pass transcription with fast provisional captions and retrospective window correction."
        }
    }

    public var isExperimental: Bool {
        switch self {
        case .nemotronStreamingExperimental, .qwen3StreamingExperimental: true
        case .appleSpeechAnalyzer, .whisperKitLargeV3Turbo: false
        }
    }
}

public enum TranslationMode: String, CaseIterable, Codable, Sendable, Identifiable {
    case off
    case translateFinalSegments

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .off: "Off"
        case .translateFinalSegments: "Live Translation (English ↔ Japanese)"
        }
    }
}

public struct LocalModelPack: Identifiable, Equatable, Sendable {
    public enum Ownership: String, Equatable, Sendable {
        case systemManaged
        case appManaged
        case experimental
    }

    public let id: String
    public let engine: TranscriptionEngineID
    public let supportedLanguages: Set<SpeechLanguage>
    public let ownership: Ownership
    public let estimatedDownloadMB: Int?
    public let recommendation: String

    public init(
        id: String,
        engine: TranscriptionEngineID,
        supportedLanguages: Set<SpeechLanguage>,
        ownership: Ownership,
        estimatedDownloadMB: Int?,
        recommendation: String
    ) {
        self.id = id
        self.engine = engine
        self.supportedLanguages = supportedLanguages
        self.ownership = ownership
        self.estimatedDownloadMB = estimatedDownloadMB
        self.recommendation = recommendation
    }
}

public enum ModelCatalog {
    public static let packs: [LocalModelPack] = [
        .init(
            id: "apple-speech-en-ja",
            engine: .appleSpeechAnalyzer,
            supportedLanguages: [.english, .japanese],
            ownership: .systemManaged,
            estimatedDownloadMB: nil,
            recommendation: "Best first choice on macOS 26: fast live results and OS-managed language assets."
        ),
        .init(
            id: "whisperkit-large-v3-626mb",
            engine: .whisperKitLargeV3Turbo,
            supportedLanguages: [.english, .japanese],
            ownership: .appManaged,
            estimatedDownloadMB: 626,
            recommendation: "Use for an accuracy pass or when Apple Speech is unavailable."
        ),
        .init(
            id: "nemotron-3.5-streaming",
            engine: .nemotronStreamingExperimental,
            supportedLanguages: [.english, .japanese],
            ownership: .experimental,
            estimatedDownloadMB: 756,
            recommendation: "Experimental Apple-silicon MLX live captions. Mimi finalizes bounded local windows at a pause or 30 seconds to keep memory predictable."
        ),
        .init(
            id: "qwen3-asr-0.6b-streaming",
            engine: .qwen3StreamingExperimental,
            supportedLanguages: [.english, .japanese],
            ownership: .experimental,
            estimatedDownloadMB: 713,
            recommendation: "Experimental dual-pass MLX captions: a fast rolling hypothesis plus agreement-based confirmation and corrected 8-second windows."
        )
    ]

    public static func pack(for engine: TranscriptionEngineID) -> LocalModelPack? {
        packs.first { $0.engine == engine }
    }
}

public enum TranscriptEvent: Equatable, Sendable {
    case partial(String)
    case final(String)
}

public struct TranscriptSegment: Identifiable, Codable, Equatable, Sendable {
    public let id: UUID
    public let text: String
    public let language: SpeechLanguage
    public let createdAt: Date

    public init(
        id: UUID = UUID(),
        text: String,
        language: SpeechLanguage,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.text = text
        self.language = language
        self.createdAt = createdAt
    }
}

public struct TranscriptDocument: Codable, Equatable, Sendable {
    public private(set) var segments: [TranscriptSegment]
    public private(set) var liveText: String

    public init(segments: [TranscriptSegment] = [], liveText: String = "") {
        self.segments = segments
        self.liveText = liveText
    }

    public var renderedText: String {
        (segments.map(\.text) + (liveText.isEmpty ? [] : [liveText]))
            .joined(separator: "\n")
    }

    /// Stable, immutable text. Translation and persistent session storage use
    /// this rather than a volatile real-time ASR hypothesis.
    public var finalizedText: String {
        segments.map(\.text).joined(separator: "\n")
    }

    public func finalizedText(for language: SpeechLanguage) -> String {
        segments
            .filter { $0.language == language }
            .map(\.text)
            .joined(separator: "\n")
    }

    public func renderedText(for language: SpeechLanguage, includingLiveText: Bool) -> String {
        let finalized = finalizedText(for: language)
        guard includingLiveText, !liveText.isEmpty else { return finalized }
        return [finalized, liveText]
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    /// A bounded tail for realtime translation. Re-translating the complete
    /// meeting on every partial makes latency grow with session length; a few
    /// recent completed phrases plus the current Apple Speech hypothesis give
    /// the translator local word-order context while keeping work constant.
    public func realtimeTranslationContext(
        for language: SpeechLanguage,
        maximumCharacterCount: Int = 480
    ) -> String {
        guard maximumCharacterCount > 0 else { return "" }

        let current = Self.normalized(liveText)
        var parts = current.isEmpty ? [] : [String(current.suffix(maximumCharacterCount))]
        var remaining = maximumCharacterCount - (parts.first?.count ?? 0)

        for segment in segments.reversed() where segment.language == language {
            let separatorCost = parts.isEmpty ? 0 : 1
            guard segment.text.count + separatorCost <= remaining else { break }
            parts.insert(segment.text, at: 0)
            remaining -= segment.text.count + separatorCost
        }
        return parts.joined(separator: "\n")
    }

    public mutating func apply(_ event: TranscriptEvent, language: SpeechLanguage, now: Date = Date()) {
        switch event {
        case let .partial(text):
            liveText = Self.normalized(text)
        case let .final(text):
            let normalized = Self.normalized(text)
            guard !normalized.isEmpty else {
                liveText = ""
                return
            }

            // Repeated phrases are legitimate speech (for example, "yes",
            // then "yes"). ASR engines must supply a stable result identity
            // before we coalesce finals; text equality is not safe.
            segments.append(.init(text: normalized, language: language, createdAt: now))
            liveText = ""
        }
    }

    @discardableResult
    public mutating func finalizeLiveText(language: SpeechLanguage, now: Date = Date()) -> TranscriptSegment? {
        let normalized = Self.normalized(liveText)
        guard !normalized.isEmpty else { return nil }
        let segment = TranscriptSegment(text: normalized, language: language, createdAt: now)
        segments.append(segment)
        liveText = ""
        return segment
    }

    private static func normalized(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

public enum RecordingState: Equatable, Sendable {
    case idle
    case preparing
    case recording
    case processing
    case failed(String)

    public var label: String {
        switch self {
        case .idle: "Ready"
        case .preparing: "Preparing local model…"
        case .recording: "Listening locally"
        case .processing: "Finalizing transcript…"
        case let .failed(message): message
        }
    }
}
