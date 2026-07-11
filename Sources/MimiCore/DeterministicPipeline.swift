import Foundation

/// A deliberately small, framework-free representation of the event contract
/// shared by microphone/process capture and every local ASR provider.
public struct DeterministicTranscriptionPipeline: Sendable {
    public private(set) var state: RecordingState = .idle
    public private(set) var document = TranscriptDocument()
    public let language: SpeechLanguage

    public init(language: SpeechLanguage) {
        self.language = language
    }

    public mutating func start() {
        precondition(state == .idle, "A pipeline can only start from idle")
        state = .recording
    }

    public mutating func receive(_ event: TranscriptEvent, now: Date = Date()) {
        precondition(state == .recording, "Only a recording pipeline can receive ASR events")
        document.apply(event, language: language, now: now)
    }

    public mutating func stop(now: Date = Date()) {
        guard state == .recording else { return }
        state = .processing
        document.finalizeLiveText(language: language, now: now)
        state = .idle
    }
}
