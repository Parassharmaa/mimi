import Foundation
import MimiCore

@MainActor
final class UnavailablePhononEngine: WhisperAccuracyTranscribing {
    var isDownloaded: Bool { false }
    var isRemovable: Bool { false }
    var runtimeAvailabilityMessage: String? { "Phonon 2 is not included in this runtime." }
    func ensureInstalled() throws { throw TranscriptionSessionError.whisperLiveUnavailable }
    func install(onProgress: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void) async throws {
        throw TranscriptionSessionError.whisperLiveUnavailable
    }
    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String {
        throw TranscriptionSessionError.whisperLiveUnavailable
    }
    func removeDownloadedModel() async throws { throw TranscriptionSessionError.whisperLiveUnavailable }
}
