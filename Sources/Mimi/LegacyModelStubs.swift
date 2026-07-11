@preconcurrency import AVFoundation
import Foundation
import MimiCore
import MimiSession

/// Decode-only compatibility for old settings and command-line probes. These
/// engines are intentionally unavailable in Mimi's Apple-only product.
@MainActor
class RemovedLiveModel {
    var runtimeAvailabilityMessage: String? { "This model is no longer included in Mimi." }
    var isDownloaded: Bool { false }

    func ensureInstalled() throws { throw RemovedModelError.unavailable }
    func install() async throws { throw RemovedModelError.unavailable }
    func install(onProgress: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void) async throws {
        _ = onProgress
        throw RemovedModelError.unavailable
    }
    func startLive(
        language: SpeechLanguage,
        inputFormat: AVAudioFormat,
        onEvent: @escaping @MainActor (TranscriptEvent) -> Void,
        onBackpressure: @escaping @MainActor (String) -> Void
    ) async throws {
        throw RemovedModelError.unavailable
    }
    func consumeLive(_ buffer: AVAudioPCMBuffer) {}
    func stopLive() async {}
    func cancelLive() async {}
    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String {
        throw RemovedModelError.unavailable
    }
    func removeDownloadedModel() async throws {}
}

@MainActor
final class NemotronMLXLiveEngine: RemovedLiveModel, NemotronMLXLiveTranscribing {}

@MainActor
final class QwenMLXLiveEngine: RemovedLiveModel, QwenMLXLiveTranscribing {}

private enum RemovedModelError: LocalizedError {
    case unavailable
    var errorDescription: String? { "Mimi now uses Apple Speech for live transcription." }
}
