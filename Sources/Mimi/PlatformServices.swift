@preconcurrency import AVFoundation
import Foundation
import MimiCore
import MimiSession

extension MicrophoneCapture: MicrophoneCapturing {}

@available(macOS 26.0, *)
extension AppleSpeechEngine: AppleLiveTranscribing {}

extension WhisperKitAccuracyEngine: WhisperAccuracyTranscribing {}

@MainActor
final class SystemAppleSpeechProvider: AppleSpeechProviding {
    var isPlatformAvailable: Bool {
        if #available(macOS 26.0, *) { return true }
        return false
    }

    func assetStatus(for language: SpeechLanguage) async -> AppleSpeechAssetStatus {
        guard #available(macOS 26.0, *) else { return .unsupported }
        return await AppleSpeechEngine.assetStatus(for: language)
    }

    func installAssets(for language: SpeechLanguage) async throws {
        guard #available(macOS 26.0, *) else {
            throw TranscriptionSessionError.appleSpeechRequiresMacOS26
        }
        try await AppleSpeechEngine.installAssets(for: language)
    }

    func makeEngine() throws -> any AppleLiveTranscribing {
        guard #available(macOS 26.0, *) else {
            throw TranscriptionSessionError.appleSpeechRequiresMacOS26
        }
        return AppleSpeechEngine()
    }
}

@MainActor
final class FileTranscriptStore: TranscriptPersisting {
    func loadLatestTranscript() -> TranscriptDocument {
        MimiStorage.loadLatestTranscript()
    }

    func saveLatestTranscript(_ document: TranscriptDocument) throws {
        try MimiStorage.saveLatestTranscript(document)
    }

    func clearLatestTranscript() throws {
        try MimiStorage.clearLatestTranscript()
    }

    func makeTemporaryRecordingURL(fileExtension: String) throws -> URL {
        try MimiStorage.makeTemporaryRecordingURL(fileExtension: fileExtension)
    }

    func removeTemporaryRecording(at url: URL) throws {
        try FileManager.default.removeItem(at: url)
    }

    func removeStaleTemporaryRecordings() {
        MimiStorage.removeStaleTemporaryRecordings()
    }
}
