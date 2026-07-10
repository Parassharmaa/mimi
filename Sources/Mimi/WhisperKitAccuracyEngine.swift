import Foundation
import MimiCore
@preconcurrency import WhisperKit

@MainActor
final class WhisperKitAccuracyEngine {
    /// Argmax's compressed multilingual Large-v3 Core ML artifact. This is not
    /// labelled Turbo because it is not the separate full Turbo artifact.
    private let modelName = "large-v3-v20240930_626MB"
    private let modelFolder: URL
    private let installMarkerURL: URL
    private var whisperKit: WhisperKit?

    init(fileManager: FileManager = .default) {
        let support = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? fileManager.temporaryDirectory
        modelFolder = support.appending(path: "Mimi/Models/WhisperKit", directoryHint: .isDirectory)
        installMarkerURL = support.appending(path: "Mimi/Models/WhisperKit/.mimi-large-v3-installed")
    }

    var isDownloaded: Bool {
        FileManager.default.fileExists(atPath: installMarkerURL.path)
    }

    func ensureInstalled() throws {
        guard isDownloaded else { throw WhisperModelError.notInstalled }
    }

    func install() async throws {
        if whisperKit == nil {
            try FileManager.default.createDirectory(at: modelFolder, withIntermediateDirectories: true)
            let config = makeConfiguration(download: true)
            whisperKit = try await WhisperKit(config)
            guard FileManager.default.createFile(atPath: installMarkerURL.path, contents: Data()) else {
                throw WhisperModelError.installMarkerFailed
            }
        }
    }

    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String {
        try await loadInstalledModel()
        guard let whisperKit else { return "" }

        let options = DecodingOptions(
            language: language.whisperLanguageCode,
            detectLanguage: false,
            wordTimestamps: false
        )
        let results = try await whisperKit.transcribe(
            audioPath: url.path,
            decodeOptions: options
        )
        return results.map(\.text).joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func removeDownloadedModel() throws {
        // Core ML may retain an operating-system specialization cache; Mimi
        // removes every app-owned model weight and tokenizer it manages.
        whisperKit = nil
        guard FileManager.default.fileExists(atPath: modelFolder.path) else { return }
        try FileManager.default.removeItem(at: modelFolder)
    }

    private func loadInstalledModel() async throws {
        try ensureInstalled()
        if whisperKit == nil {
            whisperKit = try await WhisperKit(makeConfiguration(download: false))
        }
    }

    private func makeConfiguration(download: Bool) -> WhisperKitConfig {
        WhisperKitConfig(
            model: modelName,
            modelFolder: modelFolder.path,
            prewarm: true,
            download: download
        )
    }
}

private enum WhisperModelError: LocalizedError {
    case notInstalled
    case installMarkerFailed

    var errorDescription: String? {
        switch self {
        case .notInstalled:
            "Download Whisper Large-v3 (626 MB) explicitly before starting an accuracy-pass recording."
        case .installMarkerFailed:
            "Whisper downloaded, but Mimi could not mark the local model as ready. Check available disk space and try again."
        }
    }
}
