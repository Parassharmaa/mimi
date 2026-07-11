import Foundation
import MimiCore
import MimiSession
@preconcurrency import WhisperKit

@MainActor
final class WhisperKitAccuracyEngine {
    /// Argmax's compressed multilingual Large-v3 Core ML artifact. This is not
    /// labelled Turbo because it is not the separate full Turbo artifact.
    private let modelName = "large-v3-v20240930_626MB"
    private let modelCacheFolder: URL
    private let installMarkerURL: URL
    private var whisperKit: WhisperKit?

    init(fileManager: FileManager = .default) {
        let support = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? fileManager.temporaryDirectory
        modelCacheFolder = support.appending(path: "Mimi/Models/WhisperKit", directoryHint: .isDirectory)
        installMarkerURL = support.appending(path: "Mimi/Models/WhisperKit/.mimi-large-v3-installed")
    }

    var isDownloaded: Bool {
        (try? installedModelFolder()) != nil
    }

    func ensureInstalled() throws {
        _ = try installedModelFolder()
    }

    func install(
        onProgress: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void
    ) async throws {
        try Task.checkCancellation()
        if isDownloaded {
            try await loadInstalledModel()
            try Task.checkCancellation()
            return
        }

        // WhisperKit treats a non-nil `modelFolder` as an already-local model
        // and skips downloading. Download to our cache base first, then load
        // the resolved snapshot folder with downloads disabled.
        whisperKit = nil
        try FileManager.default.createDirectory(at: modelCacheFolder, withIntermediateDirectories: true)
        let progressRelay = WhisperDownloadProgressRelay(onProgress)
        let downloadedFolder = try await Self.downloadModel(
            named: modelName,
            to: modelCacheFolder,
            progressRelay: progressRelay
        )
        // A cancelled download can still return a reusable partial/full
        // snapshot from the underlying hub. Keep those files for retry, but
        // never mark the model ready unless prewarm also completes.
        try Task.checkCancellation()
        let loadedWhisperKit = try await WhisperKit(makeConfiguration(modelFolder: downloadedFolder))
        try Task.checkCancellation()
        do {
            try Data(downloadedFolder.path.utf8).write(to: installMarkerURL, options: .atomic)
        } catch {
            whisperKit = nil
            throw WhisperModelError.installMarkerFailed
        }
        do {
            try Task.checkCancellation()
        } catch {
            // The snapshot remains cached for a later retry; only the marker
            // is removed so this cancelled task is never treated as ready.
            try? FileManager.default.removeItem(at: installMarkerURL)
            whisperKit = nil
            throw error
        }
        whisperKit = loadedWhisperKit
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

    func removeDownloadedModel() async throws {
        // Core ML may retain an operating-system specialization cache; Mimi
        // removes every app-owned model weight and tokenizer it manages.
        whisperKit = nil
        guard FileManager.default.fileExists(atPath: modelCacheFolder.path) else { return }
        try FileManager.default.removeItem(at: modelCacheFolder)
    }

    private func loadInstalledModel() async throws {
        let folder = try installedModelFolder()
        if whisperKit == nil {
            whisperKit = try await WhisperKit(makeConfiguration(modelFolder: folder))
        }
    }

    private func installedModelFolder() throws -> URL {
        guard FileManager.default.fileExists(atPath: installMarkerURL.path),
              let data = try? Data(contentsOf: installMarkerURL),
              let path = String(data: data, encoding: .utf8),
              !path.isEmpty else {
            throw WhisperModelError.notInstalled
        }

        let folder = URL(fileURLWithPath: path).standardizedFileURL
        let cachePath = modelCacheFolder.standardizedFileURL.path + "/"
        guard folder.path.hasPrefix(cachePath),
              FileManager.default.fileExists(atPath: folder.path) else {
            throw WhisperModelError.notInstalled
        }
        return folder
    }

    private func makeConfiguration(modelFolder: URL) -> WhisperKitConfig {
        WhisperKitConfig(
            model: modelName,
            modelFolder: modelFolder.path,
            prewarm: true,
            download: false
        )
    }

    nonisolated private static func downloadModel(
        named modelName: String,
        to modelCacheFolder: URL,
        progressRelay: WhisperDownloadProgressRelay
    ) async throws -> URL {
        try await WhisperKit.download(
            variant: modelName,
            downloadBase: modelCacheFolder,
            progressCallback: { progress in
                progressRelay.report(progress)
            }
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

/// WhisperKit reports Foundation `Progress` from a downloader callback that
/// is not actor-isolated. Copy its scalar values before hopping to Mimi's
/// main-actor session state; never pass `Progress` itself across executors.
private final class WhisperDownloadProgressRelay: @unchecked Sendable {
    private let callback: @MainActor @Sendable (ModelDownloadProgress) -> Void

    init(_ callback: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void) {
        self.callback = callback
    }

    func report(_ progress: Progress) {
        let snapshot = ModelDownloadProgress(
            completedUnitCount: progress.completedUnitCount,
            totalUnitCount: progress.totalUnitCount
        )
        Task { @MainActor [callback] in
            callback(snapshot)
        }
    }
}
