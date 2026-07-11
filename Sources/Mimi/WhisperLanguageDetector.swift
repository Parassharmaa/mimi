import Foundation
import MimiSession
@preconcurrency import WhisperKit

struct AcousticLanguageDecision: Sendable {
    let language: String
    let scores: [String: Float]
}

/// A small multilingual Whisper encoder/decoder used only for acoustic
/// language identification. It never produces transcript text and is kept in
/// a separate removable cache from Mimi's Large-v3 accuracy model.
@MainActor
final class WhisperLanguageDetector {
    static let modelName = "tiny"

    private let cacheFolder: URL
    private let markerURL: URL
    private var whisperKit: WhisperKit?

    init(fileManager: FileManager = .default) {
        let support = (try? MimiStorage.applicationDirectory(fileManager: fileManager))
            ?? fileManager.temporaryDirectory.appending(path: "Mimi", directoryHint: .isDirectory)
        cacheFolder = support.appending(path: "Models/LanguageID", directoryHint: .isDirectory)
        markerURL = cacheFolder.appending(path: ".mimi-whisper-tiny-language-id")
    }

    var isDownloaded: Bool { (try? installedFolder()) != nil }

    func install(
        onProgress: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void = { _ in }
    ) async throws {
        if isDownloaded {
            try await loadInstalledModel()
            return
        }
        try FileManager.default.createDirectory(at: cacheFolder, withIntermediateDirectories: true)
        let relay = LanguageDetectorProgressRelay(onProgress)
        let folder = try await Self.downloadModel(to: cacheFolder, relay: relay)
        try Task.checkCancellation()
        whisperKit = try await WhisperKit(.init(
            model: Self.modelName,
            modelFolder: folder.path,
            prewarm: true,
            download: false
        ))
        try Data(folder.path.utf8).write(to: markerURL, options: .atomic)
    }

    func detect(samples: [Float]) async throws -> AcousticLanguageDecision {
        try await loadInstalledModel()
        guard let whisperKit else { throw WhisperLanguageDetectorError.notInstalled }
        let result = try await whisperKit.detectLangauge(audioArray: samples)
        return .init(language: result.language, scores: result.langProbs)
    }

    func runBenchmark(
        recordingAt url: URL,
        windowSeconds: [Double] = [1, 2, 3]
    ) async throws -> AcousticLanguageIDReport {
        let loadStartedAt = ContinuousClock.now
        try await loadInstalledModel()
        let modelLoadSeconds = loadStartedAt.duration(to: .now).seconds
        let samples = try AudioProcessor.loadAudioAsFloatArray(fromPath: url.path)
        let sampleRate = Double(WhisperKit.sampleRate)
        let duration = Double(samples.count) / sampleRate
        var windows: [AcousticLanguageIDWindow] = []
        for requestedSeconds in windowSeconds where requestedSeconds <= duration + 0.05 {
            let count = min(samples.count, max(1, Int(requestedSeconds * sampleRate)))
            let startedAt = ContinuousClock.now
            let decision = try await detect(samples: Array(samples[..<count]))
            windows.append(.init(
                audioSeconds: Double(count) / sampleRate,
                decodeSeconds: startedAt.duration(to: .now).seconds,
                detectedLanguage: decision.language,
                englishScore: decision.scores["en"],
                japaneseScore: decision.scores["ja"]
            ))
        }
        return .init(
            model: Self.modelName,
            modelLoadSeconds: modelLoadSeconds,
            audioDurationSeconds: duration,
            windows: windows
        )
    }

    func removeDownloadedModel() throws {
        whisperKit = nil
        guard FileManager.default.fileExists(atPath: cacheFolder.path) else { return }
        try FileManager.default.removeItem(at: cacheFolder)
    }

    private func loadInstalledModel() async throws {
        guard whisperKit == nil else { return }
        let folder = try installedFolder()
        whisperKit = try await WhisperKit(.init(
            model: Self.modelName,
            modelFolder: folder.path,
            prewarm: true,
            download: false
        ))
    }

    private func installedFolder() throws -> URL {
        guard let data = try? Data(contentsOf: markerURL),
              let path = String(data: data, encoding: .utf8),
              !path.isEmpty else {
            throw WhisperLanguageDetectorError.notInstalled
        }
        let folder = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
        guard FileManager.default.fileExists(atPath: folder.path) else {
            throw WhisperLanguageDetectorError.notInstalled
        }
        return folder
    }

    nonisolated private static func downloadModel(
        to cacheFolder: URL,
        relay: LanguageDetectorProgressRelay
    ) async throws -> URL {
        try await WhisperKit.download(
            variant: Self.modelName,
            downloadBase: cacheFolder,
            progressCallback: { relay.report($0) }
        )
    }
}

private final class LanguageDetectorProgressRelay: @unchecked Sendable {
    private let callback: @MainActor @Sendable (ModelDownloadProgress) -> Void

    init(_ callback: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void) {
        self.callback = callback
    }

    func report(_ progress: Progress) {
        let snapshot = ModelDownloadProgress(
            completedUnitCount: progress.completedUnitCount,
            totalUnitCount: progress.totalUnitCount
        )
        Task { @MainActor [callback] in callback(snapshot) }
    }
}

private enum WhisperLanguageDetectorError: LocalizedError {
    case notInstalled

    var errorDescription: String? {
        "Download Mimi's small multilingual language detector before using Auto language mode."
    }
}
