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
    private let benchmarkModelFolderOverride: URL?
    private var whisperKit: WhisperKit?

    init(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        let support = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? fileManager.temporaryDirectory
        modelCacheFolder = support.appending(path: "Mimi/Models/WhisperKit", directoryHint: .isDirectory)
        installMarkerURL = support.appending(path: "Mimi/Models/WhisperKit/.mimi-large-v3-installed")
        benchmarkModelFolderOverride = environment["MIMI_WHISPER_MODEL_DIR"].map {
            URL(fileURLWithPath: $0, isDirectory: true).standardizedFileURL
        }
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
        if benchmarkModelFolderOverride != nil {
            throw WhisperModelError.externalBenchmarkModelCannotBeRemoved
        }
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
        if let benchmarkModelFolderOverride,
           FileManager.default.fileExists(atPath: benchmarkModelFolderOverride.path) {
            return benchmarkModelFolderOverride
        }

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

    func runRollingBenchmark(
        recordingAt url: URL,
        language: SpeechLanguage,
        stepSeconds: Double
    ) async throws -> RealtimeBenchmarkReport {
        let loadStartedAt = ContinuousClock.now
        try await loadInstalledModel()
        let modelLoadSeconds = loadStartedAt.duration(to: .now).seconds
        guard let whisperKit else { throw WhisperModelError.notInstalled }

        let samples = try AudioProcessor.loadAudioAsFloatArray(fromPath: url.path)
        let sampleRate = Double(WhisperKit.sampleRate)
        let audioDuration = Double(samples.count) / sampleRate
        let stepSampleCount = max(1, Int(stepSeconds * sampleRate))
        var updates: [String] = []
        var decodeDurations: [Double] = []
        var firstTextAt: Double?
        let runStartedAt = ContinuousClock.now

        var endSample = min(stepSampleCount, samples.count)
        while endSample > 0, endSample <= samples.count {
            try Task.checkCancellation()
            let prefix = Array(samples[..<endSample])
            let prefixDuration = Double(endSample) / sampleRate
            let decodeStartedAt = ContinuousClock.now
            let options = DecodingOptions(
                language: language.whisperLanguageCode,
                detectLanguage: false,
                wordTimestamps: true
            )
            let results = try await whisperKit.transcribe(
                audioArray: prefix,
                decodeOptions: options
            )
            let decodeSeconds = decodeStartedAt.duration(to: .now).seconds
            decodeDurations.append(decodeSeconds)
            let text = results.map(\.text).joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                updates.append(text)
                if firstTextAt == nil {
                    firstTextAt = prefixDuration + decodeSeconds
                }
            }

            if endSample == samples.count { break }
            endSample = min(samples.count, endSample + stepSampleCount)
        }

        let wallSeconds = runStartedAt.duration(to: .now).seconds
        return RealtimeBenchmarkReport(
            engine: "whisperkit",
            mode: "rolling-\(String(format: "%.2f", stepSeconds))s",
            language: language.rawValue,
            audioDurationSeconds: audioDuration,
            wallSeconds: wallSeconds,
            modelLoadSeconds: modelLoadSeconds,
            firstTextAtSeconds: firstTextAt,
            firstFinalAtSeconds: nil,
            updateCount: updates.count,
            meanDecodeSeconds: decodeDurations.isEmpty ? nil : decodeDurations.reduce(0, +) / Double(decodeDurations.count),
            maxDecodeSeconds: decodeDurations.max(),
            realTimeFactor: audioDuration > 0 ? decodeDurations.reduce(0, +) / audioDuration : nil,
            hypothesisChurn: RealtimeBenchmarkReport.hypothesisChurn(updates),
            finalText: updates.last ?? "",
            firstUpdates: Array(updates.prefix(8))
        )
    }

    func runLanguageDetectionBenchmark(
        recordingAt url: URL,
        windowSeconds: [Double] = [1, 2, 3]
    ) async throws -> AcousticLanguageIDReport {
        let loadStartedAt = ContinuousClock.now
        try await loadInstalledModel()
        let modelLoadSeconds = loadStartedAt.duration(to: .now).seconds
        guard let whisperKit else { throw WhisperModelError.notInstalled }

        let samples = try AudioProcessor.loadAudioAsFloatArray(fromPath: url.path)
        let sampleRate = Double(WhisperKit.sampleRate)
        let duration = Double(samples.count) / sampleRate
        var windows: [AcousticLanguageIDWindow] = []
        for requestedSeconds in windowSeconds where requestedSeconds <= duration + 0.05 {
            let sampleCount = min(samples.count, max(1, Int(requestedSeconds * sampleRate)))
            let startedAt = ContinuousClock.now
            let result = try await whisperKit.detectLangauge(audioArray: Array(samples[..<sampleCount]))
            windows.append(.init(
                audioSeconds: Double(sampleCount) / sampleRate,
                decodeSeconds: startedAt.duration(to: .now).seconds,
                detectedLanguage: result.language,
                englishScore: result.langProbs["en"],
                japaneseScore: result.langProbs["ja"]
            ))
        }
        return .init(
            model: modelName,
            modelLoadSeconds: modelLoadSeconds,
            audioDurationSeconds: duration,
            windows: windows
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

struct AcousticLanguageIDReport: Codable {
    let model: String
    let modelLoadSeconds: Double
    let audioDurationSeconds: Double
    let windows: [AcousticLanguageIDWindow]

    func printJSON() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        guard let text = String(data: data, encoding: .utf8) else {
            throw WhisperModelError.languageReportEncodingFailed
        }
        print(text)
    }
}

struct AcousticLanguageIDWindow: Codable {
    let audioSeconds: Double
    let decodeSeconds: Double
    let detectedLanguage: String
    let englishScore: Float?
    let japaneseScore: Float?
}

private enum WhisperModelError: LocalizedError {
    case notInstalled
    case installMarkerFailed
    case externalBenchmarkModelCannotBeRemoved
    case languageReportEncodingFailed

    var errorDescription: String? {
        switch self {
        case .notInstalled:
            "Download Whisper Large-v3 (626 MB) explicitly before starting an accuracy-pass recording."
        case .installMarkerFailed:
            "Whisper downloaded, but Mimi could not mark the local model as ready. Check available disk space and try again."
        case .externalBenchmarkModelCannotBeRemoved:
            "Mimi will not remove a model supplied through the developer benchmark override."
        case .languageReportEncodingFailed:
            "Mimi could not encode the acoustic language-identification benchmark report."
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
