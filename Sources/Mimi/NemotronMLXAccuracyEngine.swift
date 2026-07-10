import Foundation
import HuggingFace
import MLXAudioCore
import MLXAudioSTT
import MimiCore
import MimiSession

/// Mimi's native Swift/MLX implementation of the optional Nemotron 3.5 ASR
/// model. The weights are pinned, downloaded only by an explicit action, and
/// loaded directly from Mimi's Application Support cache—there is no Python,
/// helper daemon, or network request during transcription.
@MainActor
final class NemotronMLXAccuracyEngine: NemotronMLXAccuracyTranscribing {
    private static let repository = "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
    private static let revision = "7279359e4481b5e9e185a318bd618e429c6d86cd"
    private static let requiredModelFiles = ["config.json", "model.safetensors", "tokenizer.model", "vocab.txt"]

    private let fileManager: FileManager
    private let rootURL: URL
    private let runtime = NativeNemotronRuntime()

    init(fileManager: FileManager = .default, rootURL: URL? = nil) {
        self.fileManager = fileManager
        if let rootURL {
            self.rootURL = rootURL
            return
        }

        if let override = ProcessInfo.processInfo.environment["MIMI_NEMOTRON_HOME"], !override.isEmpty {
            self.rootURL = URL(fileURLWithPath: override, isDirectory: true)
            return
        }

        let support = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? fileManager.temporaryDirectory
        self.rootURL = support.appending(path: "Mimi/Models/NemotronMLX", directoryHint: .isDirectory)
    }

    var isDownloaded: Bool {
        (try? installedModelDirectory()) != nil
    }

    /// MLX Swift deliberately keeps its Metal shaders outside the linked
    /// library. The app bundle must ship the matching shader file; checking it
    /// before touching MLX turns a packaging mistake into a useful UI state
    /// instead of a failed native-model launch.
    var runtimeAvailabilityMessage: String? {
#if arch(arm64)
        guard metalLibraryURL != nil else {
            return "Nemotron MLX needs Mimi's bundled Metal runtime. Reinstall an Apple-silicon build made with full Xcode."
        }
        return nil
#else
        return "Nemotron MLX requires an Apple-silicon Mac."
#endif
    }

    func ensureInstalled() throws {
        try ensureRuntimeAvailable()
        _ = try installedModelDirectory()
    }

    func install() async throws {
        try ensureRuntimeAvailable()
        if let modelDirectory = try? installedModelDirectory() {
            try await runtime.load(modelDirectory: modelDirectory)
            return
        }

        guard let repositoryID = Repo.ID(rawValue: Self.repository) else {
            throw NemotronMLXError.invalidRepository
        }

        try fileManager.createDirectory(at: modelCacheURL, withIntermediateDirectories: true)
        let cache = HubCache(cacheDirectory: modelCacheURL)
        let client = HubClient(cache: cache)
        let snapshotDirectory = try await client.downloadSnapshot(
            of: repositoryID,
            kind: .model,
            revision: Self.revision,
            matching: ["*.safetensors", "*.json", "*.model", "*.txt"],
            maxConcurrentDownloads: 2
        )
        try validateModelDirectory(snapshotDirectory)
        try writeInstalledModelMarker(for: snapshotDirectory)
        try await runtime.load(modelDirectory: snapshotDirectory)
    }

    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String {
        try ensureRuntimeAvailable()
        let modelDirectory = try installedModelDirectory()
        try await runtime.load(modelDirectory: modelDirectory)
        return try await runtime.transcribe(recordingAt: url, language: language.rawValue)
    }

    func removeDownloadedModel() async throws {
        await runtime.unload()
        guard fileManager.fileExists(atPath: rootURL.path) else { return }
        try fileManager.removeItem(at: rootURL)
    }

    private var modelCacheURL: URL {
        rootURL.appending(path: "huggingface-cache", directoryHint: .isDirectory)
    }

    private var installMarkerURL: URL {
        rootURL.appending(path: "mimi-nemotron-installed.json")
    }

    private func installedModelDirectory() throws -> URL {
        guard let data = try? Data(contentsOf: installMarkerURL),
              let marker = try? JSONDecoder().decode(InstalledModelMarker.self, from: data),
              marker.repository == Self.repository,
              marker.revision == Self.revision else {
            throw NemotronMLXError.notInstalled
        }

        let directory = URL(fileURLWithPath: marker.modelDirectory, isDirectory: true).standardizedFileURL
        let cachePrefix = modelCacheURL.standardizedFileURL.path + "/"
        guard directory.path.hasPrefix(cachePrefix) else {
            throw NemotronMLXError.notInstalled
        }
        try validateModelDirectory(directory)
        return directory
    }

    private func validateModelDirectory(_ directory: URL) throws {
        let missing = Self.requiredModelFiles.filter {
            !fileManager.fileExists(atPath: directory.appending(path: $0).path)
        }
        guard missing.isEmpty else { throw NemotronMLXError.incompleteModel(missing) }
    }

    private func writeInstalledModelMarker(for directory: URL) throws {
        let marker = InstalledModelMarker(
            repository: Self.repository,
            revision: Self.revision,
            modelDirectory: directory.standardizedFileURL.path
        )
        try fileManager.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(marker)
        try data.write(to: installMarkerURL, options: .atomic)
    }

    private var metalLibraryURL: URL? {
        var candidates: [URL] = []
        if let executablePath = CommandLine.arguments.first, !executablePath.isEmpty {
            let executableDirectory = URL(fileURLWithPath: executablePath).deletingLastPathComponent()
            candidates.append(executableDirectory.appending(path: "mlx.metallib"))
            candidates.append(executableDirectory.appending(path: "Resources/mlx.metallib"))
        }
        return candidates.first { fileManager.fileExists(atPath: $0.path) }
    }

    private func ensureRuntimeAvailable() throws {
        if let message = runtimeAvailabilityMessage {
            throw NemotronMLXError.runtimeUnavailable(message)
        }
    }
}

private actor NativeNemotronRuntime {
    private var model: NemotronASRModel?
    private var loadedDirectory: URL?

    func load(modelDirectory: URL) throws {
        let standardizedDirectory = modelDirectory.standardizedFileURL
        if loadedDirectory == standardizedDirectory, model != nil { return }
        model = try NemotronASRModel.fromDirectory(standardizedDirectory)
        loadedDirectory = standardizedDirectory
    }

    func transcribe(recordingAt url: URL, language: String) throws -> String {
        guard let model else { throw NemotronMLXError.notInstalled }
        let (_, audio) = try loadAudioArray(
            from: url,
            sampleRate: model.preprocessConfig.sampleRate
        )
        let result = model.generate(
            audio: audio,
            generationParameters: .init(
                language: language,
                // This is an accuracy pass after Stop. The model's streaming
                // engine is available for a later bounded live-session path.
                chunkDuration: 120.0
            )
        )
        return result.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func unload() {
        model = nil
        loadedDirectory = nil
    }
}

private struct InstalledModelMarker: Codable {
    let repository: String
    let revision: String
    let modelDirectory: String
}

private enum NemotronMLXError: LocalizedError {
    case notInstalled
    case invalidRepository
    case incompleteModel([String])
    case runtimeUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .notInstalled:
            "Download the local Nemotron MLX pack before starting an accuracy-pass recording."
        case .invalidRepository:
            "Mimi could not identify the pinned Nemotron MLX model repository."
        case let .incompleteModel(files):
            "Nemotron's local download is incomplete (missing \(files.joined(separator: ", "))). Remove it and download again."
        case let .runtimeUnavailable(message):
            message
        }
    }
}
