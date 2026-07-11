@preconcurrency import AVFoundation
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
final class NemotronMLXLiveEngine: NemotronMLXLiveTranscribing {
    private static let repository = "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
    private static let revision = "7279359e4481b5e9e185a318bd618e429c6d86cd"
    private static let requiredModelFiles = ["config.json", "model.safetensors", "tokenizer.model", "vocab.txt"]

    private let fileManager: FileManager
    private let rootURL: URL
    private let runtime = NativeNemotronRuntime()
    private var audioConverter: AVAudioConverter?
    private var normalizedFormat: AVAudioFormat?
    private var pendingSamples: BoundedAudioSampleQueue
    private var liveEvent: (@MainActor (TranscriptEvent) -> Void)?
    private var liveBackpressure: (@MainActor (String) -> Void)?
    private var liveSessionID: UUID?
    private var liveDrainTask: Task<Void, Never>?
    private var isStoppingLiveSession = false
    private var hasReportedBackpressure = false

    /// Nemotron's pinned streaming implementation intentionally favors exact
    /// short-utterance output: it retains the raw buffer and recomputes mel
    /// from it on each step. Keep each internal session bounded so a long
    /// meeting cannot grow memory/CPU without limit. A silence boundary is
    /// preferred; the hard cap is a privacy/performance safety valve.
    private static let streamChunkSamples = 16_000 * 560 / 1_000
    /// A slow local model must not turn a long meeting into unbounded pending
    /// PCM. Six seconds lets a brief inference spike recover while ensuring
    /// Stop only needs to flush a bounded amount of audio.
    private static let maximumPendingSamples = 16_000 * 6

    init(fileManager: FileManager = .default, rootURL: URL? = nil) {
        self.fileManager = fileManager
        pendingSamples = BoundedAudioSampleQueue(
            maximumSampleCount: Self.maximumPendingSamples,
            preferredChunkSize: Self.streamChunkSamples
        )
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

    func startLive(
        language: SpeechLanguage,
        inputFormat: AVAudioFormat,
        onEvent: @escaping @MainActor (TranscriptEvent) -> Void,
        onBackpressure: @escaping @MainActor (String) -> Void
    ) async throws {
        try ensureRuntimeAvailable()
        let modelDirectory = try installedModelDirectory()
        guard let normalizedFormat = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1),
              let converter = AVAudioConverter(from: inputFormat, to: normalizedFormat) else {
            throw NemotronMLXError.noCompatibleLiveAudioFormat
        }

        await cancelLive()
        try await runtime.load(modelDirectory: modelDirectory)
        await runtime.startLive(language: language.rawValue)

        audioConverter = converter
        self.normalizedFormat = normalizedFormat
        pendingSamples.removeAll(keepingCapacity: true)
        liveEvent = onEvent
        liveBackpressure = onBackpressure
        liveSessionID = UUID()
        isStoppingLiveSession = false
        hasReportedBackpressure = false
    }

    func consumeLive(_ buffer: AVAudioPCMBuffer) {
        guard let sessionID = liveSessionID,
              !isStoppingLiveSession,
              let samples = normalizedSamples(from: buffer),
              !samples.isEmpty else {
            return
        }

        if pendingSamples.append(samples) > 0 {
            reportBackpressureIfNeeded()
        }
        scheduleLiveDrain(for: sessionID)
    }

    func stopLive() async {
        guard let sessionID = liveSessionID else { return }
        isStoppingLiveSession = true

        if let liveDrainTask {
            await liveDrainTask.value
        }
        await drainLiveSamples(flush: true, sessionID: sessionID)

        if let update = await runtime.finishLive() {
            publish(update, for: sessionID)
        }
        resetLiveState()
    }

    func cancelLive() async {
        let task = liveDrainTask
        liveSessionID = nil
        liveDrainTask = nil
        task?.cancel()
        resetLiveState()
        await runtime.cancelLive()
    }

    func removeDownloadedModel() async throws {
        await cancelLive()
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
        // Explicit developer/CI fixture override. Normal app installs always
        // use the revision-pinned marker below; this path exists so the
        // opt-in direct-executable smoke can exercise conversion and lifecycle
        // against already-downloaded weights without copying them into
        // Application Support or triggering a download. A sandboxed release
        // app cannot read an arbitrary external path and therefore continues
        // to use the marker-backed app-owned cache below.
        if let override = ProcessInfo.processInfo.environment["MIMI_NEMOTRON_MODEL_DIR"], !override.isEmpty {
            let directory = URL(fileURLWithPath: override, isDirectory: true).standardizedFileURL
            try validateModelDirectory(directory)
            return directory
        }

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

    private func scheduleLiveDrain(for sessionID: UUID) {
        guard liveDrainTask == nil,
              pendingSamples.count >= Self.streamChunkSamples,
              !isStoppingLiveSession else {
            return
        }

        liveDrainTask = Task { [weak self] in
            guard let self else { return }
            await self.drainLiveSamples(flush: false, sessionID: sessionID)
        }
    }

    private func drainLiveSamples(flush: Bool, sessionID: UUID) async {
        defer {
            if liveSessionID == sessionID {
                liveDrainTask = nil
                if !isStoppingLiveSession {
                    scheduleLiveDrain(for: sessionID)
                }
            }
        }

        while liveSessionID == sessionID {
            let sampleCount: Int
            if pendingSamples.count >= Self.streamChunkSamples {
                sampleCount = Self.streamChunkSamples
            } else if flush, !pendingSamples.isEmpty {
                sampleCount = pendingSamples.count
            } else {
                return
            }

            let samples = pendingSamples.dequeue(upTo: sampleCount)
            if let update = await runtime.appendLive(samples: samples) {
                publish(update, for: sessionID)
            }
        }
    }

    private func publish(_ update: NativeNemotronLiveUpdate, for sessionID: UUID) {
        guard liveSessionID == sessionID else { return }
        if let finalizedText = update.finalizedText {
            liveEvent?(.final(finalizedText))
        } else if let provisionalText = update.provisionalText {
            liveEvent?(.partial(provisionalText))
        }
    }

    private func resetLiveState() {
        audioConverter = nil
        normalizedFormat = nil
        pendingSamples.removeAll(keepingCapacity: false)
        liveEvent = nil
        liveBackpressure = nil
        liveSessionID = nil
        liveDrainTask = nil
        isStoppingLiveSession = false
        hasReportedBackpressure = false
    }

    private func normalizedSamples(from buffer: AVAudioPCMBuffer) -> [Float]? {
        guard let audioConverter, let normalizedFormat else { return nil }

        let ratio = normalizedFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(max(1, Double(buffer.frameLength) * ratio + 2))
        guard let convertedBuffer = AVAudioPCMBuffer(
            pcmFormat: normalizedFormat,
            frameCapacity: capacity
        ) else {
            return nil
        }

        let input = NemotronAudioConverterInput(buffer: buffer)
        var conversionError: NSError?
        let status = audioConverter.convert(to: convertedBuffer, error: &conversionError) { _, outputStatus in
            input.next(outputStatus)
        }
        guard status != .error,
              conversionError == nil,
              convertedBuffer.frameLength > 0,
              let samples = convertedBuffer.floatChannelData?[0] else {
            return nil
        }
        return Array(UnsafeBufferPointer(start: samples, count: Int(convertedBuffer.frameLength)))
    }

    private func reportBackpressureIfNeeded() {
        guard !hasReportedBackpressure else { return }
        hasReportedBackpressure = true
        liveBackpressure?("Nemotron MLX is slower than this audio source. Mimi skipped some queued audio to stay live; try Apple Speech or Whisper for this session.")
    }
}

private actor NativeNemotronRuntime {
    private var model: NemotronASRModel?
    private var loadedDirectory: URL?
    private var liveSession: NemotronASRStreamSession?
    private var liveLanguage: String?
    private var liveWindowPolicy = BoundedLiveWindowPolicy()

    func load(modelDirectory: URL) throws {
        let standardizedDirectory = modelDirectory.standardizedFileURL
        if loadedDirectory == standardizedDirectory, model != nil { return }
        model = try NemotronASRModel.fromDirectory(standardizedDirectory)
        loadedDirectory = standardizedDirectory
        clearLiveSession()
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
                // This remains available for fixture comparisons and future
                // offline accuracy work. Mimi's interactive path uses the
                // bounded native streaming session below.
                chunkDuration: 120.0
            )
        )
        return result.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func startLive(language: String) {
        guard let model else { return }
        liveSession = model.makeStreamSession(language: language, chunkMs: 560)
        liveLanguage = language
        liveWindowPolicy.reset()
    }

    func appendLive(samples: [Float]) -> NativeNemotronLiveUpdate? {
        guard let model, let liveSession, !samples.isEmpty else { return nil }

        let boundary = liveWindowPolicy.append(samples)

        _ = liveSession.step(samples)
        guard boundary != .none else {
            let provisionalText = liveSession.text.trimmingCharacters(in: .whitespacesAndNewlines)
            return NativeNemotronLiveUpdate(
                provisionalText: provisionalText.isEmpty ? nil : provisionalText,
                finalizedText: nil
            )
        }

        _ = liveSession.finish()
        let finalizedText = liveSession.text.trimmingCharacters(in: .whitespacesAndNewlines)
        self.liveSession = model.makeStreamSession(language: liveLanguage, chunkMs: 560)
        liveWindowPolicy.reset()
        return NativeNemotronLiveUpdate(provisionalText: nil, finalizedText: finalizedText)
    }

    func finishLive() -> NativeNemotronLiveUpdate? {
        guard let liveSession else { return nil }
        _ = liveSession.finish()
        let finalizedText = liveSession.text.trimmingCharacters(in: .whitespacesAndNewlines)
        clearLiveSession()
        return NativeNemotronLiveUpdate(provisionalText: nil, finalizedText: finalizedText)
    }

    func cancelLive() {
        clearLiveSession()
    }

    func unload() {
        clearLiveSession()
        model = nil
        loadedDirectory = nil
    }

    private func clearLiveSession() {
        liveSession = nil
        liveLanguage = nil
        liveWindowPolicy.reset()
    }
}

private struct NativeNemotronLiveUpdate: Sendable {
    let provisionalText: String?
    /// An empty string is meaningful: it clears a stale live hypothesis when
    /// a bounded window contained only silence.
    let finalizedText: String?
}

private final class NemotronAudioConverterInput: @unchecked Sendable {
    private let buffer: AVAudioPCMBuffer
    private let lock = NSLock()
    private var wasProvided = false

    init(buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }

    func next(_ outputStatus: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
        lock.lock()
        defer { lock.unlock() }
        guard !wasProvided else {
            outputStatus.pointee = .noDataNow
            return nil
        }
        wasProvided = true
        outputStatus.pointee = .haveData
        return buffer
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
    case noCompatibleLiveAudioFormat

    var errorDescription: String? {
        switch self {
        case .notInstalled:
            "Download the local Nemotron MLX pack before starting live transcription."
        case .invalidRepository:
            "Mimi could not identify the pinned Nemotron MLX model repository."
        case let .incompleteModel(files):
            "Nemotron's local download is incomplete (missing \(files.joined(separator: ", "))). Remove it and download again."
        case let .runtimeUnavailable(message):
            message
        case .noCompatibleLiveAudioFormat:
            "Nemotron MLX could not convert this selected audio source to its local 16 kHz format. Choose another input or use Whisper Large-v3."
        }
    }
}
