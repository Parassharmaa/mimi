@preconcurrency import AVFoundation
import Foundation
import HuggingFace
import MLXAudioCore
import MLXAudioSTT
import MimiCore
import MimiSession

/// Qwen3-ASR's native MLX streaming path. The package session intentionally
/// combines a frequent provisional decoder with agreement-based confirmation
/// and a higher-accuracy pass when each cached encoder window completes.
@MainActor
final class QwenMLXLiveEngine: QwenMLXLiveTranscribing {
    private static let repository = "mlx-community/Qwen3-ASR-0.6B-4bit"
    private static let revision = "313d850181767edf09f00a9c289becca70e58cd0"
    private static let requiredModelFiles = [
        "config.json", "model.safetensors", "merges.txt", "vocab.json",
        "tokenizer_config.json", "preprocessor_config.json"
    ]
    private static let feedChunkSamples = 16_000 / 5 // 200 ms
    private static let maximumPendingSamples = 16_000 * 4

    private let fileManager: FileManager
    private let rootURL: URL
    private let modelDirectory: URL
    private let runtime = NativeQwenRuntime()
    private var audioConverter: AVAudioConverter?
    private var normalizedFormat: AVAudioFormat?
    private var pendingSamples = BoundedAudioSampleQueue(
        maximumSampleCount: maximumPendingSamples,
        preferredChunkSize: feedChunkSamples
    )
    private var liveEvent: (@MainActor (TranscriptEvent) -> Void)?
    private var liveBackpressure: (@MainActor (String) -> Void)?
    private var liveSessionID: UUID?
    private var liveDrainTask: Task<Void, Never>?
    private var liveEventTask: Task<Void, Never>?
    private var displayPublishTask: Task<Void, Never>?
    private var pendingDisplayText = ""
    private var lastPublishedDisplayText = ""
    private var isStopping = false
    private var hasReportedBackpressure = false

    init(fileManager: FileManager = .default, rootURL: URL? = nil, modelDirectory: URL? = nil) {
        self.fileManager = fileManager
        let support = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? fileManager.temporaryDirectory
        self.rootURL = rootURL ?? support.appending(
            path: "Mimi/Models/Qwen3ASRMLX",
            directoryHint: .isDirectory
        )

        if let modelDirectory {
            self.modelDirectory = modelDirectory
        } else if let override = ProcessInfo.processInfo.environment["MIMI_QWEN_MODEL_DIR"], !override.isEmpty {
            self.modelDirectory = URL(fileURLWithPath: override, isDirectory: true).standardizedFileURL
        } else {
            let cache = HubCache.default.cacheDirectory
            self.modelDirectory = cache
                .appending(path: "mlx-audio", directoryHint: .isDirectory)
                .appending(path: "mlx-community_Qwen3-ASR-0.6B-4bit", directoryHint: .isDirectory)
        }
    }

    var runtimeAvailabilityMessage: String? {
#if arch(arm64)
        guard metalLibraryURL != nil else {
            return "Qwen3-ASR MLX needs Mimi's bundled Metal runtime. Reinstall an Apple-silicon build made with full Xcode."
        }
        return nil
#else
        return "Qwen3-ASR MLX requires an Apple-silicon Mac."
#endif
    }

    var isDownloaded: Bool {
        (try? installedModelDirectory()) != nil
    }

    func ensureInstalled() throws {
        try ensureRuntimeAvailable()
        _ = try installedModelDirectory()
    }

    func install() async throws {
        try await install(onProgress: { _ in })
    }

    func install(
        onProgress: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void
    ) async throws {
        try ensureRuntimeAvailable()
        if let directory = try? installedModelDirectory() {
            try await runtime.load(modelDirectory: directory)
            return
        }
        guard let repositoryID = Repo.ID(rawValue: Self.repository) else {
            throw QwenMLXError.invalidRepository
        }

        try fileManager.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        let client = HubClient(cache: .default)
        let downloadedDirectory = try await client.downloadSnapshot(
            of: repositoryID,
            kind: .model,
            to: modelDirectory,
            revision: Self.revision,
            matching: ["*.safetensors", "*.json", "*.txt"],
            maxConcurrentDownloads: 2,
            progressHandler: { progress in
                onProgress(.init(
                    completedUnitCount: progress.completedUnitCount,
                    totalUnitCount: progress.totalUnitCount
                ))
            }
        )
        try validateModelDirectory(downloadedDirectory)
        try writeInstallMarker(for: downloadedDirectory)
        try await runtime.load(modelDirectory: downloadedDirectory)
    }

    func startLive(
        language: SpeechLanguage,
        inputFormat: AVAudioFormat,
        onEvent: @escaping @MainActor (TranscriptEvent) -> Void,
        onBackpressure: @escaping @MainActor (String) -> Void
    ) async throws {
        try ensureRuntimeAvailable()
        let directory = try installedModelDirectory()
        guard let normalizedFormat = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1),
              let converter = AVAudioConverter(from: inputFormat, to: normalizedFormat) else {
            throw QwenMLXError.noCompatibleLiveAudioFormat
        }

        await cancelLive()
        try await runtime.load(modelDirectory: directory)
        let events = try await runtime.start(language: language.displayName)
        let sessionID = UUID()
        liveSessionID = sessionID
        liveEvent = onEvent
        liveBackpressure = onBackpressure
        audioConverter = converter
        self.normalizedFormat = normalizedFormat
        pendingSamples.removeAll(keepingCapacity: true)
        isStopping = false
        hasReportedBackpressure = false

        liveEventTask = Task { [weak self] in
            guard let self else { return }
            for await event in events {
                guard liveSessionID == sessionID else { return }
                switch event {
                case let .displayUpdate(confirmedText, provisionalText):
                    let display = [confirmedText, provisionalText]
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }
                        .joined(separator: " ")
                    queueDisplayUpdate(display, for: sessionID)
                case let .ended(fullText):
                    displayPublishTask?.cancel()
                    displayPublishTask = nil
                    liveEvent?(.final(fullText))
                case .provisional, .confirmed, .stats:
                    // displayUpdate is the coalesced, UI-safe stream. Publishing
                    // token events too would duplicate work and create churn.
                    break
                }
            }
        }
    }

    func consumeLive(_ buffer: AVAudioPCMBuffer) {
        guard let sessionID = liveSessionID,
              !isStopping,
              let samples = normalizedSamples(from: buffer),
              !samples.isEmpty else { return }
        if pendingSamples.append(samples) > 0 {
            reportBackpressureIfNeeded()
        }
        scheduleDrain(for: sessionID)
    }

    func stopLive() async {
        guard let sessionID = liveSessionID else { return }
        isStopping = true
        if let liveDrainTask { await liveDrainTask.value }
        await drainSamples(flush: true, sessionID: sessionID)
        let eventTask = liveEventTask
        await runtime.stop()
        await eventTask?.value
        resetLiveState()
    }

    func cancelLive() async {
        liveSessionID = nil
        liveDrainTask?.cancel()
        liveEventTask?.cancel()
        await runtime.cancel()
        resetLiveState()
    }

    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String {
        try ensureRuntimeAvailable()
        let directory = try installedModelDirectory()
        try await runtime.load(modelDirectory: directory)
        return try await runtime.transcribe(recordingAt: url, language: language.displayName)
    }

    func removeDownloadedModel() async throws {
        await cancelLive()
        await runtime.unload()
        if fileManager.fileExists(atPath: modelDirectory.path) {
            try fileManager.removeItem(at: modelDirectory)
        }
        if fileManager.fileExists(atPath: installMarkerURL.path) {
            try fileManager.removeItem(at: installMarkerURL)
        }
    }

    private var installMarkerURL: URL {
        rootURL.appending(path: "mimi-qwen3-asr-installed.json")
    }

    private func installedModelDirectory() throws -> URL {
        if ProcessInfo.processInfo.environment["MIMI_QWEN_MODEL_DIR"] != nil {
            try validateModelDirectory(modelDirectory)
            return modelDirectory
        }
        guard let data = try? Data(contentsOf: installMarkerURL),
              let marker = try? JSONDecoder().decode(QwenInstalledModelMarker.self, from: data),
              marker.repository == Self.repository,
              marker.revision == Self.revision,
              URL(fileURLWithPath: marker.modelDirectory).standardizedFileURL == modelDirectory.standardizedFileURL else {
            throw QwenMLXError.notInstalled
        }
        try validateModelDirectory(modelDirectory)
        return modelDirectory
    }

    private func validateModelDirectory(_ directory: URL) throws {
        let missing = Self.requiredModelFiles.filter {
            !fileManager.fileExists(atPath: directory.appending(path: $0).path)
        }
        guard missing.isEmpty else { throw QwenMLXError.incompleteModel(missing) }
    }

    private func writeInstallMarker(for directory: URL) throws {
        try fileManager.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let marker = QwenInstalledModelMarker(
            repository: Self.repository,
            revision: Self.revision,
            modelDirectory: directory.standardizedFileURL.path
        )
        try JSONEncoder().encode(marker).write(to: installMarkerURL, options: .atomic)
    }

    private var metalLibraryURL: URL? {
        guard let executablePath = CommandLine.arguments.first, !executablePath.isEmpty else { return nil }
        let directory = URL(fileURLWithPath: executablePath).deletingLastPathComponent()
        return [
            directory.appending(path: "mlx.metallib"),
            directory.appending(path: "Resources/mlx.metallib")
        ].first { fileManager.fileExists(atPath: $0.path) }
    }

    private func ensureRuntimeAvailable() throws {
        if let runtimeAvailabilityMessage {
            throw QwenMLXError.runtimeUnavailable(runtimeAvailabilityMessage)
        }
    }

    private func scheduleDrain(for sessionID: UUID) {
        guard liveDrainTask == nil,
              pendingSamples.count >= Self.feedChunkSamples,
              !isStopping else { return }
        liveDrainTask = Task { [weak self] in
            guard let self else { return }
            await drainSamples(flush: false, sessionID: sessionID)
        }
    }

    private func drainSamples(flush: Bool, sessionID: UUID) async {
        defer {
            if liveSessionID == sessionID {
                liveDrainTask = nil
                if !isStopping { scheduleDrain(for: sessionID) }
            }
        }
        while liveSessionID == sessionID {
            let count: Int
            if pendingSamples.count >= Self.feedChunkSamples {
                count = Self.feedChunkSamples
            } else if flush, !pendingSamples.isEmpty {
                count = pendingSamples.count
            } else {
                return
            }
            await runtime.feed(samples: pendingSamples.dequeue(upTo: count))
        }
    }

    private func normalizedSamples(from buffer: AVAudioPCMBuffer) -> [Float]? {
        guard let audioConverter, let normalizedFormat else { return nil }
        let ratio = normalizedFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(max(1, Double(buffer.frameLength) * ratio + 2))
        guard let converted = AVAudioPCMBuffer(pcmFormat: normalizedFormat, frameCapacity: capacity) else {
            return nil
        }
        let input = QwenAudioConverterInput(buffer: buffer)
        var conversionError: NSError?
        let status = audioConverter.convert(to: converted, error: &conversionError) { _, outputStatus in
            input.next(outputStatus)
        }
        guard status != .error,
              conversionError == nil,
              converted.frameLength > 0,
              let samples = converted.floatChannelData?[0] else { return nil }
        return Array(UnsafeBufferPointer(start: samples, count: Int(converted.frameLength)))
    }

    private func reportBackpressureIfNeeded() {
        guard !hasReportedBackpressure else { return }
        hasReportedBackpressure = true
        liveBackpressure?("Qwen3-ASR is behind this audio source. Mimi skipped some queued audio to keep captions current; use Apple Speech for the lowest latency.")
    }

    private func queueDisplayUpdate(_ text: String, for sessionID: UUID) {
        pendingDisplayText = text
        guard displayPublishTask == nil else { return }
        displayPublishTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(160))
            guard let self, liveSessionID == sessionID, !Task.isCancelled else { return }
            let display = pendingDisplayText
            if display != lastPublishedDisplayText {
                lastPublishedDisplayText = display
                liveEvent?(.partial(display))
            }
            displayPublishTask = nil
            if pendingDisplayText != display {
                queueDisplayUpdate(pendingDisplayText, for: sessionID)
            }
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
        liveEventTask = nil
        displayPublishTask?.cancel()
        displayPublishTask = nil
        pendingDisplayText = ""
        lastPublishedDisplayText = ""
        isStopping = false
        hasReportedBackpressure = false
    }
}

private actor NativeQwenRuntime {
    private var model: Qwen3ASRModel?
    private var loadedDirectory: URL?
    private var session: StreamingInferenceSession?

    func load(modelDirectory: URL) async throws {
        let directory = modelDirectory.standardizedFileURL
        if loadedDirectory == directory, model != nil { return }
        session?.cancel()
        session = nil
        model = try await Qwen3ASRModel.fromModelDirectory(directory)
        loadedDirectory = directory
    }

    func start(language: String) throws -> AsyncStream<MLXAudioSTT.TranscriptionEvent> {
        guard let model else { throw QwenMLXError.notInstalled }
        session?.cancel()
        let config = StreamingConfig(
            decodeIntervalSeconds: 0.5,
            boundaryDecodeIntervalSeconds: 0.2,
            boundaryBoostSeconds: 1,
            encoderWindowOverlapSeconds: 1,
            maxCachedWindows: 4,
            delayPreset: .realtime,
            language: language,
            temperature: 0,
            maxTokensPerPass: 256,
            minAgreementPasses: 2,
            boundaryMinAgreementPasses: 3,
            maxDecodeWindows: 2,
            finalizeCompletedWindows: true
        )
        let session = StreamingInferenceSession(model: model, config: config)
        self.session = session
        return session.events
    }

    func feed(samples: [Float]) {
        session?.feedAudio(samples: samples)
    }

    func stop() {
        session?.stop()
    }

    func cancel() {
        session?.cancel()
        session = nil
    }

    func unload() {
        cancel()
        model = nil
        loadedDirectory = nil
    }

    func transcribe(recordingAt url: URL, language: String) throws -> String {
        guard let model else { throw QwenMLXError.notInstalled }
        let (_, audio) = try loadAudioArray(from: url, sampleRate: model.sampleRate)
        return model.generate(audio: audio, language: language).text
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private final class QwenAudioConverterInput: @unchecked Sendable {
    private let buffer: AVAudioPCMBuffer
    private let lock = NSLock()
    private var provided = false

    init(buffer: AVAudioPCMBuffer) { self.buffer = buffer }

    func next(_ outputStatus: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
        lock.lock()
        defer { lock.unlock() }
        guard !provided else {
            outputStatus.pointee = .noDataNow
            return nil
        }
        provided = true
        outputStatus.pointee = .haveData
        return buffer
    }
}

private struct QwenInstalledModelMarker: Codable {
    let repository: String
    let revision: String
    let modelDirectory: String
}

private enum QwenMLXError: LocalizedError {
    case notInstalled
    case invalidRepository
    case incompleteModel([String])
    case runtimeUnavailable(String)
    case noCompatibleLiveAudioFormat

    var errorDescription: String? {
        switch self {
        case .notInstalled:
            "Download Qwen3-ASR MLX before starting dual-pass live transcription."
        case .invalidRepository:
            "Mimi could not identify the pinned Qwen3-ASR model repository."
        case let .incompleteModel(files):
            "Qwen3-ASR's local download is incomplete (missing \(files.joined(separator: ", "))). Remove it and download again."
        case let .runtimeUnavailable(message):
            message
        case .noCompatibleLiveAudioFormat:
            "Qwen3-ASR could not convert this audio source to 16 kHz mono PCM. Choose another input or use Apple Speech."
        }
    }
}
