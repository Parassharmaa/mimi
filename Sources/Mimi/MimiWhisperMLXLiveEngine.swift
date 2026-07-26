@preconcurrency import AVFoundation
import CryptoKit
import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXAudioSTT
import MimiCore
import MimiSession

/// Mimi's pinned bilingual speech model.
///
/// The model pack is app-managed and hash-checked before loading. Live capture
/// uses bounded overlapping windows so compute and memory do not grow with a
/// meeting's duration. Apple Speech remains available while this development
/// engine completes the paced promotion benchmark.
@MainActor
final class MimiWhisperMLXLiveEngine: WhisperAccuracyTranscribing {
    private static let repository = "mlx-community/whisper-large-v3-turbo-asr-4bit"
    private static let revision = "321a6ead9f6e0646bc8188a54d2a470e275c6b76"
    private static let bundledDirectoryName = "mimi-whisper-large-v3-turbo-q4"
    private static let requiredFiles: [String: MimiWhisperModelFileRequirement] = [
        "README.md": .init(
            bytes: 1_057,
            sha256: "5f8e786ebf20a20d2ca1bd02d6d74939c8e106a2846b4c8ceb3242a2abf6beb0"
        ),
        "added_tokens.json": .init(
            bytes: 34_648,
            sha256: "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0"
        ),
        "config.json": .init(
            bytes: 1_506,
            sha256: "9135b2ae07e6450a8f4e87ad1124abe970f705d72ea426030f969cb5014b82e9"
        ),
        "generation_config.json": .init(
            bytes: 3_772,
            sha256: "cce11bfe3aaa6ae9e072ea2637caaec8795e68d9b67e655a5af16ee509681a4c"
        ),
        "merges.txt": .init(
            bytes: 493_869,
            sha256: "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6"
        ),
        "model.safetensors": .init(
            bytes: 463_462_815,
            sha256: "45298f6dc48df8c11e0a8d1dc5e0197c688bfa530646fa21f1a0238d2b0ecda3"
        ),
        "model.safetensors.index.json": .init(
            bytes: 68_118,
            sha256: "d408891e3b45a13abcb1ccf0a4af6eb50f38331bb71275eec627b182120be015"
        ),
        "normalizer.json": .init(
            bytes: 52_666,
            sha256: "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd"
        ),
        "preprocessor_config.json": .init(
            bytes: 340,
            sha256: "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"
        ),
        "special_tokens_map.json": .init(
            bytes: 2_186,
            sha256: "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74"
        ),
        "tokenizer.json": .init(
            bytes: 2_710_337,
            sha256: "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd"
        ),
        "tokenizer_config.json": .init(
            bytes: 282_843,
            sha256: "844b642c73a91359722f47b35705f7174686df33d252695d8572cf9ac03a6389"
        ),
        "vocab.json": .init(
            bytes: 1_036_558,
            sha256: "e2aa043ef015641d363d8288e7c241c85e36a5c761fb303598e0710233344387"
        )
    ]
    private static let downloadedFiles = [
        "*.safetensors", "*.json", "*.txt", "README.md"
    ]
    // Keep endpoint detection finer than the 750 ms silence threshold. With
    // 500 ms blocks, an unaligned one-second pause can contain only one fully
    // silent block and fail to end the utterance.
    private static let feedChunkSamples = 16_000 / 10
    // A paced six-minute Japanese soak made the prior four-second bound discard
    // a cumulative two seconds during transient final-decode backlog even though
    // aggregate compute remained faster than real time. Eight seconds gives the
    // burst room to drain while keeping Stop latency and retained PCM bounded.
    private static let maximumPendingSamples = 16_000 * 8

    private let fileManager: FileManager
    private let rootURL: URL
    private let runtime = NativeMimiWhisperRuntime()
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
    private var isStopping = false
    private var hasReportedBackpressure = false
    private var liveInputSampleCount = 0
    private var liveMaximumQueuedAudioSamples = 0
    private var liveDroppedSampleCount = 0
    private var liveAudioDropEventCount = 0
    private var liveFirstAudioDropAtInputSample: Int?
    private var liveBackpressureEventCount = 0

    init(fileManager: FileManager = .default, rootURL: URL? = nil) {
        self.fileManager = fileManager
        if let rootURL {
            self.rootURL = rootURL
            return
        }
        let support = (try? MimiStorage.applicationDirectory(fileManager: fileManager))
            ?? fileManager.temporaryDirectory.appending(path: "Mimi", directoryHint: .isDirectory)
        self.rootURL = support.appending(
            path: "Models/MimiWhisperMLX",
            directoryHint: .isDirectory
        )
    }

    var supportsLiveTranscription: Bool { true }

    var runtimeAvailabilityMessage: String? {
#if arch(arm64)
        guard metalLibraryURL != nil else {
            return "Mimi Speech needs the bundled MLX Metal runtime. Reinstall an Apple-silicon development build."
        }
        return nil
#else
        return "Mimi Speech requires an Apple-silicon Mac."
#endif
    }

    var isDownloaded: Bool {
        (try? installedModelDirectory()) != nil
    }

    var isRemovable: Bool {
        bundledModelDirectory == nil && isDownloaded
    }

    func ensureInstalled() throws {
        try ensureRuntimeAvailable()
        _ = try installedModelDirectory()
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
            throw MimiWhisperMLXError.invalidRepository
        }

        try fileManager.createDirectory(at: modelCacheURL, withIntermediateDirectories: true)
        let cache = HubCache(cacheDirectory: modelCacheURL)
        let client = HubClient(cache: cache)
        let directory = try await client.downloadSnapshot(
            of: repositoryID,
            kind: .model,
            revision: Self.revision,
            matching: Self.downloadedFiles,
            maxConcurrentDownloads: 2,
            progressHandler: { progress in
                onProgress(.init(
                    completedUnitCount: progress.completedUnitCount,
                    totalUnitCount: progress.totalUnitCount
                ))
            }
        )
        try validateModelDirectory(directory)
        do {
            try await runtime.load(modelDirectory: directory)
            try writeInstallMarker(for: directory)
        } catch {
            try? fileManager.removeItem(at: installMarkerURL)
            if directory.standardizedFileURL.path.hasPrefix(
                modelCacheURL.standardizedFileURL.path + "/"
            ) {
                try? fileManager.removeItem(at: directory)
            }
            throw error
        }
    }

    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String {
        try ensureRuntimeAvailable()
        let directory = try installedModelDirectory()
        try await runtime.load(modelDirectory: directory)
        return try await runtime.transcribe(recordingAt: url, language: language)
    }

    func startLive(
        language: SpeechLanguage,
        inputFormat: AVAudioFormat,
        onEvent: @escaping @MainActor (TranscriptEvent) -> Void,
        onBackpressure: @escaping @MainActor (String) -> Void
    ) async throws {
        try ensureRuntimeAvailable()
        let directory = try installedModelDirectory()
        guard let normalizedFormat = AVAudioFormat(
            standardFormatWithSampleRate: 16_000,
            channels: 1
        ),
        let converter = AVAudioConverter(from: inputFormat, to: normalizedFormat) else {
            throw MimiWhisperMLXError.noCompatibleLiveAudioFormat
        }

        await cancelLive()
        try await runtime.load(modelDirectory: directory)
        await runtime.startLive(language: language)

        audioConverter = converter
        self.normalizedFormat = normalizedFormat
        pendingSamples.removeAll(keepingCapacity: true)
        liveEvent = onEvent
        liveBackpressure = onBackpressure
        liveSessionID = UUID()
        isStopping = false
        hasReportedBackpressure = false
        liveInputSampleCount = 0
        liveMaximumQueuedAudioSamples = 0
        liveDroppedSampleCount = 0
        liveAudioDropEventCount = 0
        liveFirstAudioDropAtInputSample = nil
        liveBackpressureEventCount = 0
    }

    func consumeLive(_ buffer: AVAudioPCMBuffer) {
        guard let sessionID = liveSessionID,
              !isStopping,
              let samples = normalizedSamples(from: buffer),
              !samples.isEmpty else {
            return
        }
        liveInputSampleCount += samples.count
        let droppedSamples = pendingSamples.append(samples)
        liveMaximumQueuedAudioSamples = max(
            liveMaximumQueuedAudioSamples,
            pendingSamples.count
        )
        liveDroppedSampleCount += droppedSamples
        if droppedSamples > 0 {
            liveAudioDropEventCount += 1
            if liveFirstAudioDropAtInputSample == nil {
                liveFirstAudioDropAtInputSample = liveInputSampleCount
            }
            reportBackpressureIfNeeded()
        }
        scheduleDrain(for: sessionID)
    }

    func stopLive() async {
        guard let sessionID = liveSessionID else { return }
        isStopping = true
        if let liveDrainTask {
            await liveDrainTask.value
        }
        await drainSamples(flush: true, sessionID: sessionID)
        if let update = await runtime.finishLive() {
            publish(update, for: sessionID)
        }
        resetLiveState()
    }

    func cancelLive() async {
        liveSessionID = nil
        liveDrainTask?.cancel()
        await runtime.cancelLive()
        resetLiveState()
    }

    func removeDownloadedModel() async throws {
        await cancelLive()
        await runtime.unload()
        guard bundledModelDirectory == nil else {
            throw MimiWhisperMLXError.bundledModelCannotBeRemoved
        }
        if fileManager.fileExists(atPath: rootURL.path) {
            try fileManager.removeItem(at: rootURL)
        }
    }

    func runBoundedBenchmark(
        recordingAt url: URL,
        language: SpeechLanguage,
        initialPartialStrideSeconds: Double? = nil,
        partialStrideSeconds: Double = MimiWhisperStreamingProfile.product.partialStrideSeconds,
        endpointSilenceSeconds: Double = MimiWhisperStreamingProfile.product.endpointSilenceSeconds
    ) async throws -> RealtimeBenchmarkReport {
        try ensureRuntimeAvailable()
        let productProfile = MimiWhisperStreamingProfile.product(for: language)
        let profile = try MimiWhisperStreamingProfile(
            initialPartialStrideSeconds: initialPartialStrideSeconds
                ?? productProfile.initialPartialStrideSeconds,
            partialStrideSeconds: partialStrideSeconds,
            endpointSilenceSeconds: endpointSilenceSeconds
        )
        let directory = try installedModelDirectory()
        let loadStartedAt = ContinuousClock.now
        try await runtime.load(modelDirectory: directory)
        let modelLoadSeconds = loadStartedAt.duration(to: .now).seconds

        let (_, audio) = try loadAudioArray(from: url, sampleRate: 16_000)
        let samples = audio.asType(.float32).asArray(Float.self)
        let result = await runtime.runBoundedBenchmark(
            samples: samples,
            language: language,
            feedChunkSamples: Self.feedChunkSamples,
            profile: profile
        )
        return RealtimeBenchmarkReport(
            engine: "mimi-whisper-mlx-q4",
            mode: profile.mode,
            language: language.rawValue,
            audioDurationSeconds: result.audioDurationSeconds,
            wallSeconds: result.wallSeconds,
            modelLoadSeconds: modelLoadSeconds,
            firstTextAtSeconds: result.firstTextAtSeconds,
            firstFinalAtSeconds: result.firstFinalAtSeconds,
            updateCount: result.updates.count,
            meanDecodeSeconds: result.decodeDurations.isEmpty
                ? nil
                : result.decodeDurations.reduce(0, +) / Double(result.decodeDurations.count),
            maxDecodeSeconds: result.decodeDurations.max(),
            realTimeFactor: result.audioDurationSeconds > 0
                ? result.decodeDurations.reduce(0, +) / result.audioDurationSeconds
                : nil,
            hypothesisChurn: RealtimeBenchmarkReport.hypothesisChurn(result.updates),
            finalText: result.finalText,
            firstUpdates: Array(result.updates.prefix(8)),
            feedChunkSeconds: Double(Self.feedChunkSamples) / 16_000,
            initialPartialStrideSeconds: profile.initialPartialStrideSeconds,
            partialStrideSeconds: profile.partialStrideSeconds,
            endpointSilenceSeconds: profile.endpointSilenceSeconds
        )
    }

    /// Replays a recording at wall-clock speed through the same converter,
    /// bounded queue, drain task, actor, and stop flush used by live capture.
    /// This complements `runBoundedBenchmark`, which intentionally measures the
    /// actor runtime directly as a compute-only control.
    func runPacedQueueBenchmark(
        recordingAt url: URL,
        language: SpeechLanguage
    ) async throws -> RealtimeBenchmarkReport {
        try ensureRuntimeAvailable()
        let audioFile = try AVAudioFile(forReading: url)
        let inputFormat = audioFile.processingFormat
        let audioDurationSeconds = Double(audioFile.length) / inputFormat.sampleRate
        let inputBufferSeconds = Double(Self.feedChunkSamples) / 16_000
        let inputFrameCount = AVAudioFrameCount(
            max(1, (inputFormat.sampleRate * inputBufferSeconds).rounded())
        )
        let profile = MimiWhisperStreamingProfile.product(for: language)
        let separator = language == .japanese ? "" : " "

        var replayStartedAt: ContinuousClock.Instant?
        var firstTextAtSeconds: Double?
        var firstFinalAtSeconds: Double?
        var lastFinalAtSeconds: Double?
        var updates: [String] = []
        var finalizedSegments: [String] = []
        var backpressureMessages: [String] = []

        let loadStartedAt = ContinuousClock.now
        try await startLive(
            language: language,
            inputFormat: inputFormat,
            onEvent: { event in
                guard let replayStartedAt else { return }
                let elapsed = replayStartedAt.duration(to: .now).seconds
                let normalized: String
                let rendered: String
                switch event {
                case let .partial(text):
                    normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !normalized.isEmpty else { return }
                    rendered = (finalizedSegments + [normalized]).joined(
                        separator: separator
                    )
                case let .final(text):
                    normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !normalized.isEmpty else { return }
                    finalizedSegments.append(normalized)
                    rendered = finalizedSegments.joined(separator: separator)
                    if firstFinalAtSeconds == nil {
                        firstFinalAtSeconds = elapsed
                    }
                    lastFinalAtSeconds = elapsed
                }
                if firstTextAtSeconds == nil {
                    firstTextAtSeconds = elapsed
                }
                updates.append(rendered)
            },
            onBackpressure: { message in
                backpressureMessages.append(message)
            }
        )
        let modelLoadSeconds = loadStartedAt.duration(to: .now).seconds
        let pacingClock = ContinuousClock()
        let startedAt = ContinuousClock.now
        replayStartedAt = startedAt
        var deliveredAudioSeconds = 0.0
        var maximumInputScheduleLatenessSeconds = 0.0

        do {
            while audioFile.framePosition < audioFile.length {
                let remaining = AVAudioFrameCount(
                    audioFile.length - audioFile.framePosition
                )
                let frameCount = min(inputFrameCount, remaining)
                guard let buffer = AVAudioPCMBuffer(
                    pcmFormat: inputFormat,
                    frameCapacity: frameCount
                ) else {
                    throw RealtimeBenchmarkError.couldNotAllocateAudioBuffer
                }
                try audioFile.read(into: buffer, frameCount: frameCount)
                deliveredAudioSeconds += (
                    Double(buffer.frameLength) / inputFormat.sampleRate
                )
                let deadline = startedAt.advanced(
                    by: .seconds(deliveredAudioSeconds)
                )
                try await pacingClock.sleep(until: deadline)
                maximumInputScheduleLatenessSeconds = max(
                    maximumInputScheduleLatenessSeconds,
                    deadline.duration(to: .now).seconds
                )
                consumeLive(buffer)
            }

            let inputDeliverySeconds = startedAt.duration(to: .now).seconds
            let maximumQueuedAudioSamples = liveMaximumQueuedAudioSamples
            let droppedAudioSamples = liveDroppedSampleCount
            let audioDropEventCount = liveAudioDropEventCount
            let firstAudioDropAtSeconds = liveFirstAudioDropAtInputSample.map {
                Double($0) / 16_000
            }
            let backpressureEventCount = liveBackpressureEventCount
            await stopLive()
            let wallSeconds = startedAt.duration(to: .now).seconds
            let finalText = finalizedSegments.isEmpty
                ? (updates.last ?? "")
                : finalizedSegments.joined(separator: separator)
            let postAudioFinalizationSeconds = lastFinalAtSeconds.map {
                max(0, $0 - audioDurationSeconds)
            }

            return RealtimeBenchmarkReport(
                engine: "mimi-whisper-mlx-q4",
                mode: "paced-live-queue-\(profile.mode)",
                language: language.rawValue,
                audioDurationSeconds: audioDurationSeconds,
                wallSeconds: wallSeconds,
                modelLoadSeconds: modelLoadSeconds,
                firstTextAtSeconds: firstTextAtSeconds,
                firstFinalAtSeconds: firstFinalAtSeconds,
                updateCount: updates.count,
                meanDecodeSeconds: nil,
                maxDecodeSeconds: nil,
                realTimeFactor: nil,
                hypothesisChurn: RealtimeBenchmarkReport.hypothesisChurn(updates),
                finalText: finalText,
                firstUpdates: Array(updates.prefix(8)),
                feedChunkSeconds: inputBufferSeconds,
                initialPartialStrideSeconds: profile.initialPartialStrideSeconds,
                partialStrideSeconds: profile.partialStrideSeconds,
                endpointSilenceSeconds: profile.endpointSilenceSeconds,
                pacedAudio: true,
                inputBufferSeconds: inputBufferSeconds,
                inputDeliverySeconds: inputDeliverySeconds,
                maximumInputScheduleLatenessSeconds: (
                    maximumInputScheduleLatenessSeconds
                ),
                queueCapacitySeconds: (
                    Double(Self.maximumPendingSamples) / 16_000
                ),
                maximumQueuedAudioSamples: maximumQueuedAudioSamples,
                droppedAudioSamples: droppedAudioSamples,
                audioDropEventCount: audioDropEventCount,
                firstAudioDropAtSeconds: firstAudioDropAtSeconds,
                backpressureEventCount: max(
                    backpressureEventCount,
                    backpressureMessages.count
                ),
                postAudioFinalizationSeconds: postAudioFinalizationSeconds
            )
        } catch {
            await cancelLive()
            throw error
        }
    }

    func runOfflineBenchmark(
        recordingAt url: URL,
        language: SpeechLanguage
    ) async throws -> RealtimeBenchmarkReport {
        try ensureRuntimeAvailable()
        let directory = try installedModelDirectory()
        let loadStartedAt = ContinuousClock.now
        try await runtime.load(modelDirectory: directory)
        let modelLoadSeconds = loadStartedAt.duration(to: .now).seconds

        let (_, audio) = try loadAudioArray(from: url, sampleRate: 16_000)
        let audioDurationSeconds = Double(audio.size) / 16_000
        let decodeStartedAt = ContinuousClock.now
        let text = try await runtime.transcribe(
            recordingAt: url,
            language: language
        )
        let decodeSeconds = decodeStartedAt.duration(to: .now).seconds
        return RealtimeBenchmarkReport(
            engine: "mimi-whisper-mlx-q4",
            mode: "record-then-decode-30s-model-chunks-no-vad",
            language: language.rawValue,
            audioDurationSeconds: audioDurationSeconds,
            wallSeconds: decodeSeconds,
            modelLoadSeconds: modelLoadSeconds,
            firstTextAtSeconds: nil,
            firstFinalAtSeconds: decodeSeconds,
            updateCount: text.isEmpty ? 0 : 1,
            meanDecodeSeconds: decodeSeconds,
            maxDecodeSeconds: decodeSeconds,
            realTimeFactor: audioDurationSeconds > 0
                ? decodeSeconds / audioDurationSeconds
                : nil,
            hypothesisChurn: 0,
            finalText: text,
            firstUpdates: text.isEmpty ? [] : [text]
        )
    }

    private var modelCacheURL: URL {
        rootURL.appending(path: "huggingface-cache", directoryHint: .isDirectory)
    }

    private var installMarkerURL: URL {
        rootURL.appending(path: "mimi-whisper-installed.json")
    }

    private var bundledModelDirectory: URL? {
        guard let resources = Bundle.main.resourceURL else { return nil }
        let directory = resources
            .appending(path: "SpeechModels", directoryHint: .isDirectory)
            .appending(path: Self.bundledDirectoryName, directoryHint: .isDirectory)
        return fileManager.fileExists(atPath: directory.path) ? directory : nil
    }

    private func installedModelDirectory() throws -> URL {
        if let override = ProcessInfo.processInfo.environment["MIMI_WHISPER_MLX_MODEL_DIR"],
           !override.isEmpty {
            let directory = URL(
                fileURLWithPath: override,
                isDirectory: true
            ).standardizedFileURL
            try validateModelDirectory(directory)
            return directory
        }
        if let bundledModelDirectory {
            try validateModelDirectory(bundledModelDirectory)
            return bundledModelDirectory
        }
        guard let data = try? Data(contentsOf: installMarkerURL),
              let marker = try? JSONDecoder().decode(
                MimiWhisperInstalledModelMarker.self,
                from: data
              ),
              marker.repository == Self.repository,
              marker.revision == Self.revision else {
            throw MimiWhisperMLXError.notInstalled
        }
        let directory = URL(
            fileURLWithPath: marker.modelDirectory,
            isDirectory: true
        ).standardizedFileURL
        let cachePrefix = modelCacheURL.standardizedFileURL.path + "/"
        guard directory.path.hasPrefix(cachePrefix) else {
            throw MimiWhisperMLXError.notInstalled
        }
        try validateModelDirectory(directory)
        return directory
    }

    private func validateModelDirectory(_ directory: URL) throws {
        let contents = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        let actualNames = Set(contents.map(\.lastPathComponent))
        let expectedNames = Set(Self.requiredFiles.keys)
        guard actualNames == expectedNames else {
            throw MimiWhisperMLXError.modelInventoryMismatch(
                missing: expectedNames.subtracting(actualNames).sorted(),
                extra: actualNames.subtracting(expectedNames).sorted()
            )
        }
        for (name, requirement) in Self.requiredFiles {
            let file = directory.appending(path: name)
            let attributes = try fileManager.attributesOfItem(atPath: file.path)
            guard let measuredBytes = attributes[.size] as? NSNumber,
                  measuredBytes.intValue == requirement.bytes else {
                throw MimiWhisperMLXError.modelSizeMismatch(name)
            }
            guard try sha256(file) == requirement.sha256 else {
                throw MimiWhisperMLXError.modelHashMismatch(name)
            }
        }
    }

    private func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func writeInstallMarker(for directory: URL) throws {
        try fileManager.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let marker = MimiWhisperInstalledModelMarker(
            repository: Self.repository,
            revision: Self.revision,
            modelDirectory: directory.standardizedFileURL.path
        )
        try JSONEncoder().encode(marker).write(to: installMarkerURL, options: .atomic)
    }

    private var metalLibraryURL: URL? {
        guard let executablePath = CommandLine.arguments.first,
              !executablePath.isEmpty else {
            return nil
        }
        let directory = URL(fileURLWithPath: executablePath).deletingLastPathComponent()
        return [
            directory.appending(path: "mlx.metallib"),
            directory.appending(path: "Resources/mlx.metallib")
        ].first { fileManager.fileExists(atPath: $0.path) }
    }

    private func ensureRuntimeAvailable() throws {
        if let runtimeAvailabilityMessage {
            throw MimiWhisperMLXError.runtimeUnavailable(runtimeAvailabilityMessage)
        }
    }

    private func scheduleDrain(for sessionID: UUID) {
        guard liveDrainTask == nil,
              pendingSamples.count >= Self.feedChunkSamples,
              !isStopping else {
            return
        }
        liveDrainTask = Task { [weak self] in
            guard let self else { return }
            await self.drainSamples(flush: false, sessionID: sessionID)
        }
    }

    private func drainSamples(flush: Bool, sessionID: UUID) async {
        defer {
            if liveSessionID == sessionID {
                liveDrainTask = nil
                if !isStopping {
                    scheduleDrain(for: sessionID)
                }
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
            if let update = await runtime.appendLive(
                samples: pendingSamples.dequeue(upTo: count)
            ) {
                publish(update, for: sessionID)
            }
        }
    }

    private func publish(_ update: NativeMimiWhisperUpdate, for sessionID: UUID) {
        guard liveSessionID == sessionID else { return }
        if let finalText = update.finalText {
            liveEvent?(.final(finalText))
        } else if let provisionalText = update.provisionalText {
            liveEvent?(.partial(provisionalText))
        }
    }

    private func normalizedSamples(from buffer: AVAudioPCMBuffer) -> [Float]? {
        guard let audioConverter, let normalizedFormat else { return nil }
        let ratio = normalizedFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(
            max(1, Double(buffer.frameLength) * ratio + 2)
        )
        guard let converted = AVAudioPCMBuffer(
            pcmFormat: normalizedFormat,
            frameCapacity: capacity
        ) else {
            return nil
        }
        let input = MimiWhisperAudioConverterInput(buffer: buffer)
        var conversionError: NSError?
        let status = audioConverter.convert(
            to: converted,
            error: &conversionError
        ) { _, outputStatus in
            input.next(outputStatus)
        }
        guard status != .error,
              conversionError == nil,
              converted.frameLength > 0,
              let samples = converted.floatChannelData?[0] else {
            return nil
        }
        return Array(
            UnsafeBufferPointer(
                start: samples,
                count: Int(converted.frameLength)
            )
        )
    }

    private func reportBackpressureIfNeeded() {
        guard !hasReportedBackpressure else { return }
        hasReportedBackpressure = true
        liveBackpressureEventCount += 1
        liveBackpressure?(
            "Mimi Speech fell behind this audio source and skipped queued audio to stay bounded. Apple Speech remains available for the lowest latency."
        )
    }

    private func resetLiveState() {
        audioConverter = nil
        normalizedFormat = nil
        pendingSamples.removeAll(keepingCapacity: false)
        liveEvent = nil
        liveBackpressure = nil
        liveSessionID = nil
        liveDrainTask = nil
        isStopping = false
        hasReportedBackpressure = false
        liveInputSampleCount = 0
        liveMaximumQueuedAudioSamples = 0
        liveDroppedSampleCount = 0
        liveAudioDropEventCount = 0
        liveFirstAudioDropAtInputSample = nil
        liveBackpressureEventCount = 0
    }
}

private struct MimiWhisperStreamingProfile: Sendable {
    static let product = MimiWhisperStreamingProfile(
        initialPartialStrideSeconds: 3,
        initialPartialStrideSamples: 48_000,
        partialStrideSeconds: 3,
        partialStrideSamples: 48_000,
        endpointSilenceSeconds: 0.75,
        endpointSilenceSamples: 12_000
    )
    static let englishProduct = MimiWhisperStreamingProfile(
        initialPartialStrideSeconds: 2,
        initialPartialStrideSamples: 32_000,
        partialStrideSeconds: 3,
        partialStrideSamples: 48_000,
        endpointSilenceSeconds: 0.75,
        endpointSilenceSamples: 12_000
    )

    static func product(for language: SpeechLanguage) -> Self {
        language == .english ? englishProduct : product
    }

    let initialPartialStrideSeconds: Double
    let initialPartialStrideSamples: Int
    let partialStrideSeconds: Double
    let partialStrideSamples: Int
    let endpointSilenceSeconds: Double
    let endpointSilenceSamples: Int

    init(
        initialPartialStrideSeconds: Double,
        partialStrideSeconds: Double,
        endpointSilenceSeconds: Double
    ) throws {
        for seconds in [initialPartialStrideSeconds, partialStrideSeconds] {
            guard seconds.isFinite, seconds >= 0.5, seconds <= 6 else {
                throw MimiWhisperMLXError.invalidPartialStride(seconds)
            }
        }
        self.initialPartialStrideSeconds = initialPartialStrideSeconds
        initialPartialStrideSamples = Int(
            (initialPartialStrideSeconds * 16_000).rounded()
        )
        self.partialStrideSeconds = partialStrideSeconds
        partialStrideSamples = Int(
            (partialStrideSeconds * 16_000).rounded()
        )
        guard endpointSilenceSeconds.isFinite,
              endpointSilenceSeconds >= 0.25,
              endpointSilenceSeconds <= 3 else {
            throw MimiWhisperMLXError.invalidEndpointSilence(
                endpointSilenceSeconds
            )
        }
        self.endpointSilenceSeconds = endpointSilenceSeconds
        endpointSilenceSamples = Int(
            (endpointSilenceSeconds * 16_000).rounded()
        )
    }

    private init(
        initialPartialStrideSeconds: Double,
        initialPartialStrideSamples: Int,
        partialStrideSeconds: Double,
        partialStrideSamples: Int,
        endpointSilenceSeconds: Double,
        endpointSilenceSamples: Int
    ) {
        self.initialPartialStrideSeconds = initialPartialStrideSeconds
        self.initialPartialStrideSamples = initialPartialStrideSamples
        self.partialStrideSeconds = partialStrideSeconds
        self.partialStrideSamples = partialStrideSamples
        self.endpointSilenceSeconds = endpointSilenceSeconds
        self.endpointSilenceSamples = endpointSilenceSamples
    }

    var mode: String {
        let initialStride = formatted(initialPartialStrideSeconds)
        let stride = formatted(partialStrideSeconds)
        let endpoint = endpointSilenceSamples
            == MimiWhisperStreamingProfile.product.endpointSilenceSamples
            ? ""
            : "-\(formatted(endpointSilenceSeconds))s-endpoint"
        if initialPartialStrideSamples != partialStrideSamples {
            return "bounded-6s-partial-\(initialStride)s-initial-\(stride)s-stride\(endpoint)-30s-final"
        }
        return "bounded-6s-partial-\(stride)s-stride\(endpoint)-30s-final"
    }

    private func formatted(_ seconds: Double) -> String {
        seconds.rounded() == seconds
            ? String(Int(seconds))
            : String(seconds)
    }
}

private actor NativeMimiWhisperRuntime {
    private static let sampleRate = 16_000
    private static let maximumWindowSamples = sampleRate * 6
    private static let maximumUtteranceSamples = sampleRate * 30
    private static let minimumSpeechSamples = sampleRate
    private static let vadCalibrationSamples = sampleRate
    private static let minimumSpeechRMSThreshold: Float = 0.0003
    // 0.0102 per 100 ms preserves the previous 0.05 per 500 ms noise-floor
    // adaptation time constant: 1 - pow(0.95, 0.1 / 0.5).
    private static let noiseFloorUpdateWeight: Float = 0.0102062

    private var model: WhisperModel?
    private var loadedDirectory: URL?
    private var language: SpeechLanguage = .english
    private var windowSamples: [Float] = []
    private var utteranceSamples: [Float] = []
    private var windowStartSample = 0
    private var totalReceivedSamples = 0
    private var samplesSinceDecode = 0
    private var trailingSilenceSamples = 0
    private var speechSamples = 0
    private var calibratedSampleCount = 0
    private var calibrationRMSValues: [Float] = []
    private var noiseFloorRMS: Float = 0.0001
    private var displayText = ""
    private var lastDecodedWindowStart: Int?
    private var profile = MimiWhisperStreamingProfile.product

    func load(modelDirectory: URL) async throws {
        let standardized = modelDirectory.standardizedFileURL
        if loadedDirectory == standardized, model != nil { return }
        model = try await WhisperModel.fromDirectory(standardized)
        loadedDirectory = standardized
        resetLive()
    }

    func transcribe(
        recordingAt url: URL,
        language: SpeechLanguage
    ) throws -> String {
        guard let model else { throw MimiWhisperMLXError.notInstalled }
        let (_, audio) = try loadAudioArray(from: url, sampleRate: Self.sampleRate)
        return decode(model: model, audio: audio.asArray(Float.self), language: language)
    }

    func startLive(language: SpeechLanguage) {
        self.language = language
        profile = .product(for: language)
        resetLive()
    }

    func appendLive(samples: [Float]) -> NativeMimiWhisperUpdate? {
        guard let model, !samples.isEmpty else { return nil }
        appendToWindow(samples)
        updateSpeechState(samples)

        let reachedEndpoint = speechSamples >= Self.minimumSpeechSamples
            && trailingSilenceSamples >= profile.endpointSilenceSamples
        let reachedMaximumDuration =
            utteranceSamples.count >= Self.maximumUtteranceSamples
        let requiredStrideSamples = lastDecodedWindowStart == nil
            ? profile.initialPartialStrideSamples
            : profile.partialStrideSamples
        let shouldDecodePartial = speechSamples >= Self.minimumSpeechSamples
            && samplesSinceDecode >= requiredStrideSamples

        if reachedMaximumDuration, speechSamples < Self.minimumSpeechSamples {
            resetUtterance()
            return nil
        }
        if reachedEndpoint || reachedMaximumDuration {
            let update = decodeUpdate(model: model, final: true)
            resetUtterance()
            return update
        }
        if shouldDecodePartial {
            return decodeUpdate(model: model, final: false)
        }
        return nil
    }

    func finishLive() -> NativeMimiWhisperUpdate? {
        guard let model,
              speechSamples >= Self.minimumSpeechSamples,
              !windowSamples.isEmpty else {
            resetLive()
            return nil
        }
        let update = decodeUpdate(model: model, final: true)
        resetLive()
        return update
    }

    func cancelLive() {
        resetLive()
    }

    func unload() {
        model = nil
        loadedDirectory = nil
        resetLive()
        Memory.clearCache()
    }

    func runBoundedBenchmark(
        samples: [Float],
        language: SpeechLanguage,
        feedChunkSamples: Int,
        profile: MimiWhisperStreamingProfile
    ) -> NativeMimiWhisperBenchmarkResult {
        self.language = language
        self.profile = profile
        defer {
            self.profile = .product
            resetLive()
        }
        resetLive()
        let startedAt = ContinuousClock.now
        var updates: [String] = []
        var finalizedSegments: [String] = []
        var decodeDurations: [Double] = []
        var firstTextAt: Double?
        var firstFinalAt: Double?

        var start = 0
        while start < samples.count {
            let end = min(samples.count, start + feedChunkSamples)
            if let update = appendLive(samples: Array(samples[start..<end])) {
                decodeDurations.append(update.decodeSeconds)
                if let finalText = update.finalText {
                    finalizedSegments.append(finalText)
                }
                if let text = renderedBenchmarkText(
                    finalizedSegments: finalizedSegments,
                    provisionalText: update.provisionalText,
                    language: language
                ) {
                    updates.append(text)
                    if firstTextAt == nil {
                        firstTextAt = Double(end) / Double(Self.sampleRate)
                            + update.decodeSeconds
                    }
                    if update.finalText != nil, firstFinalAt == nil {
                        firstFinalAt = Double(end) / Double(Self.sampleRate)
                            + update.decodeSeconds
                    }
                }
            }
            start = end
        }
        if let update = finishLive() {
            if let finalText = update.finalText {
                finalizedSegments.append(finalText)
            }
            let text = renderedBenchmarkText(
                finalizedSegments: finalizedSegments,
                provisionalText: update.provisionalText,
                language: language
            )
            if let text {
                updates.append(text)
            }
            decodeDurations.append(update.decodeSeconds)
            if firstTextAt == nil {
                firstTextAt = Double(samples.count) / Double(Self.sampleRate)
                    + update.decodeSeconds
            }
            if text != nil, firstFinalAt == nil {
                firstFinalAt = Double(samples.count) / Double(Self.sampleRate)
                    + update.decodeSeconds
            }
        }
        return NativeMimiWhisperBenchmarkResult(
            audioDurationSeconds: Double(samples.count) / Double(Self.sampleRate),
            wallSeconds: startedAt.duration(to: .now).seconds,
            firstTextAtSeconds: firstTextAt,
            firstFinalAtSeconds: firstFinalAt,
            decodeDurations: decodeDurations,
            updates: updates,
            finalText: updates.last ?? ""
        )
    }

    private func renderedBenchmarkText(
        finalizedSegments: [String],
        provisionalText: String?,
        language: SpeechLanguage
    ) -> String? {
        var parts = finalizedSegments
        if let provisionalText, !provisionalText.isEmpty {
            parts.append(provisionalText)
        }
        guard !parts.isEmpty else { return nil }
        return parts.joined(separator: language == .japanese ? "" : " ")
    }

    private func appendToWindow(_ samples: [Float]) {
        totalReceivedSamples += samples.count
        samplesSinceDecode += samples.count
        utteranceSamples.append(contentsOf: samples)
        windowSamples.append(contentsOf: samples)
        if windowSamples.count > Self.maximumWindowSamples {
            let removed = windowSamples.count - Self.maximumWindowSamples
            windowSamples.removeFirst(removed)
            windowStartSample += removed
        }
    }

    private func updateSpeechState(_ samples: [Float]) {
        let energy = samples.reduce(Float.zero) { partial, sample in
            partial + sample * sample
        } / Float(max(samples.count, 1))
        let rms = sqrt(energy)
        if calibratedSampleCount < Self.vadCalibrationSamples {
            calibratedSampleCount += samples.count
            calibrationRMSValues.append(rms)
            guard calibratedSampleCount >= Self.vadCalibrationSamples else {
                return
            }

            let sorted = calibrationRMSValues.sorted()
            let quietest = sorted.first ?? rms
            let loudest = sorted.last ?? rms
            let median = sorted[sorted.count / 2]
            let calibrationContainsSpeech =
                loudest >= 0.005
                || loudest >= max(0.001, quietest * 4)
            noiseFloorRMS = min(
                0.00012,
                max(
                    0.00005,
                    calibrationContainsSpeech ? quietest : median
                )
            )
            let calibratedThreshold = max(
                Self.minimumSpeechRMSThreshold,
                noiseFloorRMS * 2.5
            )
            if calibrationContainsSpeech || median >= calibratedThreshold {
                speechSamples += calibratedSampleCount
                trailingSilenceSamples = 0
            }
            return
        }

        let speechThreshold = min(
            0.008,
            max(Self.minimumSpeechRMSThreshold, noiseFloorRMS * 2.5)
        )
        if rms >= speechThreshold {
            speechSamples += samples.count
            trailingSilenceSamples = 0
        } else {
            trailingSilenceSamples += samples.count
            noiseFloorRMS =
                noiseFloorRMS * (1 - Self.noiseFloorUpdateWeight)
                + rms * Self.noiseFloorUpdateWeight
        }
    }

    private func decodeUpdate(
        model: WhisperModel,
        final: Bool
    ) -> NativeMimiWhisperUpdate? {
        let decodeSamples = final ? utteranceSamples : windowSamples
        guard !decodeSamples.isEmpty else { return nil }
        let startedAt = ContinuousClock.now
        let hypothesis = decode(
            model: model,
            audio: decodeSamples,
            language: language
        )
        let decodeSeconds = startedAt.duration(to: .now).seconds
        samplesSinceDecode = 0
        guard !hypothesis.isEmpty else { return nil }

        if final {
            // The final event is decoded once from the complete bounded
            // utterance. It replaces rolling overlap approximations instead
            // of preserving any partial-window duplication.
            displayText = hypothesis
        } else if lastDecodedWindowStart == nil || windowStartSample == 0 {
            displayText = hypothesis
        } else if lastDecodedWindowStart != windowStartSample {
            displayText = mergeTranscripts(
                existing: displayText,
                incoming: hypothesis,
                language: language
            )
        } else {
            displayText = hypothesis
        }
        lastDecodedWindowStart = windowStartSample
        return NativeMimiWhisperUpdate(
            provisionalText: final ? nil : displayText,
            finalText: final ? displayText : nil,
            decodeSeconds: decodeSeconds
        )
    }

    private func decode(
        model: WhisperModel,
        audio: [Float],
        language: SpeechLanguage
    ) -> String {
        let parameters = STTGenerateParameters(
            maxTokens: 448,
            temperature: 0,
            topP: 1,
            topK: 0,
            verbose: false,
            language: language.whisperLanguageCode,
            chunkDuration: 30,
            minChunkDuration: 0.1
        )
        let output = model.generate(
            audio: MLXArray(audio),
            generationParameters: parameters
        )
        return output.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func mergeTranscripts(
        existing: String,
        incoming: String,
        language: SpeechLanguage
    ) -> String {
        guard !existing.isEmpty else { return incoming }
        guard !incoming.isEmpty else { return existing }
        if incoming.hasPrefix(existing) { return incoming }
        if existing.hasSuffix(incoming) { return existing }

        let existingUnits = alignmentUnits(existing, language: language)
        let incomingUnits = alignmentUnits(incoming, language: language)
        let maximumOverlap = min(existingUnits.count, incomingUnits.count)
        let minimumOverlap = language == .japanese ? 4 : 3
        if maximumOverlap >= minimumOverlap {
            for existingLength in stride(
                from: maximumOverlap,
                through: minimumOverlap,
                by: -1
            ) {
                let incomingLowerBound = max(
                    minimumOverlap,
                    existingLength - 3
                )
                let incomingUpperBound = min(
                    incomingUnits.count,
                    existingLength + 3
                )
                var bestIncomingLength: Int?
                var bestDistance = Int.max
                for incomingLength in incomingLowerBound...incomingUpperBound {
                    let left = existingUnits
                        .suffix(existingLength)
                        .map(\.value)
                    let right = incomingUnits
                        .prefix(incomingLength)
                        .map(\.value)
                    let distance = levenshteinDistance(left, right)
                    let allowedDistance =
                        Int(Double(max(existingLength, incomingLength)) * 0.18)
                    if distance <= allowedDistance,
                       distance < bestDistance {
                        bestIncomingLength = incomingLength
                        bestDistance = distance
                    }
                }
                if let bestIncomingLength {
                    let overlapEnd = incomingUnits[bestIncomingLength - 1].endIndex
                    return existing + incoming[overlapEnd...]
                }
            }
        }
        let separator = language == .japanese ? "" : " "
        return existing + separator + incoming
    }

    private func alignmentUnits(
        _ text: String,
        language: SpeechLanguage
    ) -> [WhisperAlignmentUnit] {
        if language == .english {
            var units: [WhisperAlignmentUnit] = []
            text.enumerateSubstrings(
                in: text.startIndex..<text.endIndex,
                options: [.byWords, .substringNotRequired]
            ) { _, range, _, _ in
                units.append(.init(
                    value: text[range].lowercased(),
                    endIndex: range.upperBound
                ))
            }
            return units
        }

        let ignored = CharacterSet.whitespacesAndNewlines
            .union(.punctuationCharacters)
            .union(.symbols)
        var units: [WhisperAlignmentUnit] = []
        var index = text.startIndex
        while index < text.endIndex {
            let next = text.index(after: index)
            let character = text[index]
            let scalars = character.unicodeScalars
            if scalars.contains(where: { !ignored.contains($0) }) {
                units.append(.init(
                    value: String(character).lowercased(),
                    endIndex: next
                ))
            }
            index = next
        }
        return units
    }

    private func levenshteinDistance(
        _ left: [String],
        _ right: [String]
    ) -> Int {
        if left.isEmpty { return right.count }
        if right.isEmpty { return left.count }
        var previous = Array(0...right.count)
        for (leftIndex, leftValue) in left.enumerated() {
            var current = Array(repeating: 0, count: right.count + 1)
            current[0] = leftIndex + 1
            for (rightIndex, rightValue) in right.enumerated() {
                current[rightIndex + 1] = min(
                    previous[rightIndex + 1] + 1,
                    current[rightIndex] + 1,
                    previous[rightIndex] + (leftValue == rightValue ? 0 : 1)
                )
            }
            previous = current
        }
        return previous[right.count]
    }

    private func resetUtterance() {
        windowSamples.removeAll(keepingCapacity: true)
        utteranceSamples.removeAll(keepingCapacity: true)
        windowStartSample = totalReceivedSamples
        samplesSinceDecode = 0
        trailingSilenceSamples = 0
        speechSamples = 0
        displayText = ""
        lastDecodedWindowStart = nil
    }

    private func resetLive() {
        windowSamples.removeAll(keepingCapacity: false)
        utteranceSamples.removeAll(keepingCapacity: false)
        windowStartSample = 0
        totalReceivedSamples = 0
        samplesSinceDecode = 0
        trailingSilenceSamples = 0
        speechSamples = 0
        calibratedSampleCount = 0
        calibrationRMSValues.removeAll(keepingCapacity: false)
        noiseFloorRMS = 0.0001
        displayText = ""
        lastDecodedWindowStart = nil
    }
}

private struct WhisperAlignmentUnit {
    let value: String
    let endIndex: String.Index
}

private struct NativeMimiWhisperUpdate: Sendable {
    let provisionalText: String?
    let finalText: String?
    let decodeSeconds: Double
}

private struct NativeMimiWhisperBenchmarkResult: Sendable {
    let audioDurationSeconds: Double
    let wallSeconds: Double
    let firstTextAtSeconds: Double?
    let firstFinalAtSeconds: Double?
    let decodeDurations: [Double]
    let updates: [String]
    let finalText: String
}

private final class MimiWhisperAudioConverterInput: @unchecked Sendable {
    private let buffer: AVAudioPCMBuffer
    private let lock = NSLock()
    private var wasProvided = false

    init(buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }

    func next(
        _ outputStatus: UnsafeMutablePointer<AVAudioConverterInputStatus>
    ) -> AVAudioBuffer? {
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

private struct MimiWhisperInstalledModelMarker: Codable {
    let repository: String
    let revision: String
    let modelDirectory: String
}

private struct MimiWhisperModelFileRequirement {
    let bytes: Int
    let sha256: String
}

private enum MimiWhisperMLXError: LocalizedError {
    case notInstalled
    case invalidRepository
    case incompleteModel([String])
    case modelInventoryMismatch(missing: [String], extra: [String])
    case modelSizeMismatch(String)
    case modelHashMismatch(String)
    case runtimeUnavailable(String)
    case noCompatibleLiveAudioFormat
    case bundledModelCannotBeRemoved
    case invalidPartialStride(Double)
    case invalidEndpointSilence(Double)

    var errorDescription: String? {
        switch self {
        case .notInstalled:
            "Download Mimi Speech before starting local transcription."
        case .invalidRepository:
            "Mimi could not identify its pinned speech-model repository."
        case let .incompleteModel(files):
            "Mimi Speech is incomplete (missing \(files.joined(separator: ", "))). Remove it and download again."
        case let .modelInventoryMismatch(missing, extra):
            "Mimi refused the speech model because its file inventory changed (missing: \(missing.joined(separator: ", ")); extra: \(extra.joined(separator: ", ")))."
        case let .modelSizeMismatch(file):
            "Mimi refused the speech model because \(file) has an unexpected size."
        case let .modelHashMismatch(file):
            "Mimi refused the speech model because \(file) does not match the evaluated artifact."
        case let .runtimeUnavailable(message):
            message
        case .noCompatibleLiveAudioFormat:
            "Mimi Speech could not convert this audio source to its local 16 kHz format."
        case .bundledModelCannotBeRemoved:
            "This development build bundles Mimi Speech, so it is removed only when the app is replaced."
        case let .invalidPartialStride(seconds):
            "Mimi Speech partial stride must be between 0.5 and 6 seconds, got \(seconds)."
        case let .invalidEndpointSilence(seconds):
            "Mimi Speech endpoint silence must be between 0.25 and 3 seconds, got \(seconds)."
        }
    }
}
