@preconcurrency import AVFoundation
import Foundation
import MimiCore
import MimiSession

@MainActor
final class MimiMoonshineLiveEngine: WhisperAccuracyTranscribing {
    private let runtime = NativeMoonshineRuntime()
    private let explicitRoot: URL?
    private var converter: AVAudioConverter?
    private var format: AVAudioFormat?
    private var pending = BoundedAudioSampleQueue(maximumSampleCount: 128_000, preferredChunkSize: 800)
    private var onEvent: (@MainActor (TranscriptEvent) -> Void)?
    private var onBackpressure: (@MainActor (String) -> Void)?
    private var sessionID: UUID?
    private var drainTask: Task<Void, Never>?
    private var stopping = false
    private var computeSeconds = 0.0
    private var maximumQueuedSamples = 0

    init(modelRoot: URL? = nil) { explicitRoot = modelRoot }

    private var modelRoot: URL? {
        explicitRoot ?? (try? MimiMoonshineModel.installedRoot())
    }
    var supportsLiveTranscription: Bool { true }
    var isRemovable: Bool { explicitRoot == nil && isDownloaded }
    var isDownloaded: Bool {
        guard let modelRoot else { return false }
        return (try? MimiMoonshineModel.validate(modelRoot, hashes: false)) != nil
    }
    var runtimeAvailabilityMessage: String? {
#if arch(arm64)
        return nil
#else
        return "Moonshine Japanese requires an Apple Silicon Mac."
#endif
    }
    func ensureInstalled() throws {
        guard runtimeAvailabilityMessage == nil, let modelRoot else { throw MimiMoonshineError.notInstalled }
        _ = try MimiMoonshineModel.validate(modelRoot, hashes: false)
    }
    func install(onProgress: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void) async throws {
        if isDownloaded {
            try await runtime.load(modelRoot!)
            return
        }
        guard explicitRoot == nil, let root = modelRoot else { throw MimiMoonshineError.notInstalled }
        let staging = root.deletingLastPathComponent().appending(path: ".moonshine-download-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }
        let total = MimiMoonshineModel.files.reduce(0) { $0 + $1.bytes }
        var received = 0
        for file in MimiMoonshineModel.files {
            try Task.checkCancellation()
            let (temporary, response) = try await URLSession.shared.download(from: MimiMoonshineModel.source.appending(path: file.name))
            defer { try? FileManager.default.removeItem(at: temporary) }
            guard let response = response as? HTTPURLResponse, response.statusCode == 200 else { throw MimiMoonshineError.downloadFailed }
            try FileManager.default.moveItem(at: temporary, to: staging.appending(path: file.name))
            received += file.bytes
            onProgress(ModelDownloadProgress(completedUnitCount: Int64(received), totalUnitCount: Int64(total)))
        }
        try MimiMoonshineModel.validate(staging, hashes: true)
        try Task.checkCancellation()
        if FileManager.default.fileExists(atPath: root.path) { try FileManager.default.removeItem(at: root) }
        try FileManager.default.moveItem(at: staging, to: root)
        try await runtime.load(root)
    }
    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String {
        guard language == .japanese else { throw MimiMoonshineError.japaneseOnly }
        try ensureInstalled()
        try await runtime.load(modelRoot!)
        let report = try await runLiveSmoke(recordingAt: url, paced: false)
        return report["text"] as? String ?? ""
    }
    func runLiveSmoke(recordingAt url: URL, paced: Bool = true) async throws -> [String: Any] {
        let file = try AVAudioFile(forReading: url)
        let prepareStarted = ContinuousClock.now
        var events: [[String: Any]] = []
        var finals: [String] = []
        var warnings: [String] = []
        var started = ContinuousClock.now
        try await startLive(language: .japanese, inputFormat: file.processingFormat, onEvent: { event in
            let elapsed = started.duration(to: .now).seconds
            switch event {
            case let .partial(text): events.append(["kind":"partial", "text":text, "seconds":elapsed])
            case let .final(text): events.append(["kind":"final", "text":text, "seconds":elapsed]); finals.append(text)
            }
        }, onBackpressure: { warnings.append($0) })
        let preparation = prepareStarted.duration(to: .now).seconds
        started = .now
        let step = AVAudioFrameCount(file.processingFormat.sampleRate * 0.05)
        while file.framePosition < file.length {
            let frames = min(step, AVAudioFrameCount(file.length - file.framePosition))
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frames) else { throw MimiMoonshineError.noAudioFormat }
            try file.read(into: buffer, frameCount: frames)
            let deadline = Double(file.framePosition) / file.processingFormat.sampleRate
            let delay = deadline - started.duration(to: .now).seconds
            if paced, delay > 0 { try await Task.sleep(for: .seconds(delay)) }
            consumeLive(buffer)
            if !paced, let drainTask { await drainTask.value }
        }
        let audioEnded = started.duration(to: .now).seconds
        await stopLive()
        let wall = started.duration(to: .now).seconds
        return ["model":"moonshine-japanese-small-native", "mode":paced ? "paced-production-queue" : "direct-production-queue", "audio_seconds":Double(file.length)/file.processingFormat.sampleRate,
                "prepare_seconds":preparation, "wall_seconds":wall, "first_text_seconds":events.first(where: { ($0["text"] as? String)?.isEmpty == false })?["seconds"] ?? NSNull(),
                "compute_seconds":computeSeconds, "compute_rtf":computeSeconds / (Double(file.length) / file.processingFormat.sampleRate),
                "maximum_queued_samples":maximumQueuedSamples, "audio_delivery_boundary":"end of each 50ms buffer",
                "post_audio_finalization_seconds":wall-audioEnded, "events":events, "warnings":warnings, "text":finals.joined(separator:" ")]
    }
    func startLive(
        language: SpeechLanguage, inputFormat: AVAudioFormat,
        onEvent: @escaping @MainActor (TranscriptEvent) -> Void,
        onBackpressure: @escaping @MainActor (String) -> Void
    ) async throws {
        guard language == .japanese else { throw MimiMoonshineError.japaneseOnly }
        try ensureInstalled()
        await cancelLive()
        guard let normalized = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1),
              let conversion = AVAudioConverter(from: inputFormat, to: normalized) else { throw MimiMoonshineError.noAudioFormat }
        try await runtime.load(modelRoot!)
        try await runtime.start()
        converter = conversion; format = normalized
        self.onEvent = onEvent; self.onBackpressure = onBackpressure
        sessionID = UUID(); stopping = false
        computeSeconds = 0; maximumQueuedSamples = 0
    }
    func consumeLive(_ buffer: AVAudioPCMBuffer) {
        guard let converter, let format, let id = sessionID, !stopping else { return }
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * format.sampleRate / buffer.format.sampleRate + 32)
        guard let converted = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return }
        let provider = MoonshineAudioInput(buffer)
        var error: NSError?
        let status = converter.convert(to: converted, error: &error) { _, state in provider.next(state) }
        guard error == nil, status != .error, let channel = converted.floatChannelData?[0] else {
            onBackpressure?("Moonshine could not convert an audio buffer.")
            return
        }
        let audio = Array(UnsafeBufferPointer(start: channel, count: Int(converted.frameLength)))
        let dropped = pending.append(audio)
        maximumQueuedSamples = max(maximumQueuedSamples, pending.count)
        if dropped > 0 { onBackpressure?("Moonshine audio queue exceeded its eight-second limit; \(dropped) samples were dropped.") }
        schedule(id)
    }
    private func schedule(_ id: UUID) {
        guard drainTask == nil, pending.count >= 800, !stopping else { return }
        drainTask = Task { [weak self] in
            guard let self else { return }
            await drain(id, flush: false)
            drainTask = nil
            if sessionID == id { schedule(id) }
        }
    }
    private func drain(_ id: UUID, flush: Bool) async {
        while sessionID == id, !Task.isCancelled, !pending.isEmpty, flush || pending.count >= 800 {
            let samples = pending.dequeue(upTo: 800)
            do {
                let workStarted = ContinuousClock.now
                let updates = try await runtime.consume(samples)
                computeSeconds += workStarted.duration(to: .now).seconds
                guard sessionID == id else { return }
                for update in updates { onEvent?(update) }
            } catch { onBackpressure?(error.localizedDescription); return }
        }
    }
    func stopLive() async {
        guard let id = sessionID else { return }
        stopping = true
        if let drainTask { await drainTask.value }
        await drain(id, flush: true)
        do {
            let workStarted = ContinuousClock.now
            let updates = try await runtime.finish()
            computeSeconds += workStarted.duration(to: .now).seconds
            if sessionID == id { for update in updates { onEvent?(update) } }
        } catch { onBackpressure?(error.localizedDescription) }
        clear()
    }
    func cancelLive() async {
        sessionID = nil
        let task = drainTask
        task?.cancel()
        if let task { await task.value }
        await runtime.reset()
        clear()
    }
    private func clear() {
        sessionID = nil; drainTask = nil; converter = nil; format = nil
        onEvent = nil; onBackpressure = nil; stopping = false
        pending.removeAll(keepingCapacity: true)
    }
    func removeDownloadedModel() async throws {
        guard explicitRoot == nil, let root = modelRoot else { throw MimiMoonshineError.notInstalled }
        await cancelLive()
        await runtime.unload()
        if FileManager.default.fileExists(atPath: root.path) { try FileManager.default.removeItem(at: root) }
    }
}

private final class MoonshineAudioInput: @unchecked Sendable {
    private let buffer: AVAudioPCMBuffer
    private var supplied = false
    init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }
    func next(_ status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioPCMBuffer? {
        if supplied { status.pointee = .noDataNow; return nil }
        supplied = true; status.pointee = .haveData; return buffer
    }
}
