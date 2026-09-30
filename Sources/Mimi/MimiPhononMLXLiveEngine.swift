@preconcurrency import AVFoundation
import Foundation
import MimiCore
import MimiSession

@MainActor
final class MimiPhononMLXLiveEngine: WhisperAccuracyTranscribing {
    private let runtime = NativeMimiPhononRuntime()
    private let explicitRoot: URL?
    private var converter: AVAudioConverter?
    private var format: AVAudioFormat?
    private var pending = BoundedAudioSampleQueue(maximumSampleCount: 128_000, preferredChunkSize: 800)
    private var onEvent: (@MainActor (TranscriptEvent) -> Void)?
    private var onBackpressure: (@MainActor (String) -> Void)?
    private var sessionID: UUID?
    private var drainTask: Task<Void, Never>?
    private var stopping = false

    init(modelRoot: URL? = nil) { explicitRoot = modelRoot }

    private var modelRoot: URL? {
        explicitRoot ?? Bundle.main.resourceURL?.appending(path: "SpeechModels/mimi-phonon2")
    }
    var supportsLiveTranscription: Bool { true }
    var isRemovable: Bool { false }
    var isDownloaded: Bool {
        guard let modelRoot else { return false }
        return (try? MimiPhononManifest.read(at: modelRoot, verifyHashes: false)) != nil
    }
    var runtimeAvailabilityMessage: String? {
#if arch(arm64)
        return nil
#else
        return "Phonon 2 requires an Apple Silicon Mac."
#endif
    }
    func ensureInstalled() throws {
        guard runtimeAvailabilityMessage == nil, let modelRoot else { throw MimiPhononError.notInstalled }
        _ = try MimiPhononManifest.read(at: modelRoot, verifyHashes: false)
    }
    func install(onProgress: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void) async throws {
        try ensureInstalled()
        try await runtime.load(modelRoot!)
    }
    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String {
        guard language == .english else { throw MimiPhononError.englishOnly }
        try ensureInstalled()
        try await runtime.load(modelRoot!)
        return try await runtime.transcribe(url)
    }
    func runLiveSmoke(recordingAt url: URL) async throws -> [String: Any] {
        let file = try AVAudioFile(forReading: url)
        let prepareStarted = ContinuousClock.now
        var events: [[String: Any]] = []
        var finals: [String] = []
        var warnings: [String] = []
        var started = ContinuousClock.now
        try await startLive(language: .english, inputFormat: file.processingFormat, onEvent: { event in
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
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frames) else { throw MimiPhononError.noAudioFormat }
            try file.read(into: buffer, frameCount: frames)
            consumeLive(buffer)
            let deadline = Double(file.framePosition) / file.processingFormat.sampleRate
            let delay = deadline - started.duration(to: .now).seconds
            if delay > 0 { try await Task.sleep(for: .seconds(delay)) }
        }
        let audioEnded = started.duration(to: .now).seconds
        await stopLive()
        let wall = started.duration(to: .now).seconds
        return ["model":"phonon2-native-mlx", "mode":"paced-production-queue", "audio_seconds":Double(file.length)/file.processingFormat.sampleRate,
                "prepare_seconds":preparation, "wall_seconds":wall, "first_text_seconds":events.first?["seconds"] ?? NSNull(),
                "post_audio_finalization_seconds":wall-audioEnded, "events":events, "warnings":warnings, "text":finals.joined(separator:" ")]
    }
    func startLive(
        language: SpeechLanguage, inputFormat: AVAudioFormat,
        onEvent: @escaping @MainActor (TranscriptEvent) -> Void,
        onBackpressure: @escaping @MainActor (String) -> Void
    ) async throws {
        guard language == .english else { throw MimiPhononError.englishOnly }
        try ensureInstalled()
        await cancelLive()
        guard let normalized = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1),
              let conversion = AVAudioConverter(from: inputFormat, to: normalized) else { throw MimiPhononError.noAudioFormat }
        try await runtime.load(modelRoot!)
        await runtime.reset()
        converter = conversion; format = normalized
        self.onEvent = onEvent; self.onBackpressure = onBackpressure
        sessionID = UUID(); stopping = false
    }
    func consumeLive(_ buffer: AVAudioPCMBuffer) {
        guard let converter, let format, let id = sessionID, !stopping else { return }
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * format.sampleRate / buffer.format.sampleRate + 32)
        guard let converted = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return }
        let provider = PhononAudioInput(buffer)
        var error: NSError?
        let status = converter.convert(to: converted, error: &error) { _, state in provider.next(state) }
        guard error == nil, status != .error, let channel = converted.floatChannelData?[0] else {
            onBackpressure?("Phonon could not convert an audio buffer.")
            return
        }
        let audio = Array(UnsafeBufferPointer(start: channel, count: Int(converted.frameLength)))
        let dropped = pending.append(audio)
        if dropped > 0 { onBackpressure?("Phonon audio queue exceeded its eight-second limit; \(dropped) samples were dropped.") }
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
                let update = try await runtime.consume(samples)
                guard sessionID == id else { return }
                if let update { onEvent?(update.final ? .final(update.text) : .partial(update.text)) }
            } catch { onBackpressure?(error.localizedDescription); return }
        }
    }
    func stopLive() async {
        guard let id = sessionID else { return }
        stopping = true
        if let drainTask { await drainTask.value }
        await drain(id, flush: true)
        do {
            let text = try await runtime.finish()
            if sessionID == id, !text.isEmpty { onEvent?(.final(text)) }
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
    func removeDownloadedModel() async throws { throw MimiPhononError.notInstalled }
}

private final class PhononAudioInput: @unchecked Sendable {
    private let buffer: AVAudioPCMBuffer
    private var supplied = false
    init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }
    func next(_ status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioPCMBuffer? {
        if supplied { status.pointee = .noDataNow; return nil }
        supplied = true; status.pointee = .haveData; return buffer
    }
}
