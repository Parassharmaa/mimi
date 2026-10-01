import Foundation
import MimiCore
import MoonshineVoice

/// All native inference and mutable stream state stay on one actor.
actor NativeMoonshineRuntime {
    private var transcriber: Transcriber?
    private var stream: MoonshineVoice.Stream?
    private var loadedRoot: URL?
    private var completed = Set<UInt64>()
    private var sampleCount = 0
    private var nextUpdate = 8_000
    private var liveText = ""

    func load(_ root: URL) throws {
        guard loadedRoot != root || transcriber == nil else { return }
        reset()
        transcriber = nil
        loadedRoot = nil
        try MimiMoonshineModel.validate(root, hashes: true)
        let created = try Transcriber(modelPath: root.path, modelArch: .smallStreaming, options: [
            .init(name: "max_tokens_per_second", value: "13.0"),
            .init(name: "transcription_interval", value: "0.5"),
            .init(name: "ort_providers", value: "CPU"),
            .init(name: "vad_window_duration", value: "0.5"),
            .init(name: "return_audio_data", value: "false")
        ])
        _ = try created.transcribeWithoutStreaming(audioData: [Float](repeating: 0, count: 16_000))
        transcriber = created
        loadedRoot = root
    }

    func start() throws {
        reset()
        guard let transcriber else { throw MimiMoonshineError.notInstalled }
        // Explicit audio-time scheduling: first pass at 0.5s, then every 1s.
        // Disable the wrapper's automatic passes, which would add extra decodes.
        let created = try transcriber.createStream(updateInterval: Double.greatestFiniteMagnitude)
        try created.start()
        stream = created
    }

    func consume(_ samples: [Float]) throws -> [MimiCore.TranscriptEvent] {
        guard let stream else { return [] }
        try stream.addAudio(samples)
        sampleCount += samples.count
        guard sampleCount >= nextUpdate else { return [] }
        nextUpdate = sampleCount + 16_000
        return events(try stream.updateTranscription())
    }

    func finish() throws -> [MimiCore.TranscriptEvent] {
        guard let stream else { return [] }
        defer { reset() }
        try stream.stop()
        return events(try stream.updateTranscription(), finishing: true)
    }

    private func events(_ transcript: MoonshineVoice.Transcript, finishing: Bool = false) -> [MimiCore.TranscriptEvent] {
        var output: [MimiCore.TranscriptEvent] = []
        var partials: [String] = []
        for line in transcript.lines {
            let text = line.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !completed.contains(line.lineId) else { continue }
            if line.isComplete || finishing {
                completed.insert(line.lineId)
                if !text.isEmpty { output.append(.final(text)) }
            } else if !text.isEmpty {
                partials.append(text)
            }
        }
        let partial = partials.joined()
        if partial != liveText {
            liveText = partial
            output.append(.partial(partial))
        }
        return output
    }

    func reset() {
        // The wrapper's deinits own native free calls. Explicit close + deinit
        // would free the same native handles twice.
        stream = nil
        completed.removeAll(keepingCapacity: true)
        sampleCount = 0; nextUpdate = 8_000; liveText = ""
    }

    func unload() {
        reset()
        transcriber = nil
        loadedRoot = nil
    }
}
