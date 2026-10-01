import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXAudioSTT
import MimiCore
import MoonshineVoice

actor NativeMimiParakeetJapaneseRuntime {
    private var model: ParakeetModel?
    private var root: URL?
    private var vad: Transcriber?
    private var stream: MoonshineVoice.Stream?
    private var audio: [Float] = []
    private var offset = 0
    private var total = 0
    private var completed = Set<UInt64>()
    private var nextPartial = 5_600
    private var activeLine: UInt64?
    private var lastPartial = ""
    private var firstVAD: [String: Double] = [:]

    static func convert(source: URL, output: URL) throws -> [String: Any] {
        let model = try ParakeetModel.fromDirectory(source, computeDType: .bfloat16)
        let encoderOnly = ProcessInfo.processInfo.environment["MIMI_PARAKEET_Q4_ENCODER_ONLY"] != "0"
        quantize(model: model, groupSize: 64, bits: 4, filter: { path, _ in !encoderOnly || path.hasPrefix("encoder.") })
        eval(model)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let arrays = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try MLX.save(arrays: arrays, url: output.appending(path: "model.safetensors"))
        guard var config = try JSONSerialization.jsonObject(with: Data(contentsOf: source.appending(path: "config.json"))) as? [String: Any] else { throw MimiParakeetJapaneseError.invalidModel }
        config["quantization"] = ["group_size": 64, "bits": 4, "mode": "affine"]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys, .prettyPrinted])
            .write(to: output.appending(path: "config.json"))
        var files: [String: Any] = [:]
        for name in ["model.safetensors", "config.json"] {
            let data = try Data(contentsOf: output.appending(path: name))
            files[name] = ["bytes": data.count, "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()]
        }
        let manifest: [String: Any] = ["format": encoderOnly ? "mimi-parakeet-ja-encoder-q4-v1" : "mimi-parakeet-ja-all-q4-probe-v1", "source": "mouri45/parakeet-tdt_ctc-0.6b-ja-bf16",
            "revision": "4f8f02b8473fa440313e02a089fc45b5dc496591", "runtime": "159c4d3c083de5881f9d0d91f81d37dce13f2763",
            "quantization": "MLX Swift affine group64 bits4; BF16 residuals; " + (encoderOnly ? "encoder-only quantizable modules" : "all standard quantizable modules"), "files": files]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys, .prettyPrinted])
            .write(to: output.appending(path: "manifest.json"))
        return manifest
    }

    func load(_ directory: URL) throws {
        guard model == nil || root != directory else { return }
        reset()
        try MimiParakeetJapaneseModel.validate(directory, hashes: true)
        let loaded = try ParakeetModel.fromDirectory(directory, computeDType: .bfloat16)
        // Pay first-use Metal compilation before capture. These silent results
        // are discarded and never enter a transcript or the live timings.
        for count in [8_192, 16_000] {
            try Task.checkCancellation()
            _ = loaded.generate(audio: MLXArray([Float](repeating: 0, count: count)), generationParameters: loaded.defaultGenerationParameters)
        }
        model = loaded
        // Native Silero VAD is embedded in the SDK. No Moonshine ASR weights
        // are loaded in skip-transcription mode.
        vad = try Transcriber(modelPath: "", modelArch: .smallStreaming, options: [
            .init(name: "skip_transcription", value: "true"),
            .init(name: "return_audio_data", value: "false"),
            .init(name: "transcription_interval", value: "0.05"),
            .init(name: "vad_max_segment_duration", value: "15"),
            .init(name: "vad_window_duration", value: "0.5"),
            .init(name: "vad_look_behind_sample_count", value: "8192"),
            .init(name: "ort_providers", value: "CPU")
        ])
        root = directory
    }

    func start() throws {
        firstVAD = [:]
        reset()
        guard let vad else { throw MimiParakeetJapaneseError.notInstalled }
        let created = try vad.createStream(updateInterval: Double.greatestFiniteMagnitude)
        try created.start()
        stream = created
    }

    func consume(_ samples: [Float]) throws -> [MimiCore.TranscriptEvent] {
        guard let stream else { return [] }
        audio.append(contentsOf: samples)
        total += samples.count
        try stream.addAudio(samples)
        let events = try decode(try stream.updateTranscription())
        if audio.count > 640_000 {
            audio.removeFirst(128_000)
            offset += 128_000
        }
        return events
    }

    func finish() throws -> [MimiCore.TranscriptEvent] {
        guard let stream else { return [] }
        defer { reset() }
        try stream.stop()
        return try decode(try stream.updateTranscription(), finishing: true)
    }

    private func decode(_ transcript: MoonshineVoice.Transcript, finishing: Bool = false) throws -> [MimiCore.TranscriptEvent] {
        guard let model else { throw MimiParakeetJapaneseError.notInstalled }
        var output: [MimiCore.TranscriptEvent] = []
        for line in transcript.lines where !completed.contains(line.lineId) {
            if firstVAD.isEmpty {
                firstVAD = ["observedAudioSeconds": Double(total) / 16_000, "segmentStartSeconds": Double(line.startTime), "segmentDurationSeconds": Double(line.duration)]
            }
            let start = max(0, Int((Double(line.startTime) * 16_000).rounded()))
            let end = min(total, Int((Double(line.startTime + line.duration) * 16_000).rounded()))
            let final = line.isComplete || finishing
            if activeLine != line.lineId {
                activeLine = line.lineId
                nextPartial = start + 5_600
                lastPartial = ""
            }
            guard end > start, final || end >= nextPartial else { continue }
            guard start >= offset, end - offset <= audio.count else { throw MimiParakeetJapaneseError.noAudioFormat }
            let samples = MLXArray(Array(audio[(start - offset)..<(end - offset)]))
            let text = model.generate(audio: samples, generationParameters: model.defaultGenerationParameters)
                .text.trimmingCharacters(in: .whitespacesAndNewlines)
            if final {
                completed.insert(line.lineId)
                output.append(text.isEmpty ? .partial("") : .final(text))
                lastPartial = ""
            } else {
                nextPartial = end + 8_000
                if text != lastPartial {
                    lastPartial = text
                    output.append(.partial(text))
                }
            }
        }
        return output
    }

    func diagnostics() -> [String: Double] { firstVAD }

    func unload() {
        reset()
        model = nil; vad = nil; root = nil
    }

    func reset() {
        stream = nil
        audio.removeAll(keepingCapacity: true)
        offset = 0; total = 0; nextPartial = 5_600
        completed.removeAll(keepingCapacity: true)
        activeLine = nil; lastPartial = ""
    }
}
