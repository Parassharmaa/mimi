import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXAudioCore
import MLXAudioSTT

struct MimiPhononManifest: Decodable {
    struct Entry: Decodable {
        let path: String
        let kind: String
        let inputDimensions: Int
        let outputDimensions: Int
        enum CodingKeys: String, CodingKey {
            case path, kind
            case inputDimensions = "input_dimensions"
            case outputDimensions = "output_dimensions"
        }
    }
    struct File: Decodable { let bytes: Int; let sha256: String }
    let format: String
    let repository: String
    let revision: String
    let languages: [String]
    let modules: [Entry]
    let files: [String: File]

    static func read(at root: URL, verifyHashes: Bool) throws -> Self {
        let result = try JSONDecoder().decode(Self.self, from: Data(contentsOf: root.appending(path: "manifest.json")))
        guard result.format == "mimi-phonon2-two-plane-v1",
              result.repository == "FermionResearch/Phonon-2",
              result.revision == "1c388bcec35d19740bf36b0b675718223fa7904e",
              result.languages == ["en"], result.modules.count == 264,
              Set(result.modules.map(\.path)).count == 264,
              Set(result.files.keys) == ["config.json", "model.safetensors"] else {
            throw MimiPhononError.invalidModel
        }
        for (name, requirement) in result.files {
            let url = root.appending(path: name)
            let values = try url.resourceValues(forKeys: [.fileSizeKey, .isSymbolicLinkKey])
            guard values.fileSize == requirement.bytes, values.isSymbolicLink != true else { throw MimiPhononError.invalidModel }
            if verifyHashes {
                let stream = try FileHandle(forReadingFrom: url)
                defer { try? stream.close() }
                var hash = SHA256()
                while let data = try stream.read(upToCount: 1024 * 1024), !data.isEmpty { hash.update(data: data) }
                guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == requirement.sha256 else { throw MimiPhononError.invalidModel }
            }
        }
        return result
    }
}

private func phononMatmul(
    _ input: MLXArray, weight: MLXArray, residual: MLXArray,
    scales: MLXArray, biases: MLXArray, residualScales: MLXArray, residualBiases: MLXArray
) -> MLXArray {
    let x = input.asType(.float32)
    let base = quantizedMM(x, weight, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2)
    let other = quantizedMM(x, residual, scales: residualScales, biases: residualBiases, transpose: true, groupSize: 128, bits: 2)
    return (base + other).asType(input.dtype)
}

private final class MimiPhononLinear: Linear {
    let residualWeight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let residualScales: MLXArray
    let residualBiases: MLXArray
    init(_ entry: MimiPhononManifest.Entry, arrays: [String: MLXArray]) throws {
        func get(_ suffix: String) throws -> MLXArray {
            guard let array = arrays[entry.path + "." + suffix] else { throw MimiPhononError.invalidModel }
            return array
        }
        residualWeight = try get("residualWeight")
        scales = try get("scales")
        biases = try get("biases")
        residualScales = try get("residualScales")
        residualBiases = try get("residualBiases")
        super.init(weight: try get("weight"), bias: arrays[entry.path + ".bias"])
    }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var output = phononMatmul(x, weight: weight, residual: residualWeight, scales: scales, biases: biases, residualScales: residualScales, residualBiases: residualBiases)
        if let bias { output = output + bias }
        return output
    }
}

private final class MimiPhononConv: Conv1d {
    let residualWeight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let residualScales: MLXArray
    let residualBiases: MLXArray
    init(_ entry: MimiPhononManifest.Entry, arrays: [String: MLXArray]) throws {
        func get(_ suffix: String) throws -> MLXArray {
            guard let array = arrays[entry.path + "." + suffix] else { throw MimiPhononError.invalidModel }
            return array
        }
        residualWeight = try get("residualWeight")
        scales = try get("scales")
        biases = try get("biases")
        residualScales = try get("residualScales")
        residualBiases = try get("residualBiases")
        super.init(inputChannels: entry.inputDimensions / 16, outputChannels: entry.outputDimensions, kernelSize: 1, bias: arrays[entry.path + ".bias"] != nil)
    }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = weight.reshaped(weight.dim(0), -1)
        let r = residualWeight.reshaped(residualWeight.dim(0), -1)
        var output = phononMatmul(x, weight: w, residual: r, scales: scales, biases: biases, residualScales: residualScales, residualBiases: residualBiases)
        if let bias { output = output + bias }
        return output
    }
}

actor NativeMimiPhononRuntime {
    private var model: ParakeetModel?
    private var loadedDirectory: URL?
    private var samples: [Float] = []
    private var voiced = false
    private var lastVoice = 0
    private var peakRMS: Float = 0
    private var nextPartial = 5_600

    func load(_ directory: URL) throws {
        guard loadedDirectory != directory || model == nil else { return }
        let manifest = try MimiPhononManifest.read(at: directory, verifyHashes: true)
        let arrays = try MLX.loadArrays(url: directory.appending(path: "model.safetensors"))
        let created = try ParakeetModel.fromConfiguration(Data(contentsOf: directory.appending(path: "config.json")), computeDType: .bfloat16)
        var replacements: [String: Module] = [:]
        for entry in manifest.modules {
            guard entry.inputDimensions > 0, entry.inputDimensions % 128 == 0, entry.outputDimensions > 0,
                  let weight = arrays[entry.path + ".weight"], weight.dtype == .uint32 else { throw MimiPhononError.invalidModel }
            switch entry.kind {
            case "linear": replacements[entry.path] = try MimiPhononLinear(entry, arrays: arrays)
            case "conv1d": replacements[entry.path] = try MimiPhononConv(entry, arrays: arrays)
            default: throw MimiPhononError.invalidModel
            }
        }
        try created.update(modules: ModuleChildren.unflattened(replacements), verify: .noUnusedKeys)
        try created.update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
        created.train(false)
        eval(created)
        model = created
        loadedDirectory = directory
        reset()
    }

    func transcribe(_ url: URL) throws -> String {
        let (_, audio) = try loadAudioArray(from: url, sampleRate: 16_000)
        return try decode(audio.asType(.float32))
    }

    func reset() {
        samples.removeAll(keepingCapacity: true)
        voiced = false; lastVoice = 0; peakRMS = 0; nextPartial = 5_600
    }

    func consume(_ input: [Float]) throws -> (text: String, final: Bool)? {
        guard !input.isEmpty else { return nil }
        samples.append(contentsOf: input)
        let rms = sqrt(input.reduce(Float(0)) { $0 + $1 * $1 } / Float(input.count))
        peakRMS = max(peakRMS, rms)
        if rms > max(0.004, peakRMS * 0.18) { voiced = true; lastVoice = samples.count }
        let final = samples.count >= 480_000 || (voiced && samples.count - lastVoice >= 11_200)
        if final {
            let text = voiced ? try decode(MLXArray(samples)) : ""
            reset()
            return text.isEmpty ? nil : (text, true)
        }
        if voiced, samples.count >= nextPartial {
            let text = try decode(MLXArray(samples))
            nextPartial = samples.count + 8_000
            return text.isEmpty ? nil : (text, false)
        }
        return nil
    }

    func finish() throws -> String {
        defer { reset() }
        return voiced ? try decode(MLXArray(samples)) : ""
    }

    private func decode(_ audio: MLXArray) throws -> String {
        guard let model else { throw MimiPhononError.notInstalled }
        let result = model.generate(audio: audio.asType(.float32), generationParameters: model.defaultGenerationParameters)
        return result.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

enum MimiPhononError: LocalizedError {
    case notInstalled, invalidModel, englishOnly, noAudioFormat
    var errorDescription: String? {
        switch self {
        case .notInstalled: "Phonon 2 is not bundled. Install a Mimi build that includes Phonon 2."
        case .invalidModel: "The bundled Phonon 2 model failed validation. Reinstall Mimi."
        case .englishOnly: "Phonon 2 supports English only. Choose Mimi Whisper or Apple Speech for Japanese."
        case .noAudioFormat: "Phonon 2 could not prepare the audio format."
        }
    }
}
