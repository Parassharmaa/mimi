import CryptoKit
import Foundation

struct MimiMoonshineModel {
    struct File: Sendable { let name: String; let bytes: Int; let sha256: String }
    static let source = URL(string: "https://download.moonshine.ai/model/small-streaming-ja/quantized_26_08_23/")!
    static let files: [File] = [
        .init(name: "adapter.ort", bytes: 2869296, sha256: "edcf20b9d0bc0a92bc0ddf37d8b96599de35a6cbee970368ed74c774b055e546"),
        .init(name: "cross_kv.ort", bytes: 5358752, sha256: "fbd18ec7b9514391d7121f4d4abec1be05a60e94a0f174061428380bf772c3e7"),
        .init(name: "decoder_kv.ort", bytes: 61314512, sha256: "1246d81eeb96df05d0163dcd48ce271a061b01aa3aea047239fbde22485248be"),
        .init(name: "encoder.ort", bytes: 44358376, sha256: "1da7cc82cb91d30ad9dc4e338fb80e7a2336b2c8ad634b826419222698fbe3a5"),
        .init(name: "frontend.model.ort", bytes: 31208, sha256: "de65092295f011a39ee569f03d073e3564d94252efd74ebde16d720bcc43e38b"),
        .init(name: "frontend.weights.ort", bytes: 7769288, sha256: "4e89578a495a6c4dd5632e7acaf2fc38841e6831738c2702ef10f873125a14ec"),
        .init(name: "streaming_config.json", bytes: 512, sha256: "12d16c7f5ea6734d197b79baf47914ea7e996d6fca8d303a19aca10d0617cecc"),
        .init(name: "tokenizer.bin", bytes: 101836, sha256: "9f2599f20c3a9b03d79dbe122e443647c6589ce8e5c73e65e5c5d6f73485d5de"),
    ]
    static func validate(_ root: URL, hashes: Bool) throws {
        for file in files {
            let url = root.appending(path: file.name)
            let values = try url.resourceValues(forKeys: [.fileSizeKey, .isSymbolicLinkKey])
            guard values.fileSize == file.bytes, values.isSymbolicLink != true else { throw MimiMoonshineError.invalidModel }
            if hashes {
                let handle = try FileHandle(forReadingFrom: url)
                defer { try? handle.close() }
                var hash = SHA256()
                while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
                    try Task.checkCancellation()
                    hash.update(data: chunk)
                }
                guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == file.sha256 else {
                    throw MimiMoonshineError.invalidModel
                }
            }
        }
    }
    static func installedRoot() throws -> URL {
        try MimiStorage.applicationDirectory().appending(path: "Models/MoonshineJapaneseSmall-v1")
    }
}

enum MimiMoonshineError: LocalizedError {
    case notInstalled, invalidModel, japaneseOnly, noAudioFormat, downloadFailed
    var errorDescription: String? {
        switch self {
        case .notInstalled: "Download Moonshine Japanese in Settings before using this model."
        case .invalidModel: "Moonshine Japanese model files failed validation. Remove and download the model again."
        case .japaneseOnly: "Moonshine Japanese supports Japanese only."
        case .noAudioFormat: "Moonshine Japanese could not prepare the audio format."
        case .downloadFailed: "Moonshine Japanese could not download a model file. Try again."
        }
    }
}
