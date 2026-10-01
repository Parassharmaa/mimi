import CryptoKit
import Foundation

struct MimiParakeetJapaneseModel {
    struct File: Sendable { let name: String; let bytes: Int; let sha256: String }
    static let source = URL(string: "https://github.com/Parassharmaa/mimi/releases/download/models-parakeet-ja-encoder-q4-v1/")!
    static let files: [File] = [
        .init(name: "model.safetensors", bytes: 481725023, sha256: "84163fa7b961a579a63ae491ca04ebfac3078fa1d13bc1c5d08accd0a8e50c1c"),
        .init(name: "config.json", bytes: 132341, sha256: "1d850ea4e1fffa94f43251707d260127b8489b73261e09bdc5f130ce5fccc9f2")
    ]
    static func url(for file: File) -> URL {
        source.appending(path: "parakeet-ja-encoder-q4-" + file.name)
    }
    static func validate(_ root: URL, hashes: Bool) throws {
        for file in files {
            let url = root.appending(path: file.name)
            let values = try url.resourceValues(forKeys: [.fileSizeKey, .isSymbolicLinkKey])
            guard values.fileSize == file.bytes, values.isSymbolicLink != true else { throw MimiParakeetJapaneseError.invalidModel }
            if hashes {
                let handle = try FileHandle(forReadingFrom: url)
                defer { try? handle.close() }
                var hash = SHA256()
                while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
                    try Task.checkCancellation()
                    hash.update(data: chunk)
                }
                guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == file.sha256 else {
                    throw MimiParakeetJapaneseError.invalidModel
                }
            }
        }
    }
    static func installedRoot() throws -> URL {
        try MimiStorage.applicationDirectory().appending(path: "Models/ParakeetJapaneseEncoderQ4-v1")
    }
}

enum MimiParakeetJapaneseError: LocalizedError {
    case notInstalled, invalidModel, japaneseOnly, noAudioFormat, downloadFailed
    var errorDescription: String? {
        switch self {
        case .notInstalled: "Download Parakeet Japanese in Settings before using this model."
        case .invalidModel: "Parakeet Japanese model files failed validation. Remove and download the model again."
        case .japaneseOnly: "Parakeet Japanese supports Japanese only."
        case .noAudioFormat: "Parakeet Japanese could not prepare the audio format."
        case .downloadFailed: "Parakeet Japanese could not download a model file. Try again."
        }
    }
}
