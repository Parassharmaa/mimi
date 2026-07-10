import Foundation
import MimiCore

enum MimiStorage {
    private static let appFolderName = "Mimi"

    static func loadLatestTranscript() -> TranscriptDocument {
        guard let url = try? latestTranscriptURL(),
              let data = try? Data(contentsOf: url),
              let document = try? JSONDecoder().decode(TranscriptDocument.self, from: data) else {
            return TranscriptDocument()
        }
        return document
    }

    static func saveLatestTranscript(_ document: TranscriptDocument) throws {
        let url = try latestTranscriptURL()
        let data = try JSONEncoder().encode(document)
        try data.write(to: url, options: .atomic)
    }

    static func clearLatestTranscript() throws {
        let url = try latestTranscriptURL()
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        try FileManager.default.removeItem(at: url)
    }

    static func makeTemporaryRecordingURL(fileExtension: String) throws -> URL {
        let normalizedExtension = fileExtension.lowercased()
        guard ["caf", "wav"].contains(normalizedExtension) else {
            throw MimiStorageError.unsupportedTemporaryAudioFormat(fileExtension)
        }
        let folder = try recordingsDirectory()
        return folder.appending(path: "recording-\(UUID().uuidString).\(normalizedExtension)")
    }

    /// Raw microphone files exist only during a Whisper accuracy pass. Remove
    /// leftovers from a forced quit before presenting any transcript UI.
    static func removeStaleTemporaryRecordings() {
        guard let folder = try? recordingsDirectory(),
              let files = try? FileManager.default.contentsOfDirectory(
                at: folder,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
              ) else {
            return
        }
        for file in files {
            try? FileManager.default.removeItem(at: file)
        }
    }

    private static func latestTranscriptURL() throws -> URL {
        let folder = try applicationDirectory().appending(path: "Transcripts", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder.appending(path: "latest-session.json")
    }

    private static func recordingsDirectory() throws -> URL {
        let folder = try applicationDirectory().appending(path: "Recordings", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    private static func applicationDirectory() throws -> URL {
        let support = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let folder = support.appending(path: appFolderName, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }
}

private enum MimiStorageError: LocalizedError {
    case unsupportedTemporaryAudioFormat(String)

    var errorDescription: String? {
        switch self {
        case let .unsupportedTemporaryAudioFormat(fileExtension):
            "Mimi cannot create a temporary recording with the \(fileExtension) format."
        }
    }
}
