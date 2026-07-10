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

    static func saveLatestTranscript(_ document: TranscriptDocument) {
        guard let url = try? latestTranscriptURL(),
              let data = try? JSONEncoder().encode(document) else {
            return
        }
        try? data.write(to: url, options: .atomic)
    }

    static func clearLatestTranscript() {
        guard let url = try? latestTranscriptURL() else { return }
        try? FileManager.default.removeItem(at: url)
    }

    static func makeTemporaryRecordingURL() throws -> URL {
        let folder = try recordingsDirectory()
        return folder.appending(path: "recording-\(UUID().uuidString).caf")
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
