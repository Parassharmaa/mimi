import Foundation
import MimiCore
import MimiSession

/// Developer fixtures never read, clear or overwrite the installed app's data.
@MainActor
final class TransientTranscriptStore: TranscriptPersisting {
    private var document = TranscriptDocument()
    private let directory = FileManager.default.temporaryDirectory
        .appending(path: "mimi-fixture-recordings-\(UUID().uuidString)")

    func loadLatestTranscript() -> TranscriptDocument { document }
    func saveLatestTranscript(_ document: TranscriptDocument) throws { self.document = document }
    func clearLatestTranscript() throws { document = TranscriptDocument() }

    func makeTemporaryRecordingURL(fileExtension: String) throws -> URL {
        guard ["wav", "caf"].contains(fileExtension.lowercased()) else {
            throw CocoaError(.fileWriteInvalidFileName)
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appending(path: "recording-\(UUID().uuidString).\(fileExtension.lowercased())")
    }

    func removeTemporaryRecording(at url: URL) throws {
        guard url.deletingLastPathComponent().standardizedFileURL == directory.standardizedFileURL else {
            throw CocoaError(.fileWriteNoPermission)
        }
        try FileManager.default.removeItem(at: url)
    }

    func removeStaleTemporaryRecordings() {}
}
