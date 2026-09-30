import Foundation
import MimiCore

struct TranscriptSessionRecord: Identifiable, Codable, Equatable {
    let id: UUID
    let startedAt: Date
    let endedAt: Date
    let source: AudioSource
    let document: TranscriptDocument

    var title: String {
        if let first = document.segments.first?.text, !first.isEmpty {
            return String(first.prefix(48))
        }
        return "Untitled session"
    }
}

@MainActor
final class TranscriptHistoryStore {
    private let fileURL: URL
    private let locationError: Error?

    init(fileURL: URL? = nil, fileManager: FileManager = .default) {
        if let fileURL {
            self.fileURL = fileURL
            locationError = nil
        } else {
            do {
                self.fileURL = try MimiStorage.applicationDirectory(fileManager: fileManager)
                    .appendingPathComponent("sessions.json")
                locationError = nil
            } catch {
                self.fileURL = fileManager.temporaryDirectory.appendingPathComponent("Mimi/sessions.json")
                locationError = error
            }
        }
    }

    func load() throws -> [TranscriptSessionRecord] {
        if let locationError { throw locationError }
        let data: Data
        do {
            data = try Data(contentsOf: fileURL)
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            return []
        } catch {
            throw TranscriptHistoryError.unreadable
        }
        do {
            return try JSONDecoder().decode([TranscriptSessionRecord].self, from: data)
                .sorted { $0.startedAt > $1.startedAt }
        } catch {
            throw TranscriptHistoryError.malformed
        }
    }

    func save(_ records: [TranscriptSessionRecord]) throws {
        // A failed load must never be mistaken for an empty history to replace.
        // Revalidate here too, including stores that skipped loading for a fixture.
        _ = try load()
        let data = try JSONEncoder().encode(records)
        try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: fileURL, options: .atomic)
    }
}

private enum TranscriptHistoryError: LocalizedError {
    case unreadable
    case malformed

    var errorDescription: String? {
        switch self {
        case .unreadable:
            "Mimi couldn’t read previous sessions. The saved history was left unchanged."
        case .malformed:
            "Mimi couldn’t open previous sessions because the history file is damaged. The saved history was left unchanged."
        }
    }
}
