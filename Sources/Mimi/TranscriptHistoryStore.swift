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

    init(fileManager: FileManager = .default) {
        let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let directory = base.appendingPathComponent("Mimi", isDirectory: true)
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        fileURL = directory.appendingPathComponent("sessions.json")
    }

    func load() -> [TranscriptSessionRecord] {
        guard let data = try? Data(contentsOf: fileURL),
              let records = try? JSONDecoder().decode([TranscriptSessionRecord].self, from: data) else {
            return []
        }
        return records.sorted { $0.startedAt > $1.startedAt }
    }

    func save(_ records: [TranscriptSessionRecord]) throws {
        let data = try JSONEncoder().encode(records)
        try data.write(to: fileURL, options: .atomic)
    }
}
