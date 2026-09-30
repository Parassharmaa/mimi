import Foundation
import MimiCore

private final class FixtureFileManager: FileManager, @unchecked Sendable {
    let root: URL

    init(root: URL) {
        self.root = root
        super.init()
    }

    override var homeDirectoryForCurrentUser: URL { root }

    override func url(
        for directory: FileManager.SearchPathDirectory,
        in domain: FileManager.SearchPathDomainMask,
        appropriateFor url: URL?,
        create shouldCreate: Bool
    ) throws -> URL {
        root
    }
}

@main
struct MimiHistorySelfTest {
    @MainActor
    static func main() throws {
        let fixtureRoot = FileManager.default.temporaryDirectory
            .appending(path: "mimi-history-regression-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureRoot) }

        let fileManager = FixtureFileManager(root: fixtureRoot)
        let fileURL = fixtureRoot.appending(path: "Mimi/sessions.json")
        let store = TranscriptHistoryStore(fileManager: fileManager)
        var document = TranscriptDocument()
        document.apply(.final("Fixture English and 日本語"), language: .english)
        let record = TranscriptSessionRecord(
            id: UUID(), startedAt: Date(timeIntervalSince1970: 1),
            endedAt: Date(timeIntervalSince1970: 2), source: .microphone, document: document
        )

        expect(try store.load().isEmpty, "Missing history is an empty collection")
        try store.save([record])
        expect(try store.load() == [record], "Valid history survives a save and reload")

        let malformed = Data("{preserve this malformed history".utf8)
        try malformed.write(to: fileURL)
        let rejectedLoad: Bool
        do {
            _ = try store.load()
            rejectedLoad = false
        } catch {
            rejectedLoad = true
        }
        let rejectedSave: Bool
        do {
            try store.save([record])
            rejectedSave = false
        } catch {
            rejectedSave = true
        }
        expect(rejectedLoad, "Malformed history reports a load failure")
        expect(rejectedSave, "Malformed history blocks overwrite")
        expect(try Data(contentsOf: fileURL) == malformed, "Failed save preserves exact original bytes")

        try JSONEncoder().encode([record]).write(to: fileURL)
        try store.save([])
        expect(try store.load().isEmpty, "A repaired history file can be saved without restarting")

        try FileManager.default.removeItem(at: fileURL)
        try FileManager.default.createDirectory(at: fileURL, withIntermediateDirectories: false)
        let rejectedUnreadableLoad: Bool
        do {
            _ = try store.load()
            rejectedUnreadableLoad = false
        } catch {
            rejectedUnreadableLoad = true
        }
        expect(rejectedUnreadableLoad, "An unreadable existing history reports failure")
        do {
            try store.save([record])
            fatalError("An unreadable existing history must block saves")
        } catch {}
        var isDirectory: ObjCBool = false
        expect(
            FileManager.default.fileExists(atPath: fileURL.path, isDirectory: &isDirectory) && isDirectory.boolValue,
            "Unreadable existing history remains untouched"
        )
        print("Mimi history safety passed: missing, valid, malformed, repaired, and unreadable history.")
    }

    private static func expect(_ condition: Bool, _ message: String) {
        guard condition else { fatalError(message) }
    }
}
