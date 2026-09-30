import Foundation
import MimiCore
import MimiSession

struct TranscriptPersistenceSafetyVerificationReport: Codable {
    let schemaVersion: Int
    let status: String
    let loadFailureIsReported: Bool
    let newSessionPreservesCurrentAfterArchiveFailure: Bool
    let startPreservesCurrentAfterArchiveFailure: Bool
    let deleteFailurePreservesRecordAndSelection: Bool
    let failedOperationsPreserveHistoryBytes: Bool
    let deletionUsesCapturedTarget: Bool
    let successfulArchiveSurvivesReload: Bool
    let clearFailurePreservesCurrent: Bool
}

@MainActor
func verifyTranscriptPersistenceSafetyContract() async throws -> TranscriptPersistenceSafetyVerificationReport {
    let directory = FileManager.default.temporaryDirectory
        .appending(path: "mimi-persistence-verification-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let historyURL = directory.appending(path: "sessions.json")
    let history = TranscriptHistoryStore(fileURL: historyURL)
    let malformed = Data("{fixture damaged history".utf8)
    try malformed.write(to: historyURL)

    let storage = PersistenceVerificationStorage()
    let store = AppStore(historyStore: history, transcriptStorage: storage)
    let loadFailureIsReported = store.lastError != nil && store.historyRecords.isEmpty

    store.applyFixture(.final("Preserve this fixture transcript."), language: .english)
    let current = store.document
    let record = TranscriptSessionRecord(
        id: UUID(), startedAt: Date(timeIntervalSince1970: 1),
        endedAt: Date(timeIntervalSince1970: 2), source: .microphone,
        document: current
    )
    store.historyRecords = [record]
    store.selectedHistoryID = record.id
    store.newSession()
    let newSessionPreservesCurrentAfterArchiveFailure = store.document == current
        && storage.clearCalls == 0 && store.historyRecords == [record]
        && store.selectedHistoryID == record.id

    store.toggleRecording()
    for _ in 0..<20 { await Task.yield() }
    let startPreservesCurrentAfterArchiveFailure = store.document == current
        && storage.clearCalls == 0 && store.historyRecords == [record]
        && store.selectedHistoryID == record.id && !store.isTranscriptionSessionBusy

    let deletionRejected = !store.clearTranscript(historyID: record.id)
    let deleteFailurePreservesRecordAndSelection = deletionRejected
        && store.historyRecords == [record] && store.selectedHistoryID == record.id
    let failedOperationsPreserveHistoryBytes = try Data(contentsOf: historyURL) == malformed

    let otherRecord = TranscriptSessionRecord(
        id: UUID(), startedAt: Date(timeIntervalSince1970: 3),
        endedAt: Date(timeIntervalSince1970: 4), source: .outputAudio,
        document: current
    )
    try JSONEncoder().encode([record, otherRecord]).write(to: historyURL)
    store.historyRecords = [record, otherRecord]
    store.selectedHistoryID = otherRecord.id
    let capturedTargetDeleted = store.clearTranscript(historyID: record.id)
    let remaining = try history.load()
    let deletionUsesCapturedTarget = capturedTargetDeleted
        && store.historyRecords == [otherRecord] && remaining == [otherRecord]
        && store.selectedHistoryID == otherRecord.id

    store.historyRecords = []
    try history.save([])
    store.newSession()
    let archived = try history.load()
    let successfulArchiveSurvivesReload = archived.count == 1
        && archived.first?.document == current && store.historyRecords == archived
        && store.document.renderedText.isEmpty && storage.clearCalls == 1
        && store.selectedHistoryID == nil

    store.applyFixture(.final("Keep this when clearing fails."), language: .japanese)
    let clearFailureDocument = store.document
    storage.failsClear = true
    let clearFailurePreservesCurrent = !store.clearTranscript(historyID: nil)
        && store.document == clearFailureDocument && store.lastError != nil

    let passed = loadFailureIsReported
        && newSessionPreservesCurrentAfterArchiveFailure
        && startPreservesCurrentAfterArchiveFailure
        && deleteFailurePreservesRecordAndSelection
        && failedOperationsPreserveHistoryBytes
        && deletionUsesCapturedTarget
        && successfulArchiveSurvivesReload
        && clearFailurePreservesCurrent
    return .init(
        schemaVersion: 1,
        status: passed ? "passed" : "failed",
        loadFailureIsReported: loadFailureIsReported,
        newSessionPreservesCurrentAfterArchiveFailure: newSessionPreservesCurrentAfterArchiveFailure,
        startPreservesCurrentAfterArchiveFailure: startPreservesCurrentAfterArchiveFailure,
        deleteFailurePreservesRecordAndSelection: deleteFailurePreservesRecordAndSelection,
        failedOperationsPreserveHistoryBytes: failedOperationsPreserveHistoryBytes,
        deletionUsesCapturedTarget: deletionUsesCapturedTarget,
        successfulArchiveSurvivesReload: successfulArchiveSurvivesReload,
        clearFailurePreservesCurrent: clearFailurePreservesCurrent
    )
}

@MainActor
private final class PersistenceVerificationStorage: TranscriptPersisting {
    private(set) var clearCalls = 0
    var failsClear = false

    func loadLatestTranscript() -> TranscriptDocument { TranscriptDocument() }
    func saveLatestTranscript(_ document: TranscriptDocument) throws {}
    func clearLatestTranscript() throws {
        if failsClear { throw CocoaError(.fileWriteNoPermission) }
        clearCalls += 1
    }
    func makeTemporaryRecordingURL(fileExtension: String) throws -> URL {
        throw CocoaError(.featureUnsupported)
    }
    func removeTemporaryRecording(at url: URL) throws {}
    func removeStaleTemporaryRecordings() {}
}
