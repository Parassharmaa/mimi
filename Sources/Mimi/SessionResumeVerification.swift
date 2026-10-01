import Foundation
import MimiCore
import MimiSession

struct SessionResumeVerificationReport: Codable {
    let status: String
    let selectedContentPreserved: Bool
    let selectedIdentityPreserved: Bool
    let originalStartDatePreserved: Bool
    let checks: [String: Bool]
}

@MainActor
func verifySessionResumeContract() async throws -> SessionResumeVerificationReport {
    let directory = FileManager.default.temporaryDirectory.appending(path: "mimi-resume-verification-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let history = TranscriptHistoryStore(fileURL: directory.appending(path: "sessions.json"))
    let existing = TranscriptDocument(segments: [.init(text: "Existing Japanese session.", language: .japanese)])
    let record = TranscriptSessionRecord(id: UUID(), startedAt: Date(timeIntervalSince1970: 100), endedAt: Date(timeIntervalSince1970: 200), source: .microphone, document: existing)
    try history.save([record])
    let storage = SessionResumeVerificationStorage()
    let store = AppStore(appleSpeech: SessionResumeUnavailableApple(), historyStore: history, transcriptStorage: storage)
    store.applyFixture(.final("Another working session."), language: .english)
    store.selectedHistoryID = record.id
    store.toggleRecording()
    for _ in 0..<100 where store.isTranscriptionSessionBusy { await Task.yield() }
    let selectedContentPreserved = store.document.segments == existing.segments
    let selectedIdentityPreserved = store.document.sessionIdentity?.id == record.id
    let originalStartDatePreserved = store.document.sessionIdentity?.startedAt == record.startedAt
    var checks: [String: Bool] = [:]
    store.applyFixture(.final("Continue in English."), language: .english)
    let continued = store.document
    // Real finalized ASR events save latest before the Stop/history commit.
    try storage.saveLatestTranscript(continued)
    let recovered = AppStore(appleSpeech: SessionResumeUnavailableApple(), historyStore: history, transcriptStorage: storage)
    checks["interruptedAppendRecoversOwnerAndText"] = recovered.currentSessionID == record.id
        && recovered.document == continued && recovered.sessions.filter { $0.id == record.id }.count == 1

    let other = TranscriptSessionRecord(id: UUID(), startedAt: Date(timeIntervalSince1970: 300), endedAt: Date(timeIntervalSince1970: 400), source: .outputAudio, document: existing)
    store.historyRecords.append(other)
    try history.save(store.historyRecords)
    store.selectedHistoryID = other.id
    let committed = store.persistCurrentSession()
    let saved = try history.load()
    checks["browsingDoesNotRedirectRecordingOwner"] = committed && store.currentSessionID == record.id
        && store.viewedDocument.segments == other.document.segments
        && saved.first(where: { $0.id == record.id })?.document == continued
        && saved.first(where: { $0.id == other.id }) == other
    checks["appendPreservesOriginalSegmentsAndLanguages"] = continued.segments.first == existing.segments.first
        && continued.segments.last?.language == .english
    let repeated = store.prepareSessionForRecording(historyID: record.id) && store.persistCurrentSession()
    checks["repeatedResumeUpsertsOriginalIDAndStartDate"] = repeated
        && store.historyRecords.filter { $0.id == record.id }.count == 1
        && store.historyRecords.first(where: { $0.id == record.id })?.startedAt == record.startedAt
        && store.document == continued

    store.selectedHistoryID = other.id
    storage.failsSave = true
    checks["failedSwitchPreservesWorkingValueAndSelection"] = !store.prepareSessionForRecording(historyID: other.id)
        && store.document == continued && store.currentSessionID == record.id && store.selectedHistoryID == other.id
        && storage.loadLatestTranscript() == continued
    store.newSession()
    checks["failedNewDraftPreservesWorkingValue"] = store.document == continued && store.currentSessionID == record.id
    storage.failsSave = false
    store.selectedHistoryID = nil
    storage.failsClear = true
    let beforeDelete = store.historyRecords
    let deleteRejected = !store.clearTranscript(historyID: record.id)
    let afterRejectedDelete = try history.load()
    checks["failedCurrentDeleteRestoresArchiveAndLatest"] = deleteRejected
        && store.document == continued && storage.loadLatestTranscript() == continued
        && Set(afterRejectedDelete.map(\.id)) == Set(beforeDelete.map(\.id))
    storage.failsClear = false
    checks["deletedCurrentSessionDoesNotReappear"] = store.clearTranscript(historyID: record.id)
        && store.currentSessionID == nil && !store.sessions.contains(where: { $0.id == record.id })

    let firstHistory = TranscriptHistoryStore(fileURL: directory.appending(path: "first-session.json"))
    let firstStorage = SessionResumeVerificationStorage()
    let first = AppStore(appleSpeech: SessionResumeUnavailableApple(), historyStore: firstHistory, transcriptStorage: firstStorage)
    checks["emptyStateStartsOwnedDraft"] = first.sessions.isEmpty && first.prepareSessionForRecording(historyID: nil)
        && first.currentSessionID != nil && first.sessions.count == 1 && first.document.renderedText.isEmpty
    let firstID = first.currentSessionID
    first.applyFixture(.final("Same text."), language: .english)
    first.newSession()
    let secondID = first.currentSessionID
    first.applyFixture(.final("Same text."), language: .english)
    let sameTextSaved = first.persistCurrentSession()
    checks["plusCreatesSeparateSessionAndSameTextIsRetained"] = sameTextSaved && firstID != secondID
        && firstID != nil && secondID != nil
        && Set(first.historyRecords.map(\.id)) == Set([firstID, secondID].compactMap { $0 })
    let firstBeforeVoice = first.document
    first.isVoiceTypingActive = { true }
    checks["voiceTypingExclusivityPreservesWorkingValue"] = !first.prepareSessionForRecording(historyID: firstID)
        && first.document == firstBeforeVoice

    let legacy = try JSONDecoder().decode(TranscriptDocument.self, from: JSONEncoder().encode(existing))
    let legacyStorage = SessionResumeVerificationStorage()
    try legacyStorage.saveLatestTranscript(legacy)
    let legacyHistory = TranscriptHistoryStore(fileURL: directory.appending(path: "legacy.json"))
    try legacyHistory.save([record])
    let migrated = AppStore(appleSpeech: SessionResumeUnavailableApple(), historyStore: legacyHistory, transcriptStorage: legacyStorage)
    checks["legacyRawTranscriptStillDecodesAndReconnects"] = legacy.sessionIdentity == nil
        && migrated.currentSessionID == record.id && migrated.prepareSessionForRecording(historyID: record.id)
        && migrated.document.segments == existing.segments && migrated.document.sessionIdentity?.id == record.id
    let startupHistory = TranscriptHistoryStore(fileURL: directory.appending(path: "startup.json"))
    try startupHistory.save([record, other])
    let waitingApple = SessionResumeWaitingApple()
    let starting = AppStore(appleSpeech: waitingApple, historyStore: startupHistory, transcriptStorage: SessionResumeVerificationStorage())
    for _ in 0..<100 where !starting.canStartRecording { await Task.yield() }
    waitingApple.suspendNextCheck = true
    starting.selectedHistoryID = record.id
    starting.toggleRecording()
    let reservedImmediately = starting.isTranscriptionSessionBusy
    starting.selectedHistoryID = other.id
    for _ in 0..<100 where waitingApple.continuation == nil { await Task.yield() }
    let destinationFrozen = starting.currentSessionID == record.id && starting.selectedHistoryID == other.id
    let calls = waitingApple.calls
    starting.toggleRecording()
    for _ in 0..<10 { await Task.yield() }
    let duplicateBlocked = waitingApple.calls == calls
    waitingApple.continuation?.resume(returning: .unsupported)
    waitingApple.continuation = nil
    for _ in 0..<100 where starting.isTranscriptionSessionBusy { await Task.yield() }
    checks["startupFreezesDestinationButPreservesNewSidebarSelection"] = reservedImmediately && destinationFrozen
        && starting.currentSessionID == record.id && starting.selectedHistoryID == other.id
    checks["rapidDoubleStartIsBlockedUntilStartupSettles"] = duplicateBlocked && !starting.isTranscriptionSessionBusy
    var richer = continued
    richer.apply(.final("Safely archived after a latest-cache failure."), language: .english)
    let staleStorage = SessionResumeVerificationStorage()
    try staleStorage.saveLatestTranscript(continued)
    staleStorage.failsSave = true
    try? staleStorage.saveLatestTranscript(richer)
    let richerHistory = TranscriptHistoryStore(fileURL: directory.appending(path: "richer.json"))
    try richerHistory.save([TranscriptSessionRecord(id: record.id, startedAt: record.startedAt, endedAt: Date(), source: record.source, document: richer)])
    let restoredRicher = AppStore(appleSpeech: SessionResumeUnavailableApple(), historyStore: richerHistory, transcriptStorage: staleStorage)
    checks["newerHistoryCannotBeOverwrittenByStaleLatestCache"] = restoredRicher.document.segments == richer.segments
        && restoredRicher.currentSessionID == record.id
    staleStorage.failsSave = false
    let safelyResumedRicher = restoredRicher.prepareSessionForRecording(historyID: record.id)
    let richerAfterResume = try richerHistory.load()
    checks["resumingRecoveredHistoryKeepsEveryArchivedSegment"] = safelyResumedRicher
        && richerAfterResume.first?.document.segments == richer.segments
    first.isVoiceTypingActive = { false }
    let capturedDeleteID = first.viewedSessionID
    first.newSession()
    let replacementDraft = first.document
    let capturedDeletion = first.clearTranscript(historyID: capturedDeleteID)
    checks["capturedDeleteCannotClearAReplacementDraft"] = capturedDeletion
        && first.document == replacementDraft && first.currentSessionID != capturedDeleteID
        && !first.historyRecords.contains(where: { $0.id == capturedDeleteID })
    first.applyFixture(.final("Unarchived working text."), language: .english)
    let unarchivedID = first.viewedSessionID
    checks["currentUUIDCanBeDeletedWithoutAnArchiveEntry"] = first.clearTranscript(historyID: unarchivedID)
        && first.document.renderedText.isEmpty && first.currentSessionID == nil
    let passed = selectedContentPreserved && selectedIdentityPreserved && originalStartDatePreserved && checks.values.allSatisfy { $0 }
    return .init(status: passed ? "passed" : "failed", selectedContentPreserved: selectedContentPreserved, selectedIdentityPreserved: selectedIdentityPreserved, originalStartDatePreserved: originalStartDatePreserved, checks: checks)
}

@MainActor
private final class SessionResumeVerificationStorage: TranscriptPersisting {
    private var latest = TranscriptDocument()
    var failsSave = false
    var failsClear = false
    func loadLatestTranscript() -> TranscriptDocument { latest }
    func saveLatestTranscript(_ document: TranscriptDocument) throws {
        if failsSave { throw CocoaError(.fileWriteNoPermission) }
        latest = document
    }
    func clearLatestTranscript() throws {
        if failsClear { throw CocoaError(.fileWriteNoPermission) }
        latest = TranscriptDocument()
    }
    func makeTemporaryRecordingURL(fileExtension: String) throws -> URL { throw CocoaError(.featureUnsupported) }
    func removeTemporaryRecording(at url: URL) throws {}
    func removeStaleTemporaryRecordings() {}
}

@MainActor
private final class SessionResumeUnavailableApple: AppleSpeechProviding {
    var isPlatformAvailable: Bool { true }
    func assetStatus(for language: SpeechLanguage) async -> AppleSpeechAssetStatus { .unsupported }
    func installAssets(for language: SpeechLanguage) async throws { throw CocoaError(.featureUnsupported) }
    func makeEngine() throws -> any AppleLiveTranscribing { throw CocoaError(.featureUnsupported) }
}

@MainActor
private final class SessionResumeWaitingApple: AppleSpeechProviding {
    var suspendNextCheck = false
    var calls = 0
    var continuation: CheckedContinuation<AppleSpeechAssetStatus, Never>?
    var isPlatformAvailable: Bool { true }
    func assetStatus(for language: SpeechLanguage) async -> AppleSpeechAssetStatus {
        calls += 1
        if suspendNextCheck {
            suspendNextCheck = false
            return await withCheckedContinuation { continuation = $0 }
        }
        return .installed
    }
    func installAssets(for language: SpeechLanguage) async throws { throw CocoaError(.featureUnsupported) }
    func makeEngine() throws -> any AppleLiveTranscribing { throw CocoaError(.featureUnsupported) }
}
