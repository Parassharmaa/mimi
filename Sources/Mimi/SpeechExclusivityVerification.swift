import Foundation
import MimiCore
import MimiSession

struct SpeechExclusivityReport: Codable {
    let status: String
    let startupReservedBeforeAssetCheck: Bool
    let competingRecordingBlocked: Bool
    let reservationReleasedAfterFailure: Bool
}

@MainActor
func verifySpeechExclusivity() async -> SpeechExclusivityReport {
    let apple = SuspendedAppleAssetProvider()
    let store = AppStore(loadPersistedTranscript: false, appleSpeech: apple)
    for _ in 0..<100 where !store.canStartRecording { await Task.yield() }
    apple.suspendNextCheck = true
    store.toggleRecording()
    let reserved = store.isTranscriptionSessionBusy && store.controlsLocked && !store.canStartRecording
    for _ in 0..<100 where apple.continuation == nil { await Task.yield() }
    let calls = apple.statusCalls
    store.toggleRecording()
    for _ in 0..<10 { await Task.yield() }
    let competingBlocked = apple.continuation != nil && apple.statusCalls == calls && apple.engineCreations == 0
    apple.continuation?.resume(returning: .unsupported)
    apple.continuation = nil
    for _ in 0..<100 where store.isTranscriptionSessionBusy { await Task.yield() }
    let released = !store.isTranscriptionSessionBusy
    return .init(status: reserved && competingBlocked && released ? "passed" : "failed",
                 startupReservedBeforeAssetCheck: reserved, competingRecordingBlocked: competingBlocked,
                 reservationReleasedAfterFailure: released)
}

@MainActor
private final class SuspendedAppleAssetProvider: AppleSpeechProviding {
    var suspendNextCheck = false
    var statusCalls = 0
    var engineCreations = 0
    var continuation: CheckedContinuation<AppleSpeechAssetStatus, Never>?
    var isPlatformAvailable: Bool { true }
    func assetStatus(for language: SpeechLanguage) async -> AppleSpeechAssetStatus {
        statusCalls += 1
        if suspendNextCheck {
            suspendNextCheck = false
            return await withCheckedContinuation { continuation = $0 }
        }
        return .installed
    }
    func installAssets(for language: SpeechLanguage) async throws {}
    func makeEngine() throws -> any AppleLiveTranscribing {
        engineCreations += 1
        throw NSError(domain: "SpeechExclusivityVerification", code: 1)
    }
}
