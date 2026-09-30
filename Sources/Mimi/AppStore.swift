import AppKit
import MimiCore
import MimiSession
import Observation

/// SwiftUI-facing facade over the testable transcription session. The session
/// owns capture/model lifecycle; this facade owns only macOS UI conveniences.
@MainActor
@Observable
final class AppStore {
    @ObservationIgnored private let session: TranscriptionSession
    @ObservationIgnored private let inputDevicesProvider: () -> [AudioInputDevice]
    @ObservationIgnored private let outputDevicesProvider: () -> [AudioOutputDevice]
    @ObservationIgnored private let historyStore: TranscriptHistoryStore
    @ObservationIgnored private var recordingStartedAt: Date?
    var historyRecords: [TranscriptSessionRecord]
    var selectedHistoryID: UUID?
    @ObservationIgnored var isVoiceTypingActive: @MainActor () -> Bool = { false }
    private var recordingStartPending = false

    init(
        loadPersistedTranscript: Bool = true,
        appleSpeech: (any AppleSpeechProviding)? = nil,
        whisper: (any WhisperAccuracyTranscribing)? = nil,
        phonon: (any WhisperAccuracyTranscribing)? = nil,
        historyStore: TranscriptHistoryStore? = nil,
        transcriptStorage: (any TranscriptPersisting)? = nil
    ) {
        let historyStore = historyStore ?? TranscriptHistoryStore()
        self.historyStore = historyStore
        let historyLoadError: Error?
        do {
            historyRecords = loadPersistedTranscript ? try historyStore.load() : []
            historyLoadError = nil
        } catch {
            historyRecords = []
            historyLoadError = error
        }
        inputDevicesProvider = AudioDeviceCatalog.inputDevices
        outputDevicesProvider = AudioDeviceCatalog.outputDevices
        let appleSpeech = appleSpeech ?? SystemAppleSpeechProvider()
        let whisper = whisper ?? MimiWhisperMLXLiveEngine()
        let phonon = phonon ?? MimiPhononMLXLiveEngine()
        let createdSession = TranscriptionSession(
            dependencies: .init(
                microphoneCapture: MicrophoneCapture(),
                outputAudioCapture: OutputAudioCapture(),
                screenAudioCapture: ScreenAudioCapture(),
                appleSpeech: appleSpeech,
                automaticAppleSpeech: AutomaticAppleSpeechEngine(appleSpeech: appleSpeech),
                whisper: whisper,
                nemotron: NemotronMLXLiveEngine(),
                qwen: QwenMLXLiveEngine(),
                storage: transcriptStorage ?? FileTranscriptStore(),
                inputDevices: AudioDeviceCatalog.inputDevices(),
                outputDevices: AudioDeviceCatalog.outputDevices(),
                phonon: phonon
            ),
            loadPersistedTranscript: loadPersistedTranscript
        )
        session = createdSession
        if let historyLoadError {
            session.lastError = historyLoadError.localizedDescription
        }
        Task { [weak createdSession] in
            await createdSession?.refreshSelectedModelReadiness()
        }
    }

    var recordingState: RecordingState { session.recordingState }
    var source: AudioSource {
        get { session.source }
        set { session.source = newValue }
    }
    var sourceLanguage: SpeechLanguage {
        get { session.sourceLanguage }
        set { session.sourceLanguage = newValue }
    }
    var languageMode: TranscriptionLanguageMode {
        get { session.languageMode }
        set { session.languageMode = newValue }
    }
    var selectableLanguageModes: [TranscriptionLanguageMode] { session.selectableLanguageModes }
    var detectedLanguage: SpeechLanguage? { session.detectedLanguage }
    var engineID: TranscriptionEngineID {
        get { session.engineID }
        set { session.engineID = newValue }
    }
    var translationMode: TranslationMode {
        get { session.translationMode }
        set { session.translationMode = newValue }
    }
    var document: TranscriptDocument { session.document }
    var viewedDocument: TranscriptDocument {
        guard let selectedHistoryID,
              let record = historyRecords.first(where: { $0.id == selectedHistoryID }) else {
            return session.document
        }
        return record.document
    }
    var lastError: String? { session.lastError }
    var screenAudioSelection: ScreenAudioSelection? { session.screenAudioSelection }
    var inputDevices: [AudioInputDevice] { session.inputDevices }
    var outputDevices: [AudioOutputDevice] { session.outputDevices }
    var selectedInputDeviceID: UInt32? {
        get { session.selectedInputDeviceID }
        set { session.selectedInputDeviceID = newValue }
    }
    var selectedOutputDeviceID: UInt32? {
        get { session.selectedOutputDeviceID }
        set { session.selectedOutputDeviceID = newValue }
    }
    var menuBarSymbolName: String { session.menuBarSymbolName }
    var isRecording: Bool { session.isRecording }
    var isTranscriptionSessionBusy: Bool { session.controlsLocked || recordingStartPending }
    var controlsLocked: Bool { isTranscriptionSessionBusy || isVoiceTypingActive() }
    var modelPack: LocalModelPack? { session.modelPack }
    var canRemoveSelectedModel: Bool { session.canRemoveSelectedModel }
    var selectedModelReadiness: ModelReadiness { session.selectedModelReadiness }
    var bilingualAppleSpeechReadiness: ModelReadiness { session.bilingualAppleSpeechReadiness }
    var modelSetupState: ModelSetupState { session.modelSetupState }
    var selectedModelSetupState: ModelSetupState { session.selectedModelSetupState }
    var isModelSetupActive: Bool { session.modelSetupState.isActive }
    var canStartRecording: Bool { session.canStartRecording && !recordingStartPending && !isVoiceTypingActive() }
    var canInstallSelectedModel: Bool { session.canInstallSelectedModel }
    var canCancelSelectedModelInstall: Bool { session.canCancelSelectedModelInstall }

    func toggleRecording() {
        guard !isVoiceTypingActive(), !recordingStartPending else { return }
        if session.isRecording {
            Task {
                await session.stopRecording()
                archiveCurrentSessionIfNeeded()
            }
        } else {
            recordingStartPending = true
            Task {
                defer { recordingStartPending = false }
                guard !isVoiceTypingActive() else { return }
                if !session.document.renderedText.isEmpty {
                    guard archiveCurrentSessionIfNeeded() else { return }
                    session.clearTranscript()
                    guard session.document.renderedText.isEmpty else { return }
                }
                selectedHistoryID = nil
                recordingStartedAt = Date()
                await session.startRecording()
            }
        }
    }

    func installSelectedModel() {
        session.installSelectedModel()
    }

    func prepareBilingualAppleSpeechNow() async {
        await session.prepareBilingualAppleSpeechNow()
    }

    func cancelSelectedModelInstall() {
        session.cancelSelectedModelInstall()
    }

    func refreshSelectedModelReadiness() {
        Task { await session.refreshSelectedModelReadiness() }
    }

    func removeSelectedModel() {
        session.removeSelectedModel()
    }

    @discardableResult
    func clearTranscript() -> Bool {
        clearTranscript(historyID: selectedHistoryID)
    }

    @discardableResult
    func clearTranscript(historyID: UUID?) -> Bool {
        if let historyID {
            let remainingRecords = historyRecords.filter { $0.id != historyID }
            guard remainingRecords.count != historyRecords.count else { return true }
            do {
                try historyStore.save(remainingRecords)
                historyRecords = remainingRecords
                if selectedHistoryID == historyID { selectedHistoryID = nil }
                return true
            } catch {
                session.lastError = error.localizedDescription
                return false
            }
        } else {
            session.clearTranscript()
            return session.document.renderedText.isEmpty
        }
    }

    func selectCurrentSession() {
        selectedHistoryID = nil
    }

    func newSession() {
        guard !controlsLocked else { return }
        if !session.document.renderedText.isEmpty {
            guard archiveCurrentSessionIfNeeded() else { return }
        }
        session.clearTranscript()
        guard session.document.renderedText.isEmpty else { return }
        selectedHistoryID = nil
        recordingStartedAt = nil
    }

    @discardableResult
    private func archiveCurrentSessionIfNeeded() -> Bool {
        let document = session.document
        guard !document.renderedText.isEmpty else {
            recordingStartedAt = nil
            return true
        }
        let start = recordingStartedAt ?? document.segments.first?.createdAt ?? Date()
        var updatedRecords = historyRecords.filter { $0.document != document }
        updatedRecords.insert(
            TranscriptSessionRecord(
                id: UUID(),
                startedAt: start,
                endedAt: Date(),
                source: session.source,
                document: document
            ),
            at: 0
        )
        do {
            try historyStore.save(updatedRecords)
            historyRecords = updatedRecords
            recordingStartedAt = nil
            return true
        } catch {
            session.lastError = error.localizedDescription
            return false
        }
    }

    func refreshInputDevices() {
        session.replaceInputDevices(inputDevicesProvider())
    }

    func refreshOutputDevices() {
        session.replaceOutputDevices(outputDevicesProvider())
    }

    func selectScreenAudioContent() {
        Task { await session.selectScreenAudioContent() }
    }

    func reportError(_ error: Error) {
        session.lastError = error.localizedDescription
    }

    func copyTranscript() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(viewedDocument.renderedText, forType: .string)
    }

    func applyFixture(_ event: TranscriptEvent, language: SpeechLanguage) {
        session.applyFixture(event, language: language)
    }

    /// Development-only fixture used by the deterministic UI smoke launch.
    func applyPresentationFixture(state: RecordingState, lastError: String? = nil) {
        session.recordingState = state
        session.lastError = lastError
    }

    func runMicrophoneCaptureSmokeTest() async throws -> Int {
        try await session.runMicrophoneCaptureSmokeTest()
    }

    func runEngineSmokeTest(engine: TranscriptionEngineID, language: SpeechLanguage) async throws {
        try await session.runEngineSmokeTest(engine: engine, language: language)
    }
}
