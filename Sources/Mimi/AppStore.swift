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
    @ObservationIgnored private let legacySessionIdentity: TranscriptSessionIdentity
    var historyRecords: [TranscriptSessionRecord]
    var selectedHistoryID: UUID?
    @ObservationIgnored var isVoiceTypingActive: @MainActor () -> Bool = { false }
    private var recordingTransitionPending = false

    init(
        loadPersistedTranscript: Bool = true,
        appleSpeech: (any AppleSpeechProviding)? = nil,
        whisper: (any WhisperAccuracyTranscribing)? = nil,
        phonon: (any WhisperAccuracyTranscribing)? = nil,
        japaneseFast: (any WhisperAccuracyTranscribing)? = nil,
        historyStore: TranscriptHistoryStore? = nil,
        transcriptStorage: (any TranscriptPersisting)? = nil
    ) {
        let historyStore = historyStore ?? (loadPersistedTranscript
            ? TranscriptHistoryStore()
            : TranscriptHistoryStore(fileURL: FileManager.default.temporaryDirectory
                .appending(path: "mimi-fixture-history-\(UUID().uuidString)/sessions.json")))
        self.historyStore = historyStore
        let historyLoadError: Error?
        let loadedRecords: [TranscriptSessionRecord]
        do {
            loadedRecords = loadPersistedTranscript ? try historyStore.load() : []
            historyRecords = loadedRecords
            historyLoadError = nil
        } catch {
            loadedRecords = []
            historyRecords = []
            historyLoadError = error
        }
        inputDevicesProvider = AudioDeviceCatalog.inputDevices
        outputDevicesProvider = AudioDeviceCatalog.outputDevices
        let appleSpeech = appleSpeech ?? SystemAppleSpeechProvider()
        let whisper = whisper ?? MimiWhisperMLXLiveEngine()
        let phonon = phonon ?? MimiPhononMLXLiveEngine()
        let japaneseFast = japaneseFast ?? MimiParakeetJapaneseLiveEngine()
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
                storage: transcriptStorage ?? (loadPersistedTranscript ? FileTranscriptStore() : TransientTranscriptStore()),
                inputDevices: AudioDeviceCatalog.inputDevices(),
                outputDevices: AudioDeviceCatalog.outputDevices(),
                phonon: phonon,
                japaneseFast: japaneseFast
            ),
            loadPersistedTranscript: loadPersistedTranscript
        )
        session = createdSession
        var loadedDocument = createdSession.document
        // Finals form an append-only segment chain. A successful history
        // commit can outlive a failed latest-cache write for the same owner.
        if let owner = loadedDocument.sessionIdentity,
           let archived = loadedRecords.first(where: { $0.id == owner.id }),
           archived.document.segments.count > loadedDocument.segments.count,
           Array(archived.document.segments.prefix(loadedDocument.segments.count)) == loadedDocument.segments {
            loadedDocument = archived.document
            loadedDocument.sessionIdentity = TranscriptSessionIdentity(id: archived.id, startedAt: archived.startedAt, source: archived.source)
            createdSession.document = loadedDocument
        }
        let recovered = loadedRecords.first {
            !loadedDocument.renderedText.isEmpty && $0.document.segments == loadedDocument.segments
                && $0.document.liveText == loadedDocument.liveText
        }
        legacySessionIdentity = loadedDocument.sessionIdentity ?? TranscriptSessionIdentity(
            id: recovered?.id ?? UUID(),
            startedAt: recovered?.startedAt ?? loadedDocument.segments.first?.createdAt ?? Date(),
            source: recovered?.source ?? createdSession.source
        )
        if let historyLoadError {
            session.lastError = historyLoadError.localizedDescription
        }
        Task { [weak createdSession] in
            await createdSession?.refreshSelectedModelReadiness()
        }
        if loadPersistedTranscript, let configuration = ExperimentalMLXTranslationConfiguration.resolved() {
            Task {
                try? await ExperimentalMLXTranslationEngine.shared.prepareForLiveTranslation(configuration: configuration)
            }
        }
    }

    var recordingState: RecordingState {
        recordingTransitionPending && session.recordingState == .idle ? .preparing : session.recordingState
    }
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
    var currentSessionIdentity: TranscriptSessionIdentity? {
        session.document.sessionIdentity ?? (session.document.renderedText.isEmpty ? nil : legacySessionIdentity)
    }
    var currentSessionID: UUID? { currentSessionIdentity?.id }
    var viewedSessionID: UUID? { selectedHistoryID ?? currentSessionID }
    var sessions: [TranscriptSessionRecord] {
        guard let identity = currentSessionIdentity else { return historyRecords }
        let working = TranscriptSessionRecord(
            id: identity.id, startedAt: identity.startedAt,
            endedAt: document.segments.last?.createdAt ?? identity.startedAt,
            source: identity.source, document: document
        )
        return [working] + historyRecords.filter { $0.id != identity.id }
    }
    var viewedDocument: TranscriptDocument {
        guard let selectedHistoryID,
              selectedHistoryID != currentSessionID,
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
    var isTranscriptionSessionBusy: Bool { session.controlsLocked || recordingTransitionPending }
    var controlsLocked: Bool { isTranscriptionSessionBusy || isVoiceTypingActive() }
    var modelPack: LocalModelPack? { session.modelPack }
    var canRemoveSelectedModel: Bool { session.canRemoveSelectedModel }
    var selectedModelReadiness: ModelReadiness { session.selectedModelReadiness }
    var bilingualAppleSpeechReadiness: ModelReadiness { session.bilingualAppleSpeechReadiness }
    var modelSetupState: ModelSetupState { session.modelSetupState }
    var selectedModelSetupState: ModelSetupState { session.selectedModelSetupState }
    var isModelSetupActive: Bool { session.modelSetupState.isActive }
    var canStartRecording: Bool { session.canStartRecording && !recordingTransitionPending && !isVoiceTypingActive() }
    var canInstallSelectedModel: Bool { session.canInstallSelectedModel }
    var canCancelSelectedModelInstall: Bool { session.canCancelSelectedModelInstall }

    func toggleRecording() {
        guard !isVoiceTypingActive(), !recordingTransitionPending else { return }
        let shouldStop = session.isRecording
        let destinationID = viewedSessionID
        recordingTransitionPending = true
        Task {
            defer { recordingTransitionPending = false }
            if shouldStop {
                await session.stopRecording()
                persistCurrentSession()
            } else {
                guard prepareSessionForRecording(historyID: destinationID) else { return }
                await session.startRecording()
            }
        }
    }

    /// The sidebar can change during model startup. The caller freezes the
    /// destination before awaiting; the working document carries its owner.
    @discardableResult
    func prepareSessionForRecording(historyID: UUID?) -> Bool {
        guard !session.controlsLocked, !isVoiceTypingActive() else { return false }
        var replacement: TranscriptDocument
        if let historyID, historyID != currentSessionID {
            guard let record = historyRecords.first(where: { $0.id == historyID }) else {
                session.lastError = "This session is no longer available. Choose another session."
                return false
            }
            replacement = record.document
            replacement.sessionIdentity = TranscriptSessionIdentity(id: record.id, startedAt: record.startedAt, source: record.source)
        } else {
            replacement = session.document
            replacement.sessionIdentity = currentSessionIdentity ?? TranscriptSessionIdentity(source: session.source)
        }
        do { _ = try historyStore.load() }
        catch { session.lastError = error.localizedDescription; return false }
        guard persistCurrentSession(), session.replaceTranscript(with: replacement) else { return false }
        if selectedHistoryID == historyID { selectedHistoryID = nil }
        return true
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
        guard !controlsLocked else { return false }
        if let historyID {
            let remainingRecords = historyRecords.filter { $0.id != historyID }
            let removesSavedRecord = remainingRecords.count != historyRecords.count
            let removesWorkingRecord = historyID == currentSessionID
            guard removesSavedRecord || removesWorkingRecord else { return true }
            do {
                if removesSavedRecord { try historyStore.save(remainingRecords) }
                if removesWorkingRecord {
                    guard session.clearTranscript() else {
                        if removesSavedRecord { try? historyStore.save(historyRecords) }
                        return false
                    }
                }
                historyRecords = remainingRecords
                if selectedHistoryID == historyID { selectedHistoryID = nil }
                if selectedHistoryID == nil && currentSessionID == nil {
                    selectedHistoryID = remainingRecords.first?.id
                }
                return true
            } catch {
                session.lastError = error.localizedDescription
                return false
            }
        } else {
            if let currentSessionID, historyRecords.contains(where: { $0.id == currentSessionID }) {
                return clearTranscript(historyID: currentSessionID)
            }
            let cleared = session.clearTranscript()
            if cleared && selectedHistoryID == nil { selectedHistoryID = historyRecords.first?.id }
            return cleared
        }
    }

    func selectCurrentSession() {
        selectedHistoryID = nil
    }

    func newSession() {
        guard !controlsLocked else { return }
        guard persistCurrentSession() else { return }
        let draft = TranscriptDocument(sessionIdentity: TranscriptSessionIdentity(source: session.source))
        guard session.replaceTranscript(with: draft) else { return }
        selectedHistoryID = nil
    }

    @discardableResult
    func persistCurrentSession() -> Bool {
        var document = session.document
        guard !document.renderedText.isEmpty, let identity = currentSessionIdentity else { return true }
        document.sessionIdentity = identity
        var updatedRecords = historyRecords.filter { $0.id != identity.id }
        updatedRecords.insert(
            TranscriptSessionRecord(
                id: identity.id,
                startedAt: identity.startedAt,
                endedAt: Date(),
                source: identity.source,
                document: document
            ),
            at: 0
        )
        do {
            try historyStore.save(updatedRecords)
            historyRecords = updatedRecords
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
