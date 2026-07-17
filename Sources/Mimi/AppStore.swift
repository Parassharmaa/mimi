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

    init(loadPersistedTranscript: Bool = true) {
        let historyStore = TranscriptHistoryStore()
        self.historyStore = historyStore
        historyRecords = loadPersistedTranscript ? historyStore.load() : []
        inputDevicesProvider = AudioDeviceCatalog.inputDevices
        outputDevicesProvider = AudioDeviceCatalog.outputDevices
        let appleSpeech = SystemAppleSpeechProvider()
        let createdSession = TranscriptionSession(
            dependencies: .init(
                microphoneCapture: MicrophoneCapture(),
                outputAudioCapture: OutputAudioCapture(),
                screenAudioCapture: ScreenAudioCapture(),
                appleSpeech: appleSpeech,
                automaticAppleSpeech: AutomaticAppleSpeechEngine(appleSpeech: appleSpeech),
                whisper: WhisperKitAccuracyEngine(),
                nemotron: NemotronMLXLiveEngine(),
                qwen: QwenMLXLiveEngine(),
                storage: FileTranscriptStore(),
                inputDevices: AudioDeviceCatalog.inputDevices(),
                outputDevices: AudioDeviceCatalog.outputDevices()
            ),
            loadPersistedTranscript: loadPersistedTranscript
        )
        session = createdSession
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
    var controlsLocked: Bool { session.controlsLocked }
    var modelPack: LocalModelPack? { session.modelPack }
    var canRemoveSelectedModel: Bool { session.canRemoveSelectedModel }
    var selectedModelReadiness: ModelReadiness { session.selectedModelReadiness }
    var bilingualAppleSpeechReadiness: ModelReadiness { session.bilingualAppleSpeechReadiness }
    var modelSetupState: ModelSetupState { session.modelSetupState }
    var selectedModelSetupState: ModelSetupState { session.selectedModelSetupState }
    var isModelSetupActive: Bool { session.modelSetupState.isActive }
    var canStartRecording: Bool { session.canStartRecording }
    var canInstallSelectedModel: Bool { session.canInstallSelectedModel }
    var canCancelSelectedModelInstall: Bool { session.canCancelSelectedModelInstall }

    func toggleRecording() {
        if session.isRecording {
            Task {
                await session.stopRecording()
                archiveCurrentSessionIfNeeded()
            }
        } else {
            Task {
                if !session.document.renderedText.isEmpty {
                    archiveCurrentSessionIfNeeded()
                    session.clearTranscript()
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

    func clearTranscript() {
        if let selectedHistoryID {
            historyRecords.removeAll { $0.id == selectedHistoryID }
            self.selectedHistoryID = nil
            try? historyStore.save(historyRecords)
        } else {
            session.clearTranscript()
        }
    }

    func selectCurrentSession() {
        selectedHistoryID = nil
    }

    func newSession() {
        guard !controlsLocked else { return }
        if !session.document.renderedText.isEmpty {
            archiveCurrentSessionIfNeeded()
        }
        session.clearTranscript()
        selectedHistoryID = nil
        recordingStartedAt = nil
    }

    private func archiveCurrentSessionIfNeeded() {
        let document = session.document
        guard !document.renderedText.isEmpty else {
            recordingStartedAt = nil
            return
        }
        let start = recordingStartedAt ?? document.segments.first?.createdAt ?? Date()
        if let existingIndex = historyRecords.firstIndex(where: { $0.document == document }) {
            historyRecords.remove(at: existingIndex)
        }
        historyRecords.insert(
            TranscriptSessionRecord(
                id: UUID(),
                startedAt: start,
                endedAt: Date(),
                source: session.source,
                document: document
            ),
            at: 0
        )
        recordingStartedAt = nil
        do {
            try historyStore.save(historyRecords)
        } catch {
            session.lastError = error.localizedDescription
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
