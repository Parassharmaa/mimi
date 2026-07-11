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

    init(loadPersistedTranscript: Bool = true) {
        inputDevicesProvider = AudioDeviceCatalog.inputDevices
        let createdSession = TranscriptionSession(
            dependencies: .init(
                microphoneCapture: MicrophoneCapture(),
                screenAudioCapture: ScreenAudioCapture(),
                appleSpeech: SystemAppleSpeechProvider(),
                whisper: WhisperKitAccuracyEngine(),
                nemotron: NemotronMLXLiveEngine(),
                storage: FileTranscriptStore(),
                inputDevices: AudioDeviceCatalog.inputDevices()
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
    var engineID: TranscriptionEngineID {
        get { session.engineID }
        set { session.engineID = newValue }
    }
    var translationMode: TranslationMode {
        get { session.translationMode }
        set { session.translationMode = newValue }
    }
    var document: TranscriptDocument { session.document }
    var lastError: String? { session.lastError }
    var screenAudioSelection: ScreenAudioSelection? { session.screenAudioSelection }
    var inputDevices: [AudioInputDevice] { session.inputDevices }
    var selectedInputDeviceID: UInt32? {
        get { session.selectedInputDeviceID }
        set { session.selectedInputDeviceID = newValue }
    }
    var menuBarSymbolName: String { session.menuBarSymbolName }
    var isRecording: Bool { session.isRecording }
    var controlsLocked: Bool { session.controlsLocked }
    var modelPack: LocalModelPack? { session.modelPack }
    var canRemoveSelectedModel: Bool { session.canRemoveSelectedModel }
    var selectedModelReadiness: ModelReadiness { session.selectedModelReadiness }
    var modelSetupState: ModelSetupState { session.modelSetupState }
    var selectedModelSetupState: ModelSetupState { session.selectedModelSetupState }
    var isModelSetupActive: Bool { session.modelSetupState.isActive }
    var canStartRecording: Bool { session.canStartRecording }
    var canInstallSelectedModel: Bool { session.canInstallSelectedModel }
    var canCancelSelectedModelInstall: Bool { session.canCancelSelectedModelInstall }

    func toggleRecording() {
        session.toggleRecording()
    }

    func installSelectedModel() {
        session.installSelectedModel()
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
        session.clearTranscript()
    }

    func refreshInputDevices() {
        session.replaceInputDevices(inputDevicesProvider())
    }

    func selectScreenAudioContent() {
        Task { await session.selectScreenAudioContent() }
    }

    func copyTranscript() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(document.renderedText, forType: .string)
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
