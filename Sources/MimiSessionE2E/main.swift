@preconcurrency import AVFoundation
import Foundation
import MimiCore
import MimiSession

@main
struct MimiSessionE2E {
    static func main() async {
        await appleStreamingEnglishSessionDrainsBoundedFrames()
        await whisperJapaneseAccuracySessionFreezesConfigurationAndCleansAudio()
        await nemotronJapaneseAccuracySessionUsesWAVAndCleansAudio()
        await screenAudioSelectionAndLifecycle()
        await screenAudioPickerCancellationAndUnexpectedStop()
        await screenAudioAccuracyPassAndUnexpectedStopCleanTemporaryAudio()
        await modelAndPermissionFailuresNeverStartCapture()
        await captureAndTranscriptionFailuresCleanUp()
        await modelLifecycleAndPersistenceEdges()
        transcriptAndTranslationRoutingEdgeCases()
        print("Mimi session E2E passed: model gates, capture lifecycle, EN/JA routing, cleanup, persistence, and realtime queue edges.")
    }

    @MainActor
    private static func appleStreamingEnglishSessionDrainsBoundedFrames() async {
        let capture = FakeCapture()
        capture.framesToEmitOnStart = 64
        let apple = FakeAppleProvider()
        let whisper = FakeWhisper(isDownloaded: true)
        let storage = FakeStorage()
        let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
        session.engineID = .appleSpeechAnalyzer
        session.sourceLanguage = .english

        await session.startRecording()
        await yieldToMainActor()

        expect(session.recordingState == .recording, "Apple session starts after an installed-capability check and microphone grant")
        expect(capture.startCount == 1, "Apple session starts one microphone capture")
        expect(capture.recordingURLs == [nil], "Apple live ASR never writes raw microphone audio")
        expect(apple.engine.consumedBufferCount == 32, "Realtime queue drops oldest frames at its bounded capacity")

        apple.engine.emit(.partial("local audio stays"))
        apple.engine.emit(.final("local audio stays on this Mac"))
        expect(session.document.liveText.isEmpty, "Apple final event clears volatile text")
        expect(session.document.segments.map(\.text) == ["local audio stays on this Mac"], "Apple final event is persisted")
        expect(storage.savedDocuments.count == 1, "Final live segments persist immediately")

        let staleCallback = capture.lastCallback
        await session.stopRecording()
        expect(session.recordingState == .idle, "Apple stop returns the session to idle")
        expect(apple.engine.stopCount == 1, "Apple engine finalizes exactly once")

        staleCallback?(FakeCapture.makeBuffer())
        await yieldToMainActor()
        expect(apple.engine.consumedBufferCount == 32, "A callback retained after Stop cannot feed stale audio into a future session")
    }

    @MainActor
    private static func whisperJapaneseAccuracySessionFreezesConfigurationAndCleansAudio() async {
        let capture = FakeCapture()
        let apple = FakeAppleProvider()
        let whisper = FakeWhisper(isDownloaded: true)
        whisper.transcription = "これはローカルの日本語文字起こしです。"
        let storage = FakeStorage()
        let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
        session.engineID = .whisperKitLargeV3Turbo
        session.sourceLanguage = .japanese

        await session.startRecording()
        expect(session.recordingState == .recording, "Installed Whisper starts an accuracy-pass recording")
        expect(storage.createdTemporaryURLs.count == 1, "Whisper gets one temporary source-audio URL")
        let temporaryURL = storage.createdTemporaryURLs[0]

        // Changing controls while capture is active must not alter the frozen
        // engine/language selected at Start.
        session.engineID = .appleSpeechAnalyzer
        session.sourceLanguage = .english
        await session.stopRecording()

        expect(
            whisper.transcribeCalls.count == 1 &&
                whisper.transcribeCalls[0].0 == temporaryURL &&
                whisper.transcribeCalls[0].1 == .japanese,
            "Whisper stop uses the frozen Japanese session configuration"
        )
        expect(session.document.segments.map(\.language) == [.japanese], "Whisper final text retains Japanese routing")
        expect(storage.removedTemporaryURLs == [temporaryURL], "Whisper source audio is deleted after transcription")
        expect(apple.makeEngineCalls == 0, "A mid-session picker change never swaps the active engine")
    }

    @MainActor
    private static func nemotronJapaneseAccuracySessionUsesWAVAndCleansAudio() async {
        let capture = FakeCapture()
        let apple = FakeAppleProvider()
        let whisper = FakeWhisper(isDownloaded: true)
        let nemotron = FakeNemotron(isDownloaded: true)
        nemotron.transcription = "これはMLXで処理した日本語の文字起こしです。"
        let storage = FakeStorage()
        let session = makeSession(
            capture: capture,
            apple: apple,
            whisper: whisper,
            nemotron: nemotron,
            storage: storage
        )
        session.engineID = .nemotronStreamingExperimental
        session.sourceLanguage = .japanese

        await session.startRecording()
        expect(session.recordingState == .recording, "Installed Nemotron starts an accuracy-pass recording")
        expect(storage.createdTemporaryURLs.count == 1, "Nemotron gets exactly one temporary source-audio URL")
        let temporaryURL = storage.createdTemporaryURLs[0]
        expect(temporaryURL.pathExtension == "wav", "Nemotron receives a portable WAV instead of Whisper's CAF")
        expect(capture.recordingURLs == [temporaryURL], "Nemotron's WAV is handed to the active microphone capture")

        // As with Whisper, the active session owns its configuration even if
        // the visible controls change while recording.
        session.engineID = .appleSpeechAnalyzer
        session.sourceLanguage = .english
        await session.stopRecording()

        expect(
            nemotron.transcribeCalls.count == 1 &&
                nemotron.transcribeCalls[0].0 == temporaryURL &&
                nemotron.transcribeCalls[0].1 == .japanese,
            "Nemotron stop uses the frozen Japanese session configuration"
        )
        expect(session.document.segments.map(\.language) == [.japanese], "Nemotron final text retains Japanese routing")
        expect(storage.removedTemporaryURLs == [temporaryURL], "Nemotron WAV is deleted after native MLX transcription")
        expect(whisper.transcribeCalls.isEmpty, "A Nemotron session never invokes Whisper")
    }

    @MainActor
    private static func screenAudioSelectionAndLifecycle() async {
        let microphone = FakeCapture(permissionGranted: false)
        let screen = FakeScreenAudioCapture()
        screen.framesToEmitOnStart = 1
        let apple = FakeAppleProvider()
        let whisper = FakeWhisper(isDownloaded: true)
        let storage = FakeStorage()
        let session = makeSession(
            capture: microphone,
            screen: screen,
            apple: apple,
            whisper: whisper,
            storage: storage
        )
        session.engineID = .appleSpeechAnalyzer
        session.source = .applicationAudio

        expect(!session.canStartRecording, "App audio cannot start until a person selects an app")
        await session.startRecording()
        expect(screen.startCount == 0, "No selection never starts ScreenCaptureKit audio")
        expect(microphone.permissionRequests == 0, "App audio never requests microphone access")
        expect(isFailed(session.recordingState), "Missing app selection is visible before recording")

        await session.selectScreenAudioContent()
        expect(
            screen.selectionRequests == [.applicationAudio] &&
                session.screenAudioSelection?.source == .applicationAudio,
            "The explicit picker action selects an app-audio source"
        )
        expect(session.canStartRecording, "A selected app makes app audio startable")

        await session.startRecording()
        await yieldToMainActor()
        expect(session.recordingState == .recording, "Selected app audio starts a live session")
        expect(screen.configureCount == 1 && screen.startCount == 1, "App audio configures and starts the screen audio lane")
        expect(microphone.startCount == 0, "App audio does not fall back to microphone capture")
        expect(apple.engine.consumedBufferCount == 1, "Screen audio PCM frames reach the live Apple engine")

        await session.stopRecording()
        expect(screen.stopCount == 1, "Stopping app audio stops the ScreenCaptureKit stream")
        expect(session.recordingState == .idle, "Stopping selected app audio returns to idle")
    }

    @MainActor
    private static func screenAudioPickerCancellationAndUnexpectedStop() async {
        let microphone = FakeCapture(permissionGranted: false)
        let screen = FakeScreenAudioCapture()
        screen.selectError = ProbeError.screenPickerCancelled
        let apple = FakeAppleProvider()
        let whisper = FakeWhisper(isDownloaded: true)
        let storage = FakeStorage()
        let session = makeSession(
            capture: microphone,
            screen: screen,
            apple: apple,
            whisper: whisper,
            storage: storage
        )
        session.engineID = .appleSpeechAnalyzer
        session.source = .systemAudio

        await session.selectScreenAudioContent()
        expect(screen.selectionRequests == [.systemAudio], "Display-audio picker uses the system-audio source")
        expect(session.recordingState == .idle && !session.isRecording, "Cancelling the picker leaves the session out of recording")
        expect(session.lastError == ProbeError.screenPickerCancelled.localizedDescription, "Picker cancellation remains a concise inline error")
        expect(session.screenAudioSelection == nil && !session.canStartRecording, "Cancelling the picker leaves no stale app or display selection")
        expect(microphone.permissionRequests == 0, "Cancelling display audio never requests microphone access")

        screen.selectError = nil
        await session.selectScreenAudioContent()
        await session.startRecording()
        expect(session.recordingState == .recording, "A selected display starts display-audio capture")
        expect(screen.startCount == 1, "Display audio uses the ScreenCaptureKit lane")

        screen.emitUnexpectedStop("The selected display stopped sharing audio.")
        await yieldToMainActor()
        expect(isFailed(session.recordingState) && !session.isRecording, "A system-driven stream stop exits recording with a recoverable error")
        expect(apple.engine.stopCount == 1, "An unexpected screen-audio stop finalizes the live engine")
        expect(screen.stopCount == 0, "The session does not try to stop a stream macOS already stopped")
    }

    @MainActor
    private static func screenAudioAccuracyPassAndUnexpectedStopCleanTemporaryAudio() async {
        do {
            let microphone = FakeCapture(permissionGranted: false)
            let screen = FakeScreenAudioCapture()
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: true)
            let nemotron = FakeNemotron(isDownloaded: true)
            nemotron.transcription = "selected display audio"
            let storage = FakeStorage()
            let session = makeSession(
                capture: microphone,
                screen: screen,
                apple: apple,
                whisper: whisper,
                nemotron: nemotron,
                storage: storage
            )
            session.engineID = .nemotronStreamingExperimental
            session.sourceLanguage = .english
            session.source = .systemAudio

            await session.selectScreenAudioContent()
            await session.startRecording()
            expect(session.recordingState == .recording, "Selected display audio can feed a native MLX accuracy pass")
            let temporaryURL = storage.createdTemporaryURLs[0]
            expect(temporaryURL.pathExtension == "wav", "ScreenCaptureKit accuracy capture uses the Nemotron WAV contract")
            expect(screen.recordingURLs == [temporaryURL], "ScreenCaptureKit receives the accuracy-pass output URL")
            expect(microphone.permissionRequests == 0 && microphone.startCount == 0, "Selected display audio never falls back to microphone capture")

            await session.stopRecording()
            expect(screen.stopCount == 1, "Stopping a ScreenCaptureKit accuracy pass stops the selected stream")
            expect(
                nemotron.transcribeCalls.count == 1 && nemotron.transcribeCalls[0].0 == temporaryURL,
                "Stopped selected display audio is handed to Nemotron exactly once"
            )
            expect(storage.removedTemporaryURLs == [temporaryURL], "Completed screen-audio WAV is deleted after transcription")
        }

        do {
            let microphone = FakeCapture(permissionGranted: false)
            let screen = FakeScreenAudioCapture()
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: true)
            let nemotron = FakeNemotron(isDownloaded: true)
            let storage = FakeStorage()
            let session = makeSession(
                capture: microphone,
                screen: screen,
                apple: apple,
                whisper: whisper,
                nemotron: nemotron,
                storage: storage
            )
            session.engineID = .nemotronStreamingExperimental
            session.source = .applicationAudio

            await session.selectScreenAudioContent()
            await session.startRecording()
            let temporaryURL = storage.createdTemporaryURLs[0]
            screen.emitUnexpectedStop("The selected app stopped sharing audio.")
            await yieldToMainActor()

            expect(isFailed(session.recordingState) && !session.isRecording, "An accuracy-pass stream stop is surfaced without entering transcription")
            expect(nemotron.transcribeCalls.isEmpty, "A system-stopped stream never transcribes incomplete temporary audio")
            expect(storage.removedTemporaryURLs == [temporaryURL], "A system-stopped screen-audio WAV is deleted immediately")
            expect(screen.stopCount == 0, "The system stream-stop path never double-stops ScreenCaptureKit")
            expect(microphone.permissionRequests == 0, "An app-audio stream stop never requests microphone permission")
        }
    }

    @MainActor
    private static func modelAndPermissionFailuresNeverStartCapture() async {
        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: false)
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
            session.engineID = .whisperKitLargeV3Turbo
            await session.startRecording()

            expect(capture.permissionRequests == 0, "Missing Whisper is rejected before a privacy prompt")
            expect(capture.startCount == 0, "Missing Whisper never starts capture")
            expect(storage.createdTemporaryURLs.isEmpty, "Missing Whisper creates no temporary audio file")
            expect(isFailed(session.recordingState), "Missing Whisper surfaces a failed state")
        }

        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: true)
            let nemotron = FakeNemotron(isDownloaded: false)
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, nemotron: nemotron, storage: storage)
            session.engineID = .nemotronStreamingExperimental
            await session.startRecording()

            expect(capture.permissionRequests == 0, "Missing Nemotron is rejected before microphone permission")
            expect(capture.startCount == 0, "Missing Nemotron never starts capture")
            expect(whisper.ensureCalls == 0, "Nemotron never touches the Whisper model")
            expect(nemotron.ensureCalls == 1, "Nemotron validates its own local model before capture")
            expect(isFailed(session.recordingState), "Missing Nemotron communicates an explicit failed state")
        }

        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: true)
            let nemotron = FakeNemotron(isDownloaded: true)
            nemotron.runtimeAvailabilityMessage = "Fixture MLX Metal runtime is missing."
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, nemotron: nemotron, storage: storage)
            session.engineID = .nemotronStreamingExperimental

            expect(
                session.selectedModelReadiness == .unavailable("Fixture MLX Metal runtime is missing."),
                "A missing MLX Metal runtime is distinguishable from a missing model download"
            )
            expect(!session.canInstallSelectedModel && !session.canStartRecording, "A missing native runtime disables both install and recording controls")
            await session.startRecording()

            expect(capture.permissionRequests == 0 && capture.startCount == 0, "A missing MLX runtime never prompts for or starts microphone capture")
            expect(storage.createdTemporaryURLs.isEmpty, "A missing MLX runtime creates no temporary source audio")
            expect(nemotron.ensureCalls == 1, "The native runtime is revalidated at the start boundary")
            expect(isFailed(session.recordingState), "A missing MLX runtime becomes a recoverable failed state when invoked directly")
        }

        do {
            let capture = FakeCapture(permissionGranted: false)
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: true)
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
            session.engineID = .appleSpeechAnalyzer
            await session.startRecording()

            expect(capture.startCount == 0, "Denied microphone permission never installs a tap")
            expect(apple.makeEngineCalls == 0, "Denied microphone permission never prepares the Apple engine")
            expect(isFailed(session.recordingState), "Denied microphone permission is visible to the user")
        }
    }

    @MainActor
    private static func captureAndTranscriptionFailuresCleanUp() async {
        do {
            let capture = FakeCapture()
            capture.startError = ProbeError.captureStart
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: true)
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
            session.engineID = .appleSpeechAnalyzer
            await session.startRecording()

            expect(apple.engine.startCount == 1, "Apple setup occurs before a failing hardware start")
            expect(apple.engine.stopCount == 1, "A failed capture start tears down the prepared Apple engine")
            expect(isFailed(session.recordingState), "Capture start failure is surfaced")
        }

        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: true)
            whisper.transcribeError = ProbeError.transcription
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
            session.engineID = .whisperKitLargeV3Turbo
            await session.startRecording()
            let temporaryURL = storage.createdTemporaryURLs[0]
            await session.stopRecording()

            expect(storage.removedTemporaryURLs == [temporaryURL], "Whisper source audio is deleted when transcription fails")
            expect(isFailed(session.recordingState), "Whisper transcription failure becomes a visible failed state")
        }

        do {
            let microphone = FakeCapture(permissionGranted: false)
            let screen = FakeScreenAudioCapture()
            screen.startError = ProbeError.captureStart
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: true)
            let storage = FakeStorage()
            let session = makeSession(
                capture: microphone,
                screen: screen,
                apple: apple,
                whisper: whisper,
                storage: storage
            )
            session.engineID = .whisperKitLargeV3Turbo
            session.source = .applicationAudio

            await session.selectScreenAudioContent()
            await session.startRecording()

            let temporaryURL = storage.createdTemporaryURLs[0]
            expect(isFailed(session.recordingState), "A ScreenCaptureKit start failure is visible to the user")
            expect(storage.removedTemporaryURLs == [temporaryURL], "A failed ScreenCaptureKit start removes its temporary source audio")
            expect(screen.stopCount == 0, "A stream that failed before activation is not stopped a second time")
            expect(microphone.permissionRequests == 0, "A screen-audio start failure does not request microphone access")
        }
    }

    @MainActor
    private static func modelLifecycleAndPersistenceEdges() async {
        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: false)
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
            session.engineID = .whisperKitLargeV3Turbo
            await session.installSelectedModelNow()

            expect(whisper.installCalls == 1 && whisper.isDownloaded, "Explicit Whisper install changes model readiness")
            expect(session.selectedModelReadiness == .ready, "Installed Whisper becomes startable")
            await session.removeSelectedModelNow()
            expect(whisper.removeCalls == 1 && !whisper.isDownloaded, "Whisper removal revokes the app-owned model")
        }

        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: true)
            let nemotron = FakeNemotron(isDownloaded: false)
            let storage = FakeStorage()
            let session = makeSession(
                capture: capture,
                apple: apple,
                whisper: whisper,
                nemotron: nemotron,
                storage: storage
            )
            session.engineID = .nemotronStreamingExperimental

            await session.installSelectedModelNow()
            expect(nemotron.installCalls == 1 && nemotron.isDownloaded, "Explicit Nemotron install changes model readiness")
            expect(session.selectedModelReadiness == .ready, "Installed Nemotron becomes startable without a hidden download")
            await session.removeSelectedModelNow()
            expect(nemotron.removeCalls == 1 && !nemotron.isDownloaded, "Nemotron removal revokes the app-owned local model")
        }

        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider(isAvailable: false)
            let whisper = FakeWhisper(isDownloaded: true)
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
            session.engineID = .appleSpeechAnalyzer
            await session.installSelectedModelNow()

            expect(apple.installCalls == 0, "Unavailable Apple Speech does not attempt asset installation")
            expect(isFailed(session.recordingState), "Unavailable Apple Speech reports a clear failure")
        }

        do {
            var initial = TranscriptDocument()
            initial.apply(.final("previous session"), language: .english)
            let storage = FakeStorage(initialDocument: initial)
            let session = makeSession(capture: FakeCapture(), apple: FakeAppleProvider(), whisper: FakeWhisper(isDownloaded: true), storage: storage)
            expect(storage.removedStaleCount == 1, "Session launch cleans stale temporary recordings")
            expect(session.document.segments.map(\.text) == ["previous session"], "Latest finalized transcript restores locally")

            session.clearTranscript()
            expect(storage.clearCalls == 1 && session.document.renderedText.isEmpty, "Clear removes persisted and in-memory transcript together")
        }
    }

    private static func transcriptAndTranslationRoutingEdgeCases() {
        var document = TranscriptDocument()
        document.apply(.partial("volatile English"), language: .english)
        document.apply(.final("final English"), language: .english)
        document.apply(.final("最終の日本語"), language: .japanese)
        document.apply(.final("final English"), language: .english)

        expect(document.finalizedText(for: .english) == "final English\nfinal English", "Translation input preserves legitimate repeated final English speech")
        expect(document.finalizedText(for: .japanese) == "最終の日本語", "Translation input filters immutable text by its source language")
        expect(document.liveText.isEmpty, "Final events never leave a volatile translation input behind")
    }

    @MainActor
    private static func makeSession(
        capture: FakeCapture,
        screen: FakeScreenAudioCapture = FakeScreenAudioCapture(),
        apple: FakeAppleProvider,
        whisper: FakeWhisper,
        nemotron: FakeNemotron = FakeNemotron(isDownloaded: false),
        storage: FakeStorage
    ) -> TranscriptionSession {
        TranscriptionSession(
            dependencies: .init(
                microphoneCapture: capture,
                screenAudioCapture: screen,
                appleSpeech: apple,
                whisper: whisper,
                nemotron: nemotron,
                storage: storage,
                inputDevices: [.init(id: 42, name: "Fixture Microphone")]
            ),
            loadPersistedTranscript: true
        )
    }

    @MainActor
    private static func yieldToMainActor() async {
        await Task.yield()
        await Task.yield()
    }

    private static func isFailed(_ state: RecordingState) -> Bool {
        if case .failed = state { return true }
        return false
    }

    private static func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
        guard condition() else { fatalError("Mimi session E2E failure: \(message)") }
    }
}

@MainActor
private final class FakeCapture: MicrophoneCapturing {
    var permissionGranted: Bool
    var configureError: Error?
    var startError: Error?
    var stopError: Error?
    var framesToEmitOnStart = 0
    private(set) var permissionRequests = 0
    private(set) var configureDeviceIDs: [UInt32?] = []
    private(set) var startCount = 0
    private(set) var stopCount = 0
    private(set) var recordingURLs: [URL?] = []
    private(set) var lastCallback: (@Sendable (AVAudioPCMBuffer) -> Void)?

    init(permissionGranted: Bool = true) {
        self.permissionGranted = permissionGranted
    }

    func requestPermission() async -> Bool {
        permissionRequests += 1
        return permissionGranted
    }

    func configureInput(deviceID: UInt32?) throws -> AVAudioFormat {
        configureDeviceIDs.append(deviceID)
        if let configureError { throw configureError }
        return AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)!
    }

    func start(
        recordingTo url: URL?,
        deviceID: UInt32?,
        onBuffer: @escaping @Sendable (AVAudioPCMBuffer) -> Void
    ) throws {
        startCount += 1
        recordingURLs.append(url)
        lastCallback = onBuffer
        if let startError { throw startError }
        for _ in 0..<framesToEmitOnStart {
            onBuffer(Self.makeBuffer())
        }
    }

    func stop() throws -> URL? {
        stopCount += 1
        if let stopError { throw stopError }
        return recordingURLs.last ?? nil
    }

    static func makeBuffer(channels: AVAudioChannelCount = 1) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: channels)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 256)!
        buffer.frameLength = 256
        return buffer
    }
}

@MainActor
private final class FakeScreenAudioCapture: ScreenAudioCapturing {
    var selectedContent: ScreenAudioSelection?
    var selectError: Error?
    var configureError: Error?
    var startError: Error?
    var stopError: Error?
    var framesToEmitOnStart = 0
    private(set) var selectionRequests: [AudioSource] = []
    private(set) var configureCount = 0
    private(set) var startCount = 0
    private(set) var stopCount = 0
    private(set) var recordingURLs: [URL?] = []
    private(set) var lastCallback: (@Sendable (AVAudioPCMBuffer) -> Void)?
    private var onStreamStopped: (@MainActor @Sendable (String?) -> Void)?

    func selectContent(for source: AudioSource) async throws {
        selectionRequests.append(source)
        if let selectError { throw selectError }
        selectedContent = ScreenAudioSelection(source: source, description: "Fixture \(source.displayName)")
    }

    func configureInput() throws -> AVAudioFormat {
        configureCount += 1
        if let configureError { throw configureError }
        return AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 2)!
    }

    func start(
        recordingTo url: URL?,
        onBuffer: @escaping @Sendable (AVAudioPCMBuffer) -> Void,
        onStreamStopped: @escaping @MainActor @Sendable (String?) -> Void
    ) async throws {
        startCount += 1
        recordingURLs.append(url)
        lastCallback = onBuffer
        self.onStreamStopped = onStreamStopped
        if let startError { throw startError }
        for _ in 0..<framesToEmitOnStart {
            onBuffer(FakeCapture.makeBuffer(channels: 2))
        }
    }

    func stop() async throws -> URL? {
        stopCount += 1
        if let stopError { throw stopError }
        onStreamStopped = nil
        return recordingURLs.last ?? nil
    }

    func emitUnexpectedStop(_ message: String?) {
        onStreamStopped?(message)
    }
}

@MainActor
private final class FakeAppleEngine: AppleLiveTranscribing {
    var startError: Error?
    private(set) var startCount = 0
    private(set) var stopCount = 0
    private(set) var consumedBufferCount = 0
    private(set) var startedLanguages: [SpeechLanguage] = []
    private var onEvent: (@MainActor (TranscriptEvent) -> Void)?

    func start(
        language: SpeechLanguage,
        inputFormat: AVAudioFormat,
        onEvent: @escaping @MainActor (TranscriptEvent) -> Void
    ) async throws {
        startCount += 1
        startedLanguages.append(language)
        if let startError { throw startError }
        self.onEvent = onEvent
    }

    func consume(_ buffer: AVAudioPCMBuffer) {
        consumedBufferCount += 1
    }

    func stop() async {
        stopCount += 1
    }

    func emit(_ event: TranscriptEvent) {
        onEvent?(event)
    }
}

@MainActor
private final class FakeAppleProvider: AppleSpeechProviding {
    let engine = FakeAppleEngine()
    var isAvailable: Bool
    var installError: Error?
    var makeEngineError: Error?
    private(set) var installCalls = 0
    private(set) var makeEngineCalls = 0

    init(isAvailable: Bool = true) {
        self.isAvailable = isAvailable
    }

    func installAssets(for language: SpeechLanguage) async throws {
        installCalls += 1
        if let installError { throw installError }
    }

    func makeEngine() throws -> any AppleLiveTranscribing {
        makeEngineCalls += 1
        if let makeEngineError { throw makeEngineError }
        return engine
    }
}

@MainActor
private final class FakeWhisper: WhisperAccuracyTranscribing {
    var isDownloaded: Bool
    var ensureError: Error?
    var installError: Error?
    var transcribeError: Error?
    var transcription = ""
    private(set) var ensureCalls = 0
    private(set) var installCalls = 0
    private(set) var removeCalls = 0
    private(set) var transcribeCalls: [(URL, SpeechLanguage)] = []

    init(isDownloaded: Bool) {
        self.isDownloaded = isDownloaded
    }

    func ensureInstalled() throws {
        ensureCalls += 1
        if let ensureError { throw ensureError }
        guard isDownloaded else { throw ProbeError.modelNotInstalled }
    }

    func install() async throws {
        installCalls += 1
        if let installError { throw installError }
        isDownloaded = true
    }

    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String {
        transcribeCalls.append((url, language))
        if let transcribeError { throw transcribeError }
        return transcription
    }

    func removeDownloadedModel() async throws {
        removeCalls += 1
        isDownloaded = false
    }
}

@MainActor
private final class FakeNemotron: NemotronMLXAccuracyTranscribing {
    var runtimeAvailabilityMessage: String?
    var isDownloaded: Bool
    var ensureError: Error?
    var installError: Error?
    var transcribeError: Error?
    var transcription = ""
    private(set) var ensureCalls = 0
    private(set) var installCalls = 0
    private(set) var removeCalls = 0
    private(set) var transcribeCalls: [(URL, SpeechLanguage)] = []

    init(isDownloaded: Bool) {
        self.isDownloaded = isDownloaded
    }

    func ensureInstalled() throws {
        ensureCalls += 1
        if let runtimeAvailabilityMessage { throw ProbeError.runtimeUnavailable(runtimeAvailabilityMessage) }
        if let ensureError { throw ensureError }
        guard isDownloaded else { throw ProbeError.modelNotInstalled }
    }

    func install() async throws {
        installCalls += 1
        if let installError { throw installError }
        isDownloaded = true
    }

    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String {
        transcribeCalls.append((url, language))
        if let transcribeError { throw transcribeError }
        return transcription
    }

    func removeDownloadedModel() async throws {
        removeCalls += 1
        isDownloaded = false
    }
}

@MainActor
private final class FakeStorage: TranscriptPersisting {
    var initialDocument: TranscriptDocument
    var saveError: Error?
    var clearError: Error?
    var makeTemporaryError: Error?
    private(set) var savedDocuments: [TranscriptDocument] = []
    private(set) var clearCalls = 0
    private(set) var createdTemporaryURLs: [URL] = []
    private(set) var removedTemporaryURLs: [URL] = []
    private(set) var removedStaleCount = 0

    init(initialDocument: TranscriptDocument = TranscriptDocument()) {
        self.initialDocument = initialDocument
    }

    func loadLatestTranscript() -> TranscriptDocument {
        initialDocument
    }

    func saveLatestTranscript(_ document: TranscriptDocument) throws {
        if let saveError { throw saveError }
        savedDocuments.append(document)
    }

    func clearLatestTranscript() throws {
        if let clearError { throw clearError }
        clearCalls += 1
    }

    func makeTemporaryRecordingURL(fileExtension: String) throws -> URL {
        if let makeTemporaryError { throw makeTemporaryError }
        let url = URL(fileURLWithPath: "/tmp/mimi-e2e-\(createdTemporaryURLs.count).\(fileExtension)")
        createdTemporaryURLs.append(url)
        return url
    }

    func removeTemporaryRecording(at url: URL) throws {
        removedTemporaryURLs.append(url)
    }

    func removeStaleTemporaryRecordings() {
        removedStaleCount += 1
    }
}

private enum ProbeError: LocalizedError {
    case captureStart
    case transcription
    case modelNotInstalled
    case screenPickerCancelled
    case runtimeUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .captureStart: "Fixture capture start failed"
        case .transcription: "Fixture Whisper transcription failed"
        case .modelNotInstalled: "Fixture Whisper model is missing"
        case .screenPickerCancelled: "No app or display was selected."
        case let .runtimeUnavailable(message): message
        }
    }
}
