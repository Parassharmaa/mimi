@preconcurrency import AVFoundation
import Foundation
import MimiCore
import MimiSession

@main
struct MimiSessionE2E {
    static func main() async {
        await appleStreamingEnglishSessionDrainsBoundedFrames()
        await whisperJapaneseAccuracySessionFreezesConfigurationAndCleansAudio()
        await nemotronJapaneseLiveSessionStreamsAndFinalizesWithoutAudioFiles()
        await screenAudioSelectionAndLifecycle()
        await screenAudioPickerCancellationAndUnexpectedStop()
        await screenAudioLiveNemotronAndUnexpectedStopCleanLiveState()
        await modelAndPermissionFailuresNeverStartCapture()
        await captureAndTranscriptionFailuresCleanUp()
        await modelLifecycleAndPersistenceEdges()
        await modelSetupStateMachinesAreTruthfulAndRecoverable()
        await appleStartBoundaryRejectsFreshMissingAssetDespiteStaleReadyCache()
        await appleStartBoundaryAbandonsSupersededSelectionAfterAssetLookup()
        liveFlowControlStaysBoundedAndFinalizesAtSafeBoundaries()
        transcriptAndTranslationRoutingEdgeCases()
        print("Mimi session E2E passed: model setup states, capture lifecycle, EN/JA routing, cleanup, persistence, and realtime queue edges.")
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
    private static func nemotronJapaneseLiveSessionStreamsAndFinalizesWithoutAudioFiles() async {
        let capture = FakeCapture()
        capture.framesToEmitOnStart = 2
        let apple = FakeAppleProvider()
        let whisper = FakeWhisper(isDownloaded: true)
        let nemotron = FakeNemotron(isDownloaded: true)
        nemotron.livePartialText = "これはMLXで処理中の日本語です。"
        nemotron.liveFinalText = "これはMLXで処理した日本語の文字起こしです。"
        nemotron.liveBackpressureMessage = "Fixture live inference is behind."
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
        await yieldToMainActor()
        expect(session.recordingState == .recording, "Installed Nemotron starts a local live session")
        expect(storage.createdTemporaryURLs.isEmpty && capture.recordingURLs == [nil], "Live Nemotron never writes raw microphone audio")
        expect(nemotron.liveStartLanguages == [.japanese], "Nemotron receives the frozen Japanese live-language route")
        expect(session.document.liveText == nemotron.livePartialText, "Nemotron emits a replaceable Japanese live hypothesis while audio arrives")
        expect(session.lastError == "Fixture live inference is behind." && session.recordingState == .recording, "Live backpressure remains nonfatal and visible while capture continues")

        // The active live model owns its configuration even if the visible
        // controls change while recording.
        session.engineID = .appleSpeechAnalyzer
        session.sourceLanguage = .english
        await session.stopRecording()

        expect(nemotron.liveStopCalls == 1, "Stopping a live Nemotron session flushes exactly once")
        expect(nemotron.liveCancelCalls == 0, "A normally finalized live session is not cancelled during cleanup")
        expect(session.document.segments.map(\.language) == [.japanese], "Nemotron final text retains the frozen Japanese route")
        expect(session.document.segments.map(\.text) == [nemotron.liveFinalText], "Nemotron finalizes the bounded live hypothesis once")
        expect(session.document.liveText.isEmpty, "Nemotron finalization clears the volatile live text")
        expect(storage.removedTemporaryURLs.isEmpty, "Live Nemotron has no temporary source audio to remove")
        expect(whisper.transcribeCalls.isEmpty, "A Nemotron session never invokes Whisper")

        let staleCallback = capture.lastCallback
        staleCallback?(FakeCapture.makeBuffer())
        await yieldToMainActor()
        expect(nemotron.liveConsumedBufferCount == 2, "A callback retained after Stop cannot feed a stale live Nemotron session")
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
    private static func screenAudioLiveNemotronAndUnexpectedStopCleanLiveState() async {
        do {
            let microphone = FakeCapture(permissionGranted: false)
            let screen = FakeScreenAudioCapture()
            screen.framesToEmitOnStart = 1
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: true)
            let nemotron = FakeNemotron(isDownloaded: true)
            nemotron.livePartialText = "selected display live audio"
            nemotron.liveFinalText = "selected display final audio"
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
            await yieldToMainActor()
            expect(session.recordingState == .recording, "Selected display audio can feed native MLX live transcription")
            expect(screen.recordingURLs == [nil], "Live ScreenCaptureKit transcription does not retain selected display audio")
            expect(microphone.permissionRequests == 0 && microphone.startCount == 0, "Selected display audio never falls back to microphone capture")
            expect(session.document.liveText == nemotron.livePartialText, "Selected display PCM reaches the live Nemotron engine")

            await session.stopRecording()
            expect(screen.stopCount == 1, "Stopping a ScreenCaptureKit live session stops the selected stream")
            expect(nemotron.liveStopCalls == 1, "Stopped selected display audio flushes the live Nemotron session exactly once")
            expect(storage.removedTemporaryURLs.isEmpty, "Live screen-audio transcription has no source-audio file to delete")

            let staleCallback = screen.lastCallback
            staleCallback?(FakeCapture.makeBuffer(channels: 2))
            await yieldToMainActor()
            expect(nemotron.liveConsumedBufferCount == 1, "A ScreenCaptureKit callback retained after Stop cannot mutate a future live session")
        }

        do {
            let microphone = FakeCapture(permissionGranted: false)
            let screen = FakeScreenAudioCapture()
            screen.framesToEmitOnStart = 1
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: true)
            let nemotron = FakeNemotron(isDownloaded: true)
            nemotron.livePartialText = "selected app live audio"
            nemotron.liveFinalText = "selected app final audio"
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
            await yieldToMainActor()
            screen.emitUnexpectedStop("The selected app stopped sharing audio.")
            await yieldToMainActor()

            expect(isFailed(session.recordingState) && !session.isRecording, "A live stream stop remains a recoverable capture error")
            expect(nemotron.liveStopCalls == 1 && nemotron.liveCancelCalls == 0, "A system-stopped stream flushes rather than discarding live MLX captions")
            expect(session.document.segments.map(\.text) == [nemotron.liveFinalText], "A system-stopped stream retains its finalized local live transcript")
            expect(storage.savedDocuments.contains(where: { $0.segments.map(\.text) == [nemotron.liveFinalText] }), "The recovered transcript persists before the capture error is shown")
            expect(storage.removedTemporaryURLs.isEmpty, "A system-stopped live session leaves no temporary audio behind")
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
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: true)
            let nemotron = FakeNemotron(isDownloaded: true)
            nemotron.liveStartError = ProbeError.transcription
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, nemotron: nemotron, storage: storage)
            session.engineID = .nemotronStreamingExperimental

            await session.startRecording()

            expect(capture.permissionRequests == 1 && capture.startCount == 0, "A failed live-model start never installs a microphone tap")
            expect(storage.createdTemporaryURLs.isEmpty, "A failed live-model start never creates source audio")
            expect(isFailed(session.recordingState), "A failed live-model start becomes a recoverable recording error")
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
            expect(
                session.selectedModelReadiness == .unavailable("Apple Speech live transcription requires macOS 26 or later."),
                "Unavailable Apple Speech remains unavailable instead of showing a misleading download action"
            )
            expect(isSetupFailed(session.selectedModelSetupState), "Unavailable Apple Speech retains a setup-specific explanation")
            expect(session.recordingState == .idle, "A setup failure does not masquerade as a failed recording")
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

        do {
            var initial = TranscriptDocument()
            initial.apply(.final("keep this transcript"), language: .english)
            let storage = FakeStorage(initialDocument: initial)
            storage.clearError = ProbeError.transcription
            let session = makeSession(capture: FakeCapture(), apple: FakeAppleProvider(), whisper: FakeWhisper(isDownloaded: true), storage: storage)

            session.clearTranscript()
            expect(session.document.renderedText == "keep this transcript" && isFailed(session.recordingState), "A failed clear keeps the local transcript visible and recoverable")

            storage.clearError = nil
            session.clearTranscript()
            expect(storage.clearCalls == 1 && session.document.renderedText.isEmpty, "Retrying a failed clear removes the transcript exactly once")
        }
    }

    @MainActor
    private static func modelSetupStateMachinesAreTruthfulAndRecoverable() async {
        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            apple.assetStatuses[.english] = .installed
            apple.assetStatuses[.japanese] = .supported
            let whisper = FakeWhisper(isDownloaded: true)
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
            session.engineID = .appleSpeechAnalyzer
            session.sourceLanguage = .japanese

            await session.refreshSelectedModelReadiness()
            expect(
                session.selectedModelReadiness == .needsDownload("Download the macOS-managed Japanese Apple Speech asset before recording."),
                "Apple setup distinguishes a supported Japanese asset from an installed one"
            )
            expect(session.canInstallSelectedModel && !session.canStartRecording, "A missing Apple language asset enables setup and disables recording")

            await session.startRecording()
            expect(capture.permissionRequests == 0 && capture.startCount == 0, "Missing Apple assets are rejected before microphone permission or capture")

            apple.statusAfterInstall = .installed
            await session.installSelectedModelNow()
            expect(apple.installCalls == 1, "Apple asset install is explicit and language-scoped")
            expect(session.selectedModelReadiness == .ready && session.canStartRecording, "Apple is ready only after a post-install status check says installed")
            expect(session.recordingState == .idle, "Successful model setup returns the recorder to idle")

            session.sourceLanguage = .english
            await session.refreshSelectedModelReadiness()
            expect(session.selectedModelReadiness == .ready, "An installed English asset remains independently ready")
            session.sourceLanguage = .japanese
            expect(session.selectedModelReadiness == .ready, "The installed Japanese status is cached independently from English")
        }

        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            apple.assetStatuses[.japanese] = .supported
            apple.statusAfterInstall = .downloading
            let whisper = FakeWhisper(isDownloaded: true)
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
            session.engineID = .appleSpeechAnalyzer
            session.sourceLanguage = .japanese

            await session.installSelectedModelNow()
            expect(isWaitingForSystem(session.selectedModelSetupState), "A post-install Apple download that continues in macOS is shown as waiting, not success")
            expect(!session.canStartRecording && !session.canInstallSelectedModel, "A macOS-managed Apple download cannot record or start duplicate installers")
            await session.startRecording()
            expect(capture.permissionRequests == 0 && capture.startCount == 0, "A waiting Apple asset still blocks capture before permission")
        }

        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            // The first two reads are the setup preflight and post-install
            // recheck. The third is the background poll scheduled while
            // macOS continues the system-owned download.
            apple.assetStatusSequence[.english] = [.supported, .downloading, .installed]
            apple.statusAfterInstall = .downloading
            let whisper = FakeWhisper(isDownloaded: true)
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)

            await session.installSelectedModelNow()
            expect(isWaitingForSystem(session.selectedModelSetupState), "Apple reports a truthful waiting state before the background system poll completes")
            expect(!session.canStartRecording, "Apple remains non-recordable while the system download is in flight")

            // `scheduleAppleSpeechDownloadRefresh` polls after five seconds.
            // Leave a small deterministic margin for the main-actor update.
            try? await Task.sleep(for: .seconds(6))
            await yieldToMainActor()

            expect(session.selectedModelSetupState == .idle, "A polled installed Apple asset clears the waiting setup state")
            expect(session.selectedModelReadiness == .ready && session.canStartRecording, "A polled installed Apple asset becomes recordable without a manual status refresh")
        }

        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: false)
            whisper.progressEvents = [.init(completedUnitCount: 1, totalUnitCount: 4)]
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
            session.engineID = .whisperKitLargeV3Turbo

            var sawProgress = false
            whisper.afterProgress = {
                guard case let .downloading(engine, _, progress) = session.selectedModelSetupState else { return }
                sawProgress = engine == .whisperKitLargeV3Turbo &&
                    progress?.completedUnitCount == 1 &&
                    progress?.totalUnitCount == 4 &&
                    progress?.fractionCompleted == 0.25 &&
                    !session.controlsLocked
            }
            await session.installSelectedModelNow()
            expect(sawProgress, "Whisper setup exposes truthful file-unit progress without locking recording controls")
            expect(session.selectedModelReadiness == .ready, "Whisper becomes ready after its explicit install and prewarm")

            whisper.isDownloaded = false
            whisper.installError = CancellationError()
            await session.installSelectedModelNow()
            expect(isSetupCancelled(session.selectedModelSetupState), "A cancelled Whisper download is not shown as a generic failure")
            expect(session.recordingState == .idle && session.canInstallSelectedModel, "A cancelled Whisper download remains retryable without failing the recorder")
        }

        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: false)
            whisper.progressEvents = [.init(completedUnitCount: 1, totalUnitCount: 2)]
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
            session.engineID = .whisperKitLargeV3Turbo

            var cancelRemainedAvailableAfterSelectionChange = false
            whisper.afterProgress = {
                // Either the session locks the picker at its boundary, or it
                // permits this change but keeps the in-flight download
                // globally cancellable. In neither design may a download be
                // orphaned merely because the visible selection changes.
                session.engineID = .appleSpeechAnalyzer
                cancelRemainedAvailableAfterSelectionChange = session.canCancelSelectedModelInstall
                session.cancelSelectedModelInstall()
            }

            session.installSelectedModel()
            await yieldToMainActor()

            expect(cancelRemainedAvailableAfterSelectionChange, "An in-flight Whisper setup remains recoverable after a model-selection attempt")
            expect(isSetupCancelled(session.modelSetupState), "Cancelling an in-flight task updates the global setup state even if the selection changed")
            expect(!whisper.isDownloaded, "Cancelling an in-flight Whisper task never marks the model ready")

            // Returning to the original model must expose a retryable state,
            // regardless of whether the selection change was accepted or
            // blocked while the task was active.
            session.engineID = .whisperKitLargeV3Turbo
            expect(isSetupCancelled(session.selectedModelSetupState) && session.canInstallSelectedModel, "The interrupted Whisper setup is retryable after returning to its model")

            whisper.afterProgress = nil
            await session.installSelectedModelNow()
            expect(session.selectedModelSetupState == .idle && session.selectedModelReadiness == .ready, "A retried Whisper setup recovers cleanly after cancellation")
        }

        do {
            let capture = FakeCapture()
            let apple = FakeAppleProvider()
            let whisper = FakeWhisper(isDownloaded: false)
            whisper.installError = ProbeError.modelDownload
            let storage = FakeStorage()
            let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)
            session.engineID = .whisperKitLargeV3Turbo

            await session.installSelectedModelNow()
            expect(isSetupFailed(session.selectedModelSetupState), "A non-cancellation Whisper download error is retained as a model-setup failure")
            expect(session.selectedModelReadiness.canStart == false && session.canInstallSelectedModel, "A failed Whisper download disables recording but exposes Retry")
            expect(session.recordingState == .idle && capture.permissionRequests == 0 && capture.startCount == 0, "A failed model download never becomes a failed recording or starts capture")

            whisper.installError = nil
            await session.installSelectedModelNow()
            expect(whisper.installCalls == 2, "Retry invokes the Whisper installer again after a non-cancellation failure")
            expect(session.selectedModelSetupState == .idle && session.selectedModelReadiness == .ready, "A successful Retry clears the Whisper failure and makes it ready")
        }
    }

    @MainActor
    private static func appleStartBoundaryRejectsFreshMissingAssetDespiteStaleReadyCache() async {
        let capture = FakeCapture()
        let apple = FakeAppleProvider()
        let whisper = FakeWhisper(isDownloaded: true)
        let storage = FakeStorage()
        let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)

        // First establish the normal, previously-ready cache value.
        apple.scriptedAssetStatusResponses = [.immediate(.installed)]
        await session.refreshSelectedModelReadiness()
        expect(session.selectedModelReadiness == .ready, "The race fixture begins from an installed Apple Speech cache")

        // Start observes a fresh, missing asset but suspends before it can
        // return. A competing refresh subsequently preserves the old ready
        // cache. The start boundary must honor its own fresh result rather
        // than trusting that newer cache generation.
        apple.scriptedAssetStatusResponses = [
            .suspended(.supported),
            .immediate(.installed)
        ]
        let startTask = Task { @MainActor in
            await session.startRecording()
        }
        await waitUntil(
            { apple.suspendedAssetStatusRequestCount == 1 },
            "The start-boundary Apple asset lookup should reach its controlled suspension"
        )

        await session.refreshSelectedModelReadiness()
        expect(session.selectedModelReadiness == .ready, "The competing refresh deliberately retains the stale ready cache")

        apple.resumeSuspendedAssetStatus()
        await startTask.value

        expect(capture.permissionRequests == 0 && capture.startCount == 0, "A fresh missing Apple asset never prompts for or starts microphone capture even when the cache remains ready")
        expect(apple.makeEngineCalls == 0, "A fresh missing Apple asset never constructs a live SpeechAnalyzer")
        expect(isFailed(session.recordingState), "The start boundary surfaces the fresh missing-asset result instead of recording")
    }

    @MainActor
    private static func appleStartBoundaryAbandonsSupersededSelectionAfterAssetLookup() async {
        let capture = FakeCapture()
        let apple = FakeAppleProvider()
        let whisper = FakeWhisper(isDownloaded: true)
        let storage = FakeStorage()
        let session = makeSession(capture: capture, apple: apple, whisper: whisper, storage: storage)

        // Mimic a normal ready menu state, then make only the next
        // start-boundary AssetInventory lookup suspend.
        apple.scriptedAssetStatusResponses = [.immediate(.installed)]
        await session.refreshSelectedModelReadiness()
        expect(session.canStartRecording, "The superseded-selection fixture starts from a recordable Apple Speech configuration")

        apple.scriptedAssetStatusResponses = [.suspended(.installed)]
        let startTask = Task { @MainActor in
            await session.startRecording()
        }
        await waitUntil(
            { apple.suspendedAssetStatusRequestCount == 1 },
            "The start-boundary installed-asset lookup should suspend before capture"
        )

        // A person may change every session-affecting control while macOS is
        // resolving the asset check. The stale start must be abandoned rather
        // than capturing with its old microphone/Apple configuration.
        session.source = .systemAudio
        session.engineID = .whisperKitLargeV3Turbo
        session.sourceLanguage = .japanese
        session.selectedInputDeviceID = 42

        apple.resumeSuspendedAssetStatus()
        await startTask.value

        expect(session.recordingState == .idle && !session.isRecording, "A superseded start attempt leaves the session idle")
        expect(capture.permissionRequests == 0 && capture.startCount == 0, "A superseded start never requests microphone access or starts a microphone tap")
        expect(apple.makeEngineCalls == 0 && apple.engine.startCount == 0, "A superseded start never constructs or starts the old Apple Speech engine")
        expect(whisper.ensureCalls == 0, "A superseded start does not silently begin the newly selected Whisper session")
        expect(storage.createdTemporaryURLs.isEmpty, "A superseded start creates no accuracy-pass recording file")
        expect(
            session.source == .systemAudio &&
                session.engineID == .whisperKitLargeV3Turbo &&
                session.sourceLanguage == .japanese &&
                session.selectedInputDeviceID == 42,
            "A superseded start preserves the person's new source, model, language, and microphone selection"
        )
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

    private static func liveFlowControlStaysBoundedAndFinalizesAtSafeBoundaries() {
        var queue = BoundedAudioSampleQueue(maximumSampleCount: 160, preferredChunkSize: 40)
        expect(queue.append(Array(repeating: 0.1, count: 120)) == 0, "A live queue accepts normal audio without dropping it")
        expect(queue.append(Array(repeating: 0.2, count: 80)) == 40, "A slow live consumer drops only oldest whole decode chunks")
        expect(queue.count == 160, "Live backpressure keeps queued PCM memory bounded")
        expect(queue.dequeue(upTo: 40).allSatisfy { $0 == 0.1 }, "Backpressure preserves newest queued audio after discarding the oldest chunk")
        expect(queue.dequeue(upTo: 200).count == 120 && queue.isEmpty, "A bounded live queue flushes deterministically at Stop")

        var policy = BoundedLiveWindowPolicy(
            minimumWindowSampleCount: 30,
            maximumWindowSampleCount: 300,
            silenceBoundarySampleCount: 8,
            silenceRMS: 0.01
        )
        expect(policy.append(Array(repeating: 0.1, count: 29)) == .none, "A short spoken live window stays provisional")
        expect(policy.append(Array(repeating: 0, count: 8)) == .silence, "A sufficient pause finalizes a bounded live window")
        policy.reset()
        expect(policy.append(Array(repeating: 0.1, count: 300)) == .maximumDuration, "Continuous speech finalizes at the hard live-window cap")
        policy.reset()
        expect(policy.append(Array(repeating: 0.1, count: 1)) == .none, "A reset starts a fresh live window after finalization")
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

    @MainActor
    private static func waitUntil(
        _ condition: @escaping @MainActor () -> Bool,
        _ failureMessage: String
    ) async {
        for _ in 0..<100 {
            if condition() { return }
            await Task.yield()
        }
        fatalError("Mimi session E2E failure: \(failureMessage)")
    }

    private static func isFailed(_ state: RecordingState) -> Bool {
        if case .failed = state { return true }
        return false
    }

    private static func isSetupFailed(_ state: ModelSetupState) -> Bool {
        if case .failed = state { return true }
        return false
    }

    private static func isSetupCancelled(_ state: ModelSetupState) -> Bool {
        if case .cancelled = state { return true }
        return false
    }

    private static func isWaitingForSystem(_ state: ModelSetupState) -> Bool {
        if case .waitingForSystem = state { return true }
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
    var isPlatformAvailable: Bool
    var assetStatuses: [SpeechLanguage: AppleSpeechAssetStatus]
    var assetStatusSequence: [SpeechLanguage: [AppleSpeechAssetStatus]] = [:]
    var scriptedAssetStatusResponses: [FakeAppleAssetStatusResponse] = []
    var statusAfterInstall: AppleSpeechAssetStatus = .installed
    var installError: Error?
    var makeEngineError: Error?
    private(set) var assetStatusRequests: [SpeechLanguage] = []
    private(set) var suspendedAssetStatusRequestCount = 0
    private var suspendedAssetStatusContinuation: CheckedContinuation<AppleSpeechAssetStatus, Never>?
    private var suspendedAssetStatus: AppleSpeechAssetStatus?
    private(set) var installCalls = 0
    private(set) var makeEngineCalls = 0

    init(isAvailable: Bool = true) {
        isPlatformAvailable = isAvailable
        let initialStatus: AppleSpeechAssetStatus = isAvailable ? .installed : .unsupported
        assetStatuses = [.english: initialStatus, .japanese: initialStatus]
    }

    func assetStatus(for language: SpeechLanguage) async -> AppleSpeechAssetStatus {
        assetStatusRequests.append(language)
        if !scriptedAssetStatusResponses.isEmpty {
            let response = scriptedAssetStatusResponses.removeFirst()
            switch response {
            case let .immediate(status):
                assetStatuses[language] = status
                return status
            case let .suspended(status):
                assetStatuses[language] = status
                suspendedAssetStatusRequestCount += 1
                suspendedAssetStatus = status
                return await withCheckedContinuation { continuation in
                    suspendedAssetStatusContinuation = continuation
                }
            }
        }
        if var sequence = assetStatusSequence[language], let next = sequence.first {
            sequence.removeFirst()
            assetStatusSequence[language] = sequence
            assetStatuses[language] = next
            return next
        }
        return assetStatuses[language] ?? .unsupported
    }

    func resumeSuspendedAssetStatus() {
        guard let continuation = suspendedAssetStatusContinuation else {
            fatalError("Mimi session E2E failure: no suspended Apple asset status request to resume")
        }
        suspendedAssetStatusContinuation = nil
        let status = suspendedAssetStatus ?? .unsupported
        suspendedAssetStatus = nil
        continuation.resume(returning: status)
    }

    func installAssets(for language: SpeechLanguage) async throws {
        installCalls += 1
        if let installError { throw installError }
        assetStatuses[language] = statusAfterInstall
    }

    func makeEngine() throws -> any AppleLiveTranscribing {
        makeEngineCalls += 1
        if let makeEngineError { throw makeEngineError }
        return engine
    }
}

private enum FakeAppleAssetStatusResponse {
    case immediate(AppleSpeechAssetStatus)
    case suspended(AppleSpeechAssetStatus)
}

@MainActor
private final class FakeWhisper: WhisperAccuracyTranscribing {
    var isDownloaded: Bool
    var ensureError: Error?
    var installError: Error?
    var progressEvents: [ModelDownloadProgress] = []
    var afterProgress: (() -> Void)?
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

    func install(
        onProgress: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void
    ) async throws {
        installCalls += 1
        for progress in progressEvents {
            onProgress(progress)
            afterProgress?()
            // Model downloads are expected to cooperate with task
            // cancellation. This lets the deterministic suite exercise the
            // public cancel action rather than only injecting a cancellation
            // error from the fake.
            try Task.checkCancellation()
        }
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
private final class FakeNemotron: NemotronMLXLiveTranscribing {
    var runtimeAvailabilityMessage: String?
    var isDownloaded: Bool
    var ensureError: Error?
    var installError: Error?
    var transcribeError: Error?
    var transcription = ""
    var liveStartError: Error?
    var livePartialText = ""
    var liveFinalText = ""
    var liveBackpressureMessage: String?
    private(set) var ensureCalls = 0
    private(set) var installCalls = 0
    private(set) var removeCalls = 0
    private(set) var transcribeCalls: [(URL, SpeechLanguage)] = []
    private(set) var liveStartLanguages: [SpeechLanguage] = []
    private(set) var liveConsumedBufferCount = 0
    private(set) var liveStopCalls = 0
    private(set) var liveCancelCalls = 0
    private var onLiveEvent: (@MainActor (TranscriptEvent) -> Void)?
    private var onLiveBackpressure: (@MainActor (String) -> Void)?

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

    func startLive(
        language: SpeechLanguage,
        inputFormat: AVAudioFormat,
        onEvent: @escaping @MainActor (TranscriptEvent) -> Void,
        onBackpressure: @escaping @MainActor (String) -> Void
    ) async throws {
        liveStartLanguages.append(language)
        if let liveStartError { throw liveStartError }
        onLiveEvent = onEvent
        onLiveBackpressure = onBackpressure
    }

    func consumeLive(_ buffer: AVAudioPCMBuffer) {
        liveConsumedBufferCount += 1
        if let liveBackpressureMessage {
            onLiveBackpressure?(liveBackpressureMessage)
            self.liveBackpressureMessage = nil
        }
        guard !livePartialText.isEmpty else { return }
        onLiveEvent?(.partial(livePartialText))
    }

    func stopLive() async {
        liveStopCalls += 1
        onLiveEvent?(.final(liveFinalText))
        onLiveEvent = nil
        onLiveBackpressure = nil
    }

    func cancelLive() async {
        liveCancelCalls += 1
        onLiveEvent = nil
        onLiveBackpressure = nil
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
    case modelDownload
    case modelNotInstalled
    case screenPickerCancelled
    case runtimeUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .captureStart: "Fixture capture start failed"
        case .transcription: "Fixture Whisper transcription failed"
        case .modelDownload: "Fixture Whisper download failed"
        case .modelNotInstalled: "Fixture Whisper model is missing"
        case .screenPickerCancelled: "No app or display was selected."
        case let .runtimeUnavailable(message): message
        }
    }
}
