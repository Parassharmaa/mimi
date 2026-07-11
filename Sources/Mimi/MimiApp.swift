import AppKit
@preconcurrency import AVFoundation
import Darwin
import MimiCore
import SwiftUI

@MainActor
final class MimiAppDelegate: NSObject, NSApplicationDelegate {
    private var e2eWindow: NSWindow?
    private var e2eStore: AppStore?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Mimi is intentionally a background/menu-bar utility. Opening the
        // transcript window remains available from the menu extra.
        NSApplication.shared.setActivationPolicy(.accessory)

        let arguments = ProcessInfo.processInfo.arguments
        switch argument(after: "--e2e-appearance", in: arguments) {
        case "light":
            NSApplication.shared.appearance = NSAppearance(named: .aqua)
        case "dark":
            NSApplication.shared.appearance = NSAppearance(named: .darkAqua)
        default:
            break
        }
        if arguments.contains("--benchmark-install-language-id") {
            Task { @MainActor in
                let status: Int32
                do {
                    let detector = WhisperLanguageDetector()
                    try await detector.install { progress in
                        if let fraction = progress.fractionCompleted {
                            print("Mimi language detector setup: \(Int((fraction * 100).rounded()))%")
                        }
                    }
                    print("Mimi Whisper tiny language detector is installed and ready.")
                    status = 0
                } catch {
                    print("Mimi language detector install failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        if arguments.contains("--e2e-output-audio-smoke") {
            Task { @MainActor in
                let status: Int32
                do {
                    let capture = OutputAudioCapture()
                    let format = try capture.configureInput(deviceID: nil)
                    let counter = OutputAudioSmokeCounter()
                    try capture.start(recordingTo: nil, deviceID: nil) { _ in
                        counter.increment()
                    }
                    let speech = AVSpeechSynthesizer()
                    speech.speak(AVSpeechUtterance(string: "Mimi output audio capture smoke test."))
                    try await Task.sleep(for: .seconds(4))
                    speech.stopSpeaking(at: .immediate)
                    _ = try capture.stop()
                    guard counter.value > 0 else {
                        throw RealtimeBenchmarkCommandError.outputAudioMissing
                    }
                    print("Mimi selected output audio smoke passed: \(counter.value) PCM callbacks at \(Int(format.sampleRate)) Hz.")
                    status = 0
                } catch {
                    print("Mimi selected output audio smoke failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        if arguments.contains("--benchmark-install-qwen") {
            Task { @MainActor in
                let status: Int32
                do {
                    let engine = QwenMLXLiveEngine()
                    try await engine.install { progress in
                        if let fraction = progress.fractionCompleted {
                            print("Mimi Qwen3-ASR setup: \(Int((fraction * 100).rounded()))%")
                        }
                    }
                    guard engine.isDownloaded else {
                        throw RealtimeBenchmarkCommandError.qwenDidNotBecomeReady
                    }
                    print("Mimi Qwen3-ASR MLX is installed and ready.")
                    status = 0
                } catch {
                    print("Mimi Qwen3-ASR install failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        if let audioPath = argument(after: "--e2e-qwen-live-smoke", in: arguments) {
            let language: SpeechLanguage = switch argument(after: "--language", in: arguments) {
            case "ja", "ja-JP": .japanese
            default: .english
            }
            Task { @MainActor in
                let status: Int32
                do {
                    let engine = QwenMLXLiveEngine()
                    let audioURL = URL(fileURLWithPath: audioPath)
                    let file = try AVAudioFile(forReading: audioURL)
                    var partialCount = 0
                    var finalText = ""
                    try await engine.startLive(
                        language: language,
                        inputFormat: file.processingFormat,
                        onEvent: { event in
                            switch event {
                            case .partial:
                                partialCount += 1
                            case let .final(text):
                                finalText = text
                            }
                        },
                        onBackpressure: { message in
                            print("Mimi Qwen3-ASR live smoke warning: \(message)")
                        }
                    )
                    while file.framePosition < file.length {
                        let remaining = AVAudioFrameCount(file.length - file.framePosition)
                        let count = min(AVAudioFrameCount(3_200), remaining)
                        guard let buffer = AVAudioPCMBuffer(
                            pcmFormat: file.processingFormat,
                            frameCapacity: count
                        ) else {
                            throw RealtimeBenchmarkCommandError.qwenFixtureBufferFailed
                        }
                        try file.read(into: buffer, frameCount: count)
                        engine.consumeLive(buffer)
                        try await Task.sleep(for: .seconds(
                            Double(buffer.frameLength) / file.processingFormat.sampleRate
                        ))
                    }
                    await engine.stopLive()
                    guard partialCount > 0, !finalText.isEmpty else {
                        throw RealtimeBenchmarkCommandError.qwenLiveOutputMissing
                    }
                    print("Mimi Qwen3-ASR \(language.displayName) live app smoke passed: \(partialCount) display updates; final text: \(finalText)")
                    status = 0
                } catch {
                    print("Mimi Qwen3-ASR live app smoke failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        if let installLanguage = argument(after: "--benchmark-install-apple-assets", in: arguments) {
            let language: SpeechLanguage = switch installLanguage {
            case "ja", "ja-JP": .japanese
            default: .english
            }
            Task { @MainActor in
                let status: Int32
                do {
                    guard #available(macOS 26.0, *) else {
                        throw RealtimeBenchmarkCommandError.appleSpeechUnavailable
                    }
                    try await AppleSpeechEngine.installAssets(for: language)
                    let assetStatus = await AppleSpeechEngine.assetStatus(for: language)
                    guard assetStatus == .installed else {
                        throw RealtimeBenchmarkCommandError.appleAssetsDidNotBecomeReady(language)
                    }
                    print("Mimi Apple Speech \(language.displayName) assets are installed for realtime benchmarking.")
                    status = 0
                } catch {
                    print("Mimi Apple Speech asset install failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        if let audioPath = argument(after: "--benchmark-language-id", in: arguments) {
            Task { @MainActor in
                let status: Int32
                do {
                    let report = try await WhisperKitAccuracyEngine().runLanguageDetectionBenchmark(
                        recordingAt: URL(fileURLWithPath: audioPath)
                    )
                    try report.printJSON()
                    status = 0
                } catch {
                    print("Mimi acoustic language-ID benchmark failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        if let audioPath = argument(after: "--benchmark-tiny-language-id", in: arguments) {
            Task { @MainActor in
                let status: Int32
                do {
                    let report = try await WhisperLanguageDetector().runBenchmark(
                        recordingAt: URL(fileURLWithPath: audioPath)
                    )
                    try report.printJSON()
                    status = 0
                } catch {
                    print("Mimi tiny acoustic language-ID benchmark failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        if let benchmarkEngine = argument(after: "--benchmark-realtime", in: arguments),
           let audioPath = argument(after: "--audio", in: arguments) {
            let language: SpeechLanguage = switch argument(after: "--language", in: arguments) {
            case "ja", "ja-JP": .japanese
            default: .english
            }
            let audioURL = URL(fileURLWithPath: audioPath)
            Task { @MainActor in
                let status: Int32
                do {
                    var report: RealtimeBenchmarkReport
                    switch benchmarkEngine {
                    case "apple-accurate":
                        guard #available(macOS 26.0, *) else {
                            throw RealtimeBenchmarkCommandError.appleSpeechUnavailable
                        }
                        report = try await RealtimeBenchmarkRunner.runApple(
                            recordingAt: audioURL,
                            language: language,
                            mode: .accurate,
                            simulateRealtime: true
                        )
                    case "apple-progressive":
                        guard #available(macOS 26.0, *) else {
                            throw RealtimeBenchmarkCommandError.appleSpeechUnavailable
                        }
                        report = try await RealtimeBenchmarkRunner.runApple(
                            recordingAt: audioURL,
                            language: language,
                            mode: .progressive,
                            simulateRealtime: true
                        )
                    case "whisper":
                        let step = Double(argument(after: "--step", in: arguments) ?? "2") ?? 2
                        report = try await WhisperKitAccuracyEngine().runRollingBenchmark(
                            recordingAt: audioURL,
                            language: language,
                            stepSeconds: step
                        )
                    case "qwen":
                        report = try await RealtimeBenchmarkRunner.runQwen(
                            recordingAt: audioURL,
                            language: language,
                            simulateRealtime: true
                        )
                    default:
                        throw RealtimeBenchmarkCommandError.unknownEngine(benchmarkEngine)
                    }
                    if let reference = argument(after: "--reference", in: arguments) {
                        report.score(against: reference, language: language)
                    }
                    try report.printJSON()
                    status = 0
                } catch {
                    print("Mimi realtime benchmark failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        if arguments.contains("--e2e-microphone-smoke") {
            let store = AppStore(loadPersistedTranscript: false)
            e2eStore = store
            Task { @MainActor in
                let status: Int32
                do {
                    let callbackCount = try await store.runMicrophoneCaptureSmokeTest()
                    print("Mimi microphone smoke passed: \(callbackCount) audio callbacks in one second.")
                    status = 0
                } catch {
                    print("Mimi microphone smoke failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        if arguments.contains("--e2e-nemotron-live-smoke") {
            let language: SpeechLanguage
            switch argument(after: "--e2e-language", in: arguments) {
            case "ja", "ja-JP":
                language = .japanese
            default:
                language = .english
            }
            Task { @MainActor in
                let status: Int32
                do {
                    let engine = NemotronMLXLiveEngine()
                    guard let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1) else {
                        throw NemotronLiveSmokeError.invalidFixtureFormat
                    }

                    var receivedFinal = false
                    try await engine.startLive(
                        language: language,
                        inputFormat: format,
                        onEvent: { event in
                            if case .final = event {
                                receivedFinal = true
                            }
                        },
                        onBackpressure: { message in
                            print("Mimi Nemotron live app smoke warning: \(message)")
                        }
                    )

                    // 640 ms of owned 48 kHz PCM exercises Mimi's converter,
                    // bounded queue, streaming session, and sub-window Stop
                    // flush without opening a microphone or retaining audio.
                    for _ in 0..<30 {
                        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1_024) else {
                            throw NemotronLiveSmokeError.invalidFixtureBuffer
                        }
                        buffer.frameLength = 1_024
                        if let samples = buffer.floatChannelData?[0] {
                            for index in 0..<Int(buffer.frameLength) {
                                samples[index] = 0
                            }
                        }
                        engine.consumeLive(buffer)
                    }
                    await engine.stopLive()
                    guard receivedFinal else { throw NemotronLiveSmokeError.missingFinalEvent }
                    print("Mimi Nemotron \(language.displayName) live app smoke passed: converted 48 kHz PCM, flushed a bounded local stream, and retained no source-audio file.")
                    status = 0
                } catch {
                    print("Mimi Nemotron \(language.displayName) live app smoke failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        if let engineSmoke = argument(after: "--e2e-engine-smoke", in: arguments) {
            let store = AppStore(loadPersistedTranscript: false)
            e2eStore = store
            let engine: TranscriptionEngineID
            switch engineSmoke {
            case "whisper":
                engine = .whisperKitLargeV3Turbo
            case "nemotron":
                engine = .nemotronStreamingExperimental
            case "qwen":
                engine = .qwen3StreamingExperimental
            default:
                engine = .appleSpeechAnalyzer
            }
            let language: SpeechLanguage
            switch argument(after: "--e2e-language", in: arguments) {
            case "ja", "ja-JP":
                language = .japanese
            default:
                language = .english
            }
            Task { @MainActor in
                let status: Int32
                do {
                    try await store.runEngineSmokeTest(engine: engine, language: language)
                    print("Mimi \(engine.displayName) \(language.displayName) smoke passed: local model started and stopped without retaining source audio.")
                    status = 0
                } catch {
                    print("Mimi \(engine.displayName) smoke failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        guard arguments.contains("--e2e-window") else { return }

        let store = AppStore(loadPersistedTranscript: false)
        store.sourceLanguage = .japanese
        store.engineID = .appleSpeechAnalyzer
        store.translationMode = .translateFinalSegments
        store.applyFixture(.final("こんにちは、Mimi はローカルで文字起こしします。"), language: .japanese)
        store.applyFixture(.final("Mimi keeps the transcript on this Mac."), language: .english)
        let fixturePreferences = UserPreferences(defaults: UserDefaults(suiteName: "MimiE2E-\(UUID().uuidString)")!)

        let screen = argument(after: "--e2e-screen", in: arguments) ?? "menu"
        let presentationState = argument(after: "--e2e-state", in: arguments) ?? "ready"
        switch presentationState {
        case "recording":
            store.applyPresentationFixture(state: .recording)
        case "backpressure":
            store.applyPresentationFixture(
                state: .recording,
                lastError: "Nemotron MLX is slower than this audio source. Mimi skipped some queued audio to stay live."
            )
        case "failed":
            store.applyPresentationFixture(state: .failed("Microphone access needs attention"))
        default:
            break
        }
        let view: AnyView
        let size: NSSize
        switch screen {
        case "onboarding":
            view = AnyView(OnboardingView(store: store, preferences: fixturePreferences))
            size = NSSize(width: 620, height: 500)
        case "captions":
            fixturePreferences.floatingCaptionsEnabled = true
            fixturePreferences.floatingCaptionContent = .both
            view = AnyView(FloatingCaptionView(store: store, preferences: fixturePreferences))
            size = NSSize(width: 820, height: 150)
        case "transcript":
            view = AnyView(TranscriptWindow(
                store: store,
                preferences: fixturePreferences,
                isConfirmingClear: presentationState == "clear-confirmation",
                fixtureTranslation: "Hello. Mimi transcribes locally on this Mac.",
                initiallyFollowingLatest: presentationState != "follow-latest-paused"
            ))
            size = NSSize(width: 820, height: 600)
        case "settings", "settings-models":
            view = AnyView(SettingsView(store: store, preferences: fixturePreferences, initialTab: .models))
            size = NSSize(width: 620, height: 540)
        case "settings-capture":
            view = AnyView(SettingsView(store: store, preferences: fixturePreferences, initialTab: .capture))
            size = NSSize(width: 620, height: 540)
        case "settings-privacy":
            view = AnyView(SettingsView(store: store, preferences: fixturePreferences, initialTab: .privacy))
            size = NSSize(width: 620, height: 540)
        case "settings-captions":
            view = AnyView(SettingsView(store: store, preferences: fixturePreferences, initialTab: .captions))
            size = NSSize(width: 620, height: 540)
        default:
            view = AnyView(MenuBarView(
                store: store,
                preferences: fixturePreferences,
                isConfirmingClear: presentationState == "clear-confirmation",
                initiallyFollowingLatest: presentationState != "follow-latest-paused"
            ))
            size = NSSize(width: 430, height: 580)
        }

        let hostingController = NSHostingController(rootView: view)
        // The smoke harness owns a fixed AppKit test window. Letting the
        // hosting controller also resize it from intrinsic content produces a
        // constraint feedback loop for a popover-width SwiftUI surface.
        hostingController.sizingOptions = []
        let window = NSWindow(contentViewController: hostingController)
        window.title = "Mimi E2E \(screen.capitalized)"
        window.styleMask = [.titled, .closable, .miniaturizable]
        if screen != "menu" {
            window.styleMask.insert(.resizable)
        }
        window.setContentSize(size)
        window.center()
        window.makeKeyAndOrderFront(nil)
        // The physical visual-QA runner may be launched while another app is
        // active. Put this deterministic test window in front so the native
        // surface can actually be inspected, rather than merely constructed.
        window.orderFrontRegardless()
        NSApplication.shared.activate(ignoringOtherApps: true)
        e2eStore = store
        e2eWindow = window

        if arguments.contains("--e2e-auto-quit") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                print("Mimi UI smoke passed: \(screen) \(presentationState) surface rendered deterministic English/Japanese sample data.")
                // TranslationSession may still own an asynchronous framework
                // task for transcript fixtures. A CLI smoke harness must exit
                // deterministically even when that framework task is pending;
                // normal app launches continue to use AppKit termination.
                fflush(stdout)
                Darwin.exit(0)
            }
        }
    }

    private func argument(after flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else {
            return nil
        }
        return arguments[index + 1]
    }
}

private enum RealtimeBenchmarkCommandError: LocalizedError {
    case appleSpeechUnavailable
    case appleAssetsDidNotBecomeReady(SpeechLanguage)
    case qwenDidNotBecomeReady
    case qwenFixtureBufferFailed
    case qwenLiveOutputMissing
    case outputAudioMissing
    case unknownEngine(String)

    var errorDescription: String? {
        switch self {
        case .appleSpeechUnavailable:
            "Apple Speech realtime benchmarking requires macOS 26 or later."
        case let .appleAssetsDidNotBecomeReady(language):
            "Apple finished the setup request, but its \(language.displayName) speech asset still is not installed."
        case .qwenDidNotBecomeReady:
            "Qwen3-ASR setup finished without a valid installed-model marker."
        case .qwenFixtureBufferFailed:
            "Mimi could not allocate a Qwen3-ASR fixture buffer."
        case .qwenLiveOutputMissing:
            "Qwen3-ASR did not produce both live and final output for the fixture."
        case .outputAudioMissing:
            "The Core Audio output tap started, but no output PCM arrived. Allow System Audio Recording and make sure the selected output is playing."
        case let .unknownEngine(engine):
            "Unknown realtime benchmark engine: \(engine)."
        }
    }
}

private final class OutputAudioSmokeCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    func increment() {
        lock.lock()
        count += 1
        lock.unlock()
    }
}

private enum NemotronLiveSmokeError: LocalizedError {
    case invalidFixtureFormat
    case invalidFixtureBuffer
    case missingFinalEvent

    var errorDescription: String? {
        switch self {
        case .invalidFixtureFormat:
            "Mimi could not create the deterministic 48 kHz live-audio fixture."
        case .invalidFixtureBuffer:
            "Mimi could not allocate the deterministic live-audio fixture buffer."
        case .missingFinalEvent:
            "Nemotron did not flush a final event when the live fixture stopped."
        }
    }
}

@main
struct MimiApp: App {
    @NSApplicationDelegateAdaptor(MimiAppDelegate.self) private var appDelegate
    @State private var store: AppStore
    @State private var preferences: UserPreferences
    private let onboardingCoordinator: OnboardingWindowCoordinator
    private let floatingCaptionController: FloatingCaptionController

    init() {
        let store = AppStore()
        let preferences = UserPreferences()
        _store = State(initialValue: store)
        _preferences = State(initialValue: preferences)
        onboardingCoordinator = OnboardingWindowCoordinator(store: store, preferences: preferences)
        floatingCaptionController = FloatingCaptionController(store: store, preferences: preferences)
    }

    var body: some Scene {
        MenuBarExtra {
            MenuBarView(store: store, preferences: preferences)
        } label: {
            Label(store.isRecording ? "Mimi REC" : "Mimi", systemImage: store.menuBarSymbolName)
                .accessibilityLabel(menuBarAccessibilityLabel)
        }
        .menuBarExtraStyle(.window)

        Settings {
            SettingsView(store: store, preferences: preferences)
        }

        WindowGroup("Mimi Transcript", id: "transcript") {
            TranscriptWindow(store: store, preferences: preferences)
        }
        .defaultSize(width: 920, height: 640)
        .defaultPosition(.center)
        .windowToolbarStyle(.unified)
        .windowResizability(.contentMinSize)
        .commands {
            CommandMenu("Recording") {
                Button(store.isRecording ? "Stop Recording" : "Start Recording") {
                    store.toggleRecording()
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .disabled(store.isRecording ? store.recordingState == .processing : !store.canStartRecording)

                Button("Copy Transcript") {
                    store.copyTranscript()
                }
                .keyboardShortcut("c", modifiers: [.command, .shift])
                .disabled(store.document.renderedText.isEmpty)

            }
        }
    }

    private var menuBarAccessibilityLabel: String {
        if store.isRecording {
            return "Mimi recording \(store.source.displayName.lowercased()) locally"
        }
        return "Mimi \(store.recordingState.label.lowercased())"
    }
}
