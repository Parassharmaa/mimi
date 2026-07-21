import AppKit
@preconcurrency import AVFoundation
import Darwin
import MimiCore
import SwiftUI

@MainActor
final class MimiAppDelegate: NSObject, NSApplicationDelegate {
    private var e2eWindow: NSWindow?
    private var e2eStore: AppStore?
    private var translationBenchmarkCoordinator: AppleTranslationBenchmarkCoordinator?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let arguments = ProcessInfo.processInfo.arguments
        if let outputPath = argument(after: "--verify-translation-runtime-cache", in: arguments) {
            let report = verifyExperimentalTranslationRuntimeCacheContract()
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(report).write(
                    to: URL(filePath: outputPath),
                    options: .atomic
                )
                print("Mimi translation runtime cache verification \(report.status): \(outputPath)")
                fflush(stdout)
                Darwin.exit(report.status == "passed" ? 0 : 1)
            } catch {
                print("Mimi translation runtime cache verification failed to write: \(error.localizedDescription)")
                fflush(stdout)
                Darwin.exit(1)
            }
        }
        if let outputPath = argument(after: "--verify-translation-fallback", in: arguments) {
            let report = verifyExperimentalTranslationFallbackContract()
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(report).write(
                    to: URL(filePath: outputPath),
                    options: .atomic
                )
                print("Mimi translation fallback verification \(report.status): \(outputPath)")
                fflush(stdout)
                Darwin.exit(report.status == "passed" ? 0 : 1)
            } catch {
                print("Mimi translation fallback verification failed to write: \(error.localizedDescription)")
                fflush(stdout)
                Darwin.exit(1)
            }
        }
        if let outputPath = argument(after: "--verify-translation-critical-token-failure", in: arguments),
           let modelRoot = argument(after: "--model-root", in: arguments),
           let text = argument(after: "--text", in: arguments) {
            let sourceLanguage: SpeechLanguage =
                argument(after: "--direction", in: arguments) == "ja-en"
                ? .japanese : .english
            Task { @MainActor in
                let report = await verifyExperimentalTranslationCriticalTokenFailure(
                    modelRoot: URL(filePath: modelRoot, directoryHint: .isDirectory),
                    source: text,
                    sourceLanguage: sourceLanguage
                )
                let status: Int32
                do {
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                    try encoder.encode(report).write(
                        to: URL(filePath: outputPath),
                        options: .atomic
                    )
                    print("Mimi critical-token failure verification \(report.status): \(outputPath)")
                    status = report.status == "passed" ? 0 : 1
                } catch {
                    print("Mimi critical-token failure verification failed to write: \(error.localizedDescription)")
                    status = 1
                }
                fflush(stdout)
                Darwin.exit(status)
            }
            return
        }
        if let outputPath = argument(after: "--verify-translation-moe-router", in: arguments),
           let enJARouterPath = argument(after: "--en-ja-router", in: arguments),
           let jaENRouterPath = argument(after: "--ja-en-router", in: arguments),
           let pythonReportPath = argument(after: "--python-report", in: arguments) {
            do {
                let report = try verifyMarianSourceExpertRouterParity(
                    enJARouterURL: URL(filePath: enJARouterPath),
                    jaENRouterURL: URL(filePath: jaENRouterPath),
                    pythonReportURL: URL(filePath: pythonReportPath)
                )
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(report).write(
                    to: URL(filePath: outputPath),
                    options: .atomic
                )
                print("Mimi source-expert router parity \(report.status): \(outputPath)")
                fflush(stdout)
                Darwin.exit(report.status == "passed" ? 0 : 1)
            } catch {
                print("Mimi source-expert router parity failed: \(error.localizedDescription)")
                fflush(stdout)
                Darwin.exit(1)
            }
        }
        if let outputPath = argument(after: "--verify-translation-memory", in: arguments),
           let memoryPath = argument(after: "--memory", in: arguments),
           let pythonReportPath = argument(after: "--python-report", in: arguments) {
            do {
                let report = try verifyTranslationMemoryParity(
                    memoryURL: URL(filePath: memoryPath),
                    pythonReportURL: URL(filePath: pythonReportPath)
                )
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(report).write(
                    to: URL(filePath: outputPath),
                    options: .atomic
                )
                print("Mimi exact translation-memory parity \(report.status): \(outputPath)")
                fflush(stdout)
                Darwin.exit(report.status == "passed" ? 0 : 1)
            } catch {
                print("Mimi exact translation-memory parity failed: \(error.localizedDescription)")
                fflush(stdout)
                Darwin.exit(1)
            }
        }
        if let modelRoot = argument(after: "--validate-translation-mlx", in: arguments) {
            do {
                let directory = URL(filePath: modelRoot, directoryHint: .isDirectory)
                try ExperimentalMLXTranslationEngine.validateModelPack(at: directory)
                print("Mimi MLX Marian model-pack validation passed: \(directory.path)")
                fflush(stdout)
                Darwin.exit(0)
            } catch {
                print("Mimi MLX Marian model-pack validation failed: \(error.localizedDescription)")
                fflush(stdout)
                Darwin.exit(1)
            }
        }
        if let outputPath = argument(after: "--verify-translation-mlx-parity", in: arguments),
           let modelRoot = argument(after: "--model-root", in: arguments),
           let suitePath = argument(after: "--suite", in: arguments),
           let pythonReportPath = argument(after: "--python-report", in: arguments) {
            Task { @MainActor in
                let status: Int32
                do {
                    let warmRuns: Int
                    if let value = argument(after: "--translation-mlx-warm-runs", in: arguments) {
                        guard let parsed = Int(value), parsed >= 0 else {
                            throw TranslationMLXBenchmarkCommandError.invalidWarmRuns(value)
                        }
                        warmRuns = parsed
                    } else {
                        warmRuns = 0
                    }
                    let report = try await verifyTranslationMLXParity(
                        modelRoot: URL(filePath: modelRoot, directoryHint: .isDirectory),
                        suiteURL: URL(filePath: suitePath),
                        pythonReportURL: URL(filePath: pythonReportPath),
                        warmRuns: warmRuns,
                        cachedDecoding: !arguments.contains("--translation-mlx-full-prefix")
                    )
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                    try encoder.encode(report).write(
                        to: URL(filePath: outputPath),
                        options: .atomic
                    )
                    print("Mimi Swift/MLX translation parity \(report.status): \(outputPath)")
                    status = report.status == "passed" ? 0 : 1
                } catch {
                    print("Mimi Swift/MLX translation parity failed: \(error.localizedDescription)")
                    status = 1
                }
                fflush(stdout)
                Darwin.exit(status)
            }
            return
        }
        if let outputPath = argument(after: "--smoke-translation-mlx-moe", in: arguments),
           let modelRoot = argument(after: "--model-root", in: arguments),
           let text = argument(after: "--text", in: arguments),
           let expectedEngine = argument(after: "--expected-engine", in: arguments),
           ["generalist", "expert", "translation-memory"].contains(expectedEngine) {
            let sourceLanguage: SpeechLanguage =
                argument(after: "--direction", in: arguments) == "ja-en"
                ? .japanese : .english
            Task { @MainActor in
                let status: Int32
                do {
                    let report = try await verifyExperimentalMLXTranslationMoESmoke(
                        modelRoot: URL(filePath: modelRoot, directoryHint: .isDirectory),
                        source: text,
                        sourceLanguage: sourceLanguage,
                        expectedEngine: expectedEngine
                    )
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                    try encoder.encode(report).write(
                        to: URL(filePath: outputPath),
                        options: .atomic
                    )
                    print("Mimi MLX Marian MoE smoke \(report.status): \(outputPath)")
                    status = report.status == "passed" ? 0 : 1
                } catch {
                    print("Mimi MLX Marian MoE smoke failed: \(error.localizedDescription)")
                    status = 1
                }
                fflush(stdout)
                Darwin.exit(status)
            }
            return
        }
        if let modelRoot = argument(after: "--smoke-translation-mlx", in: arguments),
           let text = argument(after: "--text", in: arguments) {
            let direction = argument(after: "--direction", in: arguments) == "ja-en"
                ? "ja-en" : "en-ja"
            Task { @MainActor in
                let status: Int32
                do {
                    let output = try await Task.detached {
                        let directory = URL(filePath: modelRoot, directoryHint: .isDirectory)
                            .appending(path: direction, directoryHint: .isDirectory)
                        let runtime = try await MarianMLXTranslationRuntime.load(directory: directory)
                        return runtime.translate(text)
                    }.value
                    print("Mimi MLX Marian \(direction) smoke passed: \(output)")
                    status = 0
                } catch {
                    print("Mimi MLX Marian smoke failed: \(error.localizedDescription)")
                    status = 1
                }
                fflush(stdout)
                Darwin.exit(status)
            }
            return
        }
        if arguments.contains("--e2e-insert-text") || arguments.contains("--e2e-stream-insert") {
            // This harness represents the already-running menu-bar app and
            // must not become active or disturb the external focused field.
            NSApplication.shared.setActivationPolicy(.accessory)
        } else {
            // Mimi lives in the menu bar, but also remains a normal installed
            // Mac app with a Dock presence and standard app menus.
            NSApplication.shared.setActivationPolicy(.regular)
        }

        switch argument(after: "--e2e-appearance", in: arguments) {
        case "light":
            NSApplication.shared.appearance = NSAppearance(named: .aqua)
        case "dark":
            NSApplication.shared.appearance = NSAppearance(named: .darkAqua)
        default:
            break
        }
        if let suitePath = argument(after: "--benchmark-translation-apple", in: arguments),
           let outputPath = argument(after: "--output", in: arguments) {
            do {
                let warmRuns = try benchmarkWarmRuns(arguments)
                let suite = try TranslationBenchmarkCase.loadJSONL(
                    from: URL(fileURLWithPath: suitePath)
                )
                let coordinator = AppleTranslationBenchmarkCoordinator(
                    suite: suite,
                    warmRuns: warmRuns
                ) { result in
                    let status: Int32
                    do {
                        let report = try result.get()
                        let encoder = JSONEncoder()
                        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
                        encoder.dateEncodingStrategy = .iso8601
                        let data = try encoder.encode(report)
                        try data.write(
                            to: URL(fileURLWithPath: outputPath),
                            options: .atomic
                        )
                        print("Mimi Apple Translation benchmark wrote \(report.results.count) cases to \(outputPath).")
                        status = 0
                    } catch {
                        print("Mimi Apple Translation benchmark failed: \(error.localizedDescription)")
                        status = 1
                    }
                    fflush(stdout)
                    Darwin.exit(status)
                }
                translationBenchmarkCoordinator = coordinator
                let hostingController = NSHostingController(rootView: AppleTranslationBenchmarkView(
                    suite: suite,
                    coordinator: coordinator
                ))
                hostingController.sizingOptions = []
                let window = NSWindow(contentViewController: hostingController)
                window.setContentSize(NSSize(width: 1, height: 1))
                window.alphaValue = 0.01
                window.orderBack(nil)
                e2eWindow = window
            } catch {
                print("Mimi Apple Translation benchmark failed: \(error.localizedDescription)")
                Darwin.exit(1)
            }
            return
        }
        if let insertionText = argument(after: "--e2e-insert-text", in: arguments) {
            Task { @MainActor in
                let status: Int32
                do {
                    // Give the harness time to return focus to the external
                    // field. Mimi itself opens no window for this smoke path.
                    try await Task.sleep(for: .seconds(3))
                    print("Mimi Voice Type smoke frontmost app: \(NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "none"); accessibility trusted: \(AXIsProcessTrusted())")
                    let target = try FocusedTextTarget.capture(promptIfNeeded: false)
                    try await target.replaceLiveText(with: insertionText)
                    print("Mimi Voice Type insertion smoke passed.")
                    status = 0
                } catch {
                    print("Mimi Voice Type insertion smoke failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        if let stream = argument(after: "--e2e-stream-insert", in: arguments) {
            Task { @MainActor in
                let status: Int32
                do {
                    let delay = Double(argument(after: "--e2e-delay", in: arguments) ?? "3") ?? 3
                    try await Task.sleep(for: .seconds(delay))
                    let target = try FocusedTextTarget.capture(promptIfNeeded: false)
                    let updates = stream.split(separator: "|", omittingEmptySubsequences: false).map(String.init)
                    for update in updates {
                        try await target.replaceLiveText(with: update)
                    }
                    try await target.rollback()
                    print("Mimi realtime Voice Type smoke passed: \(updates.count) verified field updates and rollback.")
                    status = 0
                } catch {
                    print("Mimi realtime Voice Type smoke failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
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
        if arguments.contains("--e2e-main-window-lifecycle") {
            runMainWindowLifecycleSmoke()
            return
        }
        guard arguments.contains("--e2e-window") else {
            if UserDefaults.standard.bool(forKey: "completedOnboarding") {
                AppWindowCoordinator.shared.showTranscript()
            }
            return
        }

        let screen = argument(after: "--e2e-screen", in: arguments) ?? "menu"
        let presentationState = argument(after: "--e2e-state", in: arguments) ?? "ready"
        let exercisesLiveTranslation = arguments.contains("--e2e-live-translation")
        let store = AppStore(loadPersistedTranscript: false)
        store.languageMode = ["incremental-translation", "translation-stream", "caption-stream"].contains(presentationState)
            ? .automatic
            : .japanese
        store.engineID = .appleSpeechAnalyzer
        store.translationMode = .translateFinalSegments
        store.applyFixture(.final("こんにちは、Mimi はローカルで文字起こしします。"), language: .japanese)
        store.applyFixture(.final("Mimi keeps the transcript on this Mac."), language: .english)
        let fixturePreferences = UserPreferences(defaults: UserDefaults(suiteName: "MimiE2E-\(UUID().uuidString)")!)
        if argument(after: "--e2e-language", in: arguments) == "japanese" {
            fixturePreferences.interfaceLanguage = .japanese
        }
        fixturePreferences.voiceTypingEnabled = false
        let fixtureVoiceTyping = VoiceTypingController(preferences: fixturePreferences)
        if presentationState == "voice-enabled" {
            fixturePreferences.voiceTypingEnabled = true
        }

        switch presentationState {
        case "recording", "translation-stream", "caption-stream":
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
            let onboardingFixture: OnboardingPreparationFixture = switch presentationState {
            case "model-preparing": .preparing
            case "model-ready": .ready
            case "model-failed": .failed
            default: .live
            }
            view = AnyView(OnboardingView(
                store: store,
                preferences: fixturePreferences,
                voiceTyping: fixtureVoiceTyping,
                initialStep: ["ready", "voice-enabled"].contains(presentationState)
                    ? 4
                    : (["model-preparing", "model-ready", "model-failed"].contains(presentationState) ? 2 : 0),
                preparationFixture: onboardingFixture
            ))
            size = NSSize(width: 620, height: 560)
        case "captions":
            fixturePreferences.floatingCaptionsEnabled = true
            fixturePreferences.floatingCaptionContent = .both
            fixturePreferences.floatingCaptionClickThrough = false
            view = AnyView(FloatingCaptionView(store: store, preferences: fixturePreferences))
            size = NSSize(width: 820, height: 150)
        case "voice-typing":
            fixtureVoiceTyping.applyPresentationFixture(text: "")
            view = AnyView(VoiceTypingPill(controller: fixtureVoiceTyping, preferences: fixturePreferences))
            size = NSSize(width: 64, height: 64)
        case "transcript":
            view = AnyView(TranscriptWindow(
                store: store,
                preferences: fixturePreferences,
                isConfirmingClear: presentationState == "clear-confirmation",
                fixtureTranslation: exercisesLiveTranslation
                    && ["incremental-translation", "translation-stream"].contains(presentationState)
                    ? nil
                    : "Hello. Mimi transcribes locally on this Mac.",
                initiallyFollowingLatest: presentationState != "follow-latest-paused"
            ))
            size = NSSize(width: 820, height: 600)
        case "settings", "settings-models":
            view = AnyView(SettingsView(store: store, preferences: fixturePreferences, voiceTyping: fixtureVoiceTyping, initialTab: .models))
            size = NSSize(width: 620, height: 540)
        case "settings-capture":
            view = AnyView(SettingsView(store: store, preferences: fixturePreferences, voiceTyping: fixtureVoiceTyping, initialTab: .capture))
            size = NSSize(width: 620, height: 540)
        case "settings-privacy":
            view = AnyView(SettingsView(store: store, preferences: fixturePreferences, voiceTyping: fixtureVoiceTyping, initialTab: .privacy))
            size = NSSize(width: 620, height: 540)
        case "settings-captions":
            view = AnyView(SettingsView(store: store, preferences: fixturePreferences, voiceTyping: fixtureVoiceTyping, initialTab: .captions))
            size = NSSize(width: 620, height: 540)
        case "settings-voice":
            view = AnyView(SettingsView(store: store, preferences: fixturePreferences, voiceTyping: fixtureVoiceTyping, initialTab: .voiceTyping))
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
        if screen == "captions" || screen == "voice-typing" {
            window.styleMask = [.borderless]
            window.isOpaque = false
            window.backgroundColor = .clear
            window.level = .floating
        } else {
            window.styleMask = [.titled, .closable, .miniaturizable]
        }
        if screen != "menu", screen != "captions", screen != "voice-typing" {
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

        if presentationState == "caption-stream" {
            Task { @MainActor in
                for tick in 1...600 {
                    try? await Task.sleep(for: .milliseconds(120))
                    store.applyFixture(
                        .partial("Live caption phrase \(tick) keeps growing without blinking"),
                        language: .english
                    )
                }
            }
        } else if presentationState == "translation-stream" {
            Task { @MainActor in
                for tick in 1...180 {
                    try? await Task.sleep(for: .milliseconds(300))
                    let language: SpeechLanguage = tick.isMultiple(of: 2) ? .english : .japanese
                    let text = language == .english
                        ? "Sustained English sentence \(tick)."
                        : "長時間テストの日本語文 \(tick)。"
                    store.applyFixture(.final(text), language: language)
                }
            }
        }

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

    func applicationShouldHandleReopen(
        _ sender: NSApplication,
        hasVisibleWindows flag: Bool
    ) -> Bool {
        AppWindowCoordinator.shared.showTranscript()
        return false
    }

    private func runMainWindowLifecycleSmoke() {
        Task { @MainActor in
            AppWindowCoordinator.shared.showTranscript()
            try? await Task.sleep(for: .milliseconds(600))
            guard let firstWindow = transcriptWindow else {
                print("Mimi main-window lifecycle smoke failed: launch did not open the transcript window.")
                Darwin.exit(1)
            }
            let firstWindowNumber = firstWindow.windowNumber
            firstWindow.close()
            try? await Task.sleep(for: .milliseconds(250))

            AppWindowCoordinator.shared.showTranscript()
            try? await Task.sleep(for: .milliseconds(600))
            guard let reopenedWindow = transcriptWindow, reopenedWindow.isVisible else {
                print("Mimi main-window lifecycle smoke failed: reopen did not restore the transcript window.")
                Darwin.exit(1)
            }
            let visibleTranscriptWindows = AppWindowCoordinator.shared.visibleTranscriptWindows
            guard visibleTranscriptWindows.count == 1 else {
                print("Mimi main-window lifecycle smoke failed: reopen created duplicate transcript windows.")
                Darwin.exit(1)
            }
            print("Mimi main-window lifecycle smoke passed: launch opened window \(firstWindowNumber), and Dock-style reopen restored one transcript window.")
            Darwin.exit(0)
        }
    }

    private var transcriptWindow: NSWindow? {
        AppWindowCoordinator.shared.visibleTranscriptWindows.first
    }

    private func benchmarkWarmRuns(_ arguments: [String]) throws -> Int {
        guard let value = argument(
            after: "--benchmark-translation-apple-warm-runs",
            in: arguments
        ) else { return 3 }
        guard let count = Int(value), count >= 0 else {
            throw AppleTranslationBenchmarkCommandError.invalidWarmRuns(value)
        }
        return count
    }

    private func argument(after flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else {
            return nil
        }
        return arguments[index + 1]
    }
}

private enum AppleTranslationBenchmarkCommandError: LocalizedError {
    case invalidWarmRuns(String)

    var errorDescription: String? {
        switch self {
        case let .invalidWarmRuns(value):
            "Apple Translation benchmark warm runs must be a non-negative integer, not \(value)."
        }
    }
}

private enum TranslationMLXBenchmarkCommandError: LocalizedError {
    case invalidWarmRuns(String)

    var errorDescription: String? {
        switch self {
        case let .invalidWarmRuns(value):
            "Translation MLX warm runs must be a nonnegative integer, got: \(value)."
        }
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
    private let voiceTypingController: VoiceTypingController
    private let voiceTypingPanelController: VoiceTypingPanelController

    init() {
        let store = AppStore()
        let preferences = UserPreferences()
        let voiceTyping = VoiceTypingController(
            preferences: preferences,
            isSessionRecording: { store.isRecording }
        )
        _store = State(initialValue: store)
        _preferences = State(initialValue: preferences)
        AppWindowCoordinator.shared.configure(store: store, preferences: preferences)
        onboardingCoordinator = OnboardingWindowCoordinator(store: store, preferences: preferences, voiceTyping: voiceTyping)
        floatingCaptionController = FloatingCaptionController(store: store, preferences: preferences)
        voiceTypingController = voiceTyping
        voiceTypingPanelController = VoiceTypingPanelController(controller: voiceTyping, preferences: preferences)
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
            SettingsView(store: store, preferences: preferences, voiceTyping: voiceTypingController)
        }

        .commands {
            CommandMenu("Recording") {
                Button(preferences.text("New Session", "新しいセッション")) {
                    store.newSession()
                }
                .keyboardShortcut("n", modifiers: .command)
                .disabled(store.controlsLocked)

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
