import AppKit
@preconcurrency import AVFoundation
import Foundation
import MimiCore
import Observation

@MainActor
@Observable
final class AppStore {
    var recordingState: RecordingState = .idle
    var source: AudioSource = .microphone
    var sourceLanguage: SpeechLanguage = .english
    var engineID: TranscriptionEngineID = .appleSpeechAnalyzer
    var translationMode: TranslationMode = .off
    var document = TranscriptDocument()
    var lastError: String?
    var lastRecordingURL: URL?
    var inputDevices = AudioDeviceCatalog.inputDevices()
    var selectedInputDeviceID: UInt32?
    private var modelStorageRevision = 0

    private let microphoneCapture = MicrophoneCapture()
    private let whisperKitEngine = WhisperKitAccuracyEngine()
    // `AnyObject` keeps the macOS 26-only type out of this macOS 15 store's
    // stored-property ABI. Every access is safely guarded below.
    @ObservationIgnored private var appleEngine: AnyObject?
    @ObservationIgnored private var activeSession: SessionConfiguration?

    init(loadPersistedTranscript: Bool = true) {
        if loadPersistedTranscript {
            document = MimiStorage.loadLatestTranscript()
        }
        MimiStorage.removeStaleTemporaryRecordings()
    }

    var menuBarSymbolName: String {
        switch recordingState {
        case .recording: "waveform.circle.fill"
        case .preparing, .processing: "arrow.triangle.2.circlepath.circle"
        case .failed: "exclamationmark.triangle.fill"
        case .idle: "ear"
        }
    }

    var isRecording: Bool {
        if case .recording = recordingState { return true }
        return false
    }

    var controlsLocked: Bool {
        switch recordingState {
        case .preparing, .recording, .processing: true
        case .idle, .failed: false
        }
    }

    var modelPack: LocalModelPack? {
        ModelCatalog.pack(for: engineID)
    }

    var canRemoveSelectedModel: Bool {
        _ = modelStorageRevision
        return engineID == .whisperKitLargeV3Turbo && whisperKitEngine.isDownloaded
    }

    func toggleRecording() {
        Task {
            if isRecording {
                await stopRecording()
            } else {
                await startRecording()
            }
        }
    }

    func installSelectedModel() {
        Task {
            recordingState = .preparing
            lastError = nil

            do {
                switch engineID {
                case .appleSpeechAnalyzer:
                    guard #available(macOS 26.0, *) else {
                        throw MimiError.appleSpeechRequiresMacOS26
                    }
                    try await AppleSpeechEngine.installAssets(for: sourceLanguage)
                case .whisperKitLargeV3Turbo:
                    try await whisperKitEngine.install()
                case .nemotronStreamingExperimental:
                    throw MimiError.nemotronRequiresBakeOff
                }
                modelStorageRevision += 1
                recordingState = .idle
            } catch {
                record(error)
            }
        }
    }

    func removeSelectedModel() {
        do {
            switch engineID {
            case .whisperKitLargeV3Turbo:
                try whisperKitEngine.removeDownloadedModel()
                modelStorageRevision += 1
            case .appleSpeechAnalyzer:
                lastError = "Apple's language assets are shared system models, so Mimi does not remove them."
            case .nemotronStreamingExperimental:
                break
            }
        } catch {
            record(error)
        }
    }

    private func startRecording() async {
        let configuration = SessionConfiguration(
            source: source,
            language: sourceLanguage,
            engine: engineID,
            inputDeviceID: selectedInputDeviceID
        )
        guard configuration.source == .microphone else {
            record(MimiError.captureSourceNotEnabled(configuration.source))
            return
        }

        recordingState = .preparing
        lastError = nil
        activeSession = configuration

        do {
            guard await microphoneCapture.requestPermission() else {
                throw MimiError.microphonePermissionDenied
            }

            let recordingURL = configuration.engine == .whisperKitLargeV3Turbo
                ? try MimiStorage.makeTemporaryRecordingURL()
                : nil
            lastRecordingURL = recordingURL

            switch configuration.engine {
            case .appleSpeechAnalyzer:
                guard #available(macOS 26.0, *) else {
                    throw MimiError.appleSpeechRequiresMacOS26
                }
                let inputFormat = try microphoneCapture.configureInput(deviceID: configuration.inputDeviceID)
                let engine = AppleSpeechEngine()
                try await engine.start(language: configuration.language, inputFormat: inputFormat) { [weak self] event in
                    self?.receive(event, for: configuration)
                }
                appleEngine = engine
            case .whisperKitLargeV3Turbo:
                // An accuracy recording never triggers a surprise model
                // download. Installation is a separate, explicit action.
                try whisperKitEngine.ensureInstalled()
            case .nemotronStreamingExperimental:
                throw MimiError.nemotronRequiresBakeOff
            }

            try microphoneCapture.start(recordingTo: recordingURL, deviceID: configuration.inputDeviceID) { [weak self] buffer in
                let transferableBuffer = SendableAudioBuffer(buffer)
                Task { @MainActor [weak self] in
                    guard let self,
                          self.activeSession == configuration,
                          #available(macOS 26.0, *),
                          let engine = self.appleEngine as? AppleSpeechEngine else {
                        return
                    }
                    engine.consume(transferableBuffer.buffer)
                }
            }
            recordingState = .recording
        } catch {
            await tearDownCapture(keepAudio: false)
            record(error)
        }
    }

    private func stopRecording() async {
        guard let configuration = activeSession else {
            recordingState = .idle
            return
        }
        recordingState = .processing

        do {
            let completedURL = try microphoneCapture.stop()
            switch configuration.engine {
            case .appleSpeechAnalyzer:
                if #available(macOS 26.0, *), let engine = appleEngine as? AppleSpeechEngine {
                    await engine.stop()
                }
                document.finalizeLiveText(language: configuration.language)
                persistDocument()
            case .whisperKitLargeV3Turbo:
                guard let completedURL else { throw MimiError.missingRecording }
                let text = try await whisperKitEngine.transcribe(
                    recordingAt: completedURL,
                    language: configuration.language
                )
                document.apply(.final(text), language: configuration.language)
                persistDocument()
            case .nemotronStreamingExperimental:
                throw MimiError.nemotronRequiresBakeOff
            }
            await tearDownCapture(keepAudio: false)
            recordingState = .idle
        } catch {
            await tearDownCapture(keepAudio: false)
            record(error)
        }
    }

    func clearTranscript() {
        document = TranscriptDocument()
        MimiStorage.clearLatestTranscript()
    }

    func refreshInputDevices() {
        inputDevices = AudioDeviceCatalog.inputDevices()
        if let selectedInputDeviceID, !inputDevices.contains(where: { $0.id == selectedInputDeviceID }) {
            self.selectedInputDeviceID = nil
        }
    }

    func copyTranscript() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(document.renderedText, forType: .string)
    }

    private func tearDownCapture(keepAudio: Bool) async {
        _ = try? microphoneCapture.stop()
        if #available(macOS 26.0, *) {
            if let engine = appleEngine as? AppleSpeechEngine {
                await engine.stop()
            }
            appleEngine = nil
        }

        if !keepAudio, let lastRecordingURL {
            try? FileManager.default.removeItem(at: lastRecordingURL)
            self.lastRecordingURL = nil
        }
        activeSession = nil
    }

    private func receive(_ event: TranscriptEvent, for configuration: SessionConfiguration) {
        guard activeSession == configuration else { return }
        document.apply(event, language: configuration.language)
        if case .final = event {
            persistDocument()
        }
    }

    private func persistDocument() {
        MimiStorage.saveLatestTranscript(document)
    }

    private func record(_ error: Error) {
        lastError = error.localizedDescription
        recordingState = .failed(error.localizedDescription)
    }
}

private struct SessionConfiguration: Equatable, Sendable {
    let source: AudioSource
    let language: SpeechLanguage
    let engine: TranscriptionEngineID
    let inputDeviceID: UInt32?
}

private struct SendableAudioBuffer: @unchecked Sendable {
    let buffer: AVAudioPCMBuffer

    init(_ buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }
}

private enum MimiError: LocalizedError {
    case microphonePermissionDenied
    case appleSpeechRequiresMacOS26
    case nemotronRequiresBakeOff
    case missingRecording
    case captureSourceNotEnabled(AudioSource)

    var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            "Mimi needs Microphone access. Enable it in System Settings, then try again."
        case .appleSpeechRequiresMacOS26:
            "Apple Speech live transcription requires macOS 26 or later. Choose Whisper Large-v3 on this Mac."
        case .nemotronRequiresBakeOff:
            "Nemotron 3.5 is visible as an experimental model but is not enabled until Mimi's local Mac accuracy and latency bake-off passes."
        case .missingRecording:
            "Mimi could not find the local audio used for this accuracy pass."
        case let .captureSourceNotEnabled(source):
            "\(source.displayName) capture is reserved for the Core Audio tap smoke test. Microphone capture is ready now."
        }
    }
}
