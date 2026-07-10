@preconcurrency import AVFoundation
import Foundation
import MimiCore
import Observation

public struct AudioInputDevice: Identifiable, Hashable, Sendable {
    public let id: UInt32
    public let name: String

    public init(id: UInt32, name: String) {
        self.id = id
        self.name = name
    }

    public var displayName: String { name }
}

@MainActor
public protocol MicrophoneCapturing: AnyObject {
    func requestPermission() async -> Bool
    func configureInput(deviceID: UInt32?) throws -> AVAudioFormat
    func start(
        recordingTo url: URL?,
        deviceID: UInt32?,
        onBuffer: @escaping @Sendable (AVAudioPCMBuffer) -> Void
    ) throws
    @discardableResult func stop() throws -> URL?
}

/// The user-selected ScreenCaptureKit lane. Unlike microphone capture, it
/// requires an explicit picker selection before it can begin and can stop when
/// macOS revokes the selected content.
public struct ScreenAudioSelection: Equatable, Sendable {
    public let source: AudioSource
    public let description: String

    public init(source: AudioSource, description: String) {
        self.source = source
        self.description = description
    }
}

@MainActor
public protocol ScreenAudioCapturing: AnyObject {
    var selectedContent: ScreenAudioSelection? { get }

    /// Presents macOS's content-sharing picker. This must only be called from
    /// an explicit person-initiated action.
    func selectContent(for source: AudioSource) async throws
    func configureInput() throws -> AVAudioFormat
    func start(
        recordingTo url: URL?,
        onBuffer: @escaping @Sendable (AVAudioPCMBuffer) -> Void,
        onStreamStopped: @escaping @MainActor @Sendable (String?) -> Void
    ) async throws
    @discardableResult func stop() async throws -> URL?
}

@MainActor
public protocol AppleLiveTranscribing: AnyObject {
    func start(
        language: SpeechLanguage,
        inputFormat: AVAudioFormat,
        onEvent: @escaping @MainActor (TranscriptEvent) -> Void
    ) async throws
    func consume(_ buffer: AVAudioPCMBuffer)
    func stop() async
}

@MainActor
public protocol AppleSpeechProviding: AnyObject {
    var isAvailable: Bool { get }
    func installAssets(for language: SpeechLanguage) async throws
    func makeEngine() throws -> any AppleLiveTranscribing
}

@MainActor
public protocol WhisperAccuracyTranscribing: AnyObject {
    var isDownloaded: Bool { get }
    func ensureInstalled() throws
    func install() async throws
    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String
    func removeDownloadedModel() async throws
}

/// Optional Apple-silicon MLX execution of the Nemotron 3.5 model. The
/// current local runner is an accuracy pass: it receives the short, temporary
/// WAV captured by Mimi after Stop and removes no source audio itself.
@MainActor
public protocol NemotronMLXAccuracyTranscribing: AnyObject {
    /// A nil value means the app contains a usable MLX Metal runtime on this
    /// machine. Keeping this separate from `isDownloaded` makes it impossible
    /// to promise a usable model merely because its weights are present.
    var runtimeAvailabilityMessage: String? { get }
    var isDownloaded: Bool { get }
    func ensureInstalled() throws
    func install() async throws
    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String
    func removeDownloadedModel() async throws
}

@MainActor
public protocol TranscriptPersisting: AnyObject {
    func loadLatestTranscript() -> TranscriptDocument
    func saveLatestTranscript(_ document: TranscriptDocument) throws
    func clearLatestTranscript() throws
    func makeTemporaryRecordingURL(fileExtension: String) throws -> URL
    func removeTemporaryRecording(at url: URL) throws
    func removeStaleTemporaryRecordings()
}

@MainActor
public struct TranscriptionSessionDependencies {
    public let microphoneCapture: any MicrophoneCapturing
    public let screenAudioCapture: any ScreenAudioCapturing
    public let appleSpeech: any AppleSpeechProviding
    public let whisper: any WhisperAccuracyTranscribing
    public let nemotron: any NemotronMLXAccuracyTranscribing
    public let storage: any TranscriptPersisting
    public let inputDevices: [AudioInputDevice]

    public init(
        microphoneCapture: any MicrophoneCapturing,
        screenAudioCapture: any ScreenAudioCapturing,
        appleSpeech: any AppleSpeechProviding,
        whisper: any WhisperAccuracyTranscribing,
        nemotron: any NemotronMLXAccuracyTranscribing,
        storage: any TranscriptPersisting,
        inputDevices: [AudioInputDevice]
    ) {
        self.microphoneCapture = microphoneCapture
        self.screenAudioCapture = screenAudioCapture
        self.appleSpeech = appleSpeech
        self.whisper = whisper
        self.nemotron = nemotron
        self.storage = storage
        self.inputDevices = inputDevices
    }
}

public enum ModelReadiness: Equatable, Sendable {
    case ready
    case needsDownload(String)
    case unavailable(String)
    case experimental(String)

    public var message: String? {
        switch self {
        case .ready:
            nil
        case let .needsDownload(message), let .unavailable(message), let .experimental(message):
            message
        }
    }

    public var canStart: Bool {
        if case .ready = self { return true }
        return false
    }
}

@MainActor
@Observable
public final class TranscriptionSession {
    public var recordingState: RecordingState = .idle
    public var source: AudioSource = .microphone
    public var sourceLanguage: SpeechLanguage = .english
    public var engineID: TranscriptionEngineID
    public var translationMode: TranslationMode = .off
    public var document: TranscriptDocument
    public var lastError: String?
    public var lastRecordingURL: URL?
    public var inputDevices: [AudioInputDevice]
    public var selectedInputDeviceID: UInt32?

    private let microphoneCapture: any MicrophoneCapturing
    private let screenAudioCapture: any ScreenAudioCapturing
    private let appleSpeech: any AppleSpeechProviding
    private let whisper: any WhisperAccuracyTranscribing
    private let nemotron: any NemotronMLXAccuracyTranscribing
    private let storage: any TranscriptPersisting
    @ObservationIgnored private var appleEngine: (any AppleLiveTranscribing)?
    @ObservationIgnored private var activeSession: SessionConfiguration?
    @ObservationIgnored private var activeAudioFrames: RealtimeAudioFramePipe?
    @ObservationIgnored private var captureIsActive = false
    @ObservationIgnored private var modelStorageRevision = 0

    public init(
        dependencies: TranscriptionSessionDependencies,
        loadPersistedTranscript: Bool = true,
        initialEngine: TranscriptionEngineID? = nil
    ) {
        microphoneCapture = dependencies.microphoneCapture
        screenAudioCapture = dependencies.screenAudioCapture
        appleSpeech = dependencies.appleSpeech
        whisper = dependencies.whisper
        nemotron = dependencies.nemotron
        storage = dependencies.storage
        inputDevices = dependencies.inputDevices
        engineID = initialEngine ?? (dependencies.appleSpeech.isAvailable ? .appleSpeechAnalyzer : .whisperKitLargeV3Turbo)
        document = loadPersistedTranscript ? dependencies.storage.loadLatestTranscript() : TranscriptDocument()
        dependencies.storage.removeStaleTemporaryRecordings()
    }

    public var menuBarSymbolName: String {
        switch recordingState {
        case .recording: "waveform.circle.fill"
        case .preparing, .processing: "arrow.triangle.2.circlepath.circle"
        case .failed: "exclamationmark.triangle.fill"
        case .idle: "ear"
        }
    }

    public var isRecording: Bool {
        if case .recording = recordingState { return true }
        return false
    }

    public var controlsLocked: Bool {
        switch recordingState {
        case .preparing, .recording, .processing: true
        case .idle, .failed: false
        }
    }

    public var modelPack: LocalModelPack? {
        ModelCatalog.pack(for: engineID)
    }

    public var canRemoveSelectedModel: Bool {
        _ = modelStorageRevision
        return switch engineID {
        case .whisperKitLargeV3Turbo: whisper.isDownloaded
        case .nemotronStreamingExperimental: nemotron.isDownloaded
        case .appleSpeechAnalyzer: false
        }
    }

    public var selectedModelReadiness: ModelReadiness {
        switch engineID {
        case .appleSpeechAnalyzer:
            return appleSpeech.isAvailable
                ? .ready
                : .unavailable("Apple Speech live transcription requires macOS 26 or later.")
        case .whisperKitLargeV3Turbo:
            return whisper.isDownloaded
                ? .ready
                : .needsDownload("Download Whisper Large-v3 (626 MB) before starting an accuracy-pass recording.")
        case .nemotronStreamingExperimental:
            if let runtimeAvailabilityMessage = nemotron.runtimeAvailabilityMessage {
                return .unavailable(runtimeAvailabilityMessage)
            }
            return nemotron.isDownloaded
                ? .ready
                : .needsDownload("Download Nemotron MLX (756 MB) before starting an accuracy-pass recording.")
        }
    }

    public var canStartRecording: Bool {
        guard !controlsLocked, selectedModelReadiness.canStart else { return false }
        return switch source {
        case .microphone:
            true
        case .applicationAudio, .systemAudio:
            screenAudioCapture.selectedContent?.source == source
        }
    }

    public var screenAudioSelection: ScreenAudioSelection? {
        screenAudioCapture.selectedContent
    }

    public var canInstallSelectedModel: Bool {
        guard !controlsLocked else { return false }
        return switch engineID {
        case .appleSpeechAnalyzer: appleSpeech.isAvailable
        case .whisperKitLargeV3Turbo: true
        case .nemotronStreamingExperimental: nemotron.runtimeAvailabilityMessage == nil
        }
    }

    public func replaceInputDevices(_ devices: [AudioInputDevice]) {
        inputDevices = devices
        if let selectedInputDeviceID, !devices.contains(where: { $0.id == selectedInputDeviceID }) {
            self.selectedInputDeviceID = nil
        }
    }

    /// Opens the system-owned picker for selected app/display audio. A cancel
    /// is a normal outcome, so the session remains idle and exposes a concise
    /// inline status rather than entering a recording failure state.
    public func selectScreenAudioContent() async {
        guard !controlsLocked else { return }
        guard source != .microphone else {
            lastError = "Choose Selected App Audio or Selected Display Audio before selecting screen audio."
            return
        }

        lastError = nil
        do {
            try await screenAudioCapture.selectContent(for: source)
        } catch {
            lastError = error.localizedDescription
            recordingState = .idle
        }
    }

    public func toggleRecording() {
        Task {
            if isRecording {
                await stopRecording()
            } else {
                await startRecording()
            }
        }
    }

    public func installSelectedModel() {
        Task { await installSelectedModelNow() }
    }

    public func installSelectedModelNow() async {
        guard !controlsLocked else { return }
        recordingState = .preparing
        lastError = nil

        do {
            switch engineID {
            case .appleSpeechAnalyzer:
                guard appleSpeech.isAvailable else { throw TranscriptionSessionError.appleSpeechRequiresMacOS26 }
                try await appleSpeech.installAssets(for: sourceLanguage)
            case .whisperKitLargeV3Turbo:
                try await whisper.install()
            case .nemotronStreamingExperimental:
                try await nemotron.install()
            }
            modelStorageRevision += 1
            recordingState = .idle
        } catch {
            record(error)
        }
    }

    public func removeSelectedModel() {
        Task { await removeSelectedModelNow() }
    }

    public func removeSelectedModelNow() async {
        guard !controlsLocked else { return }
        recordingState = .preparing
        lastError = nil
        do {
            switch engineID {
            case .whisperKitLargeV3Turbo:
                try await whisper.removeDownloadedModel()
                modelStorageRevision += 1
            case .appleSpeechAnalyzer:
                lastError = "Apple's language assets are shared system models, so Mimi does not remove them."
            case .nemotronStreamingExperimental:
                try await nemotron.removeDownloadedModel()
                modelStorageRevision += 1
            }
            recordingState = .idle
        } catch {
            record(error)
        }
    }

    public func startRecording() async {
        guard !controlsLocked else { return }
        let configuration = SessionConfiguration(
            source: source,
            language: sourceLanguage,
            engine: engineID,
            inputDeviceID: selectedInputDeviceID
        )
        recordingState = .preparing
        lastError = nil
        activeSession = configuration

        do {
            // Model eligibility is checked before privacy prompts or capture.
            // A gated/missing model should never start a recording or trigger a
            // surprise download.
            switch configuration.engine {
            case .appleSpeechAnalyzer:
                guard appleSpeech.isAvailable else { throw TranscriptionSessionError.appleSpeechRequiresMacOS26 }
            case .whisperKitLargeV3Turbo:
                try whisper.ensureInstalled()
            case .nemotronStreamingExperimental:
                try nemotron.ensureInstalled()
            }

            let recordingURL: URL?
            switch configuration.engine {
            case .appleSpeechAnalyzer:
                recordingURL = nil
            case .whisperKitLargeV3Turbo:
                recordingURL = try storage.makeTemporaryRecordingURL(fileExtension: "caf")
            case .nemotronStreamingExperimental:
                // The MLX runner intentionally consumes a portable mono WAV
                // file so it never needs a system codec or conversion helper.
                recordingURL = try storage.makeTemporaryRecordingURL(fileExtension: "wav")
            }
            lastRecordingURL = recordingURL

            let inputFormat: AVAudioFormat
            switch configuration.source {
            case .microphone:
                guard await microphoneCapture.requestPermission() else {
                    throw TranscriptionSessionError.microphonePermissionDenied
                }
                inputFormat = try microphoneCapture.configureInput(deviceID: configuration.inputDeviceID)
            case .applicationAudio, .systemAudio:
                guard screenAudioCapture.selectedContent?.source == configuration.source else {
                    throw TranscriptionSessionError.screenAudioSelectionRequired(configuration.source)
                }
                inputFormat = try screenAudioCapture.configureInput()
            }

            let audioFrames: RealtimeAudioFramePipe?
            switch configuration.engine {
            case .appleSpeechAnalyzer:
                let engine = try appleSpeech.makeEngine()
                try await engine.start(language: configuration.language, inputFormat: inputFormat) { [weak self] event in
                    self?.receive(event, for: configuration)
                }
                appleEngine = engine
                let frames = RealtimeAudioFramePipe(capacity: 32)
                activeAudioFrames = frames
                audioFrames = frames
            case .whisperKitLargeV3Turbo:
                audioFrames = nil
            case .nemotronStreamingExperimental:
                audioFrames = nil
            }

            let onBuffer: @Sendable (AVAudioPCMBuffer) -> Void
            if let audioFrames {
                onBuffer = { @Sendable [weak self, configuration, audioFrames] buffer in
                    guard audioFrames.enqueueCopy(of: buffer) else { return }
                    Task { @MainActor [weak self, configuration, audioFrames] in
                        self?.drainAudioFrames(audioFrames, for: configuration)
                    }
                }
            } else {
                onBuffer = { _ in }
            }

            switch configuration.source {
            case .microphone:
                try microphoneCapture.start(
                    recordingTo: recordingURL,
                    deviceID: configuration.inputDeviceID,
                    onBuffer: onBuffer
                )
            case .applicationAudio, .systemAudio:
                try await screenAudioCapture.start(
                    recordingTo: recordingURL,
                    onBuffer: onBuffer,
                    onStreamStopped: { [weak self, configuration] message in
                        Task { @MainActor [weak self, configuration] in
                            await self?.screenAudioStreamStopped(message, for: configuration)
                        }
                    }
                )
            }
            captureIsActive = true
            recordingState = .recording
        } catch {
            await tearDownCapture(keepAudio: false)
            record(error)
        }
    }

    public func stopRecording() async {
        guard let configuration = activeSession else {
            recordingState = .idle
            return
        }
        recordingState = .processing

        do {
            let completedURL = try await stopCaptureIfNeeded()
            switch configuration.engine {
            case .appleSpeechAnalyzer:
                if let activeAudioFrames {
                    drainAudioFrames(activeAudioFrames, for: configuration)
                }
                if let engine = appleEngine {
                    await engine.stop()
                }
                appleEngine = nil
                document.finalizeLiveText(language: configuration.language)
                try persistDocument()
            case .whisperKitLargeV3Turbo:
                guard let completedURL else { throw TranscriptionSessionError.missingRecording }
                let text = try await whisper.transcribe(recordingAt: completedURL, language: configuration.language)
                document.apply(.final(text), language: configuration.language)
                try persistDocument()
            case .nemotronStreamingExperimental:
                guard let completedURL else { throw TranscriptionSessionError.missingRecording }
                let text = try await nemotron.transcribe(recordingAt: completedURL, language: configuration.language)
                document.apply(.final(text), language: configuration.language)
                try persistDocument()
            }
            await tearDownCapture(keepAudio: false)
            recordingState = .idle
        } catch {
            await tearDownCapture(keepAudio: false)
            record(error)
        }
    }

    public func clearTranscript() {
        do {
            try storage.clearLatestTranscript()
            document = TranscriptDocument()
        } catch {
            record(error)
        }
    }

    public func applyFixture(_ event: TranscriptEvent, language: SpeechLanguage) {
        document.apply(event, language: language)
    }

    /// Used only by the opt-in physical-Mac smoke command. It captures no
    /// source audio and verifies that the live tap receives callbacks.
    public func runMicrophoneCaptureSmokeTest() async throws -> Int {
        guard await microphoneCapture.requestPermission() else {
            throw TranscriptionSessionError.microphonePermissionDenied
        }

        let callbackCounter = RealtimeCallbackCounter()
        do {
            try microphoneCapture.start(recordingTo: nil, deviceID: selectedInputDeviceID) { @Sendable _ in
                callbackCounter.increment()
            }
            captureIsActive = true
            try await Task.sleep(nanoseconds: 1_000_000_000)
            _ = try await stopCaptureIfNeeded()
        } catch {
            _ = try? await stopCaptureIfNeeded()
            throw error
        }

        let callbackCount = callbackCounter.value
        guard callbackCount > 0 else { throw TranscriptionSessionError.captureSmokeReceivedNoAudio }
        return callbackCount
    }

    /// An opt-in physical-Mac model check. The caller must have explicitly
    /// installed the selected local model first; this method never downloads
    /// a model or retains source audio after the one-second session.
    public func runEngineSmokeTest(
        engine: TranscriptionEngineID,
        language: SpeechLanguage,
        durationNanoseconds: UInt64 = 1_000_000_000
    ) async throws {
        source = .microphone
        engineID = engine
        sourceLanguage = language
        await startRecording()
        guard isRecording else {
            throw TranscriptionSessionError.engineSmokeDidNotStart(lastError ?? "The model did not enter recording state.")
        }
        try await Task.sleep(nanoseconds: durationNanoseconds)
        await stopRecording()
        guard recordingState == .idle else {
            throw TranscriptionSessionError.engineSmokeDidNotFinish(lastError ?? "The model did not finish cleanly.")
        }
    }

    private func stopCaptureIfNeeded() async throws -> URL? {
        guard captureIsActive else { return nil }
        captureIsActive = false
        switch activeSession?.source ?? source {
        case .microphone:
            return try microphoneCapture.stop()
        case .applicationAudio, .systemAudio:
            return try await screenAudioCapture.stop()
        }
    }

    private func tearDownCapture(keepAudio: Bool) async {
        _ = try? await stopCaptureIfNeeded()
        activeAudioFrames?.discard()
        activeAudioFrames = nil
        if let engine = appleEngine {
            await engine.stop()
        }
        appleEngine = nil

        if !keepAudio, let lastRecordingURL {
            try? storage.removeTemporaryRecording(at: lastRecordingURL)
            self.lastRecordingURL = nil
        }
        activeSession = nil
    }

    private func screenAudioStreamStopped(_ message: String?, for configuration: SessionConfiguration) async {
        guard activeSession == configuration, captureIsActive else { return }

        // The ScreenCaptureKit delegate has already stopped the stream. Mark
        // the capture inactive before teardown so it cannot call stopCapture
        // a second time while handling a system-driven stop.
        captureIsActive = false
        activeAudioFrames?.discard()
        activeAudioFrames = nil
        if let engine = appleEngine {
            await engine.stop()
        }
        appleEngine = nil

        if let lastRecordingURL {
            try? storage.removeTemporaryRecording(at: lastRecordingURL)
            self.lastRecordingURL = nil
        }
        activeSession = nil
        record(TranscriptionSessionError.screenAudioStreamStopped(message))
    }

    private func drainAudioFrames(_ audioFrames: RealtimeAudioFramePipe, for configuration: SessionConfiguration) {
        guard activeSession == configuration,
              activeAudioFrames === audioFrames,
              let engine = appleEngine else {
            audioFrames.discard()
            return
        }

        while let frame = audioFrames.dequeueForDrain() {
            guard activeSession == configuration, activeAudioFrames === audioFrames else {
                audioFrames.discard()
                return
            }
            engine.consume(frame.buffer)
        }
    }

    private func receive(_ event: TranscriptEvent, for configuration: SessionConfiguration) {
        guard activeSession == configuration else { return }
        document.apply(event, language: configuration.language)
        if case .final = event {
            do {
                try persistDocument()
            } catch {
                lastError = error.localizedDescription
            }
        }
    }

    private func persistDocument() throws {
        try storage.saveLatestTranscript(document)
    }

    private func record(_ error: Error) {
        lastError = error.localizedDescription
        recordingState = .failed(error.localizedDescription)
    }
}

private struct SessionConfiguration: Equatable, Sendable {
    let id: UUID
    let source: AudioSource
    let language: SpeechLanguage
    let engine: TranscriptionEngineID
    let inputDeviceID: UInt32?

    init(source: AudioSource, language: SpeechLanguage, engine: TranscriptionEngineID, inputDeviceID: UInt32?) {
        id = UUID()
        self.source = source
        self.language = language
        self.engine = engine
        self.inputDeviceID = inputDeviceID
    }
}

private final class RealtimeAudioFramePipe: @unchecked Sendable {
    private let lock = NSLock()
    private let capacity: Int
    private var frames: [SendableAudioBuffer] = []
    private var drainScheduled = false

    init(capacity: Int) {
        self.capacity = max(1, capacity)
    }

    /// Returns true only when the caller must schedule a main-actor drain.
    /// Dropping the oldest frame when full keeps live transcription current.
    func enqueueCopy(of buffer: AVAudioPCMBuffer) -> Bool {
        guard let frame = SendableAudioBuffer(copying: buffer) else { return false }
        lock.lock()
        defer { lock.unlock() }

        if frames.count == capacity {
            frames.removeFirst()
        }
        frames.append(frame)

        guard !drainScheduled else { return false }
        drainScheduled = true
        return true
    }

    func dequeueForDrain() -> SendableAudioBuffer? {
        lock.lock()
        defer { lock.unlock() }

        guard !frames.isEmpty else {
            drainScheduled = false
            return nil
        }
        return frames.removeFirst()
    }

    func discard() {
        lock.lock()
        frames.removeAll(keepingCapacity: true)
        lock.unlock()
    }
}

/// AVAudioEngine owns a tap buffer only for the duration of its callback. This
/// deep copy is the explicit hand-off to the bounded, cross-actor queue.
private final class SendableAudioBuffer: @unchecked Sendable {
    let buffer: AVAudioPCMBuffer

    init?(copying source: AVAudioPCMBuffer) {
        guard let copy = AVAudioPCMBuffer(pcmFormat: source.format, frameCapacity: source.frameLength) else {
            return nil
        }

        copy.frameLength = source.frameLength
        let sourceBuffers = UnsafeMutableAudioBufferListPointer(source.mutableAudioBufferList)
        let destinationBuffers = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
        guard sourceBuffers.count == destinationBuffers.count else { return nil }

        for index in sourceBuffers.indices {
            let sourceBuffer = sourceBuffers[index]
            guard let sourceData = sourceBuffer.mData,
                  let destinationData = destinationBuffers[index].mData else {
                return nil
            }
            let byteCount = min(Int(sourceBuffer.mDataByteSize), Int(destinationBuffers[index].mDataByteSize))
            guard byteCount > 0 else { continue }
            memcpy(destinationData, sourceData, byteCount)
            destinationBuffers[index].mDataByteSize = UInt32(byteCount)
        }

        buffer = copy
    }
}

private final class RealtimeCallbackCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    func increment() {
        lock.lock()
        count += 1
        lock.unlock()
    }

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }
}

public enum TranscriptionSessionError: LocalizedError {
    case microphonePermissionDenied
    case appleSpeechRequiresMacOS26
    case appleAssetsNeedExplicitDownload
    case missingRecording
    case screenAudioSelectionRequired(AudioSource)
    case screenAudioStreamStopped(String?)
    case captureSmokeReceivedNoAudio
    case engineSmokeDidNotStart(String)
    case engineSmokeDidNotFinish(String)

    public var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            "Mimi needs Microphone access. Enable it in System Settings, then try again."
        case .appleSpeechRequiresMacOS26:
            "Apple Speech live transcription requires macOS 26 or later. Choose Whisper Large-v3 on this Mac."
        case .appleAssetsNeedExplicitDownload:
            "Download Apple Speech's local language assets in Settings before starting."
        case .missingRecording:
            "Mimi could not find the local audio used for this accuracy pass."
        case let .screenAudioSelectionRequired(source):
            "Choose a \(source.displayName.lowercased()) source in the macOS picker before recording."
        case let .screenAudioStreamStopped(message):
            message ?? "The selected app or display stopped sharing audio. Choose it again to continue."
        case .captureSmokeReceivedNoAudio:
            "Mimi received no microphone buffers during the one-second smoke test. Check the selected input device."
        case let .engineSmokeDidNotStart(message):
            "Mimi could not start the requested model smoke test: \(message)"
        case let .engineSmokeDidNotFinish(message):
            "Mimi could not finish the requested model smoke test: \(message)"
        }
    }
}
