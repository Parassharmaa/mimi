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

/// The real availability of the selected Apple Speech language asset. This is
/// deliberately distinct from whether the Mac supports the SpeechAnalyzer API:
/// a capable Mac can still need a per-language system download.
public enum AppleSpeechAssetStatus: Equatable, Sendable {
    case unsupported
    case supported
    case downloading
    case installed
}

@MainActor
public protocol AppleSpeechProviding: AnyObject {
    /// Whether this Mac can use the SpeechAnalyzer API at all.
    var isPlatformAvailable: Bool { get }
    /// The actual asset state for the selected language on this Mac.
    func assetStatus(for language: SpeechLanguage) async -> AppleSpeechAssetStatus
    func installAssets(for language: SpeechLanguage) async throws
    func makeEngine() throws -> any AppleLiveTranscribing
}

public struct ModelDownloadProgress: Equatable, Sendable {
    /// The downloader's aggregate work units. Do not assume these are bytes:
    /// WhisperKit currently reports equally weighted model-file units.
    public let completedUnitCount: Int64
    public let totalUnitCount: Int64

    public init(completedUnitCount: Int64, totalUnitCount: Int64) {
        self.completedUnitCount = completedUnitCount
        self.totalUnitCount = totalUnitCount
    }

    public var fractionCompleted: Double? {
        guard totalUnitCount > 0 else { return nil }
        return min(1, max(0, Double(completedUnitCount) / Double(totalUnitCount)))
    }
}

@MainActor
public protocol WhisperAccuracyTranscribing: AnyObject {
    var isDownloaded: Bool { get }
    func ensureInstalled() throws
    func install(
        onProgress: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void
    ) async throws
    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String
    func removeDownloadedModel() async throws
}

/// Optional Apple-silicon MLX execution of the Nemotron 3.5 model. Unlike the
/// post-stop Whisper accuracy path, this protocol receives Mimi's selected
/// microphone/app/display PCM stream directly and reports replaceable live
/// hypotheses plus final bounded-window segments.
@MainActor
public protocol LocalLiveTranscribing: AnyObject {
    /// A nil value means the app contains a usable MLX Metal runtime on this
    /// machine. Keeping this separate from `isDownloaded` makes it impossible
    /// to promise a usable model merely because its weights are present.
    var runtimeAvailabilityMessage: String? { get }
    var isDownloaded: Bool { get }
    func ensureInstalled() throws
    func install() async throws
    func install(
        onProgress: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void
    ) async throws
    func startLive(
        language: SpeechLanguage,
        inputFormat: AVAudioFormat,
        onEvent: @escaping @MainActor (TranscriptEvent) -> Void,
        onBackpressure: @escaping @MainActor (String) -> Void
    ) async throws
    func consumeLive(_ buffer: AVAudioPCMBuffer)
    func stopLive() async
    func cancelLive() async
    func transcribe(recordingAt url: URL, language: SpeechLanguage) async throws -> String
    func removeDownloadedModel() async throws
}

public extension LocalLiveTranscribing {
    func install(
        onProgress: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void
    ) async throws {
        _ = onProgress
        try await install()
    }
}

@MainActor
public protocol NemotronMLXLiveTranscribing: LocalLiveTranscribing {}

@MainActor
public protocol QwenMLXLiveTranscribing: LocalLiveTranscribing {}

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
    public let nemotron: any NemotronMLXLiveTranscribing
    public let qwen: any QwenMLXLiveTranscribing
    public let storage: any TranscriptPersisting
    public let inputDevices: [AudioInputDevice]

    public init(
        microphoneCapture: any MicrophoneCapturing,
        screenAudioCapture: any ScreenAudioCapturing,
        appleSpeech: any AppleSpeechProviding,
        whisper: any WhisperAccuracyTranscribing,
        nemotron: any NemotronMLXLiveTranscribing,
        qwen: any QwenMLXLiveTranscribing,
        storage: any TranscriptPersisting,
        inputDevices: [AudioInputDevice]
    ) {
        self.microphoneCapture = microphoneCapture
        self.screenAudioCapture = screenAudioCapture
        self.appleSpeech = appleSpeech
        self.whisper = whisper
        self.nemotron = nemotron
        self.qwen = qwen
        self.storage = storage
        self.inputDevices = inputDevices
    }
}

public enum ModelReadiness: Equatable, Sendable {
    case checking(String)
    case ready
    case needsDownload(String)
    case downloading(String)
    case unavailable(String)
    case experimental(String)

    public var message: String? {
        switch self {
        case .ready:
            nil
        case let .checking(message), let .needsDownload(message), let .downloading(message), let .unavailable(message), let .experimental(message):
            message
        }
    }

    public var canStart: Bool {
        if case .ready = self { return true }
        return false
    }

    public var canInstall: Bool {
        if case .needsDownload = self { return true }
        return false
    }
}

/// A model download is a separate background operation, not a recording state.
/// This keeps a ready engine usable while another optional model is downloading.
public enum ModelSetupState: Equatable, Sendable {
    case idle
    case checking(engine: TranscriptionEngineID, language: SpeechLanguage?)
    case downloading(
        engine: TranscriptionEngineID,
        language: SpeechLanguage?,
        progress: ModelDownloadProgress?
    )
    case prewarming(engine: TranscriptionEngineID, language: SpeechLanguage?)
    case removing(engine: TranscriptionEngineID, language: SpeechLanguage?)
    case waitingForSystem(engine: TranscriptionEngineID, language: SpeechLanguage?)
    case cancelled(engine: TranscriptionEngineID, language: SpeechLanguage?)
    case failed(engine: TranscriptionEngineID, language: SpeechLanguage?, message: String)

    public var engine: TranscriptionEngineID? {
        switch self {
        case .idle:
            nil
        case let .checking(engine, _), let .downloading(engine, _, _), let .prewarming(engine, _),
             let .removing(engine, _), let .waitingForSystem(engine, _), let .cancelled(engine, _),
             let .failed(engine, _, _):
            engine
        }
    }

    public var language: SpeechLanguage? {
        switch self {
        case .idle:
            nil
        case let .checking(_, language), let .downloading(_, language, _), let .prewarming(_, language),
             let .removing(_, language), let .waitingForSystem(_, language), let .cancelled(_, language),
             let .failed(_, language, _):
            language
        }
    }

    public var isActive: Bool {
        switch self {
        case .checking, .downloading, .prewarming, .removing:
            true
        case .idle, .waitingForSystem, .cancelled, .failed:
            false
        }
    }

    public func matches(engine: TranscriptionEngineID, language: SpeechLanguage?) -> Bool {
        self.engine == engine && self.language == language
    }
}

@MainActor
@Observable
public final class TranscriptionSession {
    public var recordingState: RecordingState = .idle
    public var source: AudioSource = .microphone
    public var sourceLanguage: SpeechLanguage = .english {
        didSet {
            guard sourceLanguage != oldValue else { return }
            scheduleSelectedModelReadinessRefresh()
        }
    }
    public var engineID: TranscriptionEngineID {
        didSet {
            guard engineID != oldValue else { return }
            scheduleSelectedModelReadinessRefresh()
        }
    }
    public var translationMode: TranslationMode = .off
    public var document: TranscriptDocument
    public var lastError: String?
    public var lastRecordingURL: URL?
    public var inputDevices: [AudioInputDevice]
    public var selectedInputDeviceID: UInt32?
    public private(set) var appleSpeechAssetStatuses: [SpeechLanguage: AppleSpeechAssetStatus] = [:]
    public private(set) var modelSetupState: ModelSetupState = .idle

    private let microphoneCapture: any MicrophoneCapturing
    private let screenAudioCapture: any ScreenAudioCapturing
    private let appleSpeech: any AppleSpeechProviding
    private let whisper: any WhisperAccuracyTranscribing
    private let nemotron: any NemotronMLXLiveTranscribing
    private let qwen: any QwenMLXLiveTranscribing
    private let storage: any TranscriptPersisting
    @ObservationIgnored private var appleEngine: (any AppleLiveTranscribing)?
    @ObservationIgnored private var nemotronLiveSessionActive = false
    @ObservationIgnored private var qwenLiveSessionActive = false
    @ObservationIgnored private var activeSession: SessionConfiguration?
    @ObservationIgnored private var activeAudioFrames: RealtimeAudioFramePipe?
    @ObservationIgnored private var captureIsActive = false
    @ObservationIgnored private var modelStorageRevision = 0
    @ObservationIgnored private var modelReadinessRefreshTask: Task<Void, Never>?
    @ObservationIgnored private var appleSpeechDownloadRefreshTask: Task<Void, Never>?
    @ObservationIgnored private var appleStatusGenerations: [SpeechLanguage: Int] = [:]
    @ObservationIgnored private var modelSetupTask: Task<Void, Never>?
    @ObservationIgnored private var modelSetupID: UUID?

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
        qwen = dependencies.qwen
        storage = dependencies.storage
        inputDevices = dependencies.inputDevices
        engineID = initialEngine ?? (dependencies.appleSpeech.isPlatformAvailable ? .appleSpeechAnalyzer : .whisperKitLargeV3Turbo)
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
        guard !modelSetupState.isActive else { return false }
        return switch engineID {
        case .whisperKitLargeV3Turbo: whisper.isDownloaded
        case .nemotronStreamingExperimental: nemotron.isDownloaded
        case .qwen3StreamingExperimental: qwen.isDownloaded
        case .appleSpeechAnalyzer: false
        }
    }

    public var selectedModelReadiness: ModelReadiness {
        if let activeReadiness = readinessDuringSelectedSetup {
            return activeReadiness
        }

        switch engineID {
        case .appleSpeechAnalyzer:
            guard appleSpeech.isPlatformAvailable else {
                return .unavailable("Apple Speech live transcription requires macOS 26 or later.")
            }
            guard let assetStatus = appleSpeechAssetStatuses[sourceLanguage] else {
                return .checking("Checking the \(sourceLanguage.displayName) Apple Speech asset…")
            }
            return appleReadiness(for: assetStatus, language: sourceLanguage)
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
                : .needsDownload("Download Nemotron MLX (756 MB) before starting experimental bounded live transcription.")
        case .qwen3StreamingExperimental:
            if let runtimeAvailabilityMessage = qwen.runtimeAvailabilityMessage {
                return .unavailable(runtimeAvailabilityMessage)
            }
            return qwen.isDownloaded
                ? .ready
                : .needsDownload("Download Qwen3-ASR MLX (713 MB) before starting experimental dual-pass live transcription.")
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
        guard !controlsLocked, !modelSetupState.isActive else { return false }
        return selectedModelReadiness.canInstall
    }

    public var selectedModelSetupState: ModelSetupState {
        // The normal UI locks the selectors while setup is active. Preserve
        // the active state anyway if a programmatic/model-selection race gets
        // through, so the in-flight download never becomes invisible.
        if modelSetupState.isActive {
            return modelSetupState
        }
        return modelSetupState.matches(engine: engineID, language: selectedModelSetupLanguage)
            ? modelSetupState
            : .idle
    }

    public var canCancelSelectedModelInstall: Bool {
        guard modelSetupState.matches(
            engine: .whisperKitLargeV3Turbo,
            language: nil
        ) else {
            return false
        }

        return switch modelSetupState {
        case .downloading, .prewarming:
            true
        case .idle, .checking, .removing, .waitingForSystem, .cancelled, .failed:
            false
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

    /// Re-checks the system-managed Apple asset for the selected language.
    /// App-managed model markers are synchronous, so they only need an
    /// observation refresh.
    public func refreshSelectedModelReadiness() async {
        switch engineID {
        case .appleSpeechAnalyzer:
            _ = await refreshAppleSpeechAssetStatus(for: sourceLanguage)
        case .whisperKitLargeV3Turbo, .nemotronStreamingExperimental, .qwen3StreamingExperimental:
            modelStorageRevision += 1
        }
    }

    public func installSelectedModel() {
        guard let request = beginModelSetup() else { return }
        modelSetupTask = Task { [weak self] in
            await self?.performModelInstall(request)
        }
    }

    public func installSelectedModelNow() async {
        guard let request = beginModelSetup() else { return }
        await performModelInstall(request)
    }

    public func cancelSelectedModelInstall() {
        guard canCancelSelectedModelInstall else { return }
        modelSetupTask?.cancel()
    }

    public func removeSelectedModel() {
        guard let request = beginModelSetup(removing: true) else { return }
        modelSetupTask = Task { [weak self] in
            await self?.performModelRemoval(request)
        }
    }

    public func removeSelectedModelNow() async {
        guard let request = beginModelSetup(removing: true) else { return }
        await performModelRemoval(request)
    }

    private func beginModelSetup(removing: Bool = false) -> ModelSetupRequest? {
        guard !controlsLocked, !modelSetupState.isActive else { return nil }

        let request = ModelSetupRequest(
            id: UUID(),
            engine: engineID,
            language: engineID == .appleSpeechAnalyzer ? sourceLanguage : nil
        )
        // Setup is its own recoverable workflow. Do not leave a stale
        // recording-start error pinned in the menu after the person chooses
        // the action that resolves it.
        lastError = nil
        if case .failed = recordingState {
            recordingState = .idle
        }
        modelSetupID = request.id
        modelSetupState = removing
            ? .removing(engine: request.engine, language: request.language)
            : .checking(engine: request.engine, language: request.language)
        return request
    }

    private func performModelInstall(_ request: ModelSetupRequest) async {
        defer { finishModelSetupTask(id: request.id) }

        do {
            try Task.checkCancellation()

            switch request.engine {
            case .appleSpeechAnalyzer:
                guard appleSpeech.isPlatformAvailable else {
                    throw TranscriptionSessionError.appleSpeechRequiresMacOS26
                }
                guard let language = request.language else { return }

                let initialStatus = await refreshAppleSpeechAssetStatus(for: language)
                try Task.checkCancellation()
                switch initialStatus {
                case .installed, .unsupported:
                    updateModelSetup(.idle, for: request)
                case .downloading:
                    updateModelSetup(.waitingForSystem(engine: request.engine, language: language), for: request)
                    scheduleAppleSpeechDownloadRefresh(for: language)
                case .supported:
                    updateModelSetup(.downloading(engine: request.engine, language: language, progress: nil), for: request)
                    try await appleSpeech.installAssets(for: language)
                    let finalStatus = await refreshAppleSpeechAssetStatus(for: language)
                    switch finalStatus {
                    case .installed, .unsupported:
                        updateModelSetup(.idle, for: request)
                    case .supported, .downloading:
                        updateModelSetup(.waitingForSystem(engine: request.engine, language: language), for: request)
                        scheduleAppleSpeechDownloadRefresh(for: language)
                    }
                }
            case .whisperKitLargeV3Turbo:
                updateModelSetup(.downloading(engine: request.engine, language: nil, progress: nil), for: request)
                try await whisper.install { [weak self, request] progress in
                    self?.updateModelDownloadProgress(progress, for: request)
                }
                modelStorageRevision += 1
                updateModelSetup(.idle, for: request)
            case .nemotronStreamingExperimental:
                updateModelSetup(.downloading(engine: request.engine, language: nil, progress: nil), for: request)
                try await nemotron.install()
                modelStorageRevision += 1
                updateModelSetup(.idle, for: request)
            case .qwen3StreamingExperimental:
                updateModelSetup(.downloading(engine: request.engine, language: nil, progress: nil), for: request)
                try await qwen.install { [weak self, request] progress in
                    self?.updateModelDownloadProgress(progress, for: request)
                }
                modelStorageRevision += 1
                updateModelSetup(.idle, for: request)
            }
        } catch {
            if isCancellation(error) {
                updateModelSetup(.cancelled(engine: request.engine, language: request.language), for: request)
            } else {
                if let language = request.language {
                    _ = await refreshAppleSpeechAssetStatus(for: language)
                }
                updateModelSetup(
                    .failed(engine: request.engine, language: request.language, message: error.localizedDescription),
                    for: request
                )
            }
        }
    }

    private func performModelRemoval(_ request: ModelSetupRequest) async {
        defer { finishModelSetupTask(id: request.id) }

        do {
            switch request.engine {
            case .whisperKitLargeV3Turbo:
                try await whisper.removeDownloadedModel()
                modelStorageRevision += 1
            case .nemotronStreamingExperimental:
                try await nemotron.removeDownloadedModel()
                modelStorageRevision += 1
            case .qwen3StreamingExperimental:
                try await qwen.removeDownloadedModel()
                modelStorageRevision += 1
            case .appleSpeechAnalyzer:
                updateModelSetup(
                    .failed(
                        engine: request.engine,
                        language: request.language,
                        message: "Apple manages this shared language asset. Mimi cannot remove it."
                    ),
                    for: request
                )
                return
            }
            updateModelSetup(.idle, for: request)
        } catch {
            updateModelSetup(
                .failed(engine: request.engine, language: request.language, message: error.localizedDescription),
                for: request
            )
        }
    }

    private var selectedModelSetupLanguage: SpeechLanguage? {
        engineID == .appleSpeechAnalyzer ? sourceLanguage : nil
    }

    private var readinessDuringSelectedSetup: ModelReadiness? {
        // Do not make a newly selected ready model look unavailable merely
        // because a different model is downloading in the background.
        guard modelSetupState.matches(engine: engineID, language: selectedModelSetupLanguage) else {
            return nil
        }
        return switch selectedModelSetupState {
        case .idle:
            nil
        case let .checking(engine, language):
            .checking(modelSetupTitle(for: engine, language: language, verb: "Checking"))
        case let .downloading(engine, language, progress):
            .downloading(modelDownloadMessage(for: engine, language: language, progress: progress))
        case let .prewarming(engine, language):
            .downloading(modelSetupTitle(for: engine, language: language, verb: "Preparing"))
        case let .removing(engine, language):
            .checking(modelSetupTitle(for: engine, language: language, verb: "Removing"))
        case let .waitingForSystem(_, language):
            .downloading(
                "macOS is downloading or waiting to download the \(language?.displayName ?? "selected") Apple Speech asset. You can use another ready model meanwhile."
            )
        case let .cancelled(engine, language):
            engine == .whisperKitLargeV3Turbo
                ? .needsDownload(
                    "\(modelSetupTitle(for: engine, language: language, verb: "Download")) paused. Retry resumes any saved Whisper data."
                )
                : nil
        case .failed:
            // Keep actual model availability authoritative. The setup card
            // retains the actionable error, while this value still prevents a
            // false download CTA for unsupported Apple hardware/languages.
            nil
        }
    }

    private func appleReadiness(
        for status: AppleSpeechAssetStatus,
        language: SpeechLanguage
    ) -> ModelReadiness {
        return switch status {
        case .installed:
            .ready
        case .supported:
            .needsDownload(
                "Download the macOS-managed \(language.displayName) Apple Speech asset before recording."
            )
        case .downloading:
            .downloading(
                "macOS is downloading or waiting to download the \(language.displayName) Apple Speech asset."
            )
        case .unsupported:
            .unavailable(
                "Apple Speech does not provide a local \(language.displayName) asset on this Mac. Choose Whisper Large-v3 instead."
            )
        }
    }

    private func modelSetupTitle(
        for engine: TranscriptionEngineID,
        language: SpeechLanguage?,
        verb: String
    ) -> String {
        return switch engine {
        case .appleSpeechAnalyzer:
            "\(verb) \(language?.displayName ?? "Apple Speech") Apple Speech"
        case .whisperKitLargeV3Turbo:
            "\(verb) Whisper Large-v3"
        case .nemotronStreamingExperimental:
            "\(verb) Nemotron MLX"
        case .qwen3StreamingExperimental:
            "\(verb) Qwen3-ASR MLX"
        }
    }

    private func modelDownloadMessage(
        for engine: TranscriptionEngineID,
        language: SpeechLanguage?,
        progress: ModelDownloadProgress?
    ) -> String {
        guard engine == .whisperKitLargeV3Turbo, let progress else {
            return modelSetupTitle(for: engine, language: language, verb: "Downloading")
        }

        if let fraction = progress.fractionCompleted {
            return "Downloading Whisper Large-v3 — \(Int((fraction * 100).rounded()))%"
        }
        return "Downloading Whisper Large-v3…"
    }

    private func scheduleSelectedModelReadinessRefresh() {
        modelReadinessRefreshTask?.cancel()
        guard engineID == .appleSpeechAnalyzer else {
            modelStorageRevision += 1
            return
        }

        let language = sourceLanguage
        modelReadinessRefreshTask = Task { [weak self] in
            guard let self, !Task.isCancelled else { return }
            _ = await self.refreshAppleSpeechAssetStatus(for: language)
        }
    }

    private func refreshAppleSpeechAssetStatus(for language: SpeechLanguage) async -> AppleSpeechAssetStatus {
        guard appleSpeech.isPlatformAvailable else {
            appleSpeechAssetStatuses[language] = .unsupported
            return .unsupported
        }

        let nextGeneration = (appleStatusGenerations[language] ?? 0) + 1
        appleStatusGenerations[language] = nextGeneration
        let status = await appleSpeech.assetStatus(for: language)
        guard appleStatusGenerations[language] == nextGeneration else {
            // The caller may be at the recording boundary. Even when a newer
            // refresh owns the cache, this call must use its own freshly
            // queried truth rather than an older cached `.installed` value.
            return status
        }

        appleSpeechAssetStatuses[language] = status
        if modelSetupState.matches(engine: .appleSpeechAnalyzer, language: language) {
            switch modelSetupState {
            case .waitingForSystem where status == .installed || status == .unsupported:
                modelSetupState = .idle
            case .failed where status == .installed:
                // A manual status check must recover an already-installed
                // asset from an older setup error without requiring relaunch.
                modelSetupState = .idle
            default:
                break
            }
        }
        return status
    }

    private func scheduleAppleSpeechDownloadRefresh(for language: SpeechLanguage) {
        appleSpeechDownloadRefreshTask?.cancel()
        appleSpeechDownloadRefreshTask = Task { [weak self] in
            var lastStatus: AppleSpeechAssetStatus = .supported
            for _ in 0..<12 {
                try? await Task.sleep(for: .seconds(5))
                guard let self, !Task.isCancelled else { return }
                let status = await self.refreshAppleSpeechAssetStatus(for: language)
                lastStatus = status
                guard status == .supported || status == .downloading else { return }
            }

            guard let self,
                  !Task.isCancelled,
                  lastStatus == .supported,
                  modelSetupState.matches(engine: .appleSpeechAnalyzer, language: language),
                  case .waitingForSystem = modelSetupState else { return }
            modelSetupState = .failed(
                engine: .appleSpeechAnalyzer,
                language: language,
                message: "macOS has not confirmed the \(language.displayName) Apple Speech asset yet. Check status, then retry if it is still missing."
            )
        }
    }

    private func updateModelDownloadProgress(
        _ progress: ModelDownloadProgress,
        for request: ModelSetupRequest
    ) {
        updateModelSetup(
            .downloading(engine: request.engine, language: request.language, progress: progress),
            for: request
        )
    }

    private func updateModelSetup(_ state: ModelSetupState, for request: ModelSetupRequest) {
        guard modelSetupID == request.id else { return }
        modelSetupState = state
    }

    private func finishModelSetupTask(id: UUID) {
        guard modelSetupID == id else { return }
        modelSetupID = nil
        modelSetupTask = nil
    }

    private func isCancellation(_ error: Error) -> Bool {
        Task.isCancelled || error is CancellationError || (error as? URLError)?.code == .cancelled
    }

    private func configurationMatchesCurrentSelection(_ configuration: SessionConfiguration) -> Bool {
        configuration.source == source &&
            configuration.language == sourceLanguage &&
            configuration.engine == engineID &&
            configuration.inputDeviceID == selectedInputDeviceID
    }

    public func startRecording() async {
        guard !controlsLocked else { return }
        let configuration = SessionConfiguration(
            source: source,
            language: sourceLanguage,
            engine: engineID,
            inputDeviceID: selectedInputDeviceID
        )

        // Re-check Apple’s per-language asset at the recording boundary. The
        // button is normally disabled until it is ready, but this protects the
        // path from a system asset eviction or a stale UI refresh before any
        // microphone permission/capture side effect can occur.
        if configuration.engine == .appleSpeechAnalyzer {
            switch await refreshAppleSpeechAssetStatus(for: configuration.language) {
            case .installed:
                break
            case .supported:
                record(TranscriptionSessionError.appleAssetsNeedExplicitDownload)
                return
            case .downloading:
                record(TranscriptionSessionError.appleAssetsDownloading)
                return
            case .unsupported:
                record(TranscriptionSessionError.appleSpeechLanguageUnavailable(configuration.language))
                return
            }
        }

        // AssetInventory is asynchronous. Do not let an older start attempt
        // begin capture after a person has switched input, language, model, or
        // microphone while macOS was checking the selected Apple asset.
        guard !controlsLocked, configurationMatchesCurrentSelection(configuration) else {
            return
        }

        recordingState = .preparing
        lastError = nil
        activeSession = configuration

        do {
            // Model eligibility is checked before privacy prompts or capture.
            // A gated/missing model should never start a recording or trigger a
            // surprise download.
            switch configuration.engine {
            case .appleSpeechAnalyzer:
                guard appleSpeech.isPlatformAvailable else { throw TranscriptionSessionError.appleSpeechRequiresMacOS26 }
            case .whisperKitLargeV3Turbo:
                try whisper.ensureInstalled()
            case .nemotronStreamingExperimental:
                try nemotron.ensureInstalled()
            case .qwen3StreamingExperimental:
                try qwen.ensureInstalled()
            }

            let recordingURL: URL?
            switch configuration.engine {
            case .appleSpeechAnalyzer:
                recordingURL = nil
            case .whisperKitLargeV3Turbo:
                recordingURL = try storage.makeTemporaryRecordingURL(fileExtension: "caf")
            case .nemotronStreamingExperimental:
                // The bounded live MLX session receives selected PCM frames
                // directly, so it neither writes nor retains raw source audio.
                recordingURL = nil
            case .qwen3StreamingExperimental:
                // Qwen's dual-pass session keeps bounded in-memory encoder
                // windows and never retains a source-audio file.
                recordingURL = nil
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
                try await nemotron.startLive(
                    language: configuration.language,
                    inputFormat: inputFormat,
                    onEvent: { [weak self] event in
                        self?.receive(event, for: configuration)
                    },
                    onBackpressure: { [weak self] message in
                        self?.receiveLiveWarning(message, for: configuration)
                    }
                )
                nemotronLiveSessionActive = true
                let frames = RealtimeAudioFramePipe(capacity: 32)
                activeAudioFrames = frames
                audioFrames = frames
            case .qwen3StreamingExperimental:
                try await qwen.startLive(
                    language: configuration.language,
                    inputFormat: inputFormat,
                    onEvent: { [weak self] event in
                        self?.receive(event, for: configuration)
                    },
                    onBackpressure: { [weak self] message in
                        self?.receiveLiveWarning(message, for: configuration)
                    }
                )
                qwenLiveSessionActive = true
                let frames = RealtimeAudioFramePipe(capacity: 32)
                activeAudioFrames = frames
                audioFrames = frames
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
                if let activeAudioFrames {
                    drainAudioFrames(activeAudioFrames, for: configuration)
                }
                await nemotron.stopLive()
                nemotronLiveSessionActive = false
                try persistDocument()
            case .qwen3StreamingExperimental:
                if let activeAudioFrames {
                    drainAudioFrames(activeAudioFrames, for: configuration)
                }
                await qwen.stopLive()
                qwenLiveSessionActive = false
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
        if nemotronLiveSessionActive {
            await nemotron.cancelLive()
            nemotronLiveSessionActive = false
        }
        if qwenLiveSessionActive {
            await qwen.cancelLive()
            qwenLiveSessionActive = false
        }

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
        // a second time while handling a system-driven stop. Finalize the
        // live engine before reporting the capture error: a selected app may
        // quit mid-sentence, but its local captions should remain available
        // to copy instead of vanishing with the stream.
        captureIsActive = false
        if let activeAudioFrames {
            drainAudioFrames(activeAudioFrames, for: configuration)
        }

        switch configuration.engine {
        case .appleSpeechAnalyzer:
            if let engine = appleEngine {
                await engine.stop()
            }
            appleEngine = nil
            document.finalizeLiveText(language: configuration.language)
            try? persistDocument()
        case .nemotronStreamingExperimental:
            if nemotronLiveSessionActive {
                await nemotron.stopLive()
                nemotronLiveSessionActive = false
            }
            try? persistDocument()
        case .qwen3StreamingExperimental:
            if qwenLiveSessionActive {
                await qwen.stopLive()
                qwenLiveSessionActive = false
            }
            try? persistDocument()
        case .whisperKitLargeV3Turbo:
            break
        }
        activeAudioFrames?.discard()
        activeAudioFrames = nil

        if let lastRecordingURL {
            try? storage.removeTemporaryRecording(at: lastRecordingURL)
            self.lastRecordingURL = nil
        }
        activeSession = nil
        record(TranscriptionSessionError.screenAudioStreamStopped(message))
    }

    private func drainAudioFrames(_ audioFrames: RealtimeAudioFramePipe, for configuration: SessionConfiguration) {
        guard activeSession == configuration,
              activeAudioFrames === audioFrames else {
            audioFrames.discard()
            return
        }

        while let frame = audioFrames.dequeueForDrain() {
            guard activeSession == configuration, activeAudioFrames === audioFrames else {
                audioFrames.discard()
                return
            }
            switch configuration.engine {
            case .appleSpeechAnalyzer:
                guard let appleEngine else {
                    audioFrames.discard()
                    return
                }
                appleEngine.consume(frame.buffer)
            case .nemotronStreamingExperimental:
                nemotron.consumeLive(frame.buffer)
            case .qwen3StreamingExperimental:
                qwen.consumeLive(frame.buffer)
            case .whisperKitLargeV3Turbo:
                audioFrames.discard()
                return
            }
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

    private func receiveLiveWarning(_ message: String, for configuration: SessionConfiguration) {
        guard activeSession == configuration else { return }
        // This is nonfatal: when local MLX inference falls behind realtime
        // input, the engine bounds memory by discarding oldest queued audio.
        // Keep capture running and make that tradeoff visible immediately.
        lastError = message
    }

    private func persistDocument() throws {
        try storage.saveLatestTranscript(document)
    }

    private func record(_ error: Error) {
        lastError = error.localizedDescription
        recordingState = .failed(error.localizedDescription)
    }
}

private struct ModelSetupRequest: Equatable, Sendable {
    let id: UUID
    let engine: TranscriptionEngineID
    let language: SpeechLanguage?
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
    case appleAssetsDownloading
    case appleSpeechLanguageUnavailable(SpeechLanguage)
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
        case .appleAssetsDownloading:
            "macOS is still downloading Apple Speech's local language asset. Use Whisper Large-v3 while it finishes, then check the model status."
        case let .appleSpeechLanguageUnavailable(language):
            "Apple Speech does not provide a local \(language.displayName) asset on this Mac. Choose Whisper Large-v3 instead."
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
