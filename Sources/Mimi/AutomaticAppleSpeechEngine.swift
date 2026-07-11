@preconcurrency import AVFoundation
import MimiCore
import MimiSession
@preconcurrency import WhisperKit

/// Apple Speech remains the transcription engine. Whisper tiny is used only
/// as a bounded acoustic router that decides whether the active Apple engine
/// should be English or Japanese.
@MainActor
final class AutomaticAppleSpeechEngine: AutomaticAppleSpeechTranscribing {
    private let appleSpeech: any AppleSpeechProviding
    private let detector: WhisperLanguageDetector
    private lazy var streamingDetector = StreamingAcousticLanguageDetector(detector: detector)

    private var activeEngine: (any AppleLiveTranscribing)?
    private var activeLanguage: SpeechLanguage?
    private var inputFormat: AVAudioFormat?
    private var generation = UUID()
    private var routeConfirmed = false
    private var isSwitching = false
    private var recentBuffers: [AVAudioPCMBuffer] = []
    private var recentFrameCount: AVAudioFramePosition = 0
    private var onEvent: (@MainActor (TranscriptEvent, SpeechLanguage) -> Void)?
    private var onLanguageChange: (@MainActor (SpeechLanguage) -> Void)?

    init(
        appleSpeech: any AppleSpeechProviding,
        detector: WhisperLanguageDetector = WhisperLanguageDetector()
    ) {
        self.appleSpeech = appleSpeech
        self.detector = detector
    }

    var isLanguageDetectorInstalled: Bool { detector.isDownloaded }

    func installLanguageDetector(
        onProgress: @escaping @MainActor @Sendable (ModelDownloadProgress) -> Void
    ) async throws {
        try await detector.install(onProgress: onProgress)
    }

    func removeLanguageDetector() throws {
        try detector.removeDownloadedModel()
    }

    func start(
        inputFormat: AVAudioFormat,
        fallbackLanguage: SpeechLanguage,
        onEvent: @escaping @MainActor (TranscriptEvent, SpeechLanguage) -> Void,
        onLanguageChange: @escaping @MainActor (SpeechLanguage) -> Void
    ) async throws {
        guard isLanguageDetectorInstalled else {
            throw AutomaticLanguageError.detectorNotInstalled
        }

        self.inputFormat = inputFormat
        self.onEvent = onEvent
        self.onLanguageChange = onLanguageChange
        routeConfirmed = false
        isSwitching = false
        recentBuffers = []
        recentFrameCount = 0

        try await replaceEngine(with: fallbackLanguage, replayRecentAudio: false)
        streamingDetector.start { [weak self] language in
            guard let self else { return }
            Task { @MainActor [weak self] in
                await self?.handleStableLanguage(language)
            }
        }
    }

    func consume(_ buffer: AVAudioPCMBuffer) {
        retainRecentCopy(of: buffer)
        streamingDetector.consume(buffer)
        guard !isSwitching else { return }
        activeEngine?.consume(buffer)
    }

    func stop() async {
        streamingDetector.stop()
        routeConfirmed = false
        generation = UUID()
        if let activeEngine {
            await activeEngine.stop()
        }
        activeEngine = nil
        activeLanguage = nil
        inputFormat = nil
        recentBuffers = []
        recentFrameCount = 0
        onEvent = nil
        onLanguageChange = nil
        isSwitching = false
    }

    private func handleStableLanguage(_ language: SpeechLanguage) async {
        guard !isSwitching else { return }
        if activeLanguage == language {
            guard !routeConfirmed else { return }
            routeConfirmed = true
            onLanguageChange?(language)
            return
        }

        isSwitching = true
        routeConfirmed = false
        do {
            try await replaceEngine(with: language, replayRecentAudio: true)
            routeConfirmed = true
            onLanguageChange?(language)
        } catch {
            // Keep capture alive on the previous route. Asset readiness is
            // checked before Auto starts, so this is a runtime recovery path.
            routeConfirmed = activeEngine != nil
        }
        isSwitching = false
    }

    private func replaceEngine(
        with language: SpeechLanguage,
        replayRecentAudio: Bool
    ) async throws {
        guard let inputFormat else { throw AutomaticLanguageError.missingInputFormat }
        let nextGeneration = UUID()
        generation = nextGeneration

        if let activeEngine {
            await activeEngine.stop()
        }

        let engine = try appleSpeech.makeEngine()
        try await engine.start(language: language, inputFormat: inputFormat) { [weak self] event in
            guard let self,
                  self.generation == nextGeneration,
                  self.routeConfirmed,
                  self.activeLanguage == language else { return }
            self.onEvent?(event, language)
        }
        activeEngine = engine
        activeLanguage = language

        if replayRecentAudio {
            for buffer in recentBuffers {
                engine.consume(buffer)
            }
        }
    }

    private func retainRecentCopy(of buffer: AVAudioPCMBuffer) {
        guard let copy = Self.copy(buffer) else { return }
        recentBuffers.append(copy)
        recentFrameCount += AVAudioFramePosition(copy.frameLength)

        let maximumFrames = AVAudioFramePosition(max(1, buffer.format.sampleRate * 1.5))
        while recentFrameCount > maximumFrames, recentBuffers.count > 1 {
            recentFrameCount -= AVAudioFramePosition(recentBuffers.removeFirst().frameLength)
        }
    }

    private static func copy(_ source: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard let copy = AVAudioPCMBuffer(
            pcmFormat: source.format,
            frameCapacity: source.frameLength
        ) else { return nil }
        copy.frameLength = source.frameLength

        let sourceBuffers = UnsafeMutableAudioBufferListPointer(source.mutableAudioBufferList)
        let destinationBuffers = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
        guard sourceBuffers.count == destinationBuffers.count else { return nil }
        for index in sourceBuffers.indices {
            guard let sourceData = sourceBuffers[index].mData,
                  let destinationData = destinationBuffers[index].mData else { return nil }
            memcpy(destinationData, sourceData, Int(sourceBuffers[index].mDataByteSize))
            destinationBuffers[index].mDataByteSize = sourceBuffers[index].mDataByteSize
        }
        return copy
    }
}

@MainActor
private final class StreamingAcousticLanguageDetector {
    private let detector: WhisperLanguageDetector
    private var samples: [Float] = []
    private var samplesSinceDetection = 0
    private var detectionTask: Task<Void, Never>?
    private var hysteresis = AutomaticLanguageHysteresis()
    private var onStableLanguage: (@MainActor (SpeechLanguage) -> Void)?

    init(detector: WhisperLanguageDetector) {
        self.detector = detector
    }

    func start(onStableLanguage: @escaping @MainActor (SpeechLanguage) -> Void) {
        stop()
        self.onStableLanguage = onStableLanguage
    }

    func consume(_ buffer: AVAudioPCMBuffer) {
        guard let resampled = AudioProcessor.resampleAudio(
            fromBuffer: buffer,
            toSampleRate: Double(WhisperKit.sampleRate),
            channelCount: 1
        ) else { return }
        let next = AudioProcessor.convertBufferToArray(buffer: resampled)
        guard !next.isEmpty else { return }

        samples.append(contentsOf: next)
        samplesSinceDetection += next.count
        let maximumSamples = WhisperKit.sampleRate * 2
        if samples.count > maximumSamples {
            samples.removeFirst(samples.count - maximumSamples)
        }

        guard samples.count >= WhisperKit.sampleRate,
              samplesSinceDetection >= 12_000,
              detectionTask == nil else { return }
        samplesSinceDetection = 0
        let snapshot = samples
        detectionTask = Task { [weak self, detector] in
            defer { self?.detectionTask = nil }
            do {
                let decision = try await detector.detect(samples: snapshot)
                guard !Task.isCancelled,
                      let language = Self.language(for: decision.language) else { return }
                let score = decision.scores[decision.language] ?? -.infinity
                let rms = Self.rms(of: snapshot)
                if let stable = self?.hysteresis.observe(
                    language,
                    logProbability: score,
                    rms: rms
                ) {
                    self?.onStableLanguage?(stable)
                }
            } catch {
                // A failed window is discarded. The next bounded window can
                // recover without interrupting capture or Apple Speech.
            }
        }
    }

    func stop() {
        detectionTask?.cancel()
        detectionTask = nil
        samples = []
        samplesSinceDetection = 0
        hysteresis = AutomaticLanguageHysteresis()
        onStableLanguage = nil
    }

    private static func language(for code: String) -> SpeechLanguage? {
        switch code {
        case "en": .english
        case "ja": .japanese
        default: nil
        }
    }

    private static func rms(of samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sum = samples.reduce(Float.zero) { $0 + $1 * $1 }
        return sqrt(sum / Float(samples.count))
    }
}

private enum AutomaticLanguageError: LocalizedError {
    case detectorNotInstalled
    case missingInputFormat

    var errorDescription: String? {
        switch self {
        case .detectorNotInstalled:
            "Set up Mimi Auto Language before starting an automatic session."
        case .missingInputFormat:
            "Mimi could not prepare the automatic language audio route."
        }
    }
}
