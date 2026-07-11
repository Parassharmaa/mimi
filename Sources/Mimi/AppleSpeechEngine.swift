@preconcurrency import AVFoundation
import Foundation
import MimiCore
import MimiSession
import Speech

@available(macOS 26.0, *)
@MainActor
final class AppleSpeechEngine {
    enum ResultMode: String, Sendable {
        case accurate
        case progressive

        var preset: SpeechTranscriber.Preset {
            switch self {
            case .accurate:
                .transcription
            case .progressive:
                .progressiveTranscription
            }
        }
    }

    private let resultMode: ResultMode
    private var analyzer: SpeechAnalyzer?
    private var audioConverter: AVAudioConverter?
    private var analyzerFormat: AVAudioFormat?
    private var inputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    private var analysisTask: Task<Void, Never>?
    private var resultTask: Task<Void, Never>?

    init(resultMode: ResultMode = .progressive) {
        self.resultMode = resultMode
    }

    static func installAssets(for language: SpeechLanguage) async throws {
        let transcriber = try await makeTranscriber(for: language, resultMode: .progressive)
        switch await AssetInventory.status(forModules: [transcriber]) {
        case .installed:
            return
        case .unsupported:
            throw TranscriptionSessionError.appleSpeechLanguageUnavailable(language)
        case .supported, .downloading:
            break
        @unknown default:
            throw TranscriptionSessionError.appleSpeechLanguageUnavailable(language)
        }

        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await request.downloadAndInstall()
        }
    }

    static func assetStatus(for language: SpeechLanguage) async -> AppleSpeechAssetStatus {
        guard let transcriber = try? await makeTranscriber(for: language, resultMode: .progressive) else {
            return .unsupported
        }

        switch await AssetInventory.status(forModules: [transcriber]) {
        case .unsupported:
            return .unsupported
        case .supported:
            return .supported
        case .downloading:
            return .downloading
        case .installed:
            return .installed
        @unknown default:
            return .unsupported
        }
    }

    func start(
        language: SpeechLanguage,
        inputFormat: AVAudioFormat,
        onEvent: @escaping @MainActor (TranscriptEvent) -> Void
    ) async throws {
        let transcriber = try await Self.makeTranscriber(for: language, resultMode: resultMode)
        switch await AssetInventory.status(forModules: [transcriber]) {
        case .installed:
            break
        case .supported:
            throw TranscriptionSessionError.appleAssetsNeedExplicitDownload
        case .downloading:
            throw TranscriptionSessionError.appleAssetsDownloading
        case .unsupported:
            throw TranscriptionSessionError.appleSpeechLanguageUnavailable(language)
        @unknown default:
            throw TranscriptionSessionError.appleSpeechLanguageUnavailable(language)
        }

        guard let analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(
            compatibleWith: [transcriber],
            considering: inputFormat
        ) else {
            throw SpeechEngineError.noCompatibleAudioFormat
        }
        let converter = AVAudioConverter(from: inputFormat, to: analyzerFormat)
        let (inputSequence, continuation) = AsyncStream.makeStream(of: AnalyzerInput.self)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        try await analyzer.prepareToAnalyze(in: analyzerFormat)

        self.analyzer = analyzer
        self.audioConverter = converter
        self.analyzerFormat = analyzerFormat
        self.inputContinuation = continuation

        resultTask = Task { [weak self] in
            do {
                for try await result in transcriber.results {
                    guard !Task.isCancelled else { return }
                    let text = String(result.text.characters)
                    if result.isFinal {
                        onEvent(.final(text))
                    } else {
                        onEvent(.partial(text))
                    }
                }
            } catch {
                // The analysis task owns terminal state. Do not replace a
                // partially useful local transcript with an error here.
                _ = self
            }
        }

        analysisTask = Task { [weak self] in
            do {
                let lastSampleTime = try await analyzer.analyzeSequence(inputSequence)
                if let lastSampleTime {
                    try await analyzer.finalizeAndFinish(through: lastSampleTime)
                } else {
                    await analyzer.cancelAndFinishNow()
                }
            } catch {
                _ = self
            }
        }
    }

    func consume(_ buffer: AVAudioPCMBuffer) {
        guard let inputContinuation, let analyzerFormat else { return }

        guard let audioConverter else {
            inputContinuation.yield(AnalyzerInput(buffer: buffer))
            return
        }

        let ratio = analyzerFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(max(1, Double(buffer.frameLength) * ratio + 1))
        guard let convertedBuffer = AVAudioPCMBuffer(
            pcmFormat: analyzerFormat,
            frameCapacity: capacity
        ) else {
            return
        }

        let inputProvider = AudioConverterInput(buffer: buffer)
        var conversionError: NSError?
        let status = audioConverter.convert(to: convertedBuffer, error: &conversionError) { _, outputStatus in
            inputProvider.next(outputStatus)
        }
        guard status != .error, conversionError == nil, convertedBuffer.frameLength > 0 else { return }
        inputContinuation.yield(AnalyzerInput(buffer: convertedBuffer))
    }

    func stop() async {
        inputContinuation?.finish()
        inputContinuation = nil
        audioConverter = nil
        analyzerFormat = nil

        // Finish the stream first: this lets SpeechAnalyzer publish the final
        // non-volatile result instead of throwing away the last spoken words.
        if let analysisTask {
            await analysisTask.value
        }
        analysisTask = nil
        if let resultTask {
            await resultTask.value
        }
        resultTask = nil
        analyzer = nil
    }

    private static func makeTranscriber(
        for language: SpeechLanguage,
        resultMode: ResultMode
    ) async throws -> SpeechTranscriber {
        let requestedLocale = Locale(identifier: language.rawValue)
        guard let supportedLocale = await SpeechTranscriber.supportedLocale(equivalentTo: requestedLocale) else {
            throw SpeechEngineError.unsupportedLanguage(language)
        }
        return SpeechTranscriber(locale: supportedLocale, preset: resultMode.preset)
    }
}

private final class AudioConverterInput: @unchecked Sendable {
    private let buffer: AVAudioPCMBuffer
    private let lock = NSLock()
    private var wasProvided = false

    init(buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }

    func next(_ outputStatus: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
        lock.lock()
        defer { lock.unlock() }
        guard !wasProvided else {
            outputStatus.pointee = .noDataNow
            return nil
        }
        wasProvided = true
        outputStatus.pointee = .haveData
        return buffer
    }
}

private enum SpeechEngineError: LocalizedError {
    case unsupportedLanguage(SpeechLanguage)
    case noCompatibleAudioFormat

    var errorDescription: String? {
        switch self {
        case let .unsupportedLanguage(language):
            "Apple Speech does not currently provide a local model for \(language.displayName) on this Mac. Choose Whisper Large-v3 instead."
        case .noCompatibleAudioFormat:
            "Apple Speech could not negotiate a local transcription format for this microphone. Choose another input or use Whisper Large-v3."
        }
    }
}
