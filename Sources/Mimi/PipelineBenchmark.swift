@preconcurrency import AVFoundation
import Foundation
import Darwin
import MimiCore
import MimiSession

private enum PipelineBenchmarkError: LocalizedError {
    case invalidRequest(String), unavailable(String)
    var errorDescription: String? {
        switch self {
        case .invalidRequest(let message), .unavailable(let message): message
        }
    }
}

@MainActor
private enum BenchmarkSpeechEngine {
    case apple(any AppleLiveTranscribing)
    case local(any WhisperAccuracyTranscribing)

    init(_ request: BenchmarkCaseRequest) throws {
        let root = request.speechModelPath.map { URL(fileURLWithPath: $0) }
        switch request.engine {
        case "apple":
            guard #available(macOS 26, *) else { throw PipelineBenchmarkError.unavailable("Apple Speech requires macOS 26.") }
            self = .apple(AppleSpeechEngine(resultMode: .progressive))
        case "mimi-whisper":
            guard let path = request.speechModelPath,
                  ProcessInfo.processInfo.environment["MIMI_WHISPER_MLX_MODEL_DIR"] == path else {
                throw PipelineBenchmarkError.invalidRequest("Set MIMI_WHISPER_MLX_MODEL_DIR to the requested speech model path.")
            }
            let engine = MimiWhisperMLXLiveEngine()
            engine.recordsBenchmarkTiming = true
            self = .local(engine)
        case "phonon":
            guard request.language == .english else { throw PipelineBenchmarkError.unavailable("Phonon supports English only.") }
            self = .local(MimiPhononMLXLiveEngine(modelRoot: root))
        case "parakeet":
            guard request.language == .japanese else { throw PipelineBenchmarkError.unavailable("Parakeet supports Japanese only.") }
            self = .local(MimiParakeetJapaneseLiveEngine(modelRoot: root))
        default: throw PipelineBenchmarkError.invalidRequest("Unknown speech engine: \(request.engine)")
        }
    }

    func start(language: SpeechLanguage, format: AVAudioFormat,
               event: @escaping @MainActor (TranscriptEvent) -> Void,
               warning: @escaping @MainActor (String) -> Void) async throws {
        switch self {
        case .apple(let engine): try await engine.start(language: language, inputFormat: format, onEvent: event)
        case .local(let engine):
            guard engine.isDownloaded, engine.runtimeAvailabilityMessage == nil else {
                throw PipelineBenchmarkError.unavailable(engine.runtimeAvailabilityMessage ?? "The requested speech model is not installed.")
            }
            try await engine.startLive(language: language, inputFormat: format, onEvent: event, onBackpressure: warning)
        }
    }

    func consume(_ buffer: AVAudioPCMBuffer) {
        switch self {
        case .apple(let engine): engine.consume(buffer)
        case .local(let engine): engine.consumeLive(buffer)
        }
    }

    func stop() async {
        switch self {
        case .apple(let engine): await engine.stop()
        case .local(let engine): await engine.stopLive()
        }
    }

    var metrics: SpeechBenchmarkQueueMetrics {
        switch self {
        case .local(let engine as MimiPhononMLXLiveEngine): engine.benchmarkQueueMetrics
        case .local(let engine as MimiParakeetJapaneseLiveEngine): engine.benchmarkQueueMetrics
        case .local(let engine as MimiWhisperMLXLiveEngine): engine.lastBenchmarkQueueMetrics
        default: .init()
        }
    }
}

/// File replay ends at production model publication. It does not measure microphone
/// capture, SwiftUI layout, display refresh, or compositor paint.
@MainActor
func runPipelineBenchmark(requestURL: URL, traceURL: URL) async throws {
    let request = try JSONDecoder().decode(BenchmarkCaseRequest.self, from: Data(contentsOf: requestURL))
    guard ["speech", "translation", "pipeline"].contains(request.lane),
          (1...20).contains(request.repetitions), (0...5).contains(request.warmups),
          request.chunkSeconds.isFinite, (0.01...1).contains(request.chunkSeconds),
          request.translationDrainSeconds.isFinite, (0.1...60).contains(request.translationDrainSeconds) else {
        throw PipelineBenchmarkError.invalidRequest("Invalid lane, repetition count, pacing interval, or drain deadline.")
    }
    let needsSpeech = request.lane != "translation"
    let needsTranslation = request.lane != "speech"
    let translationConfiguration = request.translationModelPath.map {
        ExperimentalMLXTranslationConfiguration(modelDirectory: URL(fileURLWithPath: $0))
    }
    guard !needsSpeech || request.audioPath != nil,
          !needsTranslation || translationConfiguration != nil else {
        throw PipelineBenchmarkError.invalidRequest("The requested lane requires audio and/or a translation model path.")
    }
    var speech: BenchmarkSpeechEngine?
    var translationPrepared = false
    for index in 0..<(1 + request.warmups + request.repetitions) {
        let trace = try BenchmarkTrace(url: traceURL, append: index > 0)
        let phase = index == 0 ? "cold" : (index <= request.warmups ? "warmup" : "warm")
        trace.begin(index: index, phase: phase)
        var document = TranscriptDocument()
        var speechError: String?
        var translationError: String?
        var translationTimedOut = false
        var stream: LocalTranslationStream?
        var audioDuration: Double?
        var deliveredAudio = 0.0
        var maximumLatenessMS = 0.0
        var deliveredBuffers = 0
        var result: [String: Any] = [:]
        let scopeID = UUID()

        if needsTranslation, let configuration = translationConfiguration {
            do {
                if !translationPrepared {
                    trace.emit("translation_prepare_started")
                    try await ExperimentalMLXTranslationEngine.shared.prepareForLiveTranslation(configuration: configuration)
                    trace.emit("translation_ready")
                    translationPrepared = true
                }
                let worker = LocalTranslationStream { text, language in
                    let requestID = await trace.activeRequestID
                    let diagnostic = try await ExperimentalMLXTranslationEngine.shared.translateWithDiagnostics(
                        text, sourceLanguage: language, configuration: configuration
                    )
                    await trace.route(requestID: requestID, memory: diagnostic.usedTranslationMemory,
                                      expert: diagnostic.routedToExpert, outputTokens: diagnostic.outputTokenIDs?.count)
                    return diagnostic.output
                }
                worker.onWork = { trace.observe($0) }
                stream = worker
            } catch {
                translationError = error.localizedDescription
                trace.emit("translation_prepare_failed", fields: ["error": error.localizedDescription])
            }
        }

        if needsSpeech, let audioPath = request.audioPath {
            do {
                let file = try AVAudioFile(forReading: URL(fileURLWithPath: audioPath))
                guard file.length > 0, file.processingFormat.sampleRate > 0 else {
                    throw PipelineBenchmarkError.invalidRequest("Audio must contain samples.")
                }
                audioDuration = Double(file.length) / file.processingFormat.sampleRate
                if speech == nil { speech = try BenchmarkSpeechEngine(request) }
                guard let speech else { throw PipelineBenchmarkError.invalidRequest("Missing speech engine.") }
                trace.emit("speech_prepare_started")
                try await speech.start(language: request.language, format: file.processingFormat, event: { event in
                    let boundary = document.segments.last?.id
                    let previousCount = document.segments.count
                    document.apply(event, language: request.language)
                    let text: String
                    let partial: Bool
                    let segmentID: UUID?
                    switch event {
                    case .partial:
                        text = document.liveText
                        partial = true
                        segmentID = nil
                    case .final(let value):
                        text = value.trimmingCharacters(in: .whitespacesAndNewlines)
                        partial = false
                        segmentID = document.segments.count > previousCount ? document.segments.last?.id : nil
                    }
                    trace.source(text, partial: partial, segmentID: segmentID, boundaryID: boundary)
                    stream?.update(.init(segments: document.segments, liveText: document.liveText,
                                         language: request.language, scopeID: scopeID))
                }, warning: { message in
                    trace.emit("speech_warning", fields: ["error": message])
                })
                trace.emit("speech_ready")
                let origin = ContinuousClock.now
                trace.emit("audio_started")
                let framesPerBuffer = AVAudioFrameCount(max(1, (file.processingFormat.sampleRate * request.chunkSeconds).rounded()))
                while file.framePosition < file.length {
                    let count = min(framesPerBuffer, AVAudioFrameCount(file.length - file.framePosition))
                    guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: count) else {
                        throw RealtimeBenchmarkError.couldNotAllocateAudioBuffer
                    }
                    try file.read(into: buffer, frameCount: count)
                    deliveredAudio += Double(buffer.frameLength) / file.processingFormat.sampleRate
                    let deadline = origin.advanced(by: .seconds(deliveredAudio))
                    try await ContinuousClock().sleep(until: deadline)
                    maximumLatenessMS = max(maximumLatenessMS, deadline.duration(to: .now).seconds * 1_000)
                    trace.delivered(deliveredAudio)
                    speech.consume(buffer)
                    deliveredBuffers += 1
                }
                trace.emit("audio_ended")
                await speech.stop()
                // Match TranscriptionSession's Apple stop policy without calling
                // a promoted provisional hypothesis an engine-final event.
                if request.engine == "apple" {
                    let boundary = document.segments.last?.id
                    if let final = document.finalizeLiveText(language: request.language) {
                        trace.source(final.text, partial: false, segmentID: final.id, boundaryID: boundary, origin: "app")
                        stream?.update(.init(segments: document.segments, liveText: "", language: request.language, scopeID: scopeID))
                    }
                }
                trace.emit("speech_stopped")
                result["queue"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(speech.metrics))
            } catch {
                speechError = error.localizedDescription
                trace.emit("speech_failed", fields: ["error": error.localizedDescription])
                await speech?.stop()
            }
        } else if needsTranslation, stream != nil {
            let segment = TranscriptSegment(text: request.source, language: request.language)
            document = TranscriptDocument(segments: [segment])
            trace.emit("text_submitted")
            trace.source(request.source, partial: false, segmentID: segment.id, boundaryID: segment.id, origin: "reference")
            stream?.update(.init(segments: document.segments, liveText: "", language: request.language, scopeID: scopeID))
        }

        if let stream {
            let drainStart = ContinuousClock.now
            // Yield once so the production worker can begin the newest pending job.
            await Task.yield()
            while stream.isTranslating || document.segments.contains(where: {
                stream.translations[$0.id] == nil && !stream.failedSegmentIDs.contains($0.id)
            }) {
                if drainStart.duration(to: .now).seconds >= request.translationDrainSeconds {
                    translationError = "Translation drain deadline exceeded."
                    translationTimedOut = true
                    trace.emit("translation_drain_timeout")
                    break
                }
                try await Task.sleep(for: .milliseconds(5))
            }
            if !stream.failedSegmentIDs.isEmpty { translationError = "One or more finalized segments failed translation." }
        }
        let finalTranscript = document.segments.map(\.text).joined(separator: request.language == .japanese ? "" : " ")
        let finalTranslation = document.segments.compactMap { stream?.translations[$0.id] }.joined(separator: "\n")
        let isSilence = request.source.isEmpty
        let speechStatus = !needsSpeech ? "not_applicable" : (speechError != nil ? "error" :
            (isSilence ? (document.renderedText.isEmpty ? "success" : "hallucination") : (finalTranscript.isEmpty ? "empty_output" : "success")))
        let translationBlocked = needsSpeech && speechError != nil && document.segments.isEmpty
        let translationStatus = !needsTranslation ? "not_applicable" : (translationBlocked ? "not_run" : (translationTimedOut ? "timeout" : (translationError != nil ? "error" :
            (isSilence ? (finalTranslation.isEmpty ? "success" : "hallucination") : (finalTranslation.isEmpty ? "empty_output" : "success")))))
        var usage = rusage()
        if getrusage(RUSAGE_SELF, &usage) == 0 { result["processPeakResidentBytes"] = usage.ru_maxrss }
        result.merge([
            "caseID": request.caseID, "lane": request.lane, "engine": request.engine,
            "language": request.language.rawValue, "speechStatus": speechStatus, "translationStatus": translationStatus,
            "speechError": speechError as Any? ?? NSNull(), "translationError": translationError as Any? ?? NSNull(),
            "transcript": finalTranscript, "lastPartial": document.liveText, "translation": finalTranslation,
            "audioDurationSeconds": audioDuration as Any? ?? NSNull(), "deliveredAudioSeconds": deliveredAudio,
            "deliveredBuffers": deliveredBuffers, "maximumInputLatenessMs": maximumLatenessMS,
            "finalSegmentCount": document.segments.count,
            "translatedFinalSegmentCount": document.segments.filter { stream?.translations[$0.id] != nil }.count
        ]) { _, new in new }
        try trace.finish(result)
        stream?.reset()
        if translationTimedOut { return }
    }
}

func pipelineBenchmarkEnvironment() -> [String: Any] {
#if DEBUG
    let configuration = "debug"
#else
    let configuration = "release"
#endif
#if arch(arm64)
    let architecture = "arm64"
#else
    let architecture = "x86_64"
#endif
    return ["schemaVersion": 1, "buildConfiguration": configuration, "architecture": architecture,
            "hardware": benchmarkHardware(), "physicalMemoryBytes": ProcessInfo.processInfo.physicalMemory,
            "operatingSystem": ProcessInfo.processInfo.operatingSystemVersionString,
            "processorCount": ProcessInfo.processInfo.processorCount,
            "thermalState": ProcessInfo.processInfo.thermalState.rawValue,
            "lowPowerMode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "sourceIdentity": MimiAdaptiveSegmentationBuildIdentity.sha256]
}
