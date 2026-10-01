import Foundation
import MimiCore
import Observation

struct LocalTranslationSnapshot: Equatable {
    let segments: [TranscriptSegment]
    let liveText: String
    let language: SpeechLanguage
    let scopeID: UUID?

    var boundaryID: UUID? { segments.last?.id }
}

/// Updates never cancel active inference. One latest snapshot replaces all pending work.
@MainActor
@Observable
final class LocalTranslationStream {
    typealias Translate = @Sendable (String, SpeechLanguage) async throws -> String

    private(set) var translations: [UUID: String] = [:]
    private(set) var liveTranslation = ""
    private(set) var isTranslating = false
    private(set) var failedSegmentIDs: Set<UUID> = []
    @ObservationIgnored var onAttempt: ((String, String?) -> Void)?
    @ObservationIgnored var onPublication: ((String, String, Bool) -> Void)?
    @ObservationIgnored private var snapshot: LocalTranslationSnapshot?
    @ObservationIgnored private var worker: Task<Void, Never>?
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var lastLiveResult: (source: String, language: SpeechLanguage, boundary: UUID?, output: String)?
    @ObservationIgnored private var attemptedLive: LocalTranslationSnapshot?
    @ObservationIgnored private var lastWorkWasLive = false
    private let translate: Translate

    init(translate: @escaping Translate) {
        self.translate = translate
    }

    convenience init(configuration: ExperimentalMLXTranslationConfiguration) {
        self.init { text, language in
            try Task.checkCancellation()
            return try await ExperimentalMLXTranslationEngine.shared.translate(text, sourceLanguage: language, configuration: configuration)
        }
    }

    func update(_ input: LocalTranslationSnapshot) {
        let normalized = LocalTranslationSnapshot(segments: input.segments, liveText: input.liveText.trimmingCharacters(in: .whitespacesAndNewlines), language: input.language, scopeID: input.scopeID)
        guard snapshot != normalized else { return }
        if let previous = snapshot, previous.scopeID != normalized.scopeID {
            reset()
        }
        // A matching final may reuse only the immediately preceding live result.
        // A corrected final always goes through the full translator again.
        if let result = lastLiveResult,
           let segment = normalized.segments.last,
           segment.id != result.boundary,
           segment.text == result.source, segment.language == result.language,
           snapshot?.boundaryID == result.boundary {
            translations[segment.id] = result.output
            onPublication?(segment.text, result.output, false)
        }
        if snapshot?.boundaryID != normalized.boundaryID || snapshot?.language != normalized.language || normalized.liveText.isEmpty {
            liveTranslation = ""
            lastLiveResult = nil
            attemptedLive = nil
        }
        snapshot = normalized
        let ids = Set(normalized.segments.map(\.id))
        translations = translations.filter { ids.contains($0.key) }
        failedSegmentIDs.formIntersection(ids)
        startIfNeeded()
    }

    func reset() {
        generation &+= 1
        worker?.cancel()
        worker = nil
        snapshot = nil
        translations = [:]
        failedSegmentIDs = []
        liveTranslation = ""
        lastLiveResult = nil
        attemptedLive = nil
        isTranslating = false
        lastWorkWasLive = false
    }

    private func startIfNeeded() {
        guard worker == nil, snapshot != nil else { return }
        let currentGeneration = generation
        worker = Task { [weak self] in
            guard let self else { return }
            await drain(generation: currentGeneration)
        }
    }

    private func drain(generation currentGeneration: Int) async {
        while !Task.isCancelled, generation == currentGeneration, let input = snapshot {
            let segment = input.segments.first { translations[$0.id] == nil && !failedSegmentIDs.contains($0.id) }
            let needsLive = !input.liveText.isEmpty
                && !(lastLiveResult?.source == input.liveText && lastLiveResult?.language == input.language)
                && !(attemptedLive?.liveText == input.liveText && attemptedLive?.language == input.language && attemptedLive?.boundaryID == input.boundaryID)
            // Alternate live work with finals so neither continuous speech nor history can starve.
            let useLive = needsLive && (segment == nil || !lastWorkWasLive)
            guard useLive || segment != nil else { break }
            let text = useLive ? input.liveText : segment!.text
            let language = useLive ? input.language : segment!.language
            lastWorkWasLive = useLive
            if useLive { attemptedLive = input }
            isTranslating = true
            do {
                let output = try await translate(text, language)
                onAttempt?(text, nil)
                guard !Task.isCancelled, generation == currentGeneration else { return }
                if useLive {
                    if snapshot?.boundaryID == input.boundaryID, snapshot?.language == language, snapshot?.liveText.isEmpty == false {
                        lastLiveResult = (text, language, input.boundaryID, output)
                        liveTranslation = output
                        onPublication?(text, output, true)
                    } else if let final = snapshot?.segments.first(where: { $0.text == text && $0.language == language && !input.segments.contains($0) }) {
                        translations[final.id] = output
                        onPublication?(text, output, false)
                    }
                } else if let segment, snapshot?.segments.contains(segment) == true {
                    translations[segment.id] = output
                    onPublication?(text, output, false)
                }
            } catch {
                onAttempt?(text, error.localizedDescription)
                guard !Task.isCancelled, generation == currentGeneration else { return }
                if let segment, !useLive { failedSegmentIDs.insert(segment.id) }
                // A failed partial retains the last good result until its boundary changes.
            }
        }
        guard generation == currentGeneration else { return }
        isTranslating = false
        worker = nil
    }
}
