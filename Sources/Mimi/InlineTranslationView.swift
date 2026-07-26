import MimiCore
import Observation
import SwiftUI
@preconcurrency import Translation
import os

/// Presents finalized translations in transcript order while two persistent,
/// direction-pinned lanes serially drive English→Japanese and Japanese→English.
/// A lane never changes direction, avoiding Translation session stalls during
/// long Auto sessions that alternate languages.
struct InlineTranslationView: View {
    let segments: [TranscriptSegment]
    let fillsAvailableSpace: Bool
    let fixtureTranslation: String?
    let initiallyFollowingLatest: Bool

    @State private var model = SegmentTranslationModel()
    @State private var retryGeneration = 0

    init(
        segments: [TranscriptSegment],
        fillsAvailableSpace: Bool = false,
        fixtureTranslation: String? = nil,
        initiallyFollowingLatest: Bool = true
    ) {
        self.segments = segments
        self.fillsAvailableSpace = fillsAvailableSpace
        self.fixtureTranslation = fixtureTranslation
        self.initiallyFollowingLatest = initiallyFollowingLatest
    }

    private var renderedTranslation: String {
        if let fixtureTranslation { return fixtureTranslation }
        return segments.compactMap { segment in
            if let translation = model.translations[segment.id] {
                return translation
            }
            return model.failedSegmentIDs.contains(segment.id)
                ? "not-translated:\(segment.id.uuidString)"
                : nil
        }.joined(separator: "\n")
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("English ↔ 日本語", systemImage: "translate")
                    .font(.callout.weight(.semibold))
                Spacer()
                if model.isTranslating {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel("Translating newest sentences locally")
                }
                Button("Refresh") {
                    model.reset(for: segments)
                    retryGeneration &+= 1
                }
                .buttonStyle(.borderless)
                .disabled(segments.isEmpty || fixtureTranslation != nil || model.isTranslating)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            if !renderedTranslation.isEmpty {
                FollowLatestScrollView(
                    contentVersion: renderedTranslation,
                    initiallyFollowing: initiallyFollowingLatest
                ) {
                    VStack(alignment: .leading, spacing: 12) {
                        if let fixtureTranslation {
                            Text(fixtureTranslation)
                        } else {
                            ForEach(segments) { segment in
                                if let translation = model.translations[segment.id] {
                                    Text(translation)
                                } else if model.failedSegmentIDs.contains(segment.id) {
                                    Label(
                                        "Not translated safely",
                                        systemImage: "exclamationmark.triangle.fill"
                                    )
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                    .font(fillsAvailableSpace ? .title3 : .body)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(18)
                }
                .frame(maxHeight: fillsAvailableSpace ? .infinity : 160)
            } else if model.isTranslating {
                ContentUnavailableView {
                    Label("Translating First Sentences", systemImage: "translate")
                } description: {
                    Text("Finalized speech is translated locally, one sentence at a time.")
                } actions: {
                    ProgressView().controlSize(.small)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ContentUnavailableView(
                    "No Translation Yet",
                    systemImage: "translate",
                    description: Text("A translation appears after a sentence is finalized.")
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            if let errorText = model.errorText {
                HStack(alignment: .firstTextBaseline) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .accessibilityHidden(true)
                    Text(errorText)
                        .font(.caption)
                        .foregroundStyle(.red)
                    Spacer()
                    Button("Try Again") {
                        model.clearErrors()
                        retryGeneration &+= 1
                    }
                    .buttonStyle(.borderless)
                }
                .padding(10)
                .background(Color.red.opacity(0.08))
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
        .frame(maxHeight: fillsAvailableSpace ? .infinity : nil, alignment: .topLeading)
        .background {
            HStack(spacing: 0) {
                ExperimentalSegmentTranslationLane(
                    segments: segments,
                    model: model,
                    retryGeneration: retryGeneration,
                    isEnabled: fixtureTranslation == nil
                )
                SegmentTranslationLane(
                    segments: segments,
                    sourceLanguage: .english,
                    model: model,
                    retryGeneration: retryGeneration,
                    isEnabled: fixtureTranslation == nil && !model.isUsingExperimentalLocalCandidate
                )
                SegmentTranslationLane(
                    segments: segments,
                    sourceLanguage: .japanese,
                    model: model,
                    retryGeneration: retryGeneration,
                    isEnabled: fixtureTranslation == nil && !model.isUsingExperimentalLocalCandidate
                )
            }
            .frame(width: 0, height: 0)
            .accessibilityHidden(true)
        }
        .onChange(of: segments.map(\.id), initial: true) { _, ids in
            model.prune(validIDs: Set(ids))
        }
    }
}

@MainActor
@Observable
final class SegmentTranslationModel {
    let experimentalConfiguration: ExperimentalMLXTranslationConfiguration?

    private(set) var translations: [UUID: String] = [:]
    private(set) var failedSegmentIDs: Set<UUID> = []
    private(set) var activeLanguage: SpeechLanguage?
    private(set) var errors: [SpeechLanguage: String] = [:]
    private(set) var workGeneration = 0
    private(set) var isUsingExperimentalLocalCandidate = false

    private let logger = Logger(subsystem: "com.paras.mimi", category: "experimental-translation")

    init(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        bundle: Bundle = .main
    ) {
        experimentalConfiguration = ExperimentalMLXTranslationConfiguration.resolved(
            environment: environment,
            bundle: bundle
        )
        isUsingExperimentalLocalCandidate = experimentalConfiguration != nil
    }

    var isTranslating: Bool { activeLanguage != nil }
    var errorText: String? { errors.values.sorted().first }

    func store(_ translation: String, for segmentID: UUID) {
        translations[segmentID] = translation
        failedSegmentIDs.remove(segmentID)
    }

    func shouldAttempt(_ segmentID: UUID) -> Bool {
        translations[segmentID] == nil && !failedSegmentIDs.contains(segmentID)
    }

    func claim(_ language: SpeechLanguage) -> Bool {
        guard activeLanguage == nil else { return false }
        activeLanguage = language
        return true
    }

    func release(_ language: SpeechLanguage) {
        guard activeLanguage == language else { return }
        activeLanguage = nil
        workGeneration &+= 1
    }

    func setError(for language: SpeechLanguage) {
        errors[language] = "Translation is unavailable until macOS has the required local English and Japanese languages."
    }

    func hasError(for language: SpeechLanguage) -> Bool {
        errors[language] != nil
    }

    func clearErrors() {
        errors = [:]
        failedSegmentIDs = []
    }

    func failLocalCandidate(
        after error: Error,
        for language: SpeechLanguage,
        segmentID: UUID
    ) {
        logger.error(
            "Experimental local translation failed closed for segment \(segmentID.uuidString, privacy: .public) without Apple fallback: \(error.localizedDescription, privacy: .public)"
        )
        activeLanguage = nil
        failedSegmentIDs.insert(segmentID)
        errors[language] = "Some sentences could not be translated safely. Their source text is preserved; try them again when ready."
        workGeneration &+= 1
    }

    func prune(validIDs: Set<UUID>) {
        translations = translations.filter { validIDs.contains($0.key) }
        failedSegmentIDs.formIntersection(validIDs)
    }

    func reset(for segments: [TranscriptSegment]) {
        translations = [:]
        failedSegmentIDs = []
        activeLanguage = nil
        errors = [:]
        isUsingExperimentalLocalCandidate = experimentalConfiguration != nil
        workGeneration &+= 1
        prune(validIDs: Set(segments.map(\.id)))
    }
}

private enum TranslationFallbackVerificationFixtureError: Error {
    case candidateFailure
}

struct TranslationFallbackVerificationReport: Codable {
    let schemaVersion: Int
    let status: String
    let appleDefaultWhenExperimentalDisabled: Bool
    let candidateFailureDoesNotUseApple: Bool
    let candidateFailurePreservesLocalResults: Bool
    let candidateFailureShowsRetryableError: Bool
    let candidateFailureIsScopedToSegment: Bool
    let candidateFailureDoesNotBlockLaterSegment: Bool
    let applePartialsWhenExperimentalDisabled: Bool
    let experimentalPartialsDoNotUseApple: Bool
    let invalidModelPackRejected: Bool
}

@MainActor
func verifyExperimentalTranslationFallbackContract() -> TranslationFallbackVerificationReport {
    let disabledEnvironment = [
        ExperimentalMLXTranslationConfiguration.enabledEnvironmentKey: "0",
    ]
    let disabled = SegmentTranslationModel(environment: disabledEnvironment)
    let environment = [
        ExperimentalMLXTranslationConfiguration.enabledEnvironmentKey: "1",
        ExperimentalMLXTranslationConfiguration.modelDirectoryEnvironmentKey: "/invalid/mimi-model-pack",
    ]
    let candidate = SegmentTranslationModel(environment: environment)
    let segmentID = UUID()
    let laterSegmentID = UUID()
    candidate.store("candidate output", for: segmentID)
    _ = candidate.claim(.english)
    candidate.failLocalCandidate(
        after: TranslationFallbackVerificationFixtureError.candidateFailure,
        for: .english,
        segmentID: segmentID
    )
    let appleDefault = !disabled.isUsingExperimentalLocalCandidate
    let failureDoesNotUseApple = candidate.isUsingExperimentalLocalCandidate
    let preservesResults = candidate.translations[segmentID] == "candidate output"
        && candidate.activeLanguage == nil
    let showsRetryableError = candidate.errorText != nil
    let failureIsScoped = candidate.failedSegmentIDs == Set([segmentID])
        && !candidate.shouldAttempt(segmentID)
    let laterSegmentIsNotBlocked = candidate.shouldAttempt(laterSegmentID)
    let appleDefaultPartials = FloatingCaptionView.usesAppleTranslationForLivePartials(
        environment: disabledEnvironment
    )
    let experimentalPartialsDoNotUseApple = !FloatingCaptionView
        .usesAppleTranslationForLivePartials(environment: environment)
    let invalidPackRejected: Bool
    do {
        try ExperimentalMLXTranslationEngine.validateModelPack(
            at: URL(filePath: "/invalid/mimi-model-pack", directoryHint: .isDirectory)
        )
        invalidPackRejected = false
    } catch {
        invalidPackRejected = true
    }
    let passed = appleDefault
        && failureDoesNotUseApple
        && preservesResults
        && showsRetryableError
        && failureIsScoped
        && laterSegmentIsNotBlocked
        && appleDefaultPartials
        && experimentalPartialsDoNotUseApple
        && invalidPackRejected
    return .init(
        schemaVersion: 1,
        status: passed ? "passed" : "failed",
        appleDefaultWhenExperimentalDisabled: appleDefault,
        candidateFailureDoesNotUseApple: failureDoesNotUseApple,
        candidateFailurePreservesLocalResults: preservesResults,
        candidateFailureShowsRetryableError: showsRetryableError,
        candidateFailureIsScopedToSegment: failureIsScoped,
        candidateFailureDoesNotBlockLaterSegment: laterSegmentIsNotBlocked,
        applePartialsWhenExperimentalDisabled: appleDefaultPartials,
        experimentalPartialsDoNotUseApple: experimentalPartialsDoNotUseApple,
        invalidModelPackRejected: invalidPackRejected
    )
}

private struct ExperimentalSegmentTranslationLane: View {
    let segments: [TranscriptSegment]
    let model: SegmentTranslationModel
    let retryGeneration: Int
    let isEnabled: Bool

    private var input: ExperimentalSegmentTranslationLaneInput {
        .init(
            segmentIDs: segments.map(\.id),
            retryGeneration: retryGeneration,
            isEnabled: isEnabled && model.isUsingExperimentalLocalCandidate
        )
    }

    var body: some View {
        Color.clear
            .task(id: input) {
                guard input.isEnabled, let configuration = model.experimentalConfiguration else { return }
                for segment in segments where model.shouldAttempt(segment.id) {
                    guard !Task.isCancelled, model.claim(segment.language) else { return }
                    do {
                        let translated = try await ExperimentalMLXTranslationEngine.shared.translate(
                            segment.text,
                            sourceLanguage: segment.language,
                            configuration: configuration
                        )
                        guard !Task.isCancelled else {
                            model.release(segment.language)
                            return
                        }
                        model.store(translated, for: segment.id)
                        model.release(segment.language)
                    } catch is CancellationError {
                        model.release(segment.language)
                        return
                    } catch {
                        model.failLocalCandidate(
                            after: error,
                            for: segment.language,
                            segmentID: segment.id
                        )
                    }
                }
            }
    }
}

private struct ExperimentalSegmentTranslationLaneInput: Equatable {
    let segmentIDs: [UUID]
    let retryGeneration: Int
    let isEnabled: Bool
}

private struct SegmentTranslationLane: View {
    let segments: [TranscriptSegment]
    let sourceLanguage: SpeechLanguage
    let model: SegmentTranslationModel
    let retryGeneration: Int
    let isEnabled: Bool

    @State private var configuration: TranslationSession.Configuration?
    @State private var queue = SegmentTranslationQueue()
    @State private var isRunning = false

    private var laneSegments: [TranscriptSegment] {
        segments.filter { $0.language == sourceLanguage }
    }

    private var input: SegmentTranslationLaneInput {
        SegmentTranslationLaneInput(
            segmentIDs: laneSegments.map(\.id),
            retryGeneration: retryGeneration,
            workGeneration: model.workGeneration,
            isEnabled: isEnabled
        )
    }

    var body: some View {
        Color.clear
            .translationTask(configuration) { @MainActor session in
                guard let activeID = queue.activeSegmentID,
                      let segment = laneSegments.first(where: { $0.id == activeID }) else {
                    releaseAfterCurrentTask()
                    return
                }
                do {
                    try await session.prepareTranslation()
                    let response = try await session.translate(segment.text)
                    guard queue.activeSegmentID == segment.id else { return }
                    model.store(response.targetText, for: segment.id)
                    _ = queue.finish(segment.id)
                    releaseAfterCurrentTask()
                } catch {
                    guard queue.activeSegmentID == segment.id else { return }
                    _ = queue.finish(segment.id)
                    releaseAfterCurrentTask()
                    if !(error is CancellationError) {
                        model.setError(for: sourceLanguage)
                    }
                }
            }
            .onChange(of: input, initial: true) { _, input in
                let validIDs = Set(input.segmentIDs)
                if let activeID = queue.activeSegmentID, !validIDs.contains(activeID) {
                    queue.reset()
                    configuration = nil
                    finishRunningState()
                }
                startNextIfNeeded()
            }
    }

    private func startNextIfNeeded() {
        guard isEnabled, !isRunning, model.activeLanguage == nil,
              !model.hasError(for: sourceLanguage),
              let globallyNext = segments.first(where: { model.translations[$0.id] == nil }),
              globallyNext.language == sourceLanguage,
              let segment = queue.beginNext(
                in: laneSegments,
                completedIDs: Set(model.translations.keys)
              ), model.claim(sourceLanguage) else { return }

        isRunning = true
        if var configuration {
            configuration.invalidate()
            self.configuration = configuration
        } else if #available(macOS 26.4, *) {
            configuration = .init(
                source: .init(identifier: segment.language.rawValue),
                target: .init(identifier: segment.language.translationTarget.rawValue),
                preferredStrategy: .highFidelity
            )
        } else {
            configuration = .init(
                source: .init(identifier: segment.language.rawValue),
                target: .init(identifier: segment.language.translationTarget.rawValue)
            )
        }
    }

    private func releaseAfterCurrentTask() {
        isRunning = false
        Task { @MainActor in
            // Let the current task unwind, but keep this direction's stable
            // configuration alive. The next sentence restarts it with
            // invalidate(); nil→new configuration cycles can be missed.
            try? await Task.sleep(for: .milliseconds(12))
            model.release(sourceLanguage)
        }
    }

    private func finishRunningState() {
        isRunning = false
        model.release(sourceLanguage)
    }
}

private struct SegmentTranslationLaneInput: Equatable {
    let segmentIDs: [UUID]
    let retryGeneration: Int
    let workGeneration: Int
    let isEnabled: Bool
}
