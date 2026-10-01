import AppKit
import MimiCore
import Observation
import SwiftUI
@preconcurrency import Translation
import os

/// Shows immutable translations plus a replaceable translation of current speech.
/// Apple fallback sessions stay pinned to their direction.
struct InlineTranslationView: View {
    let segments: [TranscriptSegment]
    let liveText: String
    let liveLanguage: SpeechLanguage
    let scopeID: UUID?
    let fillsAvailableSpace: Bool
    let fixtureTranslation: String?
    let initiallyFollowingLatest: Bool
    let preferences: UserPreferences?

    @State private var model = SegmentTranslationModel()
    @State private var retryGeneration = 0
    @State private var localStream: LocalTranslationStream?

    init(
        segments: [TranscriptSegment],
        liveText: String = "",
        liveLanguage: SpeechLanguage = .english,
        scopeID: UUID? = nil,
        fillsAvailableSpace: Bool = false,
        fixtureTranslation: String? = nil,
        initiallyFollowingLatest: Bool = true,
        preferences: UserPreferences? = nil
    ) {
        self.segments = segments
        self.liveText = liveText
        self.liveLanguage = liveLanguage
        self.scopeID = scopeID
        self.fillsAvailableSpace = fillsAvailableSpace
        self.fixtureTranslation = fixtureTranslation
        self.initiallyFollowingLatest = initiallyFollowingLatest
        self.preferences = preferences
    }

    private var renderedTranslation: String {
        if let fixtureTranslation { return fixtureTranslation }
        let completed = segments.compactMap { translations[$0.id] }
        return (completed + (liveOutput.isEmpty ? [] : [liveOutput])).joined(separator: "\n")
    }

    private var translations: [UUID: String] { localStream?.translations ?? model.translations }
    private var liveOutput: String { localStream?.liveTranslation ?? "" }
    private var isTranslating: Bool { localStream?.isTranslating ?? model.isTranslating }
    private var liveInput: LocalTranslationSnapshot {
        .init(segments: segments, liveText: liveText, language: liveLanguage, scopeID: scopeID)
    }

    var body: some View {
        VStack(spacing: 0) {
            MimiPaneHeader("English ↔ 日本語", symbol: "translate") {
                if isTranslating {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel(t("Translating newest sentences locally", "新しい文をローカルで翻訳中"))
                }
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(renderedTranslation, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(MimiQuietButtonStyle())
                .help(t("Copy Translation", "翻訳をコピー"))
                .accessibilityLabel(t("Copy Translation", "翻訳をコピー"))
                .disabled(renderedTranslation.isEmpty)
                Button(t("Refresh", "更新")) {
                    localStream?.reset()
                    localStream?.update(liveInput)
                    model.reset(for: segments)
                    retryGeneration &+= 1
                }
                .buttonStyle(MimiQuietButtonStyle())
                .disabled((segments.isEmpty && liveText.isEmpty) || fixtureTranslation != nil || isTranslating)
            }
            Divider()

            if !renderedTranslation.isEmpty {
                FollowLatestScrollView(
                    contentVersion: renderedTranslation,
                    initiallyFollowing: initiallyFollowingLatest,
                    preferences: preferences
                ) {
                    VStack(alignment: .leading, spacing: 12) {
                        if let fixtureTranslation {
                            Text(fixtureTranslation)
                        } else {
                            ForEach(segments) { segment in
                                if let translation = translations[segment.id] {
                                    Text(translation)
                                }
                            }
                            if !liveOutput.isEmpty {
                                Text(liveOutput)
                                    .foregroundStyle(.secondary)
                                    .accessibilityLabel(t("Current translation: \(liveOutput)", "現在の翻訳：\(liveOutput)"))
                            }
                        }
                    }
                    .transaction { $0.animation = nil }
                    .font(fillsAvailableSpace ? .title3 : .body)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(18)
                }
                .frame(maxHeight: fillsAvailableSpace ? .infinity : 160)
            } else if isTranslating {
                ContentUnavailableView {
                    Label(t("Translating First Sentences", "最初の文を翻訳中"), systemImage: "translate")
                } description: {
                    Text(t("Speech is translated locally as words arrive.", "音声を認識するたびにローカルで翻訳します。"))
                } actions: {
                    ProgressView().controlSize(.small)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ContentUnavailableView(
                    t("No Translation Yet", "翻訳はまだありません"),
                    systemImage: "translate",
                    description: Text(t("Translations appear as you speak and may change until the sentence is complete.", "話している間に翻訳が表示され、文が確定するまで更新されます。"))
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
                    Button(t("Try Again", "再試行")) {
                        model.clearErrors()
                        retryGeneration &+= 1
                    }
                    .buttonStyle(MimiQuietButtonStyle())
                }
                .padding(10)
                .background(Color.red.opacity(0.08))
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
        .frame(maxHeight: fillsAvailableSpace ? .infinity : nil, alignment: .topLeading)
        .background {
            HStack(spacing: 0) {
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
        .onChange(of: liveInput, initial: true) { _, input in
            guard fixtureTranslation == nil, let configuration = model.experimentalConfiguration else { return }
            if localStream == nil { localStream = LocalTranslationStream(configuration: configuration) }
            localStream?.update(input)
        }
        .onAppear { localStream?.update(liveInput) }
        .onDisappear { localStream?.reset() }
        .onChange(of: segments.map(\.id), initial: true) { _, ids in
            model.prune(validIDs: Set(ids))
        }
    }

    private func t(_ english: String, _ japanese: String) -> String {
        preferences?.text(english, japanese) ?? english
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
            "Experimental local translation failed closed for \(language.rawValue, privacy: .public) segment \(segmentID.uuidString, privacy: .public) without Apple fallback: \(error.localizedDescription, privacy: .public)"
        )
        activeLanguage = nil
        failedSegmentIDs.insert(segmentID)
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
    let candidateFailureIsSilent: Bool
    let availabilityFailureShowsRetryableError: Bool
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
    let candidateFailureIsSilent = candidate.errorText == nil
    let unavailable = SegmentTranslationModel(environment: disabledEnvironment)
    unavailable.setError(for: .english)
    let availabilityFailureShowsRetryableError = unavailable.errorText != nil
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
        && candidateFailureIsSilent
        && availabilityFailureShowsRetryableError
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
        candidateFailureIsSilent: candidateFailureIsSilent,
        availabilityFailureShowsRetryableError: availabilityFailureShowsRetryableError,
        candidateFailureIsScopedToSegment: failureIsScoped,
        candidateFailureDoesNotBlockLaterSegment: laterSegmentIsNotBlocked,
        applePartialsWhenExperimentalDisabled: appleDefaultPartials,
        experimentalPartialsDoNotUseApple: experimentalPartialsDoNotUseApple,
        invalidModelPackRejected: invalidPackRejected
    )
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
