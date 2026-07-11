import MimiCore
import Observation
import SwiftUI
@preconcurrency import Translation

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
        return segments.compactMap { model.translations[$0.id] }.joined(separator: "\n")
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
                SegmentTranslationLane(
                    segments: segments,
                    sourceLanguage: .english,
                    model: model,
                    retryGeneration: retryGeneration,
                    isEnabled: fixtureTranslation == nil
                )
                SegmentTranslationLane(
                    segments: segments,
                    sourceLanguage: .japanese,
                    model: model,
                    retryGeneration: retryGeneration,
                    isEnabled: fixtureTranslation == nil
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
private final class SegmentTranslationModel {
    private(set) var translations: [UUID: String] = [:]
    private(set) var activeLanguage: SpeechLanguage?
    private(set) var errors: [SpeechLanguage: String] = [:]
    private(set) var workGeneration = 0

    var isTranslating: Bool { activeLanguage != nil }
    var errorText: String? { errors.values.sorted().first }

    func store(_ translation: String, for segmentID: UUID) {
        translations[segmentID] = translation
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
    }

    func prune(validIDs: Set<UUID>) {
        translations = translations.filter { validIDs.contains($0.key) }
    }

    func reset(for segments: [TranscriptSegment]) {
        translations = [:]
        activeLanguage = nil
        errors = [:]
        workGeneration &+= 1
        prune(validIDs: Set(segments.map(\.id)))
    }
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
                preferredStrategy: .lowLatency
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
