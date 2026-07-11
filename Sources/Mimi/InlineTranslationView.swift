import MimiCore
import SwiftUI
@preconcurrency import Translation

/// Translates immutable transcript segments one at a time. This deliberately
/// avoids feeding a growing context block back through Translation whenever a
/// new sentence arrives, which was slow and incorrect for mixed-language Auto
/// sessions.
struct InlineTranslationView: View {
    let segments: [TranscriptSegment]
    let fillsAvailableSpace: Bool
    let fixtureTranslation: String?
    let initiallyFollowingLatest: Bool

    @State private var configuration: TranslationSession.Configuration?
    @State private var translations: [UUID: String] = [:]
    @State private var pendingSegmentID: UUID?
    @State private var errorText: String?
    @State private var isTranslating = false

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
        return segments.compactMap { translations[$0.id] }.joined(separator: "\n")
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("English ↔ 日本語", systemImage: "translate")
                    .font(.callout.weight(.semibold))
                Spacer()
                if isTranslating {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel("Translating newest sentence locally")
                }
                Button("Refresh") {
                    resetAndTranslate()
                }
                .buttonStyle(.borderless)
                .disabled(segments.isEmpty || fixtureTranslation != nil)
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
                                if let translation = translations[segment.id] {
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
            } else if isTranslating {
                ContentUnavailableView {
                    Label("Translating Newest Sentence", systemImage: "translate")
                } description: {
                    Text("New finalized speech is translated locally, one sentence at a time.")
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

            if let errorText {
                HStack(alignment: .firstTextBaseline) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .accessibilityHidden(true)
                    Text(errorText)
                        .font(.caption)
                        .foregroundStyle(.red)
                    Spacer()
                    Button("Try Again") { translateNextSegment() }
                        .buttonStyle(.borderless)
                }
                .padding(10)
                .background(Color.red.opacity(0.08))
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
        .frame(maxHeight: fillsAvailableSpace ? .infinity : nil, alignment: .topLeading)
        .translationTask(configuration) { @MainActor session in
            guard let pendingSegmentID,
                  let segment = segments.first(where: { $0.id == pendingSegmentID }) else { return }
            let segmentID = segment.id
            let text = segment.text
            do {
                try await session.prepareTranslation()
                let response = try await session.translate(text)
                guard self.pendingSegmentID == segmentID,
                      segments.contains(where: { $0.id == segmentID && $0.text == text }) else { return }
                translations[segmentID] = response.targetText
                errorText = nil
                finishCurrentTranslation()
                Task { @MainActor in
                    await Task.yield()
                    translateNextSegment()
                }
            } catch {
                guard self.pendingSegmentID == segmentID else { return }
                isTranslating = false
                errorText = "Translation is unavailable until macOS has the required local English and Japanese languages."
            }
        }
        .onChange(of: segments.map(\.id), initial: true) { _, currentIDs in
            let validIDs = Set(currentIDs)
            translations = translations.filter { validIDs.contains($0.key) }
            if let pendingSegmentID, !validIDs.contains(pendingSegmentID) {
                finishCurrentTranslation()
            }
            translateNextSegment()
        }
    }

    private func resetAndTranslate() {
        configuration = nil
        translations = [:]
        pendingSegmentID = nil
        isTranslating = false
        errorText = nil
        translateNextSegment()
    }

    private func translateNextSegment() {
        guard fixtureTranslation == nil, !isTranslating,
              let segment = TranscriptDocument(segments: segments)
                .nextSegmentNeedingTranslation(completedIDs: Set(translations.keys)) else { return }

        pendingSegmentID = segment.id
        isTranslating = true
        errorText = nil
        configuration = nil
        Task { @MainActor in
            await Task.yield()
            guard pendingSegmentID == segment.id else { return }
            if #available(macOS 26.4, *) {
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
    }

    private func finishCurrentTranslation() {
        configuration = nil
        pendingSegmentID = nil
        isTranslating = false
    }
}
