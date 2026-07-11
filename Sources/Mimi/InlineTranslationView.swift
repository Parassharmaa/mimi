import MimiCore
import SwiftUI
@preconcurrency import Translation

struct InlineTranslationView: View {
    let sourceText: String
    let sourceLanguage: SpeechLanguage
    let isLive: Bool
    let fillsAvailableSpace: Bool
    let fixtureTranslation: String?
    let initiallyFollowingLatest: Bool

    @State private var configuration: TranslationSession.Configuration?
    @State private var translatedText = ""
    @State private var errorText: String?
    @State private var isTranslating = false
    @State private var requestedText = ""
    @State private var pendingText = ""

    init(
        sourceText: String,
        sourceLanguage: SpeechLanguage,
        isLive: Bool = false,
        fillsAvailableSpace: Bool = false,
        fixtureTranslation: String? = nil,
        initiallyFollowingLatest: Bool = true
    ) {
        self.sourceText = sourceText
        self.sourceLanguage = sourceLanguage
        self.isLive = isLive
        self.fillsAvailableSpace = fillsAvailableSpace
        self.fixtureTranslation = fixtureTranslation
        self.initiallyFollowingLatest = initiallyFollowingLatest
        _translatedText = State(initialValue: fixtureTranslation ?? "")
    }

    private var targetLanguage: SpeechLanguage {
        sourceLanguage.translationTarget
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label(targetLanguage.nativeName, systemImage: "translate")
                    .font(.callout.weight(.semibold))
                Spacer()
                if isTranslating {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel("Updating local translation")
                }
                if !isLive {
                    Button("Refresh") {
                        beginTranslation(of: sourceText)
                    }
                    .buttonStyle(.borderless)
                    .disabled(sourceText.isEmpty || fixtureTranslation != nil)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            if !translatedText.isEmpty {
                FollowLatestScrollView(
                    contentVersion: translatedText,
                    initiallyFollowing: initiallyFollowingLatest
                ) {
                    Text(translatedText)
                        .font(fillsAvailableSpace ? .title3 : .body)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(18)
                }
                .frame(maxHeight: fillsAvailableSpace ? .infinity : 160)
            } else if isTranslating {
                ContentUnavailableView {
                    Label("Preparing Translation", systemImage: "translate")
                } description: {
                    Text("Mimi is translating the newest stable speech locally.")
                } actions: {
                    ProgressView()
                        .controlSize(.small)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ContentUnavailableView(
                    "No Translation Yet",
                    systemImage: "translate",
                    description: Text(isLive ? "Translation appears as speech becomes stable." : "Start with transcript text to translate it locally.")
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
                    Button("Try Again") {
                        beginTranslation(of: sourceText)
                    }
                    .buttonStyle(.borderless)
                }
                .padding(10)
                .background(Color.red.opacity(0.08))
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
        .frame(maxHeight: fillsAvailableSpace ? .infinity : nil, alignment: .topLeading)
        .translationTask(configuration) { @MainActor session in
            let text = requestedText
            guard !text.isEmpty else { return }
            do {
                try await session.prepareTranslation()
                let response = try await session.translate(text)
                guard requestedText == text else { return }
                translatedText = response.targetText
                errorText = nil
                isTranslating = false
            } catch {
                guard requestedText == text else { return }
                errorText = "Translation is unavailable until macOS has the required local language pair."
                isTranslating = false
            }
        }
        .onChange(of: sourceText, initial: true) { _, newText in
            pendingText = newText
            guard fixtureTranslation == nil else { return }
            guard !newText.isEmpty else {
                translatedText = ""
                requestedText = ""
                isTranslating = false
                return
            }
        }
        .task(id: isLive) {
            guard isLive, fixtureTranslation == nil else { return }
            while !Task.isCancelled {
                // Translate the newest snapshot at a steady cadence even when
                // ASR partials keep changing continuously. Never queue more
                // than one Translation framework request at a time.
                try? await Task.sleep(for: .milliseconds(700))
                guard !Task.isCancelled else { return }
                let text = pendingText
                if !text.isEmpty, text != requestedText, !isTranslating {
                    beginTranslation(of: text)
                }
            }
        }
        .task(id: sourceText) {
            guard fixtureTranslation == nil, !isLive, !sourceText.isEmpty else { return }
            do {
                try await Task.sleep(for: .milliseconds(150))
                try Task.checkCancellation()
                beginTranslation(of: sourceText)
            } catch {
                // A newer finalized snapshot superseded this request.
            }
        }
        .onChange(of: sourceLanguage) { _, _ in
            configuration = nil
            translatedText = ""
            requestedText = ""
            pendingText = ""
            errorText = nil
            isTranslating = false
        }
    }

    private func beginTranslation(of text: String) {
        guard fixtureTranslation == nil, !text.isEmpty else { return }
        requestedText = text
        errorText = nil
        isTranslating = true
        if var configuration {
            configuration.invalidate()
            self.configuration = configuration
        } else if #available(macOS 26.4, *) {
            configuration = .init(
                source: .init(identifier: sourceLanguage.rawValue),
                target: .init(identifier: targetLanguage.rawValue),
                preferredStrategy: .lowLatency
            )
        } else {
            configuration = .init(
                source: .init(identifier: sourceLanguage.rawValue),
                target: .init(identifier: targetLanguage.rawValue)
            )
        }
    }
}
