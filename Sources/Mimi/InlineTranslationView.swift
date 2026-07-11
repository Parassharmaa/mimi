import MimiCore
import SwiftUI
@preconcurrency import Translation

struct InlineTranslationView: View {
    let sourceText: String
    let sourceLanguage: SpeechLanguage
    let isLive: Bool

    @State private var configuration: TranslationSession.Configuration?
    @State private var translatedText = ""
    @State private var errorText: String?
    @State private var isTranslating = false
    @State private var requestedText = ""
    @State private var pendingText = ""

    init(sourceText: String, sourceLanguage: SpeechLanguage, isLive: Bool = false) {
        self.sourceText = sourceText
        self.sourceLanguage = sourceLanguage
        self.isLive = isLive
    }

    private var targetLanguage: SpeechLanguage {
        sourceLanguage.translationTarget
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Label("\(sourceLanguage.nativeName) → \(targetLanguage.nativeName)", systemImage: "translate")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if isTranslating {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel("Updating local translation")
                }
                Button("Refresh") {
                    beginTranslation(of: sourceText)
                }
                .buttonStyle(.borderless)
                .disabled(sourceText.isEmpty)
            }

            if !translatedText.isEmpty {
                ScrollView {
                    Text(translatedText)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 160)
            } else if isTranslating {
                Text("Preparing local translation…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text(isLive ? "Translation appears as speech becomes stable." : "Translate transcript text locally.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let errorText {
                HStack(alignment: .firstTextBaseline) {
                    Text(errorText)
                        .font(.caption)
                        .foregroundStyle(.red)
                    Spacer()
                    Button("Try Again") {
                        beginTranslation(of: sourceText)
                    }
                    .buttonStyle(.borderless)
                }
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
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
            guard !newText.isEmpty else {
                translatedText = ""
                requestedText = ""
                isTranslating = false
                return
            }
        }
        .task(id: isLive) {
            guard isLive else { return }
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
            guard !isLive, !sourceText.isEmpty else { return }
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
        guard !text.isEmpty else { return }
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
