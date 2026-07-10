import MimiCore
import SwiftUI
@preconcurrency import Translation

struct InlineTranslationView: View {
    let sourceText: String
    let sourceLanguage: SpeechLanguage

    @State private var configuration: TranslationSession.Configuration?
    @State private var translatedText = ""
    @State private var errorText: String?
    @State private var isTranslating = false

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
                        .accessibilityLabel("Preparing local translation")
                }
                Button(isTranslating ? "Translating…" : "Translate Transcript") {
                    beginTranslation()
                }
                .buttonStyle(.borderless)
                .disabled(isTranslating)
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
                Text("Translate finalized transcript text locally.")
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
                        beginTranslation()
                    }
                    .buttonStyle(.borderless)
                }
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .translationTask(configuration) { @MainActor session in
            do {
                try await session.prepareTranslation()
                let response = try await session.translate(sourceText)
                translatedText = response.targetText
                errorText = nil
                isTranslating = false
            } catch {
                translatedText = ""
                errorText = "Translation is unavailable until macOS has the required local language pair."
                isTranslating = false
            }
        }
        .onChange(of: sourceText) { _, _ in
            configuration = nil
            translatedText = ""
            errorText = nil
            isTranslating = false
        }
        .onChange(of: sourceLanguage) { _, _ in
            configuration = nil
            translatedText = ""
            errorText = nil
            isTranslating = false
        }
    }

    private func beginTranslation() {
        configuration?.invalidate()
        translatedText = ""
        errorText = nil
        isTranslating = true
        configuration = .init(
            source: .init(identifier: sourceLanguage.rawValue),
            target: .init(identifier: targetLanguage.rawValue)
        )
    }
}
