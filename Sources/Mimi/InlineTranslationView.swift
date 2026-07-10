import MimiCore
import SwiftUI
@preconcurrency import Translation

struct InlineTranslationView: View {
    let sourceText: String
    let sourceLanguage: SpeechLanguage

    @State private var configuration: TranslationSession.Configuration?
    @State private var translatedText = ""
    @State private var errorText: String?

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
                Button("Translate") {
                    beginTranslation()
                }
                .buttonStyle(.borderless)
            }

            if translatedText.isEmpty {
                Text("Translate finalized transcript text locally.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text(translatedText)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if let errorText {
                Text(errorText)
                    .font(.caption)
                    .foregroundStyle(.red)
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
            } catch {
                translatedText = ""
                errorText = "Translation is unavailable until macOS has the required local language pair."
            }
        }
        .onChange(of: sourceText) { _, _ in
            configuration = nil
            translatedText = ""
            errorText = nil
        }
        .onChange(of: sourceLanguage) { _, _ in
            configuration = nil
            translatedText = ""
            errorText = nil
        }
    }

    private func beginTranslation() {
        if configuration == nil {
            configuration = .init(
                source: .init(identifier: sourceLanguage.rawValue),
                target: .init(identifier: targetLanguage.rawValue)
            )
        } else {
            configuration?.invalidate()
        }
    }
}
