import SwiftUI

struct TranscriptCopyButton: View {
    let text: String
    let title: String
    var copiesEntireTranscript = false
    let preferences: UserPreferences?
    @State private var copied = false
    @State private var copyGeneration = 0

    var body: some View {
        Button {
            copied = TranscriptClipboard.copy(text)
            copyGeneration += 1
        } label: {
            Label(copied ? t("Copied", "コピーしました") : (copiesEntireTranscript ? t("Copy All", "すべてコピー") : t("Copy", "コピー")), systemImage: copied ? "checkmark" : "doc.on.doc")
        }
        .labelStyle(.titleAndIcon)
        .buttonStyle(MimiQuietButtonStyle())
        .help(title)
        .accessibilityLabel(copied ? t("Copied", "コピーしました") : title)
        .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        .task(id: copyGeneration) {
            guard copied else { return }
            do { try await Task.sleep(for: .seconds(2)) } catch { return }
            copied = false
        }
    }

    private func t(_ english: String, _ japanese: String) -> String {
        preferences?.text(english, japanese) ?? english
    }
}
