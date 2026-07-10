import MimiCore
import SwiftUI

/// Keeps final and still-changing ASR text visually distinct. A live
/// hypothesis is intentionally secondary so people do not mistake it for a
/// persisted final segment.
struct TranscriptContentView: View {
    let document: TranscriptDocument
    let emptyMessage: String
    let font: Font

    init(document: TranscriptDocument, emptyMessage: String, font: Font = .body) {
        self.document = document
        self.emptyMessage = emptyMessage
        self.font = font
    }

    var body: some View {
        if document.segments.isEmpty && document.liveText.isEmpty {
            Text(emptyMessage)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            LazyVStack(alignment: .leading, spacing: 10) {
                ForEach(document.segments) { segment in
                    Text(segment.text)
                        .font(font)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                if !document.liveText.isEmpty {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Live — not finalized")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Text(document.liveText)
                            .font(font)
                            .italic()
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Live transcription, not finalized: \(document.liveText)")
                }
            }
        }
    }
}
