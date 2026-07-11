import MimiCore
import SwiftUI

/// Shows finalized and still-changing ASR text as one quiet transcript flow.
/// The current hypothesis remains subtly secondary without adding a status
/// label that competes with the words themselves.
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
                        .lineSpacing(3)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                if !document.liveText.isEmpty {
                    Text(document.liveText)
                        .font(font)
                        .lineSpacing(3)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .accessibilityLabel("Current transcription: \(document.liveText)")
                }
            }
        }
    }
}
