import AppKit
import MimiCore
import SwiftUI

/// Shows finalized and still-changing ASR text as one quiet transcript flow.
/// The current hypothesis remains subtly secondary without adding a status
/// label that competes with the words themselves.
struct TranscriptContentView: View {
    let document: TranscriptDocument
    let emptyMessage: String
    let font: NSFont
    let preferences: UserPreferences?
    var onSelectionChange: (Bool) -> Void = { _ in }

    init(document: TranscriptDocument, emptyMessage: String, font: NSFont = .preferredFont(forTextStyle: .body), preferences: UserPreferences? = nil, onSelectionChange: @escaping (Bool) -> Void = { _ in }) {
        self.document = document
        self.emptyMessage = emptyMessage
        self.font = font
        self.preferences = preferences
        self.onSelectionChange = onSelectionChange
    }

    var body: some View {
        if document.segments.isEmpty && document.liveText.isEmpty {
            Text(emptyMessage)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            SelectableTranscriptText(
                text: document.segments.map(\.text).joined(separator: "\n"),
                provisionalText: document.liveText,
                font: font,
                accessibilityLabel: preferences?.text("Original transcript", "原文") ?? "Original transcript",
                onSelectionChange: onSelectionChange
            )
        }
    }
}
