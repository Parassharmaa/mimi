import MimiCore
import MimiSession
import SwiftUI

/// A compact, explicit entry point to Apple's own ScreenCaptureKit picker.
/// The app never presents a custom list of windows or processes, and the
/// capture implementation registers only an audio output with the stream.
struct ScreenAudioSelectionControl: View {
    @Bindable var store: AppStore
    var compact = false

    init(store: AppStore, compact: Bool = false) {
        self.store = store
        self.compact = compact
    }

    var body: some View {
        if store.source != .microphone {
            VStack(alignment: .leading, spacing: compact ? 6 : 8) {
                HStack(spacing: 8) {
                    Button(selectionButtonTitle) {
                        store.selectScreenAudioContent()
                    }
                    .disabled(store.controlsLocked)

                    if isSelected {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .accessibilityLabel("Audio source selected")
                    }
                }

                if let selection, isSelected {
                    Label(selection.description, systemImage: "speaker.wave.2")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .accessibilityElement(children: .combine)
                } else {
                    Label(selectionRequiredText, systemImage: "exclamationmark.circle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .accessibilityElement(children: .combine)
                }

                if !compact {
                    Text(selectionGuidance)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if let message = store.lastError, !isSelected {
                    Label(message, systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .accessibilityElement(children: .combine)
                }
            }
            .accessibilityElement(children: .contain)
        }
    }

    private var selection: ScreenAudioSelection? {
        store.screenAudioSelection
    }

    private var isSelected: Bool {
        selection?.source == store.source
    }

    private var selectionButtonTitle: String {
        switch store.source {
        case .applicationAudio:
            "Choose App Audio…"
        case .systemAudio:
            "Choose Display Audio…"
        case .microphone:
            ""
        }
    }

    private var selectionRequiredText: String {
        switch store.source {
        case .applicationAudio:
            "Choose an app before recording."
        case .systemAudio:
            "Choose a display before recording."
        case .microphone:
            ""
        }
    }

    private var selectionGuidance: String {
        switch store.source {
        case .applicationAudio:
            "macOS will show its content picker. Choose Zoom, Chrome, or another app; Mimi captures only that app's audio."
        case .systemAudio:
            "macOS will show its display picker. Choose the display carrying your speaker output; Mimi captures only associated audio."
        case .microphone:
            ""
        }
    }
}
