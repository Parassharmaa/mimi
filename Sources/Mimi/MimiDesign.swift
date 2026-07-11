import MimiCore
import SwiftUI

enum MimiMetrics {
    static let compactSpacing: CGFloat = 8
    static let sectionSpacing: CGFloat = 16
    static let cardRadius: CGFloat = 12
    static let cardPadding: CGFloat = 12
}

struct MimiSectionLabel: View {
    let title: String
    let symbol: String?

    init(_ title: String, symbol: String? = nil) {
        self.title = title
        self.symbol = symbol
    }

    var body: some View {
        if let symbol {
            Label(title, systemImage: symbol)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
        } else {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
        }
    }
}

struct MimiStatusHeader: View {
    let state: RecordingState
    let source: AudioSource

    var body: some View {
        HStack(spacing: 11) {
            Image(systemName: symbolName)
                .font(.title2)
                .foregroundStyle(tint)
                .frame(width: 28, height: 28)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 2) {
                Text("Mimi")
                    .font(.headline)
                Text(statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Spacer(minLength: 8)

            Text(badgeText)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(tint)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(tint.opacity(0.12), in: Capsule())
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Mimi, \(statusText)")
    }

    private var statusText: String {
        switch state {
        case .recording:
            "Listening to \(source.displayName.lowercased()) on this Mac"
        case .idle:
            "Ready for local transcription"
        case .preparing, .processing, .failed:
            state.label
        }
    }

    private var badgeText: String {
        switch state {
        case .idle: "Ready"
        case .preparing: "Preparing"
        case .recording: "Recording"
        case .processing: "Finalizing"
        case .failed: "Attention"
        }
    }

    private var symbolName: String {
        switch state {
        case .idle: "ear"
        case .preparing, .processing: "waveform.badge.magnifyingglass"
        case .recording: "waveform.circle.fill"
        case .failed: "exclamationmark.triangle.fill"
        }
    }

    private var tint: Color {
        switch state {
        case .recording: .red
        case .failed: .orange
        case .idle, .preparing, .processing: .accentColor
        }
    }
}

struct MimiControlRow<Control: View>: View {
    let title: String
    let detail: String?
    let symbol: String
    private let control: Control

    init(
        _ title: String,
        detail: String? = nil,
        symbol: String,
        @ViewBuilder control: () -> Control
    ) {
        self.title = title
        self.detail = detail
        self.symbol = symbol
        self.control = control()
    }

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: symbol)
                .foregroundStyle(.secondary)
                .frame(width: 18)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.callout)
                if let detail {
                    Text(detail)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }

            Spacer(minLength: 8)
            control
                .labelsHidden()
        }
        .padding(.vertical, 6)
    }
}

private struct MimiCardModifier: ViewModifier {
    @Environment(\.colorSchemeContrast) private var contrast

    let padding: CGFloat

    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: MimiMetrics.cardRadius, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: MimiMetrics.cardRadius, style: .continuous)
                    .strokeBorder(
                        contrast == .increased ? Color.primary.opacity(0.32) : Color.primary.opacity(0.08),
                        lineWidth: 1
                    )
            }
    }
}

extension View {
    func mimiCard(padding: CGFloat = MimiMetrics.cardPadding) -> some View {
        modifier(MimiCardModifier(padding: padding))
    }
}

extension AudioSource {
    var symbolName: String {
        switch self {
        case .microphone: "mic"
        case .outputAudio: "speaker.wave.2"
        case .applicationAudio: "macwindow"
        case .systemAudio: "display"
        }
    }
}

extension SpeechLanguage {
    var symbolName: String {
        switch self {
        case .english: "character.book.closed"
        case .japanese: "character"
        }
    }
}
