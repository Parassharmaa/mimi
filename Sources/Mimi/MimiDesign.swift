import MimiCore
import SwiftUI

struct MimiAccessibilityPreview: OptionSet, Sendable {
    let rawValue: Int
    static let reduceMotion = Self(rawValue: 1)
    static let reduceTransparency = Self(rawValue: 2)
    static let increaseContrast = Self(rawValue: 4)
}

private struct MimiAccessibilityPreviewKey: EnvironmentKey {
    static let defaultValue: MimiAccessibilityPreview = []
}

extension EnvironmentValues {
    var mimiAccessibilityPreview: MimiAccessibilityPreview {
        get { self[MimiAccessibilityPreviewKey.self] }
        set { self[MimiAccessibilityPreviewKey.self] = newValue }
    }
}

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
    let preferences: UserPreferences

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
            preferences.text("Listening to \(source.displayName.lowercased()) on this Mac", "このMacで音声を文字起こし中")
        case .idle:
            preferences.text("Ready for local transcription", "ローカル文字起こしの準備完了")
        case .preparing: preferences.text("Preparing", "準備中")
        case .processing: preferences.text("Finalizing", "確定処理中")
        case .failed:
            state.label
        }
    }

    private var badgeText: String {
        switch state {
        case .idle: preferences.text("Ready", "準備完了")
        case .preparing: preferences.text("Preparing", "準備中")
        case .recording: preferences.text("Recording", "録音中")
        case .processing: preferences.text("Finalizing", "確定処理中")
        case .failed: preferences.text("Attention", "確認が必要")
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
    @Environment(\.mimiAccessibilityPreview) private var preview

    let padding: CGFloat

    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: MimiMetrics.cardRadius, style: .continuous))
            .overlay {
                if contrast == .increased || preview.contains(.increaseContrast) {
                    RoundedRectangle(cornerRadius: MimiMetrics.cardRadius, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.32), lineWidth: 1)
                }
            }
    }
}

extension View {
    func mimiCard(padding: CGFloat = MimiMetrics.cardPadding) -> some View {
        modifier(MimiCardModifier(padding: padding))
    }

    /// Chrome only. Transcript text retains a stable opaque content surface.
    func mimiChrome(padding: CGFloat = MimiMetrics.cardPadding, radius: CGFloat = 16) -> some View {
        modifier(MimiChromeModifier(padding: padding, radius: radius))
    }
}

private struct MimiChromeModifier: ViewModifier {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.colorSchemeContrast) private var contrast
    @Environment(\.mimiAccessibilityPreview) private var preview
    let padding: CGFloat
    let radius: CGFloat

    @ViewBuilder func body(content: Content) -> some View {
        if reduceTransparency || contrast == .increased || preview.contains(.reduceTransparency) || preview.contains(.increaseContrast) {
            content.padding(padding)
                .background(Color(nsColor: .windowBackgroundColor), in: .rect(cornerRadius: radius))
                .overlay { RoundedRectangle(cornerRadius: radius).strokeBorder(.primary.opacity(0.35)) }
        } else if #available(macOS 26, *) {
            content.padding(padding).glassEffect(.regular, in: .rect(cornerRadius: radius))
        } else {
            content.padding(padding).background(.regularMaterial, in: .rect(cornerRadius: radius))
        }
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
