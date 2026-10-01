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
    static let controlRadius: CGFloat = 8
    static let pagePadding: CGFloat = 24
    static let popoverWidth: CGFloat = 370
    static let hoverDuration: Double = 0.12
}

/// Shared interaction treatment for flat controls. Native prominent buttons,
/// menus, pickers and list selection keep their platform behavior.
struct MimiQuietButtonStyle: ButtonStyle {
    var horizontalPadding: CGFloat = 8
    var verticalPadding: CGFloat = 6

    func makeBody(configuration: Configuration) -> some View {
        QuietButton(configuration: configuration, horizontalPadding: horizontalPadding, verticalPadding: verticalPadding)
    }

    private struct QuietButton: View {
        let configuration: ButtonStyleConfiguration
        let horizontalPadding: CGFloat
        let verticalPadding: CGFloat
        @Environment(\.isEnabled) private var isEnabled
        @Environment(\.accessibilityReduceMotion) private var reduceMotion
        @Environment(\.colorSchemeContrast) private var contrast
        @Environment(\.mimiAccessibilityPreview) private var preview
        @State private var isHovered = false

        private var reducesMotion: Bool { reduceMotion || preview.contains(.reduceMotion) }
        private var opacity: Double {
            guard isEnabled else { return 0 }
            if configuration.isPressed { return 0.12 }
            if isHovered { return contrast == .increased || preview.contains(.increaseContrast) ? 0.12 : 0.06 }
            return 0
        }

        var body: some View {
            configuration.label
                .foregroundStyle(configuration.role == .destructive ? Color.red : Color.primary)
                .padding(.horizontal, horizontalPadding)
                .padding(.vertical, verticalPadding)
                .background(Color.primary.opacity(opacity), in: .rect(cornerRadius: MimiMetrics.controlRadius))
                .contentShape(.rect(cornerRadius: MimiMetrics.controlRadius))
                .opacity(isEnabled ? 1 : 0.4)
                .scaleEffect(isEnabled && isHovered && configuration.isPressed && !reducesMotion ? 0.98 : 1)
                .animation(reducesMotion ? nil : .easeOut(duration: MimiMetrics.hoverDuration), value: isHovered)
                .onHover { isHovered = $0 }
        }
    }
}

struct MimiChoiceRow: View {
    let title: String
    let detail: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.callout.weight(.semibold))
                    Text(detail).font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            .padding(MimiMetrics.cardPadding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(isSelected ? Color.accentColor.opacity(0.09) : Color.primary.opacity(0.025), in: .rect(cornerRadius: MimiMetrics.cardRadius))
        }
        .buttonStyle(MimiQuietButtonStyle(horizontalPadding: 0, verticalPadding: 0))
        .accessibilityLabel("\(title), \(detail)")
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

struct MimiPaneHeader<Actions: View>: View {
    let title: String
    let symbol: String
    private let actions: Actions

    init(_ title: String, symbol: String, @ViewBuilder actions: () -> Actions) {
        self.title = title
        self.symbol = symbol
        self.actions = actions()
    }

    var body: some View {
        HStack(spacing: MimiMetrics.compactSpacing) {
            Label(title, systemImage: symbol).font(.callout.weight(.semibold))
            Spacer()
            actions
        }
        .frame(minHeight: 28)
        .padding(.horizontal, MimiMetrics.sectionSpacing)
        .padding(.vertical, 10)
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
