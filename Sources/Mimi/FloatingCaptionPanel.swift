import AppKit
import MimiCore
import Observation
import SwiftUI
@preconcurrency import Translation

@MainActor
final class FloatingCaptionController: NSObject, NSWindowDelegate {
    private let store: AppStore
    private let preferences: UserPreferences
    private var panel: NSPanel?
    private var isPositioningPanel = false

    init(store: AppStore, preferences: UserPreferences) {
        self.store = store
        self.preferences = preferences
        super.init()
        observePreferences()
        updatePanel()
    }

    private func observePreferences() {
        withObservationTracking {
            _ = preferences.floatingCaptionsEnabled
            _ = preferences.floatingCaptionPosition
            _ = preferences.floatingCaptionClickThrough
            _ = preferences.floatingCaptionUsesCustomPosition
        } onChange: { [weak self] in
            DispatchQueue.main.async {
                self?.updatePanel()
                self?.observePreferences()
            }
        }
    }

    private func updatePanel() {
        guard preferences.floatingCaptionsEnabled else {
            panel?.orderOut(nil)
            return
        }

        let panel = panel ?? makePanel()
        panel.ignoresMouseEvents = preferences.floatingCaptionClickThrough
        position(panel)
        panel.orderFrontRegardless()
    }

    private func makePanel() -> NSPanel {
        let root = FloatingCaptionView(store: store, preferences: preferences)
        let controller = NSHostingController(rootView: root)
        controller.sizingOptions = []
        let panel = NSPanel(
            contentRect: .zero,
            styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        panel.contentViewController = controller
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.hidesOnDeactivate = false
        panel.animationBehavior = .utilityWindow
        panel.isReleasedWhenClosed = false
        panel.isMovableByWindowBackground = true
        panel.delegate = self
        self.panel = panel
        return panel
    }

    private func position(_ panel: NSPanel) {
        guard let screen = NSScreen.main ?? NSScreen.screens.first else { return }
        let visible = screen.visibleFrame
        let margin: CGFloat = 24
        let size: NSSize = switch preferences.floatingCaptionPosition {
        case .subtitles, .top: NSSize(width: min(820, visible.width - 80), height: 150)
        case .topRight, .bottomRight: NSSize(width: min(460, visible.width - 80), height: 190)
        }
        let presetOrigin: NSPoint = switch preferences.floatingCaptionPosition {
        case .subtitles:
            NSPoint(x: visible.midX - size.width / 2, y: visible.minY + margin)
        case .top:
            NSPoint(x: visible.midX - size.width / 2, y: visible.maxY - size.height - margin)
        case .topRight:
            NSPoint(x: visible.maxX - size.width - margin, y: visible.maxY - size.height - margin)
        case .bottomRight:
            NSPoint(x: visible.maxX - size.width - margin, y: visible.minY + margin)
        }
        let origin = preferences.floatingCaptionUsesCustomPosition
            ? clamped(preferences.floatingCaptionCustomOrigin ?? presetOrigin, size: size, in: visible)
            : presetOrigin
        isPositioningPanel = true
        panel.setFrame(NSRect(origin: origin, size: size), display: true, animate: panel.isVisible)
        isPositioningPanel = false
    }

    func windowDidMove(_ notification: Notification) {
        guard !isPositioningPanel,
              let movedPanel = notification.object as? NSPanel,
              movedPanel === panel else { return }
        preferences.rememberFloatingCaptionOrigin(movedPanel.frame.origin)
    }

    private func clamped(_ origin: CGPoint, size: CGSize, in visibleFrame: CGRect) -> CGPoint {
        CGPoint(
            x: min(max(origin.x, visibleFrame.minX), visibleFrame.maxX - size.width),
            y: min(max(origin.y, visibleFrame.minY), visibleFrame.maxY - size.height)
        )
    }
}

struct FloatingCaptionView: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @State private var configuration: TranslationSession.Configuration?
    @State private var translatedText = ""
    @State private var requestedText = ""
    @State private var pendingText = ""
    @State private var isTranslating = false

    private var sourceLanguage: SpeechLanguage {
        if store.isRecording {
            return store.detectedLanguage ?? store.document.contentLanguage(fallback: store.sourceLanguage)
        }
        return store.document.contentLanguage(fallback: store.sourceLanguage)
    }
    private var sourceText: String {
        store.document.realtimeTranslationContext(for: sourceLanguage, maximumCharacterCount: 320)
    }

    var body: some View {
        ZStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 7) {
                if preferences.floatingCaptionContent != .translation {
                    caption(sourceText, secondary: preferences.floatingCaptionContent == .both)
                }
                if preferences.floatingCaptionContent != .original {
                    caption(translatedText, secondary: false)
                }
                if sourceText.isEmpty && translatedText.isEmpty {
                    Text(preferences.text("Captions will appear here", "字幕がここに表示されます"))
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)

            if !preferences.floatingCaptionClickThrough {
                Image(systemName: "line.3.horizontal")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(.quaternary, in: Capsule())
                    .help(preferences.text("Drag captions", "字幕をドラッグ"))
                    .accessibilityLabel(preferences.text("Drag captions", "字幕をドラッグ"))
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 18)
        .background {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(
                    reduceTransparency
                        ? AnyShapeStyle(Color(nsColor: .windowBackgroundColor))
                        : AnyShapeStyle(.ultraThinMaterial.opacity(0.58))
                )
                .overlay {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .strokeBorder(.white.opacity(reduceTransparency ? 0.18 : 0.10))
                }
        }
        .overlay {
            if !preferences.floatingCaptionClickThrough {
                CaptionDragSurface()
                    .accessibilityHidden(true)
            }
        }
        .padding(8)
        .translationTask(configuration) { @MainActor session in
            let text = requestedText
            guard !text.isEmpty else { return }
            do {
                try await session.prepareTranslation()
                let response = try await session.translate(text)
                guard requestedText == text else { return }
                translatedText = response.targetText
            } catch {
                guard requestedText == text else { return }
                translatedText = preferences.text("Translation is not ready yet.", "翻訳の準備がまだできていません。")
            }
            isTranslating = false
        }
        .onChange(of: sourceText, initial: true) { _, text in
            pendingText = text
            if text.isEmpty { translatedText = "" }
        }
        .task(id: sourceLanguage) {
            configuration = nil
            translatedText = ""
            requestedText = ""
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(700))
                guard !Task.isCancelled,
                      preferences.floatingCaptionContent != .original,
                      !pendingText.isEmpty,
                      pendingText != requestedText,
                      !isTranslating else { continue }
                requestedText = pendingText
                isTranslating = true
                if var configuration {
                    configuration.invalidate()
                    self.configuration = configuration
                } else if #available(macOS 26.4, *) {
                    configuration = .init(
                        source: .init(identifier: sourceLanguage.rawValue),
                        target: .init(identifier: sourceLanguage.translationTarget.rawValue),
                        preferredStrategy: .lowLatency
                    )
                } else {
                    configuration = .init(
                        source: .init(identifier: sourceLanguage.rawValue),
                        target: .init(identifier: sourceLanguage.translationTarget.rawValue)
                    )
                }
            }
        }
    }

    private func caption(_ text: String, secondary: Bool) -> some View {
        Text(text)
            .font(secondary ? .title3 : .title2.weight(.semibold))
            .foregroundStyle(secondary ? .secondary : .primary)
            .lineLimit(preferences.floatingCaptionContent == .both ? 2 : 4)
            .multilineTextAlignment(.leading)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentTransition(.opacity)
    }
}

private struct CaptionDragSurface: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        CaptionDragNSView()
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}

private final class CaptionDragNSView: NSView {
    override func mouseDown(with event: NSEvent) {
        window?.performDrag(with: event)
    }

    override func resetCursorRects() {
        addCursorRect(bounds, cursor: .openHand)
    }

    override var acceptsFirstResponder: Bool { false }
}
