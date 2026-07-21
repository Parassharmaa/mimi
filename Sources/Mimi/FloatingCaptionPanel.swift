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
    static func usesAppleTranslationForLivePartials(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        bundle: Bundle = .main
    ) -> Bool {
        ExperimentalMLXTranslationConfiguration.resolved(
            environment: environment,
            bundle: bundle
        ) == nil
    }

    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @State private var configuration: TranslationSession.Configuration?
    @State private var pipeline = LiveTranslationPipeline()
    @State private var configuredLanguage: SpeechLanguage?
    @State private var latestTranslationInput: CaptionTranslationInput?
    @State private var retryAfter: Date?
    @State private var localTranslation = ""

    private var sourceLanguage: SpeechLanguage {
        if store.isRecording {
            return store.detectedLanguage ?? store.document.contentLanguage(fallback: store.sourceLanguage)
        }
        return store.document.contentLanguage(fallback: store.sourceLanguage)
    }
    private var sourceText: String {
        store.document.latestCaptionText
    }
    private var usesAppleTranslationForLivePartials: Bool {
        Self.usesAppleTranslationForLivePartials()
    }
    private var localTranslationConfiguration: ExperimentalMLXTranslationConfiguration? {
        ExperimentalMLXTranslationConfiguration.resolved()
    }
    private var localTranslationInput: CaptionTranslationInput {
        CaptionTranslationInput(
            text: sourceText,
            language: sourceLanguage,
            isEnabled: preferences.floatingCaptionContent != .original
                && localTranslationConfiguration != nil
        )
    }
    private var displayedTranslation: String {
        localTranslation.isEmpty ? pipeline.displayedTranslation : localTranslation
    }
    private var translationInput: CaptionTranslationInput {
        CaptionTranslationInput(
            text: sourceText,
            language: sourceLanguage,
            isEnabled: preferences.floatingCaptionContent != .original
                && usesAppleTranslationForLivePartials
        )
    }

    var body: some View {
        ZStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 7) {
                if preferences.floatingCaptionContent != .translation {
                    caption(sourceText, secondary: preferences.floatingCaptionContent == .both)
                }
                if preferences.floatingCaptionContent != .original {
                    if !displayedTranslation.isEmpty {
                        caption(displayedTranslation, secondary: false)
                    } else if !sourceText.isEmpty && usesAppleTranslationForLivePartials {
                        Text("…")
                            .font(.title2.weight(.semibold))
                            .foregroundStyle(.secondary)
                    } else if !sourceText.isEmpty,
                              preferences.floatingCaptionContent == .translation {
                        caption(sourceText, secondary: true)
                    }
                }
                if sourceText.isEmpty && displayedTranslation.isEmpty {
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
            guard let request = pipeline.activeRequest else { return }
            do {
                try await session.prepareTranslation()
                let response = try await session.translate(request.text)
                guard pipeline.activeRequest?.id == request.id else { return }
                let next = pipeline.complete(requestID: request.id, translation: response.targetText)
                retryAfter = nil
                startAfterCurrentTask(next)
            } catch {
                guard pipeline.activeRequest?.id == request.id else { return }
                let next = pipeline.fail(requestID: request.id)
                retryAfter = Date().addingTimeInterval(1.5)
                startAfterCurrentTask(next)
            }
        }
        .onChange(of: translationInput, initial: true) { _, input in
            latestTranslationInput = input
            guard input.text.isEmpty else { return }
            pipeline.clear()
            configuration = nil
            configuredLanguage = nil
            retryAfter = nil
        }
        .task(id: localTranslationInput) {
            let input = localTranslationInput
            guard input.isEnabled,
                  !input.text.isEmpty,
                  let localTranslationConfiguration else {
                if input.text.isEmpty {
                    localTranslation = ""
                }
                return
            }
            // Coalesce rapidly changing speech partials, then use the same
            // integrity-checked local engine as finalized transcript rows.
            try? await Task.sleep(for: .milliseconds(140))
            guard !Task.isCancelled else { return }
            do {
                let output = try await ExperimentalMLXTranslationEngine.shared.translate(
                    input.text,
                    sourceLanguage: input.language,
                    configuration: localTranslationConfiguration
                )
                guard !Task.isCancelled,
                      sourceText == input.text,
                      sourceLanguage == input.language else { return }
                localTranslation = output
            } catch is CancellationError {
                return
            } catch {
                guard sourceText == input.text else { return }
                localTranslation = ""
            }
        }
        .task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(180))
                guard !Task.isCancelled,
                      let input = latestTranslationInput,
                      input.isEnabled,
                      !input.text.isEmpty,
                      retryAfter.map({ Date() >= $0 }) ?? true else { continue }
                // Sample at a steady cadence rather than debouncing: continuous
                // speech must keep producing translations even if partials
                // change more frequently than the sampling interval.
                if let request = pipeline.enqueue(text: input.text, language: input.language) {
                    startTranslation(request)
                }
            }
        }
    }

    private func startAfterCurrentTask(_ request: LiveTranslationPipeline.Request?) {
        guard let request else { return }
        Task { @MainActor in
            await Task.yield()
            guard pipeline.activeRequest?.id == request.id else { return }
            startTranslation(request)
        }
    }

    private func startTranslation(_ request: LiveTranslationPipeline.Request) {
        if configuredLanguage == request.language, var configuration {
            configuration.invalidate()
            self.configuration = configuration
            return
        }

        configuredLanguage = request.language
        if #available(macOS 26.4, *) {
            configuration = .init(
                source: .init(identifier: request.language.rawValue),
                target: .init(identifier: request.language.translationTarget.rawValue),
                preferredStrategy: .lowLatency
            )
        } else {
            configuration = .init(
                source: .init(identifier: request.language.rawValue),
                target: .init(identifier: request.language.translationTarget.rawValue)
            )
        }
    }

    private func caption(_ text: String, secondary: Bool) -> some View {
        Text(text)
            .font(secondary ? .title3 : .title2.weight(.semibold))
            .foregroundStyle(secondary ? .secondary : .primary)
            .lineLimit(preferences.floatingCaptionContent == .both ? 2 : 4)
            .multilineTextAlignment(.leading)
            .frame(maxWidth: .infinity, alignment: .leading)
            // Captions are a hot streaming path. Cross-fading every partial
            // reads as blinking, so text updates are deliberately immediate.
            .transaction { $0.animation = nil }
    }
}

private struct CaptionTranslationInput: Equatable {
    let text: String
    let language: SpeechLanguage
    let isEnabled: Bool
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
