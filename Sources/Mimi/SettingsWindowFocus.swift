import AppKit
import SwiftUI

/// Keeps the Settings scene discoverable from Mimi's menu-bar popover.
///
/// `openSettings()` creates or selects the SwiftUI scene, while this tiny
/// coordinator remembers the native window once it exists and explicitly
/// brings it forward. That matters for an accessory/menu-bar app: without the
/// AppKit activation, the settings scene can appear behind the currently
/// active app and look as if the command did nothing.
@MainActor
final class SettingsWindowFocusCoordinator {
    static let shared = SettingsWindowFocusCoordinator()

    private weak var settingsWindow: NSWindow?
    private var focusRequested = false

    private init() {}

    /// Call immediately before SwiftUI's `openSettings()` action.
    func requestFocus() {
        focusRequested = true
        NSApp.activate(ignoringOtherApps: true)
        focusRegisteredWindowIfNeeded()
    }

    /// The Settings scene registers its actual `NSWindow` as soon as it is
    /// attached. This covers first launch, when no window exists yet.
    func register(_ window: NSWindow) {
        settingsWindow = window
        focusRegisteredWindowIfNeeded()
    }

    private func focusRegisteredWindowIfNeeded() {
        guard focusRequested, let settingsWindow else { return }
        focusRequested = false
        NSApp.activate(ignoringOtherApps: true)
        settingsWindow.makeKeyAndOrderFront(nil)
    }
}

/// A zero-visual bridge used only to identify the native Settings window.
struct SettingsWindowRegistrar: NSViewRepresentable {
    func makeNSView(context: Context) -> SettingsWindowRegistrationView {
        SettingsWindowRegistrationView()
    }

    func updateNSView(_ nsView: SettingsWindowRegistrationView, context: Context) {}
}

final class SettingsWindowRegistrationView: NSView {
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if let window {
            SettingsWindowFocusCoordinator.shared.register(window)
        }
    }
}
