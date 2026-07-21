import AppKit
import SwiftUI

/// Owns Mimi's single primary transcript window.
///
/// Mimi has a persistent menu-bar scene, so SwiftUI doesn't create its other
/// scenes at launch. A small AppKit coordinator gives launch and Dock reopen
/// events the normal Mac behavior while the window content remains SwiftUI.
@MainActor
final class AppWindowCoordinator {
    static let shared = AppWindowCoordinator()

    private weak var store: AppStore?
    private weak var preferences: UserPreferences?
    private var windowController: NSWindowController?

    private init() {}

    func configure(store: AppStore, preferences: UserPreferences) {
        self.store = store
        self.preferences = preferences
    }

    func showTranscript() {
        guard let store, let preferences else { return }
        let window = windowController?.window ?? makeWindow(store: store, preferences: preferences)
        if window.isMiniaturized { window.deminiaturize(nil) }
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
    }

    var visibleTranscriptWindows: [NSWindow] {
        NSApp.windows.filter { $0.identifier?.rawValue == "MimiTranscript" && $0.isVisible }
    }

    private func makeWindow(store: AppStore, preferences: UserPreferences) -> NSWindow {
        let hostingController = NSHostingController(
            rootView: TranscriptWindow(store: store, preferences: preferences)
        )
        hostingController.sizingOptions = []
        let window = NSWindow(contentViewController: hostingController)
        window.identifier = NSUserInterfaceItemIdentifier("MimiTranscript")
        window.title = preferences.text("Mimi Transcript", "Mimi 文字起こし")
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView]
        window.titlebarAppearsTransparent = true
        window.toolbarStyle = .unified
        window.minSize = NSSize(width: 680, height: 480)
        window.setContentSize(NSSize(width: 920, height: 640))
        window.collectionBehavior.insert(.fullScreenPrimary)
        window.isReleasedWhenClosed = false
        window.setFrameAutosaveName("MimiTranscript")
        if UserDefaults.standard.object(forKey: "NSWindow Frame MimiTranscript") == nil {
            window.center()
        }
        let controller = NSWindowController(window: window)
        windowController = controller
        return window
    }
}
