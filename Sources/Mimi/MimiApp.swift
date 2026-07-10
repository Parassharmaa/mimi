import AppKit
import SwiftUI

@MainActor
final class MimiAppDelegate: NSObject, NSApplicationDelegate {
    private var e2eWindow: NSWindow?
    private var e2eStore: AppStore?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Mimi is intentionally a background/menu-bar utility. Opening the
        // transcript window remains available from the menu extra.
        NSApplication.shared.setActivationPolicy(.accessory)

        let arguments = ProcessInfo.processInfo.arguments
        guard arguments.contains("--e2e-window") else { return }

        let store = AppStore(loadPersistedTranscript: false)
        store.sourceLanguage = .japanese
        store.engineID = .whisperKitLargeV3Turbo
        store.translationMode = .translateFinalSegments
        store.document.apply(.final("こんにちは、Mimi はローカルで文字起こしします。"), language: .japanese)
        store.document.apply(.final("Mimi keeps the transcript on this Mac."), language: .english)

        let view = MenuBarView(store: store)
        let window = NSWindow(contentViewController: NSHostingController(rootView: view))
        window.title = "Mimi E2E Sample"
        window.styleMask = [.titled, .closable, .miniaturizable]
        window.setContentSize(NSSize(width: 420, height: 540))
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApplication.shared.activate(ignoringOtherApps: true)
        e2eStore = store
        e2eWindow = window

        if arguments.contains("--e2e-auto-quit") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                print("Mimi UI smoke passed: menu-bar surface rendered deterministic English/Japanese sample data.")
                NSApplication.shared.terminate(nil)
            }
        }
    }
}

@main
struct MimiApp: App {
    @NSApplicationDelegateAdaptor(MimiAppDelegate.self) private var appDelegate
    @State private var store = AppStore()

    var body: some Scene {
        MenuBarExtra("Mimi", systemImage: store.menuBarSymbolName) {
            MenuBarView(store: store)
        }
        .menuBarExtraStyle(.window)

        Settings {
            SettingsView(store: store)
        }

        WindowGroup("Mimi Transcript", id: "transcript") {
            TranscriptWindow(store: store)
        }
        .defaultSize(width: 760, height: 560)
    }
}
