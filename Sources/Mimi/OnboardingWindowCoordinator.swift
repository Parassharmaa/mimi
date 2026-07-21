import AppKit
import SwiftUI

@MainActor
final class OnboardingWindowCoordinator {
    private var window: NSWindow?

    init(store: AppStore, preferences: UserPreferences, voiceTyping: VoiceTypingController) {
        let arguments = ProcessInfo.processInfo.arguments
        guard !preferences.completedOnboarding,
              !arguments.contains("--e2e-window"),
              !arguments.contains("--e2e-main-window-lifecycle") else { return }
        DispatchQueue.main.async { [weak self, weak store, weak preferences, weak voiceTyping] in
            guard let self, let store, let preferences, let voiceTyping else { return }
            self.show(store: store, preferences: preferences, voiceTyping: voiceTyping)
        }
    }

    func show(store: AppStore, preferences: UserPreferences, voiceTyping: VoiceTypingController) {
        if let window {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        let controller = NSHostingController(rootView: OnboardingView(store: store, preferences: preferences, voiceTyping: voiceTyping))
        let window = NSWindow(contentViewController: controller)
        window.title = preferences.text("Welcome to Mimi", "Mimiへようこそ")
        window.styleMask = [.titled, .closable, .fullSizeContentView]
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = true
        window.setContentSize(NSSize(width: 620, height: 560))
        window.center()
        window.setFrameAutosaveName("MimiOnboarding")
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        self.window = window
    }
}
