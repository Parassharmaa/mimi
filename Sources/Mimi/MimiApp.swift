import AppKit
import Darwin
import MimiCore
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
        if arguments.contains("--e2e-microphone-smoke") {
            let store = AppStore(loadPersistedTranscript: false)
            e2eStore = store
            Task { @MainActor in
                let status: Int32
                do {
                    let callbackCount = try await store.runMicrophoneCaptureSmokeTest()
                    print("Mimi microphone smoke passed: \(callbackCount) audio callbacks in one second.")
                    status = 0
                } catch {
                    print("Mimi microphone smoke failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        if let engineSmoke = argument(after: "--e2e-engine-smoke", in: arguments) {
            let store = AppStore(loadPersistedTranscript: false)
            e2eStore = store
            let engine: TranscriptionEngineID
            switch engineSmoke {
            case "whisper":
                engine = .whisperKitLargeV3Turbo
            case "nemotron":
                engine = .nemotronStreamingExperimental
            default:
                engine = .appleSpeechAnalyzer
            }
            let language: SpeechLanguage
            switch argument(after: "--e2e-language", in: arguments) {
            case "ja", "ja-JP":
                language = .japanese
            default:
                language = .english
            }
            Task { @MainActor in
                let status: Int32
                do {
                    try await store.runEngineSmokeTest(engine: engine, language: language)
                    print("Mimi \(engine.displayName) \(language.displayName) smoke passed: local model started and stopped without retaining source audio.")
                    status = 0
                } catch {
                    print("Mimi \(engine.displayName) smoke failed: \(error.localizedDescription)")
                    status = 1
                }
                Darwin.exit(status)
            }
            return
        }
        guard arguments.contains("--e2e-window") else { return }

        let store = AppStore(loadPersistedTranscript: false)
        store.sourceLanguage = .japanese
        store.engineID = .whisperKitLargeV3Turbo
        store.translationMode = .translateFinalSegments
        store.applyFixture(.final("こんにちは、Mimi はローカルで文字起こしします。"), language: .japanese)
        store.applyFixture(.final("Mimi keeps the transcript on this Mac."), language: .english)

        let screen = argument(after: "--e2e-screen", in: arguments) ?? "menu"
        let presentationState = argument(after: "--e2e-state", in: arguments) ?? "ready"
        switch presentationState {
        case "recording":
            store.applyPresentationFixture(state: .recording)
        case "failed":
            store.applyPresentationFixture(state: .failed("Microphone access needs attention"))
        default:
            break
        }
        let view: AnyView
        let size: NSSize
        switch screen {
        case "transcript":
            view = AnyView(TranscriptWindow(store: store))
            size = NSSize(width: 820, height: 600)
        case "settings":
            view = AnyView(SettingsView(store: store))
            size = NSSize(width: 560, height: 540)
        default:
            view = AnyView(MenuBarView(store: store))
            size = NSSize(width: 430, height: 580)
        }

        let hostingController = NSHostingController(rootView: view)
        // The smoke harness owns a fixed AppKit test window. Letting the
        // hosting controller also resize it from intrinsic content produces a
        // constraint feedback loop for a popover-width SwiftUI surface.
        hostingController.sizingOptions = []
        let window = NSWindow(contentViewController: hostingController)
        window.title = "Mimi E2E \(screen.capitalized)"
        window.styleMask = [.titled, .closable, .miniaturizable]
        if screen != "menu" {
            window.styleMask.insert(.resizable)
        }
        window.setContentSize(size)
        window.center()
        window.makeKeyAndOrderFront(nil)
        // The physical visual-QA runner may be launched while another app is
        // active. Put this deterministic test window in front so the native
        // surface can actually be inspected, rather than merely constructed.
        window.orderFrontRegardless()
        NSApplication.shared.activate(ignoringOtherApps: true)
        e2eStore = store
        e2eWindow = window

        if arguments.contains("--e2e-auto-quit") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                print("Mimi UI smoke passed: \(screen) \(presentationState) surface rendered deterministic English/Japanese sample data.")
                NSApplication.shared.terminate(nil)
            }
        }
    }

    private func argument(after flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else {
            return nil
        }
        return arguments[index + 1]
    }
}

@main
struct MimiApp: App {
    @NSApplicationDelegateAdaptor(MimiAppDelegate.self) private var appDelegate
    @State private var store = AppStore()

    var body: some Scene {
        MenuBarExtra {
            MenuBarView(store: store)
        } label: {
            Label(store.isRecording ? "Mimi REC" : "Mimi", systemImage: store.menuBarSymbolName)
                .accessibilityLabel(menuBarAccessibilityLabel)
        }
        .menuBarExtraStyle(.window)

        Settings {
            SettingsView(store: store)
        }

        WindowGroup("Mimi Transcript", id: "transcript") {
            TranscriptWindow(store: store)
        }
        .defaultSize(width: 760, height: 560)
        .windowToolbarStyle(.unified)
        .windowResizability(.contentMinSize)
        .commands {
            CommandMenu("Recording") {
                Button(store.isRecording ? "Stop Recording" : "Start Recording") {
                    store.toggleRecording()
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .disabled(store.isRecording ? store.recordingState == .processing : !store.canStartRecording)

                Button("Copy Transcript") {
                    store.copyTranscript()
                }
                .keyboardShortcut("c", modifiers: [.command, .shift])
                .disabled(store.document.renderedText.isEmpty)
            }
        }
    }

    private var menuBarAccessibilityLabel: String {
        if store.isRecording {
            return "Mimi recording \(store.source.displayName.lowercased()) locally"
        }
        return "Mimi \(store.recordingState.label.lowercased())"
    }
}
