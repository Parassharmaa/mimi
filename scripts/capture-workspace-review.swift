import AppKit
import CoreGraphics
import Foundation

@main
struct WorkspaceReviewCapture {
    @MainActor static func main() async throws {
        guard CommandLine.arguments.count == 3 else { throw CocoaError(.fileReadInvalidFileName) }
        let app = CommandLine.arguments[1]
        let output = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let cases: [(String, [String])] = [
            ("01-workspace-light", ["--e2e-screen", "transcript", "--e2e-state", "history", "--e2e-history-count", "72", "--e2e-appearance", "light"]),
            ("02-workspace-dark", ["--e2e-screen", "transcript", "--e2e-state", "history", "--e2e-history-count", "72", "--e2e-appearance", "dark"]),
            ("03-workspace-japanese", ["--e2e-screen", "transcript", "--e2e-state", "history", "--e2e-history-count", "72", "--e2e-language", "japanese", "--e2e-appearance", "light"]),
            ("04-menu-bar", ["--e2e-screen", "menu", "--e2e-state", "ready", "--e2e-appearance", "light"]),
            ("05-voice-type-settings", ["--e2e-screen", "settings-voice", "--e2e-state", "ready", "--e2e-appearance", "light"]),
            ("06-speech-settings", ["--e2e-screen", "settings-models", "--e2e-state", "ready", "--e2e-appearance", "light"]),
            ("07-onboarding", ["--e2e-screen", "onboarding", "--e2e-state", "welcome", "--e2e-appearance", "light"]),
            ("08-voice-type-hud", ["--e2e-screen", "voice-typing", "--e2e-state", "ready", "--e2e-appearance", "dark"]),
            ("09-captions", ["--e2e-screen", "captions", "--e2e-state", "ready", "--e2e-appearance", "dark"]),
            ("10-model-picker", ["--e2e-screen", "transcript", "--e2e-state", "history", "--e2e-history-count", "72", "--e2e-appearance", "light"]),
            ("11-model-control-hover", ["--e2e-screen", "transcript", "--e2e-state", "history", "--e2e-history-count", "72", "--e2e-appearance", "light"])
        ]
        for (name, arguments) in cases {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: app)
            process.arguments = ["--e2e-window"] + arguments
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            try process.run()
            do {
                var selected: [String: Any]?
                for _ in 0..<20 {
                    try await Task.sleep(for: .milliseconds(400))
                    let windows = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] ?? []
                    selected = windows.first(where: {
                        ($0[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value == process.processIdentifier
                        && (($0[kCGWindowLayer as String] as? NSNumber)?.intValue ?? 99) <= 3
                    })
                    if selected != nil { break }
                }
                guard let window = selected, let number = window[kCGWindowNumber as String] as? NSNumber else {
                    throw CocoaError(.fileReadUnknown)
                }
                NSRunningApplication(processIdentifier: process.processIdentifier)?.activate(options: [])
                try await Task.sleep(for: .milliseconds(600))
                if name == "10-model-picker" || name == "11-model-control-hover" {
                    let interaction = Process()
                    interaction.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
                        .deletingLastPathComponent().appendingPathComponent("mimi-ax-audit")
                    interaction.arguments = ["--pid", String(process.processIdentifier)] + (name == "10-model-picker"
                        ? ["--press", "Choose speech model", "--step-delay", "0.5"]
                        : ["--hover", "Choose speech model"])
                    interaction.standardOutput = FileHandle.nullDevice
                    interaction.standardError = FileHandle.nullDevice
                    try interaction.run()
                    interaction.waitUntilExit()
                    guard interaction.terminationStatus == 0 else { throw CocoaError(.featureUnsupported) }
                    try await Task.sleep(for: .milliseconds(400))
                }
                let capture = Process()
                capture.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
                capture.arguments = ["-x", "-o", "-l\(number.uint32Value)", output.appendingPathComponent("\(name).png").path]
                try capture.run()
                capture.waitUntilExit()
                guard capture.terminationStatus == 0 else { throw CocoaError(.fileWriteUnknown) }
                print("Captured \(name)")
                process.terminate()
                process.waitUntilExit()
            } catch {
                if process.isRunning { process.terminate(); process.waitUntilExit() }
                throw error
            }
        }
    }
}
