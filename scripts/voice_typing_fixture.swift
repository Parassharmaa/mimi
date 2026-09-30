import AppKit

/// An isolated, synthetic keyboard destination. Never uses the user's documents.
@MainActor
final class FixtureDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var fields: [NSTextField] = []
    private var timer: Timer?
    private var ticks = 0

    func applicationDidFinishLaunching(_ notification: Notification) {
        window = NSWindow(
            contentRect: NSRect(x: 100, y: 100, width: 520, height: 210),
            styleMask: [.titled, .closable], backing: .buffered, defer: false
        )
        window.title = "Mimi synthetic dictation fixture"
        let initial = argument("--text") ?? "hello world"
        for (index, text) in [initial, "untouched second field"].enumerated() {
            let field: NSTextField = index == 0 && CommandLine.arguments.contains("--secure")
                ? NSSecureTextField(string: text) : NSTextField(string: text)
            field.frame = NSRect(x: 24, y: 130 - index * 52, width: 470, height: 28)
            field.setAccessibilityLabel("Fixture field \(index + 1)")
            window.contentView?.addSubview(field)
            fields.append(field)
        }
        window.makeKeyAndOrderFront(nil)
        NSApplication.shared.activate(ignoringOtherApps: true)
        selectFirstField()
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
        emit()
    }

    private func selectFirstField() {
        window.makeFirstResponder(fields[0])
        guard let editor = fields[0].currentEditor() else { return }
        let location = Int(argument("--location") ?? "6") ?? 6
        let length = Int(argument("--length") ?? "5") ?? 5
        editor.selectedRange = NSRange(location: location, length: length)
    }

    private func tick() {
        ticks += 1
        if let seconds = argument("--switch-after").flatMap(Double.init),
           ticks == Int(seconds * 4) {
            window.makeFirstResponder(fields[1])
            fields[1].currentEditor()?.selectedRange = NSRange(location: 0, length: 0)
        }
        if let seconds = argument("--edit-after").flatMap(Double.init), ticks == Int(seconds * 4) {
            fields[0].stringValue = "user changed this field"
            fields[0].currentEditor()?.string = "user changed this field"
        }
        if let seconds = argument("--move-caret-after").flatMap(Double.init), ticks == Int(seconds * 4) {
            fields[0].currentEditor()?.selectedRange = NSRange(location: 0, length: 0)
        }
        emit()
        if ticks >= 60 { NSApplication.shared.terminate(nil) }
    }

    private func emit() {
        let values = fields.map { $0.currentEditor()?.string ?? $0.stringValue }
        if let data = try? JSONSerialization.data(withJSONObject: values) {
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        }
    }

    private func argument(_ flag: String) -> String? {
        let arguments = CommandLine.arguments
        guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else { return nil }
        return arguments[index + 1]
    }
}

@main
struct VoiceTypingFixture {
    @MainActor static func main() {
        let application = NSApplication.shared
        let delegate = FixtureDelegate()
        application.delegate = delegate
        application.setActivationPolicy(.regular)
        application.run()
    }
}
