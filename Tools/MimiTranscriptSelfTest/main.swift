import AppKit
import SwiftUI

@main
struct MimiTranscriptSelfTest {
    @MainActor
    static func main() async throws {
        _ = NSApplication.shared
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        let view = TranscriptTextView()
        let selectionWindow = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 500, height: 300), styleMask: [.titled], backing: .buffered, defer: false)
        selectionWindow.contentView = view
        selectionWindow.makeFirstResponder(view)
        var hasSelection = false
        view.onSelectionChange = { hasSelection = $0 }
        let original = "First sentence.\n日本語の文章です。\nA third sentence with an emoji 😀."
        let draft = "Still listening…"
        let font = NSFont.preferredFont(forTextStyle: .title3)
        view.update(text: original, provisionalText: draft, font: font)
        let fullText = original + "\n" + draft
        try check(view.isSelectable && !view.isEditable, "transcripts are selectable and read-only")
        view.selectAll(nil)
        try await Task.sleep(for: .milliseconds(20))
        try check(hasSelection, "selection tells live scrolling to pause")
        try check(view.selectedRange().length == (fullText as NSString).length, "Select All spans English, Japanese and the draft")
        pasteboard.declareTypes(view.writablePasteboardTypes, owner: nil)
        try check(view.writeSelection(to: pasteboard, types: view.writablePasteboardTypes), "native selection writes to clipboard")
        try check(pasteboard.string(forType: .string) == fullText, "copy includes every paragraph without UI labels")

        let selection = (fullText as NSString).range(of: "sentence.\n日本語")
        view.setSelectedRange(selection)
        view.update(text: original, provisionalText: draft + " more words", font: font)
        try check(view.selectedRange() == selection, "new speech preserves the selected text")
        pasteboard.declareTypes(view.writablePasteboardTypes, owner: nil)
        _ = view.writeSelection(to: pasteboard, types: view.writablePasteboardTypes)
        try check(pasteboard.string(forType: .string) == "sentence.\n日本語", "partial copy crosses a sentence boundary")
        view.update(text: "Replacement session", provisionalText: "", font: font)
        try check(view.selectedRange().length == 0, "replaced text clears stale selection")
        try await Task.sleep(for: .milliseconds(20))
        try check(!hasSelection, "cleared selection releases live scrolling")

        let longText = Array(repeating: original + " This deliberately long sentence checks that resizing a transcript pane wraps its text without clipping words or requiring horizontal scrolling.", count: 25).joined(separator: "\n")
        view.update(text: longText, provisionalText: "", font: font)
        let wide = view.fittingSize(for: 700)
        let narrow = view.fittingSize(for: 240)
        try check(narrow.height > wide.height && wide.height > 100, "long transcripts wrap at narrow window widths")
        view.setFrameSize(narrow)
        view.layoutManager?.ensureLayout(for: view.textContainer!)
        try check(view.layoutManager!.usedRect(for: view.textContainer!).height <= narrow.height + 1, "long transcript sizing includes every wrapped line")
        view.selectAll(nil)
        view.update(text: longText, provisionalText: "", font: font)
        try check(view.selectedRange().length == (longText as NSString).length, "unchanged renders retain Select All")
        try check(TranscriptClipboard.copy(fullText, to: pasteboard), "copy button writes the complete transcript")
        try check(!TranscriptClipboard.copy(" \n\t", to: pasteboard), "empty transcript cannot clear the clipboard")
        try check(pasteboard.string(forType: .string) == fullText, "disabled copy preserves previous clipboard contents")

        // Exercise SwiftUI's size negotiation with the actual native text view.
        let host = NSHostingView(rootView: ScrollView {
            SelectableTranscriptText(text: original, provisionalText: draft)
                .padding(18)
        }.frame(width: 500, height: 300).background(Color(nsColor: .textBackgroundColor)))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 500, height: 300), styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(250))
        let rendered = findTranscript(in: host)
        try check(rendered?.string == fullText, "SwiftUI renders the complete transcript")
        try check((rendered?.frame.height ?? 0) > 40, "native transcript gets a multiline layout")
        if let directory = ProcessInfo.processInfo.environment["MIMI_UI_EVIDENCE"] {
            try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
            for (name, appearance) in [("light", NSAppearance.Name.aqua), ("dark", .darkAqua)] {
                host.appearance = NSAppearance(named: appearance)
                try await Task.sleep(for: .milliseconds(150))
                host.display()
                if let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) {
                    host.cacheDisplay(in: host.bounds, to: bitmap)
                    try bitmap.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: directory).appendingPathComponent("transcript-\(name).png"))
                }
            }
        }
        rendered?.setSelectedRange(NSRange(location: 0, length: 5))
        host.rootView = ScrollView {
            SelectableTranscriptText(text: original + "\n" + longText, provisionalText: draft)
                .padding(18)
        }.frame(width: 500, height: 300).background(Color(nsColor: .textBackgroundColor))
        try await Task.sleep(for: .milliseconds(250))
        host.layoutSubtreeIfNeeded()
        let updated = findTranscript(in: host)
        try check(updated?.string == original + "\n" + longText + "\n" + draft, "live SwiftUI updates retain the entire growing transcript")
        try check(updated?.selectedRange() == NSRange(location: 0, length: 5), "live SwiftUI updates preserve the native selection")
        try check((updated?.frame.height ?? 0) > 300, "growing transcripts expand inside the scroll view")
        print("Transcript usability verification passed.")
    }

    @MainActor
    private static func findTranscript(in view: NSView) -> TranscriptTextView? {
        if let text = view as? TranscriptTextView { return text }
        return view.subviews.lazy.compactMap { findTranscript(in: $0) }.first
    }

    private static func check(_ condition: Bool, _ description: String) throws {
        guard condition else { throw NSError(domain: "MimiTranscriptSelfTest", code: 1, userInfo: [NSLocalizedDescriptionKey: description]) }
        print("PASS: \(description)")
    }
}
