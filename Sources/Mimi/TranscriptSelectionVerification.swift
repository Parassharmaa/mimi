import AppKit

@MainActor
func verifyTranscriptSelection(in root: NSView, expectedOriginal: String) -> Bool {
    func textViews(in view: NSView) -> [TranscriptTextView] {
        if let text = view as? TranscriptTextView { return [text] }
        return view.subviews.flatMap { textViews(in: $0) }
    }
    let views = textViews(in: root)
    guard views.count == 2, views.contains(where: { $0.string == expectedOriginal }) else {
        print("Transcript selection failed: expected original and translation text views.")
        return false
    }
    let pasteboard = NSPasteboard.withUniqueName()
    defer { pasteboard.releaseGlobally() }
    for view in views {
        guard view.isSelectable, !view.isEditable, view.frame.height > 0 else { return false }
        guard let container = view.textContainer,
              abs(container.containerSize.width - view.bounds.width) < 1 else {
            print("Transcript width mismatch: \(view.textContainer?.containerSize.width ?? -1) vs \(view.bounds.width)")
            return false
        }
        view.layoutManager?.ensureLayout(for: container)
        guard (view.layoutManager?.usedRect(for: container).height ?? 0) <= view.bounds.height + 1 else {
            print("Transcript selection failed: text is clipped vertically.")
            return false
        }
        guard view.window?.makeFirstResponder(view) == true,
              NSApp.sendAction(#selector(NSText.selectAll(_:)), to: nil, from: nil),
              view.selectedRange().length == (view.string as NSString).length else { return false }
        pasteboard.declareTypes(view.writablePasteboardTypes, owner: nil)
        guard view.writeSelection(to: pasteboard, types: view.writablePasteboardTypes),
              pasteboard.string(forType: .string) == view.string else { return false }
        view.setSelectedRange(NSRange(location: 0, length: 0))
    }
    print("Transcript selection passed: both visible panes copy complete text.")
    return true
}
