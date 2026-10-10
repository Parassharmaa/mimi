import AppKit
import SwiftUI

/// A single native selection spans every sentence, including the live draft.
struct SelectableTranscriptText: NSViewRepresentable {
    let text: String
    var provisionalText = ""
    var font: NSFont = .preferredFont(forTextStyle: .body)
    var accessibilityLabel = "Transcript"
    var onSelectionChange: (Bool) -> Void = { _ in }

    func makeNSView(context: Context) -> TranscriptTextView {
        let view = TranscriptTextView()
        view.onSelectionChange = onSelectionChange
        return view
    }

    func updateNSView(_ view: TranscriptTextView, context: Context) {
        view.onSelectionChange = onSelectionChange
        view.setAccessibilityLabel(accessibilityLabel)
        view.update(text: text, provisionalText: provisionalText, font: font)
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: TranscriptTextView, context: Context) -> CGSize? {
        guard let width = proposal.width, width.isFinite, width > 0 else { return nil }
        return nsView.fittingSize(for: width)
    }
}

final class TranscriptTextView: NSTextView, NSTextViewDelegate {
    var onSelectionChange: (Bool) -> Void = { _ in }

    init() {
        let storage = NSTextStorage()
        let layout = NSLayoutManager()
        let container = NSTextContainer(size: CGSize(width: 0, height: CGFloat.greatestFiniteMagnitude))
        storage.addLayoutManager(layout)
        layout.addTextContainer(container)
        super.init(frame: .zero, textContainer: container)
        delegate = self
        isEditable = false
        isSelectable = true
        isRichText = false
        drawsBackground = false
        textContainerInset = .zero
        textContainer?.lineFragmentPadding = 0
        isHorizontallyResizable = false
        isVerticallyResizable = true
        textContainer?.widthTracksTextView = false
        setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    func update(text: String, provisionalText: String, font: NSFont) {
        let paragraph = NSMutableParagraphStyle()
        paragraph.lineSpacing = 3
        paragraph.paragraphSpacing = 10
        let value = NSMutableAttributedString(string: text, attributes: [
            .font: font, .foregroundColor: NSColor.labelColor, .paragraphStyle: paragraph
        ])
        if !provisionalText.isEmpty {
            if value.length > 0 { value.append(NSAttributedString(string: "\n")) }
            value.append(NSAttributedString(string: provisionalText, attributes: [
                .font: font, .foregroundColor: NSColor.secondaryLabelColor, .paragraphStyle: paragraph
            ]))
        }
        guard textStorage?.isEqual(to: value) != true else { return }
        let selection = selectedRange()
        let oldText = string as NSString
        let newText = value.string as NSString
        // Keep a selection stable as speech is appended; discard it if a draft
        // correction changes any text before the end of the selection.
        let end = selection.location == NSNotFound ? Int.max : NSMaxRange(selection)
        let preservesSelection = end <= oldText.length && end <= newText.length
            && oldText.substring(to: end) == newText.substring(to: end)
        textStorage?.setAttributedString(value)
        setSelectedRange(preservesSelection ? selection : NSRange(location: 0, length: 0))
        invalidateIntrinsicContentSize()
    }

    func fittingSize(for width: CGFloat) -> CGSize {
        // SwiftUI probes several widths. Measure separately so a probe cannot
        // change the wrapping width of the text currently on screen.
        let bounds = textStorage?.boundingRect(
            with: CGSize(width: width, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading]
        ) ?? .zero
        return CGSize(width: width, height: max(1, ceil(bounds.height)))
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        textContainer?.containerSize = CGSize(width: newSize.width, height: .greatestFiniteMagnitude)
    }

    func textViewDidChangeSelection(_ notification: Notification) {
        // Selection can change during a SwiftUI view update. Read the latest
        // range on the next turn, after any text replacement has settled.
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            onSelectionChange(selectedRanges.contains { $0.rangeValue.length > 0 })
        }
    }
}
