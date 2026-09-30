import AppKit
import ApplicationServices

struct AccessibilityNode: Codable {
    let role: String
    let label: String
    let value: String?
    let enabled: Bool?
    let selected: Bool?
    let children: [AccessibilityNode]
}

@main
struct AccessibilityAudit {
    static func attribute(_ name: String, _ element: AXUIElement) -> CFTypeRef? {
        var value: CFTypeRef?
        return AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success ? value : nil
    }

    static func children(_ element: AXUIElement) -> [AXUIElement] {
        attribute(kAXChildrenAttribute, element) as? [AXUIElement] ?? []
    }

    static func label(_ element: AXUIElement) -> String {
        [kAXDescriptionAttribute, kAXTitleAttribute, kAXHelpAttribute].compactMap {
            attribute($0, element) as? String
        }.first { !$0.isEmpty } ?? ""
    }

    static func node(_ element: AXUIElement, depth: Int = 0) -> AccessibilityNode {
        AccessibilityNode(
            role: attribute(kAXRoleAttribute, element) as? String ?? "",
            label: label(element),
            value: attribute(kAXValueAttribute, element).map { value in
                (value as? String) ?? (value as? NSNumber)?.stringValue ?? ""
            },
            enabled: (attribute(kAXEnabledAttribute, element) as? NSNumber)?.boolValue,
            selected: (attribute(kAXSelectedAttribute, element) as? NSNumber)?.boolValue,
            children: depth < 24 ? children(element).map { node($0, depth: depth + 1) } : []
        )
    }

    static func find(_ element: AXUIElement, label expected: String, depth: Int = 0) -> AXUIElement? {
        if label(element) == expected { return element }
        guard depth < 24 else { return nil }
        for child in children(element) {
            if let found = find(child, label: expected, depth: depth + 1) { return found }
        }
        return nil
    }

    static func argument(_ flag: String) -> String? {
        let args = CommandLine.arguments
        guard let index = args.firstIndex(of: flag), index + 1 < args.count else { return nil }
        return args[index + 1]
    }

    static func sheetCount(_ node: AccessibilityNode) -> Int {
        (node.role == "AXSheet" ? 1 : 0) + node.children.reduce(0) { $0 + sheetCount($1) }
    }

    static func searchField(_ element: AXUIElement, depth: Int = 0) -> AXUIElement? {
        if attribute(kAXSubroleAttribute, element) as? String == "AXSearchField" { return element }
        guard depth < 24 else { return nil }
        for child in children(element) {
            if let result = searchField(child, depth: depth + 1) { return result }
        }
        return nil
    }

    @MainActor static func main() throws {
        guard let text = argument("--pid"), let pid = Int32(text), AXIsProcessTrusted() else {
            throw CocoaError(.userCancelled, userInfo: [NSLocalizedDescriptionKey: "Supply a synthetic fixture PID and allow Accessibility access."])
        }
        let app = AXUIElementCreateApplication(pid)
        if let query = argument("--type-search") {
            NSRunningApplication(processIdentifier: pid)?.activate(options: [])
            Thread.sleep(forTimeInterval: 0.15)
            guard NSWorkspace.shared.frontmostApplication?.processIdentifier == pid,
                  let field = searchField(app),
                  AXUIElementSetAttributeValue(field, kAXFocusedAttribute as CFString, kCFBooleanTrue) == .success else {
                throw CocoaError(.userCancelled)
            }
            let current = attribute(kAXValueAttribute, field) as? String ?? ""
            var range = CFRange(location: 0, length: current.utf16.count)
            if let value = AXValueCreate(.cfRange, &range) {
                _ = AXUIElementSetAttributeValue(field, kAXSelectedTextRangeAttribute as CFString, value)
            }
            let down = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true)!
            let up = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false)!
            let units = Array(query.utf16)
            units.withUnsafeBufferPointer {
                down.keyboardSetUnicodeString(stringLength: $0.count, unicodeString: $0.baseAddress)
            }
            down.post(tap: .cghidEventTap)
            up.post(tap: .cghidEventTap)
            Thread.sleep(forTimeInterval: 0.15)
        }
        if let query = argument("--search") {
            guard let field = searchField(app),
                  AXUIElementSetAttributeValue(field, kAXValueAttribute as CFString, query as CFString) == .success else {
                throw CocoaError(.featureUnsupported)
            }
        }
        let actions = (argument("--press-sequence") ?? argument("--press") ?? "")
            .split(separator: "|").map(String.init)
        if !actions.isEmpty {
            NSRunningApplication(processIdentifier: pid)?.activate(options: [])
            Thread.sleep(forTimeInterval: 0.15)
            guard NSWorkspace.shared.frontmostApplication?.processIdentifier == pid else {
                throw CocoaError(.userCancelled)
            }
            for title in actions {
                guard let target = find(app, label: title) else {
                    throw CocoaError(.featureUnsupported, userInfo: [NSLocalizedDescriptionKey: "No accessible action named \(title)"])
                }
                let sheetsBefore = sheetCount(node(app))
                let result = AXUIElementPerformAction(target, kAXPressAction as CFString)
                Thread.sleep(forTimeInterval: Double(argument("--step-delay") ?? "0.3") ?? 0.3)
                // Native sheet dismissal can invalidate the AX button before
                // the IPC call returns. Verify the effect, never blindly retry.
                let confirmedDismissal = title == "Cancel" && sheetsBefore > 0 && sheetCount(node(app)) < sheetsBefore
                guard result == .success || confirmedDismissal else {
                    var available: CFArray?
                    _ = AXUIElementCopyActionNames(target, &available)
                    FileHandle.standardError.write(Data("Available actions: \(available as? [String] ?? [])\n".utf8))
                    throw CocoaError(.featureUnsupported, userInfo: [NSLocalizedDescriptionKey: "AX action \(title) failed with status \(result.rawValue)"])
                }
                FileHandle.standardError.write(Data("PASS native action: \(title), AX status \(result.rawValue), confirmed dismissal \(confirmedDismissal)\n".utf8))
            }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        FileHandle.standardOutput.write(try encoder.encode(node(app)))
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
}
