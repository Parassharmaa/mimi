import AppKit
@preconcurrency import AVFoundation
import Carbon.HIToolbox
import MimiCore
import MimiSession
import Observation
import SwiftUI

enum VoiceTypingState: Equatable {
    case idle
    case listening
    case finishing
    case message(String, isError: Bool)

    var isVisible: Bool { self != .idle }
}

private struct HotKeyDefinition {
    let keyCode: UInt32
    let modifiers: UInt32
}

private extension VoiceTypingShortcut {
    var hotKey: HotKeyDefinition {
        switch self {
        case .optionSpace:
            HotKeyDefinition(keyCode: UInt32(kVK_Space), modifiers: UInt32(optionKey))
        case .commandShiftD:
            HotKeyDefinition(keyCode: UInt32(kVK_ANSI_D), modifiers: UInt32(cmdKey | shiftKey))
        }
    }
}

private final class HotKeyCallbackBox: @unchecked Sendable {
    let action: @MainActor (UInt32) -> Void
    init(action: @escaping @MainActor (UInt32) -> Void) { self.action = action }
}

private let voiceTypingHotKeyHandler: EventHandlerUPP = { _, event, userData in
    guard let event, let userData else { return noErr }
    var identifier = EventHotKeyID()
    let status = GetEventParameter(
        event,
        EventParamName(kEventParamDirectObject),
        EventParamType(typeEventHotKeyID),
        nil,
        MemoryLayout<EventHotKeyID>.size,
        nil,
        &identifier
    )
    guard status == noErr else { return status }
    let box = Unmanaged<HotKeyCallbackBox>.fromOpaque(userData).takeUnretainedValue()
    DispatchQueue.main.async { box.action(identifier.id) }
    return noErr
}

@MainActor
private final class GlobalHotKeyRegistration {
    private let callbackBox: HotKeyCallbackBox
    private var eventHandler: EventHandlerRef?
    private var primary: EventHotKeyRef?
    private var cancel: EventHotKeyRef?

    init(action: @escaping @MainActor (UInt32) -> Void) {
        callbackBox = HotKeyCallbackBox(action: action)
        var eventType = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )
        InstallEventHandler(
            GetApplicationEventTarget(),
            voiceTypingHotKeyHandler,
            1,
            &eventType,
            Unmanaged.passUnretained(callbackBox).toOpaque(),
            &eventHandler
        )
    }

    func registerPrimary(_ shortcut: VoiceTypingShortcut) -> Bool {
        if let primary { UnregisterEventHotKey(primary) }
        primary = nil
        let definition = shortcut.hotKey
        var reference: EventHotKeyRef?
        let status = RegisterEventHotKey(
            definition.keyCode,
            definition.modifiers,
            EventHotKeyID(signature: Self.signature, id: 1),
            GetApplicationEventTarget(),
            OptionBits(kEventHotKeyExclusive),
            &reference
        )
        primary = status == noErr ? reference : nil
        return status == noErr
    }

    func unregisterPrimary() {
        if let primary { UnregisterEventHotKey(primary) }
        primary = nil
    }

    func setCancelEnabled(_ enabled: Bool) {
        if let cancel { UnregisterEventHotKey(cancel) }
        cancel = nil
        guard enabled else { return }
        var reference: EventHotKeyRef?
        let status = RegisterEventHotKey(
            UInt32(kVK_Escape),
            0,
            EventHotKeyID(signature: Self.signature, id: 2),
            GetApplicationEventTarget(),
            OptionBits(kEventHotKeyExclusive),
            &reference
        )
        cancel = status == noErr ? reference : nil
    }

    private static let signature: OSType = 0x4D_49_4D_49 // MIMI
}

@MainActor
struct FocusedTextTarget {
    let element: AXUIElement
    let processIdentifier: pid_t

    static func capture(promptIfNeeded: Bool) throws -> FocusedTextTarget {
        if !AXIsProcessTrusted() {
            if promptIfNeeded {
                let options = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
                AXIsProcessTrustedWithOptions(options)
            }
            throw VoiceTypingError.accessibilityPermission
        }
        let system = AXUIElementCreateSystemWide()
        let application = NSWorkspace.shared.frontmostApplication.map {
            AXUIElementCreateApplication($0.processIdentifier)
        }
        var value: CFTypeRef?
        let applicationStatus = application.map {
            AXUIElementCopyAttributeValue($0, kAXFocusedUIElementAttribute as CFString, &value)
        }
        if applicationStatus != .success {
            value = nil
            _ = AXUIElementCopyAttributeValue(system, kAXFocusedUIElementAttribute as CFString, &value)
        }
        guard
              let value,
              CFGetTypeID(value) == AXUIElementGetTypeID() else {
            throw VoiceTypingError.noTextField
        }
        let element = unsafeDowncast(value as AnyObject, to: AXUIElement.self)
        if copyString(kAXSubroleAttribute as CFString, from: element) == (kAXSecureTextFieldSubrole as String) {
            throw VoiceTypingError.secureTextField
        }
        var pid: pid_t = 0
        AXUIElementGetPid(element, &pid)
        return FocusedTextTarget(element: element, processIdentifier: pid)
    }

    func insert(_ text: String) throws {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw VoiceTypingError.noSpeech
        }
        var settable = DarwinBoolean(false)
        let canSet = AXUIElementIsAttributeSettable(
            element,
            kAXSelectedTextAttribute as CFString,
            &settable
        ) == .success && settable.boolValue
        if canSet,
           AXUIElementSetAttributeValue(
               element,
               kAXSelectedTextAttribute as CFString,
               text as CFTypeRef
           ) == .success {
            return
        }
        try paste(text)
    }

    private func paste(_ text: String) throws {
        let pasteboard = NSPasteboard.general
        let snapshot = PasteboardSnapshot(pasteboard)
        pasteboard.clearContents()
        guard pasteboard.setString(text, forType: .string) else { throw VoiceTypingError.insertionFailed }
        NSRunningApplication(processIdentifier: processIdentifier)?.activate()
        guard let source = CGEventSource(stateID: .hidSystemState),
              let down = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(kVK_ANSI_V), keyDown: true),
              let up = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(kVK_ANSI_V), keyDown: false) else {
            snapshot.restore(to: pasteboard)
            throw VoiceTypingError.insertionFailed
        }
        down.flags = .maskCommand
        up.flags = .maskCommand
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { snapshot.restore(to: pasteboard) }
    }

    private static func copyString(_ attribute: CFString, from element: AXUIElement) -> String? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success else { return nil }
        return value as? String
    }
}

private struct PasteboardSnapshot: @unchecked Sendable {
    let items: [[NSPasteboard.PasteboardType: Data]]

    init(_ pasteboard: NSPasteboard) {
        items = (pasteboard.pasteboardItems ?? []).map { item in
            Dictionary(uniqueKeysWithValues: item.types.compactMap { type in
                item.data(forType: type).map { (type, $0) }
            })
        }
    }

    @MainActor func restore(to pasteboard: NSPasteboard) {
        pasteboard.clearContents()
        let restored = items.map { values -> NSPasteboardItem in
            let item = NSPasteboardItem()
            for (type, data) in values { item.setData(data, forType: type) }
            return item
        }
        if !restored.isEmpty { pasteboard.writeObjects(restored) }
    }
}

private enum VoiceTypingError: LocalizedError {
    case accessibilityPermission, noTextField, secureTextField, microphonePermission
    case assetsUnavailable(SpeechLanguage), shortcutUnavailable, sessionRecording, noSpeech, insertionFailed

    var errorDescription: String? {
        switch self {
        case .accessibilityPermission: "Allow Mimi in System Settings › Privacy & Security › Accessibility, then try again."
        case .noTextField: "Place the cursor in a text field, then try again."
        case .secureTextField: "Voice Type is unavailable in password fields."
        case .microphonePermission: "Allow microphone access to use Voice Type."
        case let .assetsUnavailable(language): "Prepare Apple Speech for \(language.displayName) in Language settings."
        case .shortcutUnavailable: "That shortcut is already used by another app. Choose another one in Settings."
        case .sessionRecording: "Stop the current transcription session before using Voice Type."
        case .noSpeech: "No speech was detected."
        case .insertionFailed: "This field does not accept inserted text."
        }
    }
}

private final class VoiceTypingAudioRelay: @unchecked Sendable {
    private let lock = NSLock()
    private var buffers: [AVAudioPCMBuffer] = []

    func enqueueCopy(of source: AVAudioPCMBuffer) -> Bool {
        guard let copy = AVAudioPCMBuffer(pcmFormat: source.format, frameCapacity: source.frameLength) else { return false }
        copy.frameLength = source.frameLength
        let sourceList = UnsafeMutableAudioBufferListPointer(source.mutableAudioBufferList)
        let destinationList = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
        guard sourceList.count == destinationList.count else { return false }
        for index in sourceList.indices {
            guard let sourceData = sourceList[index].mData,
                  let destinationData = destinationList[index].mData else { return false }
            memcpy(destinationData, sourceData, Int(sourceList[index].mDataByteSize))
            destinationList[index].mDataByteSize = sourceList[index].mDataByteSize
        }
        lock.lock()
        if buffers.count >= 24 { buffers.removeFirst() }
        buffers.append(copy)
        lock.unlock()
        return true
    }

    func drain() -> [AVAudioPCMBuffer] {
        lock.lock()
        defer { lock.unlock() }
        let result = buffers
        buffers.removeAll(keepingCapacity: true)
        return result
    }
}

@MainActor
@Observable
final class VoiceTypingController {
    private let preferences: UserPreferences
    private let isSessionRecording: @MainActor () -> Bool
    private let microphone = MicrophoneCapture()
    private let speechProvider = SystemAppleSpeechProvider()
    private var engine: (any AppleLiveTranscribing)?
    private var audioRelay: VoiceTypingAudioRelay?
    private var target: FocusedTextTarget?
    private var hotKeys: GlobalHotKeyRegistration!
    private var messageTask: Task<Void, Never>?
    private var observingPreferences = false
    private var finalizedPhrases: [String] = []

    private(set) var state: VoiceTypingState = .idle
    private(set) var text = ""
    private(set) var shortcutRegistered = false

    init(
        preferences: UserPreferences,
        isSessionRecording: @escaping @MainActor () -> Bool = { false }
    ) {
        self.preferences = preferences
        self.isSessionRecording = isSessionRecording
        hotKeys = GlobalHotKeyRegistration { [weak self] identifier in
            if identifier == 2 { self?.cancel() } else { self?.toggle() }
        }
        applyShortcutPreference()
        observePreferences()
    }

    var shortcutLabel: String { preferences.voiceTypingShortcut.displayName }
    var language: SpeechLanguage { preferences.voiceTypingLanguage }

    func toggle() {
        switch state {
        case .idle, .message: start()
        case .listening: finishAndInsert()
        case .finishing: break
        }
    }

    func requestAccessibilityAccess() {
        let options = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
        AXIsProcessTrustedWithOptions(options)
    }

    var hasAccessibilityAccess: Bool { AXIsProcessTrusted() }

    func refreshShortcut() { applyShortcutPreference() }

    func applyPresentationFixture(text: String) {
        self.text = text
        state = .listening
    }

    func cancel() {
        guard state == .listening || state == .finishing else { return }
        Task {
            _ = try? microphone.stop()
            audioRelay = nil
            if let engine { await engine.stop() }
            self.engine = nil
            target = nil
            text = ""
            hotKeys.setCancelEnabled(false)
            showMessage("Cancelled", isError: false)
        }
    }

    private func start() {
        messageTask?.cancel()
        Task {
            do {
                guard shortcutRegistered else { throw VoiceTypingError.shortcutUnavailable }
                guard !isSessionRecording() else { throw VoiceTypingError.sessionRecording }
                let capturedTarget = try FocusedTextTarget.capture(promptIfNeeded: true)
                guard await microphone.requestPermission() else { throw VoiceTypingError.microphonePermission }
                guard await speechProvider.assetStatus(for: language) == .installed else {
                    throw VoiceTypingError.assetsUnavailable(language)
                }
                let format = try microphone.configureInput(deviceID: nil)
                let engine = try speechProvider.makeEngine()
                text = ""
                finalizedPhrases = []
                try await engine.start(language: language, inputFormat: format) { [weak self] event in
                    guard let self else { return }
                    switch event {
                    case let .partial(value):
                        self.updateDisplayedText(livePhrase: value)
                    case let .final(value):
                        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !normalized.isEmpty { self.finalizedPhrases.append(normalized) }
                        self.updateDisplayedText(livePhrase: "")
                    }
                }
                self.engine = engine
                target = capturedTarget
                let relay = VoiceTypingAudioRelay()
                audioRelay = relay
                try microphone.start(recordingTo: nil, deviceID: nil) { [weak self, relay] buffer in
                    guard relay.enqueueCopy(of: buffer) else { return }
                    Task { @MainActor [weak self, relay] in self?.drain(relay) }
                }
                state = .listening
                hotKeys.setCancelEnabled(true)
            } catch {
                _ = try? microphone.stop()
                audioRelay = nil
                if let engine { await engine.stop() }
                self.engine = nil
                target = nil
                showMessage(error.localizedDescription, isError: true)
            }
        }
    }

    private func finishAndInsert() {
        state = .finishing
        Task {
            _ = try? microphone.stop()
            if let audioRelay { drain(audioRelay) }
            audioRelay = nil
            if let engine { await engine.stop() }
            self.engine = nil
            hotKeys.setCancelEnabled(false)
            do {
                guard let target else { throw VoiceTypingError.noTextField }
                try target.insert(text)
                self.target = nil
                showMessage("Inserted", isError: false)
            } catch {
                target = nil
                showMessage(error.localizedDescription, isError: true)
            }
        }
    }

    private func drain(_ relay: VoiceTypingAudioRelay) {
        guard audioRelay === relay, let engine else { return }
        for buffer in relay.drain() { engine.consume(buffer) }
    }

    private func updateDisplayedText(livePhrase: String) {
        let normalized = livePhrase.trimmingCharacters(in: .whitespacesAndNewlines)
        text = (finalizedPhrases + (normalized.isEmpty ? [] : [normalized])).joined(separator: " ")
    }

    private func showMessage(_ message: String, isError: Bool) {
        state = .message(message, isError: isError)
        hotKeys.setCancelEnabled(false)
        messageTask?.cancel()
        messageTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(isError ? 4 : 1.2))
            guard !Task.isCancelled else { return }
            self?.state = .idle
            self?.text = ""
        }
    }

    private func applyShortcutPreference() {
        guard preferences.voiceTypingEnabled else {
            hotKeys.unregisterPrimary()
            shortcutRegistered = false
            return
        }
        shortcutRegistered = hotKeys.registerPrimary(preferences.voiceTypingShortcut)
    }

    private func observePreferences() {
        guard !observingPreferences else { return }
        observingPreferences = true
        withObservationTracking {
            _ = preferences.voiceTypingEnabled
            _ = preferences.voiceTypingShortcut
        } onChange: { [weak self] in
            DispatchQueue.main.async {
                guard let self else { return }
                self.observingPreferences = false
                self.applyShortcutPreference()
                self.observePreferences()
            }
        }
    }
}

@MainActor
final class VoiceTypingPanelController {
    private let controller: VoiceTypingController
    private let preferences: UserPreferences
    private var panel: NSPanel?

    init(controller: VoiceTypingController, preferences: UserPreferences) {
        self.controller = controller
        self.preferences = preferences
        observe()
    }

    private func observe() {
        withObservationTracking { _ = controller.state } onChange: { [weak self] in
            DispatchQueue.main.async { self?.update(); self?.observe() }
        }
    }

    private func update() {
        guard controller.state.isVisible else { panel?.orderOut(nil); return }
        let panel = panel ?? makePanel()
        guard let screen = NSScreen.main ?? NSScreen.screens.first else { return }
        let size = NSSize(width: 460, height: 86)
        panel.setFrame(NSRect(
            x: screen.visibleFrame.midX - size.width / 2,
            y: screen.visibleFrame.minY + 48,
            width: size.width,
            height: size.height
        ), display: true)
        panel.orderFrontRegardless()
    }

    private func makePanel() -> NSPanel {
        let host = NSHostingController(rootView: VoiceTypingPill(controller: controller, preferences: preferences))
        host.sizingOptions = []
        let panel = NSPanel(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView], backing: .buffered, defer: false)
        panel.contentViewController = host
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.hidesOnDeactivate = false
        panel.ignoresMouseEvents = true
        self.panel = panel
        return panel
    }
}

struct VoiceTypingPill: View {
    @Bindable var controller: VoiceTypingController
    let preferences: UserPreferences

    var body: some View {
        HStack(spacing: 13) {
            Image(systemName: icon)
                .font(.title3.weight(.semibold))
                .foregroundStyle(iconColor)
                .symbolEffect(.pulse, options: .repeating, isActive: controller.state == .listening)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.headline).lineLimit(1)
                Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 8)
            if controller.state == .listening {
                Text("Esc")
                    .font(.caption.monospaced())
                    .padding(.horizontal, 7).padding(.vertical, 4)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
            }
        }
        .padding(.horizontal, 20)
        .frame(width: 460, height: 78)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 20).stroke(.separator.opacity(0.5)) }
        .padding(4)
    }

    private var title: String {
        switch controller.state {
        case .listening: controller.text.isEmpty ? preferences.text("Listening…", "聞き取り中…") : controller.text
        case .finishing: preferences.text("Finishing…", "確定中…")
        case let .message(message, _): message
        case .idle: ""
        }
    }

    private var detail: String {
        switch controller.state {
        case .listening: preferences.text("Press \(controller.shortcutLabel) to insert · Esc to cancel", "\(controller.shortcutLabel)で入力 · Escでキャンセル")
        case .finishing: preferences.text("Inserting into the selected field", "選択した入力欄に入力しています")
        case .message: preferences.text("Voice Type", "音声入力")
        case .idle: ""
        }
    }

    private var icon: String {
        switch controller.state {
        case .listening: "waveform"
        case .finishing: "ellipsis"
        case let .message(_, isError): isError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill"
        case .idle: "mic"
        }
    }

    private var iconColor: Color {
        if case let .message(_, isError) = controller.state { return isError ? .orange : .green }
        return .accentColor
    }
}
