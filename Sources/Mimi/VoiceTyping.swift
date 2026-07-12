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
final class FocusedTextTarget {
    let element: AXUIElement
    let processIdentifier: pid_t
    private let insertionLocation: Int
    private let replacedText: String
    private var insertedUTF16Length = 0

    private init(
        element: AXUIElement,
        processIdentifier: pid_t,
        insertionRange: CFRange,
        replacedText: String
    ) {
        self.element = element
        self.processIdentifier = processIdentifier
        insertionLocation = insertionRange.location
        self.replacedText = replacedText
    }

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
        guard let insertionRange = copyRange(kAXSelectedTextRangeAttribute as CFString, from: element) else {
            throw VoiceTypingError.noTextField
        }
        var pid: pid_t = 0
        AXUIElementGetPid(element, &pid)
        return FocusedTextTarget(
            element: element,
            processIdentifier: pid,
            insertionRange: insertionRange,
            replacedText: copyString(kAXSelectedTextAttribute as CFString, from: element) ?? ""
        )
    }

    func replaceLiveText(with text: String) async throws {
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == processIdentifier else {
            throw VoiceTypingError.focusChanged
        }
        try selectInsertedText()
        try postReplacementText(text)
        try await Task.sleep(for: .milliseconds(70))
        guard containsInsertedText(text) else {
            throw VoiceTypingError.insertionFailed
        }
        insertedUTF16Length = text.utf16.count
    }

    func rollback() async throws {
        guard insertedUTF16Length > 0 || !replacedText.isEmpty else { return }
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == processIdentifier else {
            throw VoiceTypingError.focusChanged
        }
        try selectInsertedText()
        try postReplacementText(replacedText)
        try await Task.sleep(for: .milliseconds(70))
        guard containsInsertedText(replacedText) else { throw VoiceTypingError.insertionFailed }
        insertedUTF16Length = replacedText.utf16.count
    }

    private func selectInsertedText() throws {
        var range = CFRange(location: insertionLocation, length: insertedUTF16Length)
        guard let value = AXValueCreate(.cfRange, &range),
              AXUIElementSetAttributeValue(
                  element,
                  kAXSelectedTextRangeAttribute as CFString,
                  value
              ) == .success else {
            throw VoiceTypingError.insertionFailed
        }
    }

    private func postReplacementText(_ text: String) throws {
        if text.isEmpty {
            guard insertedUTF16Length > 0 else { return }
            try postKey(CGKeyCode(kVK_Delete))
            return
        }
        guard let source = CGEventSource(stateID: .hidSystemState),
              let down = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true),
              let up = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false) else {
            throw VoiceTypingError.insertionFailed
        }
        let codeUnits = Array(text.utf16)
        codeUnits.withUnsafeBufferPointer { buffer in
            down.keyboardSetUnicodeString(stringLength: buffer.count, unicodeString: buffer.baseAddress)
        }
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
    }

    private func postKey(_ key: CGKeyCode) throws {
        guard let source = CGEventSource(stateID: .hidSystemState),
              let down = CGEvent(keyboardEventSource: source, virtualKey: key, keyDown: true),
              let up = CGEvent(keyboardEventSource: source, virtualKey: key, keyDown: false) else {
            throw VoiceTypingError.insertionFailed
        }
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
    }

    private func containsInsertedText(_ expected: String) -> Bool {
        guard let value = Self.copyString(kAXValueAttribute as CFString, from: element) else {
            return false
        }
        let utf16 = value.utf16
        guard insertionLocation >= 0,
              insertionLocation + expected.utf16.count <= utf16.count else {
            return false
        }
        let start = String.Index(utf16Offset: insertionLocation, in: value)
        let end = String.Index(
            utf16Offset: insertionLocation + expected.utf16.count,
            in: value
        )
        return String(value[start..<end]) == expected
    }

    private static func copyString(_ attribute: CFString, from element: AXUIElement) -> String? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success else { return nil }
        return value as? String
    }

    private static func copyRange(_ attribute: CFString, from element: AXUIElement) -> CFRange? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success,
              let value,
              CFGetTypeID(value) == AXValueGetTypeID() else { return nil }
        let axValue = unsafeDowncast(value as AnyObject, to: AXValue.self)
        var range = CFRange()
        guard AXValueGetValue(axValue, .cfRange, &range) else { return nil }
        return range
    }
}

private enum VoiceTypingError: LocalizedError {
    case accessibilityPermission, noTextField, secureTextField, microphonePermission
    case assetsUnavailable(SpeechLanguage), shortcutUnavailable, sessionRecording, focusChanged, insertionFailed

    var errorDescription: String? {
        switch self {
        case .accessibilityPermission: "Allow Mimi in System Settings › Privacy & Security › Accessibility, then try again."
        case .noTextField: "Place the cursor in a text field, then try again."
        case .secureTextField: "Voice Type is unavailable in password fields."
        case .microphonePermission: "Allow microphone access to use Voice Type."
        case let .assetsUnavailable(language): "Prepare Apple Speech for \(language.displayName) in Language settings."
        case .shortcutUnavailable: "That shortcut is already used by another app. Choose another one in Settings."
        case .sessionRecording: "Stop the current transcription session before using Voice Type."
        case .focusChanged: "Voice Type stopped because focus moved to another app."
        case .insertionFailed: "Mimi couldn’t update this field. No success was reported."
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
    private var fieldUpdateTask: Task<Void, Never>?
    private var pendingFieldText: String?
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

    var language: SpeechLanguage { preferences.voiceTypingLanguage }

    func toggle() {
        switch state {
        case .idle, .message: start()
        case .listening: finish()
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
        state = .finishing
        Task {
            _ = try? microphone.stop()
            if let audioRelay { drain(audioRelay) }
            audioRelay = nil
            if let engine { await engine.stop() }
            self.engine = nil
            pendingFieldText = nil
            if let fieldUpdateTask { await fieldUpdateTask.value }
            self.fieldUpdateTask = nil
            try? await target?.rollback()
            target = nil
            text = ""
            hotKeys.setCancelEnabled(false)
            state = .idle
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
                pendingFieldText = nil
                fieldUpdateTask = nil
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

    private func finish() {
        state = .finishing
        Task {
            _ = try? microphone.stop()
            if let audioRelay { drain(audioRelay) }
            audioRelay = nil
            if let engine { await engine.stop() }
            self.engine = nil
            hotKeys.setCancelEnabled(false)
            if let fieldUpdateTask { await fieldUpdateTask.value }
            self.fieldUpdateTask = nil
            target = nil
            guard case .finishing = state else { return }
            state = .idle
            text = ""
        }
    }

    private func drain(_ relay: VoiceTypingAudioRelay) {
        guard audioRelay === relay, let engine else { return }
        for buffer in relay.drain() { engine.consume(buffer) }
    }

    private func updateDisplayedText(livePhrase: String) {
        let normalized = livePhrase.trimmingCharacters(in: .whitespacesAndNewlines)
        text = (finalizedPhrases + (normalized.isEmpty ? [] : [normalized])).joined(separator: " ")
        scheduleFieldUpdate(text)
    }

    private func scheduleFieldUpdate(_ text: String) {
        guard target != nil, state == .listening || state == .finishing else { return }
        pendingFieldText = text
        guard fieldUpdateTask == nil else { return }
        fieldUpdateTask = Task { [weak self] in
            await self?.runFieldUpdates()
        }
    }

    private func runFieldUpdates() async {
        while let next = pendingFieldText {
            pendingFieldText = nil
            do {
                guard let target else { return }
                try await target.replaceLiveText(with: next)
            } catch {
                fieldUpdateTask = nil
                await stopAfterFieldFailure(error)
                return
            }
        }
        fieldUpdateTask = nil
    }

    private func stopAfterFieldFailure(_ error: Error) async {
        pendingFieldText = nil
        target = nil
        _ = try? microphone.stop()
        audioRelay = nil
        if let engine { await engine.stop() }
        self.engine = nil
        hotKeys.setCancelEnabled(false)
        showMessage(error.localizedDescription, isError: true)
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
        let size: NSSize = switch controller.state {
        case .message: NSSize(width: 420, height: 70)
        case .listening, .finishing: NSSize(width: 64, height: 64)
        case .idle: .zero
        }
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
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        Group {
            switch controller.state {
            case let .message(message, isError):
                HStack(spacing: 10) {
                    Image(systemName: isError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                        .foregroundStyle(isError ? .orange : .green)
                    Text(message)
                        .font(.callout.weight(.medium))
                        .lineLimit(2)
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 16)
                .frame(width: 412, height: 62)
                .background(surface, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            case .listening, .finishing:
                Image(systemName: controller.state == .listening ? "waveform" : "ellipsis")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(.tint)
                    .symbolEffect(
                        .pulse,
                        options: .repeating.speed(1.15),
                        isActive: controller.state == .listening && !reduceMotion
                    )
                    .frame(width: 52, height: 52)
                    .background(surface, in: Circle())
                    .overlay { Circle().stroke(.separator.opacity(0.45)) }
                    .accessibilityLabel(preferences.text("Voice Type is listening", "音声入力中"))
            case .idle:
                EmptyView()
            }
        }
        .padding(4)
    }

    private var surface: AnyShapeStyle {
        reduceTransparency
            ? AnyShapeStyle(Color(nsColor: .windowBackgroundColor))
            : AnyShapeStyle(.regularMaterial)
    }
}
