import Foundation
import MimiCore
import Observation
import ServiceManagement

enum InterfaceLanguage: String, CaseIterable, Identifiable {
    case english
    case japanese

    var id: String { rawValue }
    var nativeName: String { self == .english ? "English" : "日本語" }
}

enum FloatingCaptionContent: String, CaseIterable, Identifiable {
    case original
    case translation
    case both

    var id: String { rawValue }
}

enum FloatingCaptionPosition: String, CaseIterable, Identifiable {
    case subtitles
    case top
    case topRight
    case bottomRight

    var id: String { rawValue }
}

enum VoiceTypingShortcut: String, CaseIterable, Identifiable {
    case optionSpace
    case commandShiftD

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .optionSpace: "⌥Space"
        case .commandShiftD: "⇧⌘D"
        }
    }
}

@MainActor
@Observable
final class UserPreferences {
    private enum Key {
        static let interfaceLanguage = "interfaceLanguage"
        static let completedOnboarding = "completedOnboarding"
        static let floatingCaptions = "floatingCaptions"
        static let floatingCaptionContent = "floatingCaptionContent"
        static let floatingCaptionPosition = "floatingCaptionPosition"
        static let floatingCaptionClickThrough = "floatingCaptionClickThrough"
        static let floatingCaptionUsesCustomPosition = "floatingCaptionUsesCustomPosition"
        static let floatingCaptionOriginX = "floatingCaptionOriginX"
        static let floatingCaptionOriginY = "floatingCaptionOriginY"
        static let voiceTypingEnabled = "voiceTypingEnabled"
        static let voiceTypingShortcut = "voiceTypingShortcut"
        static let voiceTypingLanguage = "voiceTypingLanguage"
    }

    private let defaults: UserDefaults

    var interfaceLanguage: InterfaceLanguage {
        didSet { defaults.set(interfaceLanguage.rawValue, forKey: Key.interfaceLanguage) }
    }
    var completedOnboarding: Bool {
        didSet { defaults.set(completedOnboarding, forKey: Key.completedOnboarding) }
    }
    var floatingCaptionsEnabled: Bool {
        didSet { defaults.set(floatingCaptionsEnabled, forKey: Key.floatingCaptions) }
    }
    var floatingCaptionContent: FloatingCaptionContent {
        didSet { defaults.set(floatingCaptionContent.rawValue, forKey: Key.floatingCaptionContent) }
    }
    var floatingCaptionPosition: FloatingCaptionPosition {
        didSet {
            defaults.set(floatingCaptionPosition.rawValue, forKey: Key.floatingCaptionPosition)
            guard floatingCaptionPosition != oldValue else { return }
            floatingCaptionUsesCustomPosition = false
        }
    }
    var floatingCaptionClickThrough: Bool {
        didSet { defaults.set(floatingCaptionClickThrough, forKey: Key.floatingCaptionClickThrough) }
    }
    var voiceTypingEnabled: Bool {
        didSet { defaults.set(voiceTypingEnabled, forKey: Key.voiceTypingEnabled) }
    }
    var voiceTypingShortcut: VoiceTypingShortcut {
        didSet { defaults.set(voiceTypingShortcut.rawValue, forKey: Key.voiceTypingShortcut) }
    }
    var voiceTypingLanguage: SpeechLanguage {
        didSet { defaults.set(voiceTypingLanguage.rawValue, forKey: Key.voiceTypingLanguage) }
    }
    private(set) var floatingCaptionUsesCustomPosition: Bool {
        didSet { defaults.set(floatingCaptionUsesCustomPosition, forKey: Key.floatingCaptionUsesCustomPosition) }
    }
    private(set) var floatingCaptionCustomOrigin: CGPoint? {
        didSet {
            guard let floatingCaptionCustomOrigin else {
                defaults.removeObject(forKey: Key.floatingCaptionOriginX)
                defaults.removeObject(forKey: Key.floatingCaptionOriginY)
                return
            }
            defaults.set(floatingCaptionCustomOrigin.x, forKey: Key.floatingCaptionOriginX)
            defaults.set(floatingCaptionCustomOrigin.y, forKey: Key.floatingCaptionOriginY)
        }
    }
    var loginItemError: String?

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        if defaults === UserDefaults.standard {
            Self.importLegacySandboxPreferencesIfNeeded(into: defaults)
        }
        interfaceLanguage = InterfaceLanguage(
            rawValue: defaults.string(forKey: Key.interfaceLanguage) ?? ""
        ) ?? .english
        completedOnboarding = defaults.bool(forKey: Key.completedOnboarding)
        floatingCaptionsEnabled = defaults.bool(forKey: Key.floatingCaptions)
        floatingCaptionContent = FloatingCaptionContent(
            rawValue: defaults.string(forKey: Key.floatingCaptionContent) ?? ""
        ) ?? .translation
        floatingCaptionPosition = FloatingCaptionPosition(
            rawValue: defaults.string(forKey: Key.floatingCaptionPosition) ?? ""
        ) ?? .subtitles
        floatingCaptionClickThrough = defaults.object(forKey: Key.floatingCaptionClickThrough) == nil
            ? true
            : defaults.bool(forKey: Key.floatingCaptionClickThrough)
        // A global shortcut should never appear silently after an update.
        // New and existing users opt in from onboarding or Voice Type settings.
        voiceTypingEnabled = defaults.bool(forKey: Key.voiceTypingEnabled)
        voiceTypingShortcut = VoiceTypingShortcut(
            rawValue: defaults.string(forKey: Key.voiceTypingShortcut) ?? ""
        ) ?? .optionSpace
        voiceTypingLanguage = SpeechLanguage(
            rawValue: defaults.string(forKey: Key.voiceTypingLanguage) ?? ""
        ) ?? .english
        floatingCaptionUsesCustomPosition = defaults.bool(forKey: Key.floatingCaptionUsesCustomPosition)
        if defaults.object(forKey: Key.floatingCaptionOriginX) != nil,
           defaults.object(forKey: Key.floatingCaptionOriginY) != nil {
            floatingCaptionCustomOrigin = CGPoint(
                x: defaults.double(forKey: Key.floatingCaptionOriginX),
                y: defaults.double(forKey: Key.floatingCaptionOriginY)
            )
        } else {
            floatingCaptionCustomOrigin = nil
        }
    }

    private static func importLegacySandboxPreferencesIfNeeded(into defaults: UserDefaults) {
        guard defaults.object(forKey: Key.completedOnboarding) == nil else { return }
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Library/Containers/dev.paras.mimi/Data/Library/Preferences/dev.paras.mimi.plist")
        guard let data = try? Data(contentsOf: url),
              let values = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any] else {
            return
        }
        for (key, value) in values where defaults.object(forKey: key) == nil {
            defaults.set(value, forKey: key)
        }
    }

    var startsAtLogin: Bool {
        SMAppService.mainApp.status == .enabled
    }

    func setStartsAtLogin(_ enabled: Bool) {
        do {
            if enabled {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            loginItemError = nil
        } catch {
            loginItemError = interfaceLanguage == .japanese
                ? "ログイン時の起動を変更できませんでした。アプリケーションフォルダからMimiを開いて、もう一度お試しください。"
                : "Mimi couldn’t change the login setting. Open Mimi from Applications and try again."
        }
    }

    func text(_ english: String, _ japanese: String) -> String {
        interfaceLanguage == .japanese ? japanese : english
    }

    func rememberFloatingCaptionOrigin(_ origin: CGPoint) {
        floatingCaptionCustomOrigin = origin
        floatingCaptionUsesCustomPosition = true
    }

    func resetFloatingCaptionPosition() {
        floatingCaptionUsesCustomPosition = false
        floatingCaptionCustomOrigin = nil
    }
}
