import Foundation
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
        didSet { defaults.set(floatingCaptionPosition.rawValue, forKey: Key.floatingCaptionPosition) }
    }
    var floatingCaptionClickThrough: Bool {
        didSet { defaults.set(floatingCaptionClickThrough, forKey: Key.floatingCaptionClickThrough) }
    }
    var loginItemError: String?

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
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
}
