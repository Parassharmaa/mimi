import AppKit
import MimiCore
import SwiftUI

enum SettingsTab: Hashable {
    case general
    case voiceTyping
    case captions
    case models
    case capture
    case privacy
}

struct SettingsView: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences
    @Bindable var voiceTyping: VoiceTypingController
    @State private var selectedTab: SettingsTab

    init(
        store: AppStore,
        preferences: UserPreferences,
        voiceTyping: VoiceTypingController,
        initialTab: SettingsTab = .general
    ) {
        self.store = store
        self.preferences = preferences
        self.voiceTyping = voiceTyping
        _selectedTab = State(initialValue: initialTab)
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            Tab(preferences.text("General", "一般"), systemImage: "gearshape", value: .general) {
                GeneralSettingsPane(preferences: preferences)
            }

            Tab(preferences.text("Voice Type", "音声入力"), systemImage: "keyboard.badge.ellipsis", value: .voiceTyping) {
                VoiceTypingSettingsPane(preferences: preferences, voiceTyping: voiceTyping)
            }

            Tab(preferences.text("Captions", "字幕"), systemImage: "captions.bubble", value: .captions) {
                CaptionSettingsPane(preferences: preferences)
            }

            Tab(preferences.text("Languages", "言語"), systemImage: "character.book.closed", value: .models) {
                ModelsSettingsPane(store: store, preferences: preferences)
            }

            Tab(preferences.text("Audio", "音声"), systemImage: "waveform", value: .capture) {
                CaptureSettingsPane(store: store)
            }

            Tab(preferences.text("Privacy", "プライバシー"), systemImage: "hand.raised", value: .privacy) {
                PrivacySettingsPane(preferences: preferences)
            }
        }
        .scenePadding()
        .frame(width: 620, height: 540)
        .background(SettingsWindowRegistrar())
    }
}

private struct VoiceTypingSettingsPane: View {
    @Bindable var preferences: UserPreferences
    @Bindable var voiceTyping: VoiceTypingController

    var body: some View {
        Form {
            Section(preferences.text("Type anywhere by speaking", "声でどこにでも入力")) {
                Toggle(preferences.text("Enable Voice Type", "音声入力を有効にする"), isOn: $preferences.voiceTypingEnabled)
                Picker(preferences.text("Shortcut", "ショートカット"), selection: $preferences.voiceTypingShortcut) {
                    ForEach(VoiceTypingShortcut.allCases) { shortcut in
                        Text(shortcut.displayName).tag(shortcut)
                    }
                }
                .disabled(!preferences.voiceTypingEnabled)
                Picker(preferences.text("Spoken language", "話す言語"), selection: $preferences.voiceTypingLanguage) {
                    ForEach(SpeechLanguage.allCases) { language in
                        Text(language.nativeName).tag(language)
                    }
                }
                .disabled(!preferences.voiceTypingEnabled)
            }

            Section(preferences.text("Access", "アクセス")) {
                LabeledContent(preferences.text("Accessibility", "アクセシビリティ")) {
                    Label(
                        voiceTyping.hasAccessibilityAccess ? preferences.text("Ready", "準備完了") : preferences.text("Permission needed", "許可が必要"),
                        systemImage: voiceTyping.hasAccessibilityAccess ? "checkmark.circle.fill" : "exclamationmark.circle"
                    )
                    .foregroundStyle(voiceTyping.hasAccessibilityAccess ? .green : .secondary)
                }
                if !voiceTyping.hasAccessibilityAccess {
                    Button(preferences.text("Allow in System Settings…", "システム設定で許可…")) {
                        voiceTyping.requestAccessibilityAccess()
                    }
                }
                if preferences.voiceTypingEnabled && !voiceTyping.shortcutRegistered {
                    Label(
                        preferences.text("That shortcut is already in use. Choose the other shortcut.", "そのショートカットは使用中です。もう一つを選んでください。"),
                        systemImage: "exclamationmark.triangle"
                    )
                    .font(.caption)
                    .foregroundStyle(.orange)
                }
            }

            Section {
                Text(preferences.text(
                    "Place the cursor in a text field and press the shortcut. Your words appear in the field as you speak. Press the shortcut again to stop, or Escape to undo this dictation. Password fields are never supported.",
                    "入力欄にカーソルを置き、ショートカットを押すと、話した内容がその場で入力されます。もう一度押すと停止し、Escで今回の音声入力を取り消せます。パスワード欄では使用できません。"
                ))
                .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}

private struct CaptionSettingsPane: View {
    @Bindable var preferences: UserPreferences

    var body: some View {
        Form {
            Section("Floating captions") {
                Toggle(preferences.text("Show captions above other apps", "他のアプリの上に字幕を表示"), isOn: $preferences.floatingCaptionsEnabled)
                Picker(preferences.text("Show", "表示内容"), selection: $preferences.floatingCaptionContent) {
                    Text(preferences.text("Original", "原文")).tag(FloatingCaptionContent.original)
                    Text(preferences.text("Translation", "翻訳")).tag(FloatingCaptionContent.translation)
                    Text(preferences.text("Original and translation", "原文と翻訳")).tag(FloatingCaptionContent.both)
                }
                Picker(preferences.text("Position", "位置"), selection: $preferences.floatingCaptionPosition) {
                    Text(preferences.text("Subtitles at bottom", "下部に字幕")).tag(FloatingCaptionPosition.subtitles)
                    Text(preferences.text("Top center", "上部中央")).tag(FloatingCaptionPosition.top)
                    Text(preferences.text("Top right", "右上")).tag(FloatingCaptionPosition.topRight)
                    Text(preferences.text("Bottom right", "右下")).tag(FloatingCaptionPosition.bottomRight)
                }
                Toggle(preferences.text("Let clicks pass through captions", "字幕の背後をクリックできるようにする"), isOn: $preferences.floatingCaptionClickThrough)
                Button(preferences.text("Reset caption position", "字幕の位置をリセット")) {
                    preferences.resetFloatingCaptionPosition()
                }
                .disabled(!preferences.floatingCaptionUsesCustomPosition)
            }

            Section {
                Text(preferences.text(
                    "Floating captions stay visible above other apps. Turn off click-through, then drag anywhere on the caption to place it exactly where you want.",
                    "字幕は他のアプリの上に表示されます。クリック透過をオフにすると、字幕のどこからでもドラッグして好きな位置に置けます。"
                ))
                .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}

private struct GeneralSettingsPane: View {
    @Bindable var preferences: UserPreferences
    @State private var startsAtLogin = false

    var body: some View {
        Form {
            Section(preferences.text("Language", "言語")) {
                Picker(preferences.text("Mimi speaks", "表示言語"), selection: $preferences.interfaceLanguage) {
                    ForEach(InterfaceLanguage.allCases) { language in
                        Text(language.nativeName).tag(language)
                    }
                }
                Text(preferences.text(
                    "Mimi can recognize and translate both English and Japanese regardless of this setting.",
                    "この設定に関係なく、Mimiは英語と日本語の認識と翻訳ができます。"
                ))
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Section(preferences.text("At login", "ログイン時")) {
                Toggle(preferences.text("Open Mimi automatically", "Mimiを自動的に開く"), isOn: $startsAtLogin)
                    .onChange(of: startsAtLogin) { _, enabled in
                        guard preferences.startsAtLogin != enabled else { return }
                        preferences.setStartsAtLogin(enabled)
                        startsAtLogin = preferences.startsAtLogin
                    }
                if let error = preferences.loginItemError {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
        }
        .formStyle(.grouped)
        .onAppear { startsAtLogin = preferences.startsAtLogin }
    }
}

private struct ModelsSettingsPane: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences

    var body: some View {
        Form {
            Section("Transcription") {
                Picker("Model", selection: $store.engineID) {
                    ForEach(TranscriptionEngineID.selectableCases) { engine in
                        Text(engine.displayName).tag(engine)
                    }
                }
                .disabled(store.controlsLocked || store.isModelSetupActive)

                Picker("Language", selection: $store.languageMode) {
                    ForEach(TranscriptionLanguageMode.allCases) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
                .disabled(store.controlsLocked || store.isModelSetupActive)

                if let pack = store.modelPack {
                    Text(pack.recommendation)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Local availability") {
                ModelSetupStatusView(
                    readiness: store.selectedModelReadiness,
                    setupState: store.selectedModelSetupState
                )

                if let pack = store.modelPack {
                    LabeledContent("Storage") {
                        Text(storageDescription(for: pack))
                            .foregroundStyle(.secondary)
                    }
                }

                modelActions
            }

            Section(preferences.text("Translation", "翻訳")) {
                LabeledContent(preferences.text("Model", "モデル")) {
                    Label(
                        translationModelAvailable
                            ? preferences.text("ElanMT ready", "ElanMT 準備完了")
                            : preferences.text("Model missing", "モデルが見つかりません"),
                        systemImage: translationModelAvailable
                            ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
                    )
                    .foregroundStyle(translationModelAvailable ? .green : .orange)
                }
                LabeledContent(preferences.text("Languages", "言語")) {
                    Text(preferences.text("English ↔ Japanese", "英語 ↔ 日本語"))
                        .foregroundStyle(.secondary)
                }
                LabeledContent(preferences.text("Storage", "ストレージ")) {
                    Text(preferences.text("73.4 MB, included with Mimi", "73.4 MB、Mimiに同梱"))
                        .foregroundStyle(.secondary)
                }
                Text(preferences.text(
                    "Translations run entirely on this Mac with Mimi's 4-bit Marian/MLX model. No text is sent to Apple or a cloud translation service.",
                    "翻訳はMimiの4ビットMarian/MLXモデルを使い、このMac上だけで実行されます。テキストはAppleやクラウド翻訳サービスには送信されません。"
                ))
                .font(.caption)
                .foregroundStyle(.secondary)

                Button(preferences.text("Open Model License…", "モデルのライセンスを開く…")) {
                    guard let translationLicenseDirectory else { return }
                    NSWorkspace.shared.open(translationLicenseDirectory)
                }
                .disabled(translationLicenseDirectory == nil)
            }

            if store.engineID.isExperimental {
                Section {
                    Label(
                        "This model is available locally, but remains experimental while Mimi evaluates Japanese accuracy, long-session stability, and thermal performance.",
                        systemImage: "flask"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }

    @ViewBuilder
    private var modelActions: some View {
        HStack(spacing: 8) {
            if store.canInstallSelectedModel {
                Button(modelActionTitle) {
                    store.installSelectedModel()
                }
                .buttonStyle(.borderedProminent)
            }

            if store.canCancelSelectedModelInstall {
                Button("Pause Download") {
                    store.cancelSelectedModelInstall()
                }
            }

            if shouldShowAppleStatusCheck {
                Button("Check Status") {
                    store.refreshSelectedModelReadiness()
                }
            }
        }

        if store.canRemoveSelectedModel {
            Button(removeButtonTitle, role: .destructive) {
                store.removeSelectedModel()
            }
            .disabled(store.controlsLocked)
        }

    }

    private func storageDescription(for pack: LocalModelPack) -> String {
        if let size = pack.estimatedDownloadMB {
            return "About \(size) MB, managed by Mimi"
        }
        return "Language asset managed by macOS"
    }

    private var translationModelAvailable: Bool {
        ExperimentalMLXTranslationConfiguration.resolved() != nil
    }

    private var translationLicenseDirectory: URL? {
        guard let resources = Bundle.main.resourceURL else { return nil }
        let directory = resources.appending(
            path: "TranslationLicenses",
            directoryHint: .isDirectory
        )
        return FileManager.default.fileExists(atPath: directory.path) ? directory : nil
    }

    private var modelActionTitle: String {
        let retry = switch store.selectedModelSetupState {
        case .cancelled, .failed: true
        case .idle, .checking, .downloading, .prewarming, .removing, .waitingForSystem: false
        }
        let base: String = switch store.engineID {
        case .appleSpeechAnalyzer:
            store.languageMode == .automatic ? "Prepare English and Japanese" : "Prepare \(store.sourceLanguage.displayName)"
        case .whisperKitLargeV3Turbo: "Download Whisper"
        case .nemotronStreamingExperimental: "Download Nemotron"
        case .qwen3StreamingExperimental: "Download Qwen3-ASR"
        }
        return retry ? "Retry \(base)" : base
    }

    private var shouldShowAppleStatusCheck: Bool {
        guard store.engineID == .appleSpeechAnalyzer else { return false }
        return switch store.selectedModelReadiness {
        case .checking, .needsDownload, .downloading, .ready: true
        case .unavailable, .experimental: false
        }
    }

    private var removeButtonTitle: String {
        switch store.engineID {
        case .whisperKitLargeV3Turbo: "Remove Whisper Download"
        case .nemotronStreamingExperimental: "Remove Nemotron Download"
        case .qwen3StreamingExperimental: "Remove Qwen3-ASR Download"
        case .appleSpeechAnalyzer: "Remove Download"
        }
    }
}

private struct CaptureSettingsPane: View {
    @Bindable var store: AppStore

    var body: some View {
        Form {
            Section("Device audio") {
                LabeledContent("Microphone") {
                    Label("Asked when recording starts", systemImage: "mic")
                        .foregroundStyle(.secondary)
                }

                LabeledContent("Audio Output") {
                    Label("System Audio Recording", systemImage: "speaker.wave.2")
                        .foregroundStyle(.secondary)
                }

                Text("Choose the exact microphone or output device in the Session sidebar. Mimi asks for the matching macOS permission only when capture begins.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Scoped meeting audio") {
                LabeledContent("App Audio") {
                    selectionStatus(for: .applicationAudio)
                }
                Button("Choose App…") {
                    chooseScreenAudio(.applicationAudio)
                }
                .disabled(store.controlsLocked)

                LabeledContent("Display Audio") {
                    selectionStatus(for: .systemAudio)
                }
                Button("Choose Display…") {
                    chooseScreenAudio(.systemAudio)
                }
                .disabled(store.controlsLocked)

                Text("For Google Meet, choose Chrome; for Zoom, choose Zoom. Mimi registers only an audio output for the selected app or display and never captures screen pixels.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let message = store.lastError, store.source != .microphone {
                Section("Capture status") {
                    Label(message, systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }

    @ViewBuilder
    private func selectionStatus(for source: AudioSource) -> some View {
        if let selection = store.screenAudioSelection, selection.source == source {
            Label("Selected", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .accessibilityLabel("\(source.displayName) selected")
        } else {
            Label("Not selected", systemImage: "circle")
                .foregroundStyle(.secondary)
                .accessibilityLabel("\(source.displayName) not selected")
        }
    }

    private func chooseScreenAudio(_ source: AudioSource) {
        store.source = source
        store.selectScreenAudioContent()
    }
}

private struct PrivacySettingsPane: View {
    @Bindable var preferences: UserPreferences

    var body: some View {
        Form {
            Section(preferences.text("Local by design", "ローカル設計")) {
                privacyRow(
                    preferences.text("Transcript", "文字起こし"),
                    detail: preferences.text(
                        "Finalized text is stored only on this Mac.",
                        "確定したテキストはこのMacにだけ保存されます。"
                    ),
                    symbol: "text.alignleft"
                )
                privacyRow(
                    preferences.text("Transcription", "音声認識"),
                    detail: preferences.text(
                        "Apple Speech turns audio into text on this Mac.",
                        "Apple SpeechがこのMac上で音声をテキストに変換します。"
                    ),
                    symbol: "waveform"
                )
                privacyRow(
                    preferences.text("Translation", "翻訳"),
                    detail: preferences.text(
                        "Mimi's bundled Marian/MLX model translates English and Japanese on this Mac.",
                        "Mimiに同梱されたMarian/MLXモデルが、このMac上で英語と日本語を翻訳します。"
                    ),
                    symbol: "translate"
                )
            }

            Section(preferences.text("Temporary audio", "一時的な音声")) {
                Text(preferences.text(
                    "Mimi processes working audio in memory and does not keep a source-audio recording after the session.",
                    "Mimiは処理中の音声をメモリ上で扱い、セッション終了後に元の音声録音を保存しません。"
                ))
                    .foregroundStyle(.secondary)
            }

            Section(preferences.text("System services", "システムサービス")) {
                Text(preferences.text(
                    "macOS owns permission prompts and speech assets. Apple frameworks may collect non-content performance metadata, but Mimi does not send transcript, translation, or source-audio content to a cloud service.",
                    "権限の確認と音声認識アセットはmacOSが管理します。Appleのフレームワークが内容を含まない性能情報を収集する場合がありますが、Mimiは文字起こし、翻訳、元音声の内容をクラウドサービスへ送信しません。"
                ))
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    private func privacyRow(_ title: String, detail: String, symbol: String) -> some View {
        Label {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } icon: {
            Image(systemName: symbol)
                .foregroundStyle(.secondary)
        }
    }
}
