import AppKit
import MimiCore
import SwiftUI

enum SettingsTab: Hashable, CaseIterable {
    case general
    case voiceTyping
    case captions
    case models
    case capture
    case privacy

    @MainActor func title(_ preferences: UserPreferences) -> String {
        switch self {
        case .general: preferences.text("General", "一般")
        case .voiceTyping: preferences.text("Voice Type", "音声入力")
        case .captions: preferences.text("Captions", "字幕")
        case .models: preferences.text("Speech & languages", "音声認識と言語")
        case .capture: preferences.text("Audio & devices", "音声とデバイス")
        case .privacy: preferences.text("Privacy", "プライバシー")
        }
    }

    var symbol: String {
        switch self {
        case .general: "gearshape"
        case .voiceTyping: "keyboard"
        case .captions: "captions.bubble"
        case .models: "cpu"
        case .capture: "waveform"
        case .privacy: "lock.shield"
        }
    }
}

struct SettingsView: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences
    @Bindable var voiceTyping: VoiceTypingController
    @State private var selectedTab: SettingsTab
    private let focusCoordinator = SettingsWindowFocusCoordinator.shared

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

    private var sidebarSelection: Binding<SettingsTab?> {
        Binding(get: { selectedTab }, set: { if let value = $0 { selectedTab = value } })
    }

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 18) {
                Text(preferences.text("Settings", "設定"))
                    .font(.title3.weight(.semibold)).padding(.horizontal, 18).padding(.top, 24)
                List(SettingsTab.allCases, id: \.self, selection: sidebarSelection) { tab in
                    Label(tab.title(preferences), systemImage: tab.symbol).padding(.vertical, 5).tag(tab)
                }
                .listStyle(.sidebar)
                .scrollContentBackground(.hidden)
                Text(preferences.text("Made for this Mac", "このMacのために"))
                    .font(.caption).foregroundStyle(.secondary).padding(18)
            }
            .frame(width: 195)
            VStack(alignment: .leading, spacing: 6) {
                Text(selectedTab.title(preferences)).font(.title2.weight(.semibold))
                    .padding(.horizontal, 28).padding(.top, 26)
                Text(preferences.text("Make Mimi work the way you do.", "自分に合ったMimiの使い方に。"))
                    .font(.callout).foregroundStyle(.secondary)
                    .padding(.horizontal, 28).padding(.bottom, 8)
                settingsContent.frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .buttonStyle(MimiQuietButtonStyle())
        .frame(minWidth: 740, idealWidth: 780, minHeight: 560, idealHeight: 600)
        .background(SettingsWindowRegistrar())
        .onAppear { applyRequestedTab() }
        .onChange(of: focusCoordinator.requestID) { applyRequestedTab() }
    }

    @ViewBuilder private var settingsContent: some View {
        switch selectedTab {
        case .general: GeneralSettingsPane(preferences: preferences)
        case .voiceTyping: VoiceTypingSettingsPane(preferences: preferences, voiceTyping: voiceTyping)
        case .captions: CaptionSettingsPane(preferences: preferences)
        case .models: ModelsSettingsPane(store: store, preferences: preferences)
        case .capture: CaptureSettingsPane(store: store, preferences: preferences)
        case .privacy: PrivacySettingsPane(store: store, preferences: preferences)
        }
    }

    private func applyRequestedTab() {
        if let tab = focusCoordinator.consumeRequestedTab() {
            selectedTab = tab
        }
    }
}

private struct VoiceTypingSettingsPane: View {
    @Bindable var preferences: UserPreferences
    @Bindable var voiceTyping: VoiceTypingController
    @State private var accessibilityTrusted = false

    var body: some View {
        Form {
            Section(preferences.text("Dictate into supported text fields", "対応する入力欄に音声入力")) {
                Toggle(preferences.text("Enable Voice Type", "音声入力を有効にする"), isOn: $preferences.voiceTypingEnabled)
                Picker(preferences.text("Shortcut", "ショートカット"), selection: $preferences.voiceTypingShortcut) {
                    ForEach(VoiceTypingShortcut.allCases) { shortcut in
                        Text(shortcut.displayName).tag(shortcut)
                    }
                }
                .disabled(!preferences.voiceTypingEnabled)
                LabeledContent(preferences.text("Model", "モデル")) {
                    VoiceModelControl(preferences: preferences, isActive: voiceTyping.state.isActive)
                }
                Picker(preferences.text("Spoken language", "話す言語"), selection: $preferences.voiceTypingLanguage) {
                    ForEach(preferences.voiceTypingModel == .phonon2 ? [.english] : SpeechLanguage.allCases) { language in
                        Text(language.nativeName).tag(language)
                    }
                }
                .disabled(!preferences.voiceTypingEnabled || voiceTyping.state.isActive)

                Text(modelDescription)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section(preferences.text("Access", "アクセス")) {
                LabeledContent(preferences.text("Accessibility", "アクセシビリティ")) {
                    Label(
                        accessibilityTrusted ? preferences.text("Ready", "準備完了") : preferences.text("Permission needed", "許可が必要"),
                        systemImage: accessibilityTrusted ? "checkmark.circle.fill" : "exclamationmark.circle"
                    )
                    .foregroundStyle(accessibilityTrusted ? .green : .secondary)
                }
                if !accessibilityTrusted {
                    Button(preferences.text("Allow in System Settings…", "システム設定で許可…")) {
                        voiceTyping.requestAccessibilityAccess()
                    }
                }
                if voiceTyping.shortcutRegistrationFailed {
                    Label(
                        preferences.text("That shortcut is already in use. Choose the other shortcut.", "そのショートカットは使用中です。もう一つを選んでください。"),
                        systemImage: "exclamationmark.triangle"
                    )
                    .font(.caption)
                    .foregroundStyle(.orange)
                }
            }

            if let error = voiceTyping.lastError {
                Section(preferences.text("Last dictation issue", "前回の音声入力の問題")) {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityElement(children: .combine)
                    Text(preferences.text(
                        "Check Accessibility access, then place the cursor in an editable text field and try your shortcut again.",
                        "アクセシビリティの許可を確認し、編集できる入力欄にカーソルを置いて、もう一度ショートカットを押してください。"
                    ))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }

            Section {
                Text(preferences.text(
                    "Place the cursor in a supported text field and press the shortcut. Press it again to stop, or Escape to undo this dictation. Password fields and Terminal prompts are not supported. Moving the cursor or editing the field stops dictation to preserve your edits.",
                    "対応する入力欄にカーソルを置き、ショートカットを押すと音声入力が始まります。もう一度押すと停止し、Escで取り消せます。パスワード欄とTerminalのプロンプトには対応していません。カーソルの移動や編集を検知すると、内容を保護するために停止します。"
                ))
                .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .onAppear { refreshAccessibilityAccess() }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            refreshAccessibilityAccess()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didBecomeKeyNotification)) { _ in
            refreshAccessibilityAccess()
        }
    }

    private func refreshAccessibilityAccess() {
        accessibilityTrusted = voiceTyping.hasAccessibilityAccess
    }

    private var modelDescription: String {
        switch preferences.voiceTypingModel {
        case .mimiWhisper:
            preferences.text(
                "Higher local accuracy. The first dictation can take a moment while Mimi loads the model.",
                "高精度のローカルモデルです。初回の音声入力では、モデルの読み込みに少し時間がかかることがあります。"
            )
        case .phonon2:
            preferences.text(
                "Fast local English dictation. Choose Mimi Speech or Apple Speech for Japanese.",
                "高速なローカル英語音声入力です。日本語には Mimi Speech または Apple Speech を選択してください。"
            )
        case .appleSpeech:
            preferences.text(
                "Faster startup using the English or Japanese speech asset managed by macOS.",
                "macOS が管理する英語または日本語の音声アセットを使い、より速く起動します。"
            )
        }
    }
}

private struct CaptionSettingsPane: View {
    @Bindable var preferences: UserPreferences

    var body: some View {
        Form {
            Section(preferences.text("Floating captions", "フローティング字幕")) {
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
            Section(preferences.text("Transcription", "文字起こし")) {
                Picker(preferences.text("Model", "モデル"), selection: $store.engineID) {
                    ForEach(TranscriptionEngineID.selectableCases) { engine in
                        Text(engine.displayName).tag(engine)
                    }
                }
                .disabled(store.controlsLocked || store.isModelSetupActive)

                Picker(preferences.text("Language", "言語"), selection: $store.languageMode) {
                    ForEach(store.selectableLanguageModes) { mode in
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

            Section(preferences.text("Local availability", "ローカルモデルの状態")) {
                ModelSetupStatusView(
                    readiness: store.selectedModelReadiness,
                    setupState: store.selectedModelSetupState
                )

                if let pack = store.modelPack {
                    LabeledContent(preferences.text("Storage", "ストレージ")) {
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
                            ? preferences.text("Mimi ready", "Mimi 準備完了")
                            : preferences.text("Apple Translation fallback", "Apple Translationに切り替え"),
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
                    Text(translationModelAvailable
                        ? preferences.text("73.4 MB, included with Mimi", "73.4 MB、Mimiに同梱")
                        : preferences.text("Language assets managed by macOS", "macOSが管理する言語データ"))
                        .foregroundStyle(.secondary)
                }
                Text(translationModelAvailable
                    ? preferences.text("Translations run entirely on this Mac with the Mimi model. No text is sent to a cloud translation service.", "翻訳はMimiモデルを使い、このMac上だけで実行されます。テキストはクラウド翻訳サービスには送信されません。")
                    : preferences.text("The Mimi translation model is unavailable. Apple Translation can prepare its language assets for on-device translation.", "Mimiの翻訳モデルを利用できません。Apple Translationの言語データを準備すると、端末内で翻訳できます。"))
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
                        store.engineID == .phonon2
                            ? preferences.text("Phonon 2 is an English-only preview. Use Mimi Speech or Apple Speech for Japanese.", "Phonon 2は英語専用のプレビューモデルです。日本語にはMimi SpeechまたはApple Speechを使用してください。")
                            : preferences.text("This local model remains a preview while Mimi evaluates accuracy, long-session stability, and thermal performance.", "このローカルモデルは、認識精度、長時間利用時の安定性、発熱を評価中のプレビューモデルです。"),
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
                Button(preferences.text("Pause Download", "ダウンロードを一時停止")) {
                    store.cancelSelectedModelInstall()
                }
            }

            if shouldShowAppleStatusCheck {
                Button(preferences.text("Check Status", "状態を確認")) {
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
            return preferences.text("About \(size) MB, managed by Mimi", "約\(size) MB、Mimiが管理")
        }
        return preferences.text("Language asset managed by macOS", "macOSが管理する言語データ")
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
            store.languageMode == .automatic
                ? preferences.text("Prepare English and Japanese", "英語と日本語を準備")
                : preferences.text("Prepare \(store.sourceLanguage.displayName)", "\(store.sourceLanguage.nativeName)を準備")
        case .whisperKitLargeV3Turbo: preferences.text("Download Mimi Speech", "Mimi Speechをダウンロード")
        case .parakeetJapanese: preferences.text("Download Parakeet Japanese", "Parakeet Japaneseをダウンロード")
        case .phonon2: preferences.text("Prepare Phonon 2", "Phonon 2を準備")
        case .nemotronStreamingExperimental: preferences.text("Download Nemotron", "Nemotronをダウンロード")
        case .qwen3StreamingExperimental: preferences.text("Download Qwen3-ASR", "Qwen3-ASRをダウンロード")
        }
        return retry ? preferences.text("Retry \(base)", "再試行: \(base)") : base
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
        case .whisperKitLargeV3Turbo: preferences.text("Remove Mimi Speech Download", "Mimi Speechのダウンロードを削除")
        case .parakeetJapanese: preferences.text("Remove Parakeet Japanese Download", "Parakeet Japaneseのダウンロードを削除")
        case .phonon2: preferences.text("Phonon 2 is bundled", "Phonon 2は同梱済み")
        case .nemotronStreamingExperimental: preferences.text("Remove Nemotron Download", "Nemotronのダウンロードを削除")
        case .qwen3StreamingExperimental: preferences.text("Remove Qwen3-ASR Download", "Qwen3-ASRのダウンロードを削除")
        case .appleSpeechAnalyzer: preferences.text("Remove Download", "ダウンロードを削除")
        }
    }
}

private struct CaptureSettingsPane: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences

    var body: some View {
        Form {
            Section(preferences.text("Device audio", "デバイスの音声")) {
                LabeledContent(preferences.text("Microphone", "マイク")) {
                    Label(preferences.text("Asked when recording starts", "録音開始時に許可を確認"), systemImage: "mic")
                        .foregroundStyle(.secondary)
                }

                LabeledContent(preferences.text("Audio Output", "音声出力")) {
                    Label(preferences.text("System Audio Recording", "システム音声の録音"), systemImage: "speaker.wave.2")
                        .foregroundStyle(.secondary)
                }

                Text(preferences.text("Choose the microphone or output device in the Session sidebar. Mimi asks for the matching macOS permission only when capture begins.", "セッションのサイドバーでマイクまたは出力デバイスを選択してください。Mimiは録音を開始するときにだけ、必要なmacOSの権限を確認します。"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section(preferences.text("Meeting audio", "会議の音声")) {
                LabeledContent(preferences.text("App Audio", "アプリの音声")) {
                    selectionStatus(for: .applicationAudio)
                }
                Button(preferences.text("Choose App…", "アプリを選択…")) {
                    chooseScreenAudio(.applicationAudio)
                }
                .disabled(store.controlsLocked)

                LabeledContent(preferences.text("Display Audio", "ディスプレイの音声")) {
                    selectionStatus(for: .systemAudio)
                }
                Button(preferences.text("Choose Display…", "ディスプレイを選択…")) {
                    chooseScreenAudio(.systemAudio)
                }
                .disabled(store.controlsLocked)

                Text(preferences.text("For Google Meet, choose Chrome; for Zoom, choose Zoom. Mimi captures only audio from the selected app or display, never screen pixels.", "Google MeetにはChrome、ZoomにはZoomを選択してください。Mimiは選択したアプリまたはディスプレイの音声だけを取り込み、画面の画像は記録しません。"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let message = store.lastError, store.source != .microphone {
                Section(preferences.text("Capture status", "録音の状態")) {
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
            Label(preferences.text("Selected", "選択済み"), systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .accessibilityLabel(preferences.text("\(source.displayName) selected", "\(source.displayName)を選択済み"))
        } else {
            Label(preferences.text("Not selected", "未選択"), systemImage: "circle")
                .foregroundStyle(.secondary)
                .accessibilityLabel(preferences.text("\(source.displayName) not selected", "\(source.displayName)は未選択"))
        }
    }

    private func chooseScreenAudio(_ source: AudioSource) {
        store.source = source
        store.selectScreenAudioContent()
    }
}

private struct PrivacySettingsPane: View {
    @Bindable var store: AppStore
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
                        "Your selected speech model, \(store.engineID.displayName), turns audio into text on this Mac. Voice Type uses \(preferences.voiceTypingModel.displayName).",
                        "選択した音声認識モデルの\(store.engineID.displayName)が、このMac上で音声をテキストに変換します。音声入力には\(preferences.voiceTypingModel.displayName)を使用します。"
                    ),
                    symbol: "waveform"
                )
                privacyRow(
                    preferences.text("Translation", "翻訳"),
                    detail: ExperimentalMLXTranslationConfiguration.resolved() != nil
                        ? preferences.text("The bundled Mimi model translates English and Japanese on this Mac.", "同梱されたMimiモデルが、このMac上で英語と日本語を翻訳します。")
                        : preferences.text("Apple Translation translates English and Japanese on this Mac using macOS language assets.", "Apple TranslationがmacOSの言語データを使い、このMac上で英語と日本語を翻訳します。"),
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
