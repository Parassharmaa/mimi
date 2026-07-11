import MimiCore
import SwiftUI

enum SettingsTab: Hashable {
    case general
    case captions
    case models
    case capture
    case privacy
}

struct SettingsView: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences
    @State private var selectedTab: SettingsTab

    init(store: AppStore, preferences: UserPreferences = UserPreferences(), initialTab: SettingsTab = .general) {
        self.store = store
        self.preferences = preferences
        _selectedTab = State(initialValue: initialTab)
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            Tab(preferences.text("General", "一般"), systemImage: "gearshape", value: .general) {
                GeneralSettingsPane(preferences: preferences)
            }

            Tab(preferences.text("Captions", "字幕"), systemImage: "captions.bubble", value: .captions) {
                CaptionSettingsPane(preferences: preferences)
            }

            Tab(preferences.text("Languages", "言語"), systemImage: "character.book.closed", value: .models) {
                ModelsSettingsPane(store: store)
            }

            Tab(preferences.text("Audio", "音声"), systemImage: "waveform", value: .capture) {
                CaptureSettingsPane(store: store)
            }

            Tab(preferences.text("Privacy", "プライバシー"), systemImage: "hand.raised", value: .privacy) {
                PrivacySettingsPane()
            }
        }
        .scenePadding()
        .frame(width: 620, height: 540)
        .background(SettingsWindowRegistrar())
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
            }

            Section {
                Text(preferences.text(
                    "Floating captions stay visible while you work in Zoom, Chrome, or another app. Turn off click-through temporarily if you need to move or inspect them.",
                    "ZoomやChromeなどを使用中も字幕が表示されます。字幕を操作するときは、クリック透過を一時的にオフにしてください。"
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
    var body: some View {
        Form {
            Section("Local by design") {
                privacyRow(
                    "Transcript",
                    detail: "Finalized text is stored only on this Mac.",
                    symbol: "text.alignleft"
                )
                privacyRow(
                    "Transcription",
                    detail: "Apple Speech turns audio into text on this Mac.",
                    symbol: "waveform"
                )
                privacyRow(
                    "Translation",
                    detail: "Apple Translation processes English and Japanese on-device.",
                    symbol: "translate"
                )
            }

            Section("Temporary audio") {
                Text("Mimi processes working audio in memory and does not keep a source-audio recording after the session.")
                    .foregroundStyle(.secondary)
            }

            Section("System services") {
                Text("macOS owns permission prompts and Apple language assets. Apple frameworks may collect non-content performance metadata, but Mimi does not send transcript or source-audio content to a cloud service.")
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
