import MimiCore
import SwiftUI

extension TranscriptionEngineID {
    var compactTitle: String {
        switch self {
        case .appleSpeechAnalyzer: "Apple Speech"
        case .whisperKitLargeV3Turbo: "Mimi Speech"
        case .parakeetJapanese: "Parakeet Japanese"
        case .phonon2: "Phonon 2"
        case .nemotronStreamingExperimental: "Nemotron"
        case .qwen3StreamingExperimental: "Qwen3 ASR"
        }
    }

    @MainActor func detail(_ preferences: UserPreferences) -> String {
        switch self {
        case .appleSpeechAnalyzer:
            preferences.text("English and Japanese · Managed by macOS", "英語と日本語 · macOSが管理")
        case .whisperKitLargeV3Turbo:
            preferences.text("English and Japanese · Custom model · 468 MB", "英語と日本語 · カスタムモデル · 468 MB")
        case .parakeetJapanese:
            preferences.text("Japanese only · Fast live preview · 482 MB", "日本語のみ · 高速ライブプレビュー · 482 MB")
        case .phonon2:
            preferences.text("English only · Fast local speech · 424 MB", "英語のみ · 高速なローカル音声認識 · 424 MB")
        case .nemotronStreamingExperimental, .qwen3StreamingExperimental:
            preferences.text("Experimental local model", "実験的なローカルモデル")
        }
    }
}

struct SpeechModelControl: View {
    @Bindable var store: AppStore
    let preferences: UserPreferences
    @State private var isPresented = false

    var body: some View {
        Button {
            isPresented.toggle()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "cpu").foregroundStyle(.secondary)
                Text(store.engineID.compactTitle)
                    .fontWeight(.medium)
                Image(systemName: "chevron.down").font(.caption2).foregroundStyle(.secondary)
            }
        }
        .buttonStyle(MimiQuietButtonStyle())
        .disabled(store.controlsLocked || store.isModelSetupActive)
        .accessibilityLabel(preferences.text("Choose speech model", "音声認識モデルを選択"))
        .accessibilityValue(store.engineID.displayName)
        .help(store.engineID.displayName)
        .popover(isPresented: $isPresented, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(preferences.text("Speech model", "音声認識モデル")).font(.headline)
                        Text(preferences.text("Choose how Mimi listens.", "Mimiの聞き取り方を選びます。"))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button { isPresented = false } label: { Image(systemName: "xmark") }
                        .buttonStyle(MimiQuietButtonStyle())
                        .accessibilityLabel(preferences.text("Close model selection", "モデル選択を閉じる"))
                }
                VStack(spacing: 6) {
                    ForEach(TranscriptionEngineID.selectableCases) { engine in
                        MimiChoiceRow(title: engine.compactTitle, detail: engine.detail(preferences), isSelected: store.engineID == engine) {
                            store.engineID = engine
                            isPresented = false
                        }
                        .disabled(store.controlsLocked || store.isModelSetupActive)
                    }
                }
                Divider()
                ModelSetupStatusView(readiness: store.selectedModelReadiness, setupState: store.selectedModelSetupState, compact: true)
                if store.canInstallSelectedModel {
                    Button(preferences.text("Prepare selected model", "選択したモデルを準備"), action: store.installSelectedModel)
                        .buttonStyle(.borderedProminent)
                }
            }
            .padding(20)
            .frame(width: MimiMetrics.popoverWidth)
        }
    }
}

struct CaptureInputControl: View {
    @Bindable var store: AppStore
    let preferences: UserPreferences
    @State private var isPresented = false

    private var sourceTitle: String {
        switch store.source {
        case .microphone: preferences.text("Microphone", "マイク")
        case .outputAudio: preferences.text("Mac audio", "Macの音声")
        case .applicationAudio: preferences.text("App audio", "アプリの音声")
        case .systemAudio: preferences.text("Display audio", "画面の音声")
        }
    }

    var body: some View {
        Button { isPresented.toggle() } label: {
            HStack(spacing: 6) {
                Image(systemName: store.source.symbolName).foregroundStyle(.secondary)
                Text(sourceTitle)
                Image(systemName: "chevron.down").font(.caption2).foregroundStyle(.secondary)
            }
        }
        .buttonStyle(MimiQuietButtonStyle())
        .disabled(store.controlsLocked)
        .accessibilityLabel(preferences.text("Choose audio input", "音声入力を選択"))
        .accessibilityValue(sourceTitle)
        .popover(isPresented: $isPresented, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 18) {
                Text(preferences.text("Audio input", "音声入力")).font(.headline)
                Picker(preferences.text("Listen to", "聞き取る音声"), selection: $store.source) {
                    ForEach(AudioSource.allCases) { source in
                        Label(source.displayName, systemImage: source.symbolName).tag(source)
                    }
                }
                .disabled(store.controlsLocked)
                switch store.source {
                case .microphone:
                    Picker(preferences.text("Microphone", "マイク"), selection: $store.selectedInputDeviceID) {
                        Text(preferences.text("System default", "システムのデフォルト")).tag(UInt32?.none)
                        ForEach(store.inputDevices) { device in Text(device.displayName).tag(Optional(device.id)) }
                    }
                    .disabled(store.controlsLocked)
                    Button(preferences.text("Refresh microphones", "マイクを更新"), action: store.refreshInputDevices)
                        .disabled(store.controlsLocked)
                case .outputAudio:
                    Picker(preferences.text("Output device", "出力デバイス"), selection: $store.selectedOutputDeviceID) {
                        Text(preferences.text("System default", "システムのデフォルト")).tag(UInt32?.none)
                        ForEach(store.outputDevices) { device in Text(device.displayName).tag(Optional(device.id)) }
                    }
                    .disabled(store.controlsLocked)
                    Button(preferences.text("Refresh outputs", "出力を更新"), action: store.refreshOutputDevices)
                        .disabled(store.controlsLocked)
                case .applicationAudio, .systemAudio:
                    ScreenAudioSelectionControl(store: store)
                }
                Text(preferences.text("Mimi listens only while a recording is running.", "Mimiは録音中だけ音声を聞き取ります。"))
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(20)
            .frame(width: 330)
            .buttonStyle(MimiQuietButtonStyle())
        }
    }
}

struct VoiceModelControl: View {
    @Bindable var preferences: UserPreferences
    let isActive: Bool
    @State private var isPresented = false

    private func engine(_ model: VoiceTypingModel) -> TranscriptionEngineID {
        switch model {
        case .appleSpeech: .appleSpeechAnalyzer
        case .mimiWhisper: .whisperKitLargeV3Turbo
        case .phonon2: .phonon2
        }
    }

    var body: some View {
        Button { isPresented.toggle() } label: {
            HStack(spacing: 6) {
                Text(engine(preferences.voiceTypingModel).compactTitle)
                Image(systemName: "chevron.down").font(.caption2).foregroundStyle(.secondary)
            }
        }
        .buttonStyle(MimiQuietButtonStyle())
        .disabled(isActive || !preferences.voiceTypingEnabled)
        .accessibilityLabel(preferences.text("Choose Voice Type model", "音声入力モデルを選択"))
        .accessibilityValue(preferences.voiceTypingModel.displayName)
        .popover(isPresented: $isPresented, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 14) {
                Text(preferences.text("Voice Type model", "音声入力モデル")).font(.headline)
                Text(preferences.text("Used when you dictate into another app.", "他のアプリへの音声入力で使用します。"))
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(VoiceTypingModel.allCases) { model in
                    MimiChoiceRow(title: engine(model).compactTitle, detail: engine(model).detail(preferences), isSelected: preferences.voiceTypingModel == model) {
                        preferences.voiceTypingModel = model
                        isPresented = false
                    }
                    .disabled(isActive)
                }
            }
            .padding(20).frame(width: MimiMetrics.popoverWidth)
        }
    }
}

struct CaptureLanguageControl: View {
    @Bindable var store: AppStore
    let preferences: UserPreferences
    @State private var isPresented = false

    private var title: String {
        store.languageMode == .automatic ? preferences.text("Auto language", "言語を自動判定") : store.sourceLanguage.nativeName
    }

    var body: some View {
        Button { isPresented.toggle() } label: {
            HStack(spacing: 6) {
                Image(systemName: "globe").foregroundStyle(.secondary)
                Text(title)
                Image(systemName: "chevron.down").font(.caption2).foregroundStyle(.secondary)
            }
        }
        .buttonStyle(MimiQuietButtonStyle())
        .fixedSize()
        .disabled(store.controlsLocked || store.isModelSetupActive)
        .accessibilityLabel(preferences.text("Spoken language", "話す言語"))
        .accessibilityValue(title)
        .popover(isPresented: $isPresented, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Text(preferences.text("Spoken language", "話す言語")).font(.headline)
                    Spacer()
                    Button { isPresented = false } label: { Image(systemName: "xmark") }
                        .buttonStyle(MimiQuietButtonStyle())
                        .accessibilityLabel(preferences.text("Close language selection", "言語選択を閉じる"))
                }
                VStack(spacing: 6) {
                    ForEach(store.selectableLanguageModes) { mode in
                        MimiChoiceRow(title: mode.displayName, detail: detail(mode), isSelected: store.languageMode == mode) {
                            store.languageMode = mode
                            isPresented = false
                        }
                        .disabled(store.controlsLocked || store.isModelSetupActive)
                    }
                }
            }
            .padding(20)
            .frame(width: MimiMetrics.popoverWidth)
        }
    }

    private func detail(_ mode: TranscriptionLanguageMode) -> String {
        switch mode {
        case .automatic: preferences.text("Switch between English and Japanese automatically.", "英語と日本語を自動で切り替えます。")
        case .english: preferences.text("Transcribe English speech.", "英語の音声を文字起こしします。")
        case .japanese: preferences.text("Transcribe Japanese speech.", "日本語の音声を文字起こしします。")
        }
    }
}

struct WorkspaceCaptureBar: View {
    @Bindable var store: AppStore
    let preferences: UserPreferences

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: store.isRecording ? "waveform" : "lock.shield")
                    .foregroundStyle(store.isRecording ? Color.red : Color.secondary)
                    .accessibilityHidden(true)
                Text(statusTitle)
                    .font(.callout.weight(.medium))
                Spacer()
                Text(preferences.text("Your audio stays on this Mac", "音声はこのMac内で処理します"))
                    .font(.caption).foregroundStyle(.secondary)
            }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 18) { controls; Spacer(minLength: 12); recordButton }
                VStack(alignment: .leading, spacing: 14) {
                    HStack(spacing: 18) { controls }
                    HStack { Spacer(); recordButton }
                }
            }
            if !store.selectedModelReadiness.canStart || store.isModelSetupActive {
                ModelSetupStatusView(readiness: store.selectedModelReadiness, setupState: store.selectedModelSetupState, compact: true)
            }
        }
        .mimiChrome(padding: 18, radius: 20)
        .padding(.horizontal, 24)
        .padding(.bottom, 20)
        .padding(.top, 12)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(preferences.text("Recording controls", "録音コントロール"))
    }

    private var controls: some View {
        HStack(spacing: 18) {
            CaptureInputControl(store: store, preferences: preferences)
            CaptureLanguageControl(store: store, preferences: preferences)
            SpeechModelControl(store: store, preferences: preferences)
        }
        .font(.callout)
        .fixedSize(horizontal: true, vertical: false)
    }

    private var statusTitle: String {
        if store.isRecording { return preferences.text("Listening on this Mac", "このMacで文字起こし中") }
        if store.controlsLocked { return store.recordingState.label }
        return store.canStartRecording ? preferences.text("Ready when you are", "いつでも録音できます") : preferences.text("Prepare your recording", "録音の準備")
    }

    private var recordButton: some View {
        Button(action: store.toggleRecording) {
            Label(store.isRecording ? preferences.text("Stop recording", "録音を停止") : preferences.text("Record", "録音を開始"), systemImage: store.isRecording ? "stop.fill" : "record.circle")
                .font(.callout.weight(.semibold))
                .padding(.horizontal, 6)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .tint(store.isRecording ? .red : .accentColor)
        .disabled(store.isRecording ? store.recordingState == .processing : !store.canStartRecording)
        .help(preferences.text("Continue recording in the open session", "開いているセッションで録音を続けます"))
    }
}
