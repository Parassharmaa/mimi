import AVFoundation
import AppKit
import MimiCore
import SwiftUI
@preconcurrency import Translation

enum OnboardingPreparationFixture {
    case live
    case preparing
    case ready
    case failed
}

private enum TranslationPreparationState: Equatable {
    case idle
    case checking
    case preparing
    case ready
    case failed(String)
}

struct OnboardingView: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences
    @Bindable var voiceTyping: VoiceTypingController
    @Environment(\.dismissWindow) private var dismissWindow
    @State private var step: Int
    @State private var microphoneStatus = AVCaptureDevice.authorizationStatus(for: .audio)
    @State private var startAtLogin = false
    @State private var preparedSpeechSelection: String?
    @State private var accessibilityTrusted = false
    @State private var hasEnteredReadyStep = false
    @State private var translationState: TranslationPreparationState = .idle
    @State private var translationSources: [SpeechLanguage] = []
    @State private var translationConfiguration: TranslationSession.Configuration?
    private let preparationFixture: OnboardingPreparationFixture

    init(
        store: AppStore,
        preferences: UserPreferences,
        voiceTyping: VoiceTypingController,
        initialStep: Int = 0,
        preparationFixture: OnboardingPreparationFixture = .live
    ) {
        self.store = store
        self.preferences = preferences
        self.voiceTyping = voiceTyping
        self.preparationFixture = preparationFixture
        _step = State(initialValue: min(4, max(0, initialStep)))
    }

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 28) {
                HStack(spacing: 10) {
                    Image(systemName: "ear").font(.title2).foregroundStyle(.tint).accessibilityHidden(true)
                    Text("Mimi").font(.title3.weight(.semibold))
                }
                VStack(alignment: .leading, spacing: 20) {
                    ForEach(0..<5, id: \.self) { index in
                        HStack(spacing: 10) {
                            Image(systemName: index < step ? "checkmark.circle.fill" : (index == step ? "circle.inset.filled" : "circle"))
                                .foregroundStyle(index <= step ? Color.accentColor : Color.secondary)
                                .accessibilityHidden(true)
                            Text(stepTitle(index)).font(.callout.weight(index == step ? .semibold : .regular))
                                .foregroundStyle(index <= step ? .primary : .secondary)
                        }
                        .accessibilityElement(children: .combine)
                        .accessibilityValue(index == step ? t("Current step", "現在のステップ") : "")
                    }
                }
                Spacer()
                Label(t("Private by design", "プライバシーを大切に"), systemImage: "lock.shield")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(24).frame(width: 180)
            .background(.quaternary.opacity(0.25))
            .accessibilityLabel(t("Setup progress", "設定の進行状況"))
            .accessibilityValue(t("Step \(step + 1) of 5", "5ステップ中\(step + 1)番目"))

            VStack(spacing: 0) {
                ScrollView {
                    Group {
                        switch step {
                        case 0: languageStep
                        case 1: listeningStep
                        case 2: preparationStep
                        case 3: permissionStep
                        default: readyStep
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(30)
                }
                HStack {
                    if step > 0 { Button(t("Back", "戻る")) { step -= 1 } }
                    Spacer()
                    Button(step == 4 ? t("Start using Mimi", "Mimiを使い始める") : t("Continue", "続ける"), action: advance)
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.defaultAction)
                        .disabled(step == 2 && !preparationIsReady)
                }
                .padding(24)
            }
        }
        .frame(width: 790, height: 590)
        .onAppear {
            startAtLogin = preferences.startsAtLogin
            refreshAccess()
            enterReadyStepIfNeeded()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            refreshAccess()
        }
        .task(id: step) {
            guard step == 2 else { return }
            await prepareLanguagesIfNeeded()
        }
        .onChange(of: step) { _, newStep in
            if newStep == 4 { enterReadyStepIfNeeded() }
        }
        .translationTask(translationConfiguration) { @MainActor session in
            await prepareCurrentTranslation(using: session)
        }
    }

    private var languageStep: some View {
        VStack(spacing: 24) {
            welcomeSymbol("character.bubble")
            VStack(spacing: 8) {
                Text("Welcome to Mimi · Mimiへようこそ")
                    .font(.largeTitle.weight(.semibold))
                Text("Choose the language Mimi uses for buttons and settings.\nボタンや設定で使用する言語を選んでください。")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            Picker("Interface language", selection: $preferences.interfaceLanguage) {
                ForEach(InterfaceLanguage.allCases) { language in
                    Text(language.nativeName).tag(language)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 300)
        }
    }

    private func stepTitle(_ index: Int) -> String {
        switch index {
        case 0: t("Welcome", "ようこそ")
        case 1: t("Listening", "聞き取り")
        case 2: t("Models", "モデル")
        case 3: t("Access", "アクセス")
        default: t("Ready", "準備完了")
        }
    }

    private var listeningStep: some View {
        VStack(spacing: 24) {
            welcomeSymbol("waveform.and.mic")
            title(t("What should Mimi listen to?", "Mimiで何を聞き取りますか？"),
                  t("You can change this any time.", "この設定はいつでも変更できます。"))
            Picker(t("Audio source", "音声ソース"), selection: $store.source) {
                Text(t("My microphone", "マイク")).tag(AudioSource.microphone)
                Text(t("Sound playing on this Mac", "このMacで再生中の音声")).tag(AudioSource.outputAudio)
                Text(t("One app, such as Zoom or Chrome", "ZoomやChromeなど1つのアプリ")).tag(AudioSource.applicationAudio)
            }
            .pickerStyle(.radioGroup)
            .frame(maxWidth: 380, alignment: .leading)
            Picker(t("Speech model", "音声認識モデル"), selection: $store.engineID) {
                ForEach(TranscriptionEngineID.selectableCases) { engine in
                    Text(engine.displayName).tag(engine)
                }
            }
            .disabled(store.controlsLocked)
            .frame(maxWidth: 380)
            Picker(t("Spoken language", "話す言語"), selection: $store.languageMode) {
                ForEach(store.selectableLanguageModes) { mode in
                    Text(mode.displayName).tag(mode)
                }
            }
            .disabled(store.controlsLocked)
            .frame(maxWidth: 380)
            if store.engineID == .phonon2 {
                Text(t("Phonon 2 supports English only. Choose Mimi Speech or Apple Speech for Japanese.", "Phonon 2は英語専用です。日本語にはMimi SpeechまたはApple Speechを選択してください。"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var permissionStep: some View {
        VStack(spacing: 22) {
            welcomeSymbol("hand.raised")
            title(t("Your audio stays on this Mac", "音声はこのMac内で処理されます"),
                  t("Mimi asks only for access needed by the source you choose.", "選んだ音声ソースに必要な権限だけをリクエストします。"))
            VStack(spacing: 12) {
                permissionRow(
                    symbol: "mic",
                    title: t("Microphone", "マイク"),
                    detail: microphonePermissionDetail,
                    isGranted: microphonePermission == .granted,
                    canRequest: microphonePermission == .requestable || microphonePermission == .denied
                )
                permissionRow(
                    symbol: "speaker.wave.2",
                    title: t("Mac audio", "Macの音声"),
                    detail: store.source == .microphone
                        ? t("Not required for microphone recording", "マイク録音では不要です")
                        : t("macOS asks when you choose Mac or app audio", "Macやアプリの音声を選ぶときに、macOSが許可を求めます"),
                    isGranted: false,
                    canRequest: false
                )
            }
        }
    }

    private var preparationStep: some View {
        VStack(spacing: 22) {
            welcomeSymbol("arrow.down.circle")
            title(
                t("Prepare your models", "モデルを準備"),
                t(
                    "Only the models you choose need setup. Included models work offline without an Apple language download.",
                    "選択したモデルだけを準備します。同梱モデルはAppleの言語データをダウンロードせず、オフラインで利用できます。"
                )
            )
            VStack(spacing: 12) {
                preparationRow(
                    symbol: "waveform",
                    title: store.engineID.displayName,
                    state: speechPreparationState
                )
                preparationRow(
                    symbol: "character.bubble",
                    title: t("English ↔ Japanese translation", "英語 ↔ 日本語の翻訳"),
                    state: translationPreparationDisplayState
                )
            }
            Text(t(
                "Apple language downloads are needed only when an Apple model is selected.",
                "Appleの言語データは、Appleのモデルを使う場合だけ必要です。"
            ))
            .font(.caption)
            .foregroundStyle(.secondary)
            if preparationHasFailed {
                Button(t("Try again", "もう一度試す")) { retryPreparation() }
            }
            if store.canInstallSelectedModel && !requirements.usesAppleSpeech {
                Button(t("Download selected speech model", "選択した音声認識モデルをダウンロード")) {
                    store.installSelectedModel()
                }
            }
            Button(t("Choose a different model", "別のモデルを選ぶ")) {
                if store.canCancelSelectedModelInstall {
                    store.cancelSelectedModelInstall()
                }
                step = 1
            }
        }
    }

    private var readyStep: some View {
        VStack(spacing: 18) {
            welcomeSymbol("checkmark.circle")
            title(t("Mimi is ready", "Mimiの準備ができました"),
                  t("It lives in the menu bar and can show captions over other apps.", "メニューバーから使え、他のアプリの上に字幕も表示できます。"))
            Toggle(t("Open Mimi when I log in", "ログイン時にMimiを開く"), isOn: $startAtLogin)
                .frame(width: 320, alignment: .leading)
            VStack(alignment: .leading, spacing: 8) {
                Toggle(t("Dictate into text fields", "入力欄に音声入力"), isOn: $preferences.voiceTypingEnabled)
                if preferences.voiceTypingEnabled {
                    LabeledContent(t("Voice Type model", "音声入力モデル")) {
                        VoiceModelControl(preferences: preferences, isActive: voiceTyping.state.isActive)
                    }
                    Picker(t("Spoken language", "話す言語"), selection: $preferences.voiceTypingLanguage) {
                        ForEach(preferences.voiceTypingModel == .phonon2 ? [.english] : SpeechLanguage.allCases) { language in
                            Text(language.nativeName).tag(language)
                        }
                    }
                    .disabled(voiceTyping.state.isActive)
                    HStack {
                        Text(t("Shortcut", "ショートカット"))
                        Spacer()
                        Picker("Shortcut", selection: $preferences.voiceTypingShortcut) {
                            ForEach(VoiceTypingShortcut.allCases) { shortcut in
                                Text(shortcut.displayName).tag(shortcut)
                            }
                        }
                        .labelsHidden()
                        if !accessibilityTrusted {
                            Button(t("Allow…", "許可…")) { voiceTyping.requestAccessibilityAccess() }
                        }
                    }
                    Text(t(
                        "Voice Type uses your microphone and its selected model. A different model may need separate setup in Settings. Accessibility access is used only to insert text into the field you selected.",
                        "音声入力はマイクと選択したモデルを使用します。別のモデルを選ぶ場合は、設定で準備が必要なことがあります。アクセシビリティ権限は選択中の入力欄に文字を入力するためだけに使用します。"
                    ))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    if !voiceTyping.shortcutRegistered {
                        Label(
                            t("That shortcut is already in use. Choose the other one.", "このショートカットは別のアプリで使用されています。もう一方を選んでください。"),
                            systemImage: "exclamationmark.triangle"
                        )
                        .font(.caption)
                        .foregroundStyle(.orange)
                    }
                }
            }
            .frame(width: 380, alignment: .leading)
            .padding(12)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
            if let error = preferences.loginItemError {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .frame(maxWidth: 380)
            }
        }
    }

    private func welcomeSymbol(_ name: String) -> some View {
        Image(systemName: name)
            .font(.system(size: 42, weight: .medium))
            .foregroundStyle(.tint)
            .frame(width: 84, height: 84)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .accessibilityHidden(true)
    }

    private func title(_ title: String, _ detail: String) -> some View {
        VStack(spacing: 7) {
            Text(title).font(.title.weight(.semibold))
            Text(detail).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
    }

    private func permissionRow(
        symbol: String,
        title: String,
        detail: String,
        isGranted: Bool,
        canRequest: Bool
    ) -> some View {
        HStack(spacing: 14) {
            Image(systemName: symbol).foregroundStyle(.secondary).frame(width: 24)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if canRequest {
                Button(microphoneStatus == .denied || microphoneStatus == .restricted
                    ? t("Open Settings…", "設定を開く…") : t("Allow…", "許可…"), action: requestMicrophone)
            } else if isGranted {
                Label(t("Allowed", "許可済み"), systemImage: "checkmark.circle.fill")
                    .labelStyle(.iconOnly)
                    .foregroundStyle(.green)
            } else {
                Image(systemName: "minus.circle")
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
            }
        }
        .padding(14)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
        .frame(width: 440)
        .accessibilityElement(children: .contain)
    }

    private func preparationRow(
        symbol: String,
        title: String,
        state: (detail: String, kind: PreparationDisplayKind)
    ) -> some View {
        HStack(spacing: 14) {
            Image(systemName: symbol).foregroundStyle(.secondary).frame(width: 24)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                Text(state.detail).font(.caption).foregroundStyle(state.kind == .failed ? .orange : .secondary)
            }
            Spacer()
            switch state.kind {
            case .working:
                ProgressView().controlSize(.small)
            case .ready:
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            case .failed:
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            }
        }
        .padding(14)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
        .frame(width: 460)
    }

    private enum PreparationDisplayKind: Equatable { case working, ready, failed }

    private var speechPreparationState: (detail: String, kind: PreparationDisplayKind) {
        if preparationFixture == .ready { return (t("Ready", "準備完了"), .ready) }
        if preparationFixture == .preparing { return (t("Downloading…", "ダウンロード中…"), .working) }
        if preparationFixture == .failed { return (t("Couldn’t finish the download", "ダウンロードを完了できませんでした"), .failed) }
        return switch store.selectedModelReadiness {
        case .ready: (requirements.usesAppleSpeech ? t("Ready", "準備完了") : t("Local model ready. Works offline.", "ローカルモデルの準備完了。オフラインで利用できます。"), .ready)
        case .needsDownload, .unavailable:
            (store.selectedModelReadiness.message ?? t("Speech model unavailable", "音声認識モデルを利用できません"), .failed)
        case .checking:
            (t("Checking…", "確認中…"), .working)
        case .downloading, .experimental:
            (t("Downloading…", "ダウンロード中…"), .working)
        }
    }

    private var translationPreparationDisplayState: (detail: String, kind: PreparationDisplayKind) {
        if preparationFixture == .ready { return (t("Ready", "準備完了"), .ready) }
        if preparationFixture == .preparing { return (t("Downloading…", "ダウンロード中…"), .working) }
        if preparationFixture == .failed { return (t("Couldn’t finish the download", "ダウンロードを完了できませんでした"), .failed) }
        if !requirements.usesAppleTranslation {
            return (t("Mimi model included. Ready offline.", "Mimiモデルを同梱。オフラインで利用できます。"), .ready)
        }
        return switch translationState {
        case .idle, .checking: (t("Checking…", "確認中…"), .working)
        case .preparing: (t("Downloading…", "ダウンロード中…"), .working)
        case .ready: (t("Ready", "準備完了"), .ready)
        case let .failed(message): (message, .failed)
        }
    }

    private var preparationIsReady: Bool {
        if preparationFixture == .ready { return true }
        guard preparationFixture == .live else { return false }
        return speechPreparationState.kind == .ready && translationPreparationDisplayState.kind == .ready
    }

    private var preparationHasFailed: Bool {
        speechPreparationState.kind == .failed || translationPreparationDisplayState.kind == .failed
    }

    private var requirements: OnboardingRequirements {
        .init(
            engineID: store.engineID,
            hasLocalTranslation: ExperimentalMLXTranslationConfiguration.resolved() != nil
        )
    }

    private var microphonePermissionDetail: String {
        switch microphonePermission {
        case .notRequired: t("Not required for the selected audio source", "選択した音声ソースでは不要です")
        case .granted: t("Ready", "準備完了")
        case .denied: t("Allow Microphone access in System Settings", "システム設定でマイクへのアクセスを許可してください")
        case .requestable: t("Needed for microphone transcription", "マイクの文字起こしに必要です")
        case .unknown: t("Permission status unknown", "権限の状態を確認できません")
        }
    }

    private var microphonePermission: OnboardingMicrophonePermission {
        .init(source: store.source, authorizationStatus: microphoneStatus)
    }

    private func refreshAccess() {
        microphoneStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        accessibilityTrusted = voiceTyping.hasAccessibilityAccess
    }

    private func requestMicrophone() {
        if microphoneStatus == .denied || microphoneStatus == .restricted {
            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                NSWorkspace.shared.open(url)
            }
            return
        }
        Task {
            _ = await AVCaptureDevice.requestAccess(for: .audio)
            microphoneStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        }
    }

    private func advance() {
        guard step == 4 else { step += 1; return }
        if preferences.startsAtLogin != startAtLogin {
            preferences.setStartsAtLogin(startAtLogin)
            if preferences.loginItemError != nil { return }
        }
        preferences.completedOnboarding = true
        dismissWindow(id: "onboarding")
        NSApplication.shared.keyWindow?.close()
    }

    private func enterReadyStepIfNeeded() {
        guard step == 4, !hasEnteredReadyStep, !preferences.completedOnboarding else { return }
        hasEnteredReadyStep = true
        preferences.configureVoiceTypingForFirstUse(
            engineID: store.engineID,
            language: store.sourceLanguage,
            speechIsReady: store.selectedModelReadiness.canStart
        )
        preferences.voiceTypingEnabled = true
    }

    private func prepareLanguagesIfNeeded() async {
        guard preparationFixture == .live else { return }
        let speechSelection = "\(store.engineID.rawValue)/\(store.languageMode.rawValue)"
        if preparedSpeechSelection != speechSelection {
            preparedSpeechSelection = speechSelection
            if requirements.usesAppleSpeech {
                store.installSelectedModel()
            } else {
                store.refreshSelectedModelReadiness()
            }
        }
        guard requirements.usesAppleTranslation else {
            translationConfiguration = nil
            translationSources = []
            translationState = .ready
            return
        }
        guard translationState == .idle else { return }
        translationState = .checking
        let availability: LanguageAvailability
        if #available(macOS 26.4, *) {
            availability = LanguageAvailability(preferredStrategy: .lowLatency)
        } else {
            availability = LanguageAvailability()
        }
        var missingSources: [SpeechLanguage] = []
        for source in SpeechLanguage.allCases {
            let status = await availability.status(
                from: Locale.Language(identifier: source.rawValue),
                to: Locale.Language(identifier: source.translationTarget.rawValue)
            )
            switch status {
            case .installed:
                continue
            case .supported:
                missingSources.append(source)
            case .unsupported:
                translationState = .failed(t(
                    "English and Japanese translation is not available on this Mac.",
                    "このMacでは英語と日本語の翻訳を利用できません。"
                ))
                return
            @unknown default:
                translationState = .failed(t(
                    "Mimi couldn’t confirm translation availability.",
                    "翻訳を利用できるか確認できませんでした。"
                ))
                return
            }
        }
        translationSources = missingSources
        guard let first = missingSources.first else {
            translationState = .ready
            return
        }
        translationState = .preparing
        translationConfiguration = translationConfiguration(for: first)
    }

    private func prepareCurrentTranslation(using session: TranslationSession) async {
        guard preparationFixture == .live, requirements.usesAppleTranslation,
              let source = translationSources.first else { return }
        do {
            try await session.prepareTranslation()
            guard translationSources.first == source else { return }
            translationSources.removeFirst()
            if let next = translationSources.first {
                translationConfiguration = translationConfiguration(for: next)
            } else {
                translationConfiguration = nil
                translationState = .ready
            }
        } catch {
            guard !(error is CancellationError) else { return }
            translationConfiguration = nil
            translationState = .failed(t(
                "Couldn’t finish translation setup. Try again.",
                "翻訳の準備を完了できませんでした。もう一度お試しください。"
            ))
        }
    }

    private func translationConfiguration(for source: SpeechLanguage) -> TranslationSession.Configuration {
        let sourceLanguage = Locale.Language(identifier: source.rawValue)
        let targetLanguage = Locale.Language(identifier: source.translationTarget.rawValue)
        if #available(macOS 26.4, *) {
            return .init(source: sourceLanguage, target: targetLanguage, preferredStrategy: .lowLatency)
        }
        return .init(source: sourceLanguage, target: targetLanguage)
    }

    private func retryPreparation() {
        preparedSpeechSelection = nil
        translationState = .idle
        translationSources = []
        translationConfiguration = nil
        Task { await prepareLanguagesIfNeeded() }
    }

    private func t(_ english: String, _ japanese: String) -> String {
        preferences.text(english, japanese)
    }
}
