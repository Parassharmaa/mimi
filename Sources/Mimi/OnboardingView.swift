import AVFoundation
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
    @State private var speechPreparationStarted = false
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
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                ForEach(0..<5, id: \.self) { index in
                    Capsule()
                        .fill(index <= step ? Color.accentColor : Color.secondary.opacity(0.2))
                        .frame(height: 4)
                }
            }
            .padding(.horizontal, 28)
            .padding(.top, 24)

            Group {
                switch step {
                case 0: languageStep
                case 1: listeningStep
                case 2: preparationStep
                case 3: permissionStep
                default: readyStep
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(32)

            Divider()
            HStack {
                if step > 0 {
                    Button(t("Back", "戻る")) { step -= 1 }
                }
                Spacer()
                Button(step == 4 ? t("Start using Mimi", "Mimiを使い始める") : t("Continue", "続ける")) {
                    advance()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(step == 2 && !preparationIsReady)
            }
            .padding(20)
        }
        .frame(width: 620, height: 560)
        .onAppear { startAtLogin = preferences.startsAtLogin }
        .task(id: step) {
            guard step == 2 else { return }
            await prepareLanguagesIfNeeded()
        }
        .onChange(of: step) { _, newStep in
            if newStep == 4, !preferences.completedOnboarding {
                preferences.voiceTypingEnabled = true
            }
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
            Toggle(t("Recognize English and Japanese automatically", "英語と日本語を自動で認識"), isOn: autoLanguageBinding)
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
                    detail: microphoneStatus == .authorized ? t("Ready", "準備完了") : t("Needed for microphone transcription", "マイクの文字起こしに必要です"),
                    canRequest: store.source == .microphone && microphoneStatus != .authorized
                )
                permissionRow(
                    symbol: "speaker.wave.2",
                    title: t("Mac audio", "Macの音声"),
                    detail: t("macOS asks the first time you choose Mac or app audio", "初めてMacやアプリの音声を選ぶときに、macOSが許可を求めます"),
                    canRequest: false
                )
            }
        }
    }

    private var preparationStep: some View {
        VStack(spacing: 22) {
            welcomeSymbol("arrow.down.circle")
            title(
                t("Preparing English + Japanese", "英語と日本語の準備をしています"),
                t(
                    "Mimi downloads the speech and translation languages now, so recording starts smoothly later.",
                    "文字起こしと翻訳に必要な言語データを今ダウンロードして、すぐに使えるようにします。"
                )
            )
            VStack(spacing: 12) {
                preparationRow(
                    symbol: "waveform",
                    title: t("Live transcription", "リアルタイム文字起こし"),
                    state: speechPreparationState
                )
                preparationRow(
                    symbol: "character.bubble",
                    title: t("English ↔ Japanese translation", "英語 ↔ 日本語の翻訳"),
                    state: translationPreparationDisplayState
                )
            }
            Text(t(
                "Downloads are managed by macOS and stay on this Mac.",
                "ダウンロードはmacOSが管理し、このMacに保存されます。"
            ))
            .font(.caption)
            .foregroundStyle(.secondary)
            if preparationHasFailed {
                Button(t("Try again", "もう一度試す")) { retryPreparation() }
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
                Toggle(t("Type anywhere by speaking", "声でどこにでも入力"), isOn: $preferences.voiceTypingEnabled)
                if preferences.voiceTypingEnabled {
                    HStack {
                        Text(t("Shortcut", "ショートカット"))
                        Spacer()
                        Picker("Shortcut", selection: $preferences.voiceTypingShortcut) {
                            ForEach(VoiceTypingShortcut.allCases) { shortcut in
                                Text(shortcut.displayName).tag(shortcut)
                            }
                        }
                        .labelsHidden()
                        if !voiceTyping.hasAccessibilityAccess {
                            Button(t("Allow…", "許可…")) { voiceTyping.requestAccessibilityAccess() }
                        }
                    }
                    Text(t(
                        "Mimi needs Accessibility access only to insert text into the field you selected.",
                        "アクセシビリティ権限は、選択中の入力欄に文字を入力するためだけに使用します。"
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
                Button(t("Allow…", "許可…"), action: requestMicrophone)
            } else {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            }
        }
        .padding(14)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
        .frame(width: 440)
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
        return switch store.bilingualAppleSpeechReadiness {
        case .ready: (t("Ready", "準備完了"), .ready)
        case .needsDownload, .unavailable:
            (t("Couldn’t finish speech setup. Try again.", "音声の準備を完了できませんでした。もう一度お試しください。"), .failed)
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

    private var autoLanguageBinding: Binding<Bool> {
        Binding(
            get: { store.languageMode == .automatic },
            set: { store.languageMode = $0 ? .automatic : TranscriptionLanguageMode(language: store.sourceLanguage) }
        )
    }

    private func requestMicrophone() {
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

    private func prepareLanguagesIfNeeded() async {
        guard preparationFixture == .live else { return }
        if !speechPreparationStarted {
            speechPreparationStarted = true
            Task { await store.prepareBilingualAppleSpeechNow() }
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
        guard preparationFixture == .live, let source = translationSources.first else { return }
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
        speechPreparationStarted = false
        translationState = .idle
        translationSources = []
        translationConfiguration = nil
        Task { await prepareLanguagesIfNeeded() }
    }

    private func t(_ english: String, _ japanese: String) -> String {
        preferences.text(english, japanese)
    }
}
