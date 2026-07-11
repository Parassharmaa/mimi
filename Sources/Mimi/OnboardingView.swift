import AVFoundation
import MimiCore
import SwiftUI

struct OnboardingView: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences
    @Environment(\.dismissWindow) private var dismissWindow
    @State private var step = 0
    @State private var microphoneStatus = AVCaptureDevice.authorizationStatus(for: .audio)
    @State private var startAtLogin = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                ForEach(0..<4, id: \.self) { index in
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
                case 2: permissionStep
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
                Button(step == 3 ? t("Start using Mimi", "Mimiを使い始める") : t("Continue", "続ける")) {
                    advance()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
            .padding(20)
        }
        .frame(width: 620, height: 500)
        .onAppear { startAtLogin = preferences.startsAtLogin }
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
                  t("Mimi asks only for access needed by the source you choose.", "選択した音声に必要なアクセスだけを求めます。"))
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
                    detail: t("macOS will ask when you first choose app or output audio", "アプリや出力音声を初めて選ぶときにmacOSが確認します"),
                    canRequest: false
                )
            }
        }
    }

    private var readyStep: some View {
        VStack(spacing: 24) {
            welcomeSymbol("checkmark.circle")
            title(t("Mimi is ready", "Mimiの準備ができました"),
                  t("It lives in the menu bar and can show captions over other apps.", "メニューバーから使え、他のアプリの上に字幕も表示できます。"))
            Toggle(t("Open Mimi when I log in", "ログイン時にMimiを開く"), isOn: $startAtLogin)
                .frame(width: 320, alignment: .leading)
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
        guard step == 3 else { step += 1; return }
        if preferences.startsAtLogin != startAtLogin {
            preferences.setStartsAtLogin(startAtLogin)
            if preferences.loginItemError != nil { return }
        }
        preferences.completedOnboarding = true
        dismissWindow(id: "onboarding")
        NSApplication.shared.keyWindow?.close()
    }

    private func t(_ english: String, _ japanese: String) -> String {
        preferences.text(english, japanese)
    }
}
