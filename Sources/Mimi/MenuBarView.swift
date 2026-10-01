import AppKit
import MimiCore
import SwiftUI

struct MenuBarView: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences
    @Environment(\.openSettings) private var openSettings

    init(store: AppStore, preferences: UserPreferences = UserPreferences()) {
        self.store = store
        self.preferences = preferences
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                Image(systemName: "ear").font(.title2).foregroundStyle(.tint).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Mimi").font(.headline)
                    Text(store.isRecording ? t("Listening on this Mac", "このMacで文字起こし中") : t("Your words, on your Mac", "言葉を、このMacで。"))
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "lock.shield").foregroundStyle(.secondary)
                    .accessibilityLabel(t("Local processing", "ローカル処理"))
            }

            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    CaptureInputControl(store: store, preferences: preferences)
                    Spacer()
                    CaptureLanguageControl(store: store, preferences: preferences)
                }
                SpeechModelControl(store: store, preferences: preferences)
                Button(action: store.toggleRecording) {
                    Label(store.isRecording ? t("Stop recording", "録音を停止") : t("Start recording", "録音を開始"), systemImage: store.isRecording ? "stop.fill" : "record.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .tint(store.isRecording ? .red : .accentColor)
                .disabled(store.isRecording ? store.recordingState == .processing : !store.canStartRecording)
                ModelSetupStatusView(readiness: store.selectedModelReadiness, setupState: store.selectedModelSetupState, compact: true)
            }
            .mimiChrome(padding: 16, radius: 16)

            if let error = store.lastError {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Toggle(t("Floating captions", "フローティング字幕"), isOn: $preferences.floatingCaptionsEnabled).toggleStyle(.switch)
            Button { AppWindowCoordinator.shared.showTranscript() } label: {
                Label(t("Open workspace", "ワークスペースを開く"), systemImage: "rectangle.on.rectangle")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(MimiQuietButtonStyle())
            Divider()
            HStack {
                Button {
                    SettingsWindowFocusCoordinator.shared.requestFocus(tab: .voiceTyping)
                    openSettings()
                } label: { Label(t("Voice Type", "音声入力"), systemImage: "keyboard") }
                .buttonStyle(MimiQuietButtonStyle())
                Spacer()
                Button {
                    SettingsWindowFocusCoordinator.shared.requestFocus()
                    openSettings()
                } label: { Image(systemName: "gearshape") }
                .buttonStyle(MimiQuietButtonStyle())
                .accessibilityLabel(t("Settings", "設定"))
                Menu {
                    Button(t("Quit Mimi", "Mimiを終了")) { NSApplication.shared.terminate(nil) }
                } label: { Image(systemName: "ellipsis") }
                .menuStyle(.borderlessButton).fixedSize()
                .accessibilityLabel(t("More actions", "その他の操作"))
            }
            .foregroundStyle(.secondary)
        }
        .buttonStyle(MimiQuietButtonStyle())
        .padding(22)
        .frame(width: 360)
    }

    private func t(_ english: String, _ japanese: String) -> String { preferences.text(english, japanese) }
}
