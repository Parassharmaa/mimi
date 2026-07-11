import MimiCore
import SwiftUI

struct SettingsView: View {
    @Bindable var store: AppStore

    init(store: AppStore) {
        self.store = store
    }

    var body: some View {
        TabView {
            Form {
                Section("Local model") {
                    Picker("Transcription model", selection: $store.engineID) {
                        ForEach(TranscriptionEngineID.allCases) { engine in
                            Text(engine.displayName).tag(engine)
                        }
                    }
                    .disabled(store.controlsLocked || store.isModelSetupActive)

                    Picker("Language", selection: $store.sourceLanguage) {
                        ForEach(SpeechLanguage.allCases) { language in
                            Text(language.nativeName).tag(language)
                        }
                    }
                    .disabled(store.controlsLocked || store.isModelSetupActive)

                    VStack(alignment: .leading, spacing: 7) {
                        Text("Status")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        ModelSetupStatusView(
                            readiness: store.selectedModelReadiness,
                            setupState: store.selectedModelSetupState
                        )
                    }

                    if let pack = store.modelPack {
                        Text(pack.recommendation)
                            .foregroundStyle(.secondary)
                        if let size = pack.estimatedDownloadMB {
                            Text("Estimated download: about \(size) MB")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else if pack.ownership == .systemManaged {
                            Text("Apple manages this language asset at the system level. Download it explicitly before recording.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    HStack(spacing: 8) {
                        if store.canInstallSelectedModel {
                            Button(modelActionTitle) {
                                store.installSelectedModel()
                            }
                        }

                        if store.canCancelSelectedModelInstall {
                            Button("Cancel Download") {
                                store.cancelSelectedModelInstall()
                            }
                        }

                        if shouldShowAppleStatusCheck {
                            Button("Check Apple Speech Status") {
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

                    if case .unavailable = store.selectedModelReadiness,
                       store.engineID == .appleSpeechAnalyzer {
                        Button("Use Whisper Large-v3 Instead") {
                            store.engineID = .whisperKitLargeV3Turbo
                        }
                        .disabled(store.controlsLocked || store.isModelSetupActive)
                    }
                }

                Section("Privacy") {
                    Text("Mimi stores finalized transcript text locally. Temporary source audio exists only during a local accuracy pass and is deleted after the session finishes.")
                    Text("Apple Translation processes text on-device; macOS may collect non-content performance metadata for the API.")
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
            .padding()
            .tabItem { Label("Models", systemImage: "cpu") }

            Form {
                Section("Microphone") {
                    LabeledContent("Microphone access") {
                        Label("Requested when recording starts", systemImage: "mic")
                            .foregroundStyle(.secondary)
                    }
                    Text("Choose a microphone from the Session panel. Mimi asks macOS for access only when you begin a microphone recording, then shows the active local recording state in its menu-bar control and transcript window.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("App and display audio") {
                    LabeledContent("Selected App Audio") {
                        selectionStatus(for: .applicationAudio)
                    }
                    Button("Choose App Audio…") {
                        chooseScreenAudio(.applicationAudio)
                    }
                    .disabled(store.controlsLocked)

                    LabeledContent("Selected Display Audio") {
                        selectionStatus(for: .systemAudio)
                    }
                    Button("Choose Display Audio…") {
                        chooseScreenAudio(.systemAudio)
                    }
                    .disabled(store.controlsLocked)

                    Text("These buttons show macOS's content picker and request Screen Recording access only when you choose one. Mimi registers only an audio output for the selected app or display; it does not save screen images.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("For Google Meet, select Chrome; for Zoom, select Zoom. Display audio follows the display you choose, which is useful when your meeting audio is playing through that display.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    if let message = store.lastError, store.source != .microphone {
                        Label(message, systemImage: "info.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Future output route") {
                    Text("A direct Core Audio tap for raw all-device speaker output is planned separately. Today's capture lane is intentionally scoped to an app or display chosen in macOS's picker.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
            .padding()
            .tabItem { Label("Capture", systemImage: "waveform") }
        }
        .frame(width: 560, height: 520)
        .background(SettingsWindowRegistrar())
    }

    private var modelActionTitle: String {
        let retry = switch store.selectedModelSetupState {
        case .cancelled, .failed:
            true
        case .idle, .checking, .downloading, .prewarming, .removing, .waitingForSystem:
            false
        }
        let base: String
        switch store.engineID {
        case .appleSpeechAnalyzer:
            base = "Download \(store.sourceLanguage.displayName) Apple Speech"
        case .whisperKitLargeV3Turbo:
            base = "Download Whisper Large-v3"
        case .nemotronStreamingExperimental:
            base = "Download Nemotron MLX"
        }
        return retry ? "Retry \(base)" : base
    }

    private var shouldShowAppleStatusCheck: Bool {
        guard store.engineID == .appleSpeechAnalyzer else { return false }
        return switch store.selectedModelReadiness {
        case .checking, .downloading, .ready:
            true
        case .needsDownload, .unavailable, .experimental:
            false
        }
    }

    private var removeButtonTitle: String {
        switch store.engineID {
        case .whisperKitLargeV3Turbo: "Remove Downloaded Whisper Model"
        case .nemotronStreamingExperimental: "Remove Downloaded Nemotron Model"
        case .appleSpeechAnalyzer: "Remove Downloaded Model"
        }
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
