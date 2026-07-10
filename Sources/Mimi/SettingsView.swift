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
                    .disabled(store.controlsLocked)

                    LabeledContent("Status") {
                        Label(modelStatusText, systemImage: modelStatusSymbol)
                            .foregroundStyle(store.selectedModelReadiness.canStart ? .primary : .secondary)
                    }

                    LabeledContent("Language pack") {
                        Text(store.sourceLanguage.nativeName)
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

                    Button(downloadButtonTitle) {
                        store.installSelectedModel()
                    }
                    .disabled(!store.canInstallSelectedModel)

                    if store.canRemoveSelectedModel {
                        Button(removeButtonTitle, role: .destructive) {
                            store.removeSelectedModel()
                        }
                        .disabled(store.controlsLocked)
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
                    LabeledContent("Microphone") {
                        Label("Ready", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    }
                    Text("Choose a microphone from the Session panel. Mimi makes the active local recording state visible in its menu-bar control and transcript window.")
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
    }

    private var modelStatusText: String {
        store.selectedModelReadiness.message ?? "Ready to record locally"
    }

    private var modelStatusSymbol: String {
        store.selectedModelReadiness.canStart ? "checkmark.circle" : "arrow.down.circle"
    }

    private var downloadButtonTitle: String {
        switch store.engineID {
        case .appleSpeechAnalyzer: "Download Apple Language Asset"
        case .whisperKitLargeV3Turbo: "Download Whisper Large-v3"
        case .nemotronStreamingExperimental: "Download Nemotron MLX"
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
