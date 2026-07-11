import MimiCore
import SwiftUI

enum SettingsTab: Hashable {
    case models
    case capture
    case privacy
}

struct SettingsView: View {
    @Bindable var store: AppStore
    @State private var selectedTab: SettingsTab

    init(store: AppStore, initialTab: SettingsTab = .models) {
        self.store = store
        _selectedTab = State(initialValue: initialTab)
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            Tab("Models", systemImage: "cpu", value: .models) {
                ModelsSettingsPane(store: store)
            }

            Tab("Capture", systemImage: "waveform", value: .capture) {
                CaptureSettingsPane(store: store)
            }

            Tab("Privacy", systemImage: "hand.raised", value: .privacy) {
                PrivacySettingsPane()
            }
        }
        .scenePadding()
        .frame(width: 620, height: 540)
        .background(SettingsWindowRegistrar())
    }
}

private struct ModelsSettingsPane: View {
    @Bindable var store: AppStore

    var body: some View {
        Form {
            Section("Transcription") {
                Picker("Model", selection: $store.engineID) {
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

        if case .unavailable = store.selectedModelReadiness,
           store.engineID == .appleSpeechAnalyzer {
            Button("Use Whisper Large-v3 Instead") {
                store.engineID = .whisperKitLargeV3Turbo
            }
            .disabled(store.controlsLocked || store.isModelSetupActive)
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
        case .appleSpeechAnalyzer: "Download \(store.sourceLanguage.displayName)"
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
                    detail: "Apple Speech, Whisper, Qwen, and Nemotron run locally.",
                    symbol: "waveform"
                )
                privacyRow(
                    "Translation",
                    detail: "Apple Translation processes English and Japanese on-device.",
                    symbol: "translate"
                )
            }

            Section("Temporary audio") {
                Text("Apple Speech and live MLX models process working audio in memory without retaining a source file. Whisper writes a temporary source file for its post-stop accuracy pass, then Mimi deletes it when the session finishes.")
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
