import AppKit
import MimiCore
import SwiftUI

struct MenuBarView: View {
    @Bindable var store: AppStore
    @Environment(\.openSettings) private var openSettings
    @Environment(\.openWindow) private var openWindow
    @State private var isConfirmingClear = false

    private let initiallyFollowingLatest: Bool

    init(
        store: AppStore,
        isConfirmingClear: Bool = false,
        initiallyFollowingLatest: Bool = true
    ) {
        self.store = store
        self.initiallyFollowingLatest = initiallyFollowingLatest
        _isConfirmingClear = State(initialValue: isConfirmingClear)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MimiMetrics.sectionSpacing) {
            MimiStatusHeader(state: store.recordingState, source: store.source)

            recordingButton

            if let message = store.lastError {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .mimiCard(padding: 10)
                    .accessibilityLabel("Recording warning: \(message)")
            }

            configuration

            if needsModelSetupAction {
                modelSetup
            }

            transcriptPreview
            footer
        }
        .padding(16)
        .frame(width: 400)
    }

    private var recordingButton: some View {
        Button {
            store.toggleRecording()
        } label: {
            Label(
                store.isRecording ? "Stop Recording" : "Start Recording",
                systemImage: store.isRecording ? "stop.fill" : "record.circle"
            )
            .frame(maxWidth: .infinity)
        }
        .keyboardShortcut(.return, modifiers: [])
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .tint(store.isRecording ? .red : .accentColor)
        .disabled(store.isRecording ? store.recordingState == .processing : !store.canStartRecording)
    }

    private var configuration: some View {
        VStack(spacing: 0) {
            MimiControlRow(
                "Input",
                detail: sourceSummary,
                symbol: store.source.symbolName
            ) {
                Picker("Input", selection: $store.source) {
                    ForEach(AudioSource.allCases) { source in
                        Text(source.displayName).tag(source)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 176)
                .disabled(store.controlsLocked)
                .accessibilityLabel("Input source")
            }

            sourceConfiguration
            rowDivider

            MimiControlRow("Language", symbol: store.sourceLanguage.symbolName) {
                Picker("Language", selection: $store.sourceLanguage) {
                    ForEach(SpeechLanguage.allCases) { language in
                        Text(language.nativeName).tag(language)
                    }
                }
                .pickerStyle(.menu)
                .disabled(store.controlsLocked || store.isModelSetupActive)
                .accessibilityLabel("Transcription language")
            }

            rowDivider

            MimiControlRow(
                "Model",
                detail: store.selectedModelReadiness.canStart ? "Local model ready" : "Setup required",
                symbol: "cpu"
            ) {
                Picker("Model", selection: $store.engineID) {
                    ForEach(TranscriptionEngineID.allCases) { engine in
                        Text(engine.displayName).tag(engine)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 176)
                .disabled(store.controlsLocked || store.isModelSetupActive)
                .accessibilityLabel("Transcription model")
            }

            rowDivider

            MimiControlRow("Translation", symbol: "translate") {
                Picker("Translation", selection: $store.translationMode) {
                    ForEach(TranslationMode.allCases) { mode in
                        Text(mode == .off ? "Off" : "English ↔ Japanese").tag(mode)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 176)
                .disabled(store.controlsLocked)
                .accessibilityLabel("Translation mode")
            }
        }
        .mimiCard()
    }

    @ViewBuilder
    private var sourceConfiguration: some View {
        switch store.source {
        case .microphone:
            rowDivider
            devicePickerRow(
                title: "Microphone",
                selection: $store.selectedInputDeviceID,
                devices: store.inputDevices.map { ($0.id, $0.displayName) },
                refreshLabel: "Refresh microphone inputs",
                refresh: store.refreshInputDevices
            )
        case .outputAudio:
            rowDivider
            devicePickerRow(
                title: "Output",
                selection: $store.selectedOutputDeviceID,
                devices: store.outputDevices.map { ($0.id, $0.displayName) },
                refreshLabel: "Refresh audio outputs",
                refresh: store.refreshOutputDevices
            )
        case .applicationAudio, .systemAudio:
            rowDivider
            ScreenAudioSelectionControl(store: store, compact: true)
                .padding(.vertical, 8)
        }
    }

    private func devicePickerRow(
        title: String,
        selection: Binding<UInt32?>,
        devices: [(UInt32, String)],
        refreshLabel: String,
        refresh: @escaping () -> Void
    ) -> some View {
        HStack(spacing: 8) {
            Picker(title, selection: selection) {
                Text("System Default").tag(UInt32?.none)
                ForEach(devices, id: \.0) { id, name in
                    Text(name).tag(Optional(id))
                }
            }
            .pickerStyle(.menu)
            .disabled(store.controlsLocked)

            Button(action: refresh) {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.borderless)
            .help(refreshLabel)
            .accessibilityLabel(refreshLabel)
            .disabled(store.controlsLocked)
        }
        .padding(.leading, 28)
        .padding(.vertical, 8)
    }

    private var modelSetup: some View {
        VStack(alignment: .leading, spacing: 9) {
            MimiSectionLabel("Model setup", symbol: "arrow.down.circle")
            ModelSetupStatusView(
                readiness: store.selectedModelReadiness,
                setupState: store.selectedModelSetupState,
                compact: true
            )

            Button("Open Model Settings…", action: openMimiSettings)
                .buttonStyle(.bordered)
                .accessibilityHint("Opens the model setup window")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .mimiCard()
    }

    private var transcriptPreview: some View {
        VStack(alignment: .leading, spacing: 8) {
            transcriptHeader

            FollowLatestScrollView(
                contentVersion: store.document.renderedText,
                initiallyFollowing: initiallyFollowingLatest
            ) {
                TranscriptContentView(
                    document: store.document,
                    emptyMessage: "Start recording to see local transcription here."
                )
                .padding(.vertical, 2)
            }
            .frame(height: 118)
        }
        .mimiCard()
    }

    @ViewBuilder
    private var transcriptHeader: some View {
        if isConfirmingClear {
            HStack(spacing: 8) {
                Text("Clear transcript?")
                    .font(.caption.weight(.semibold))
                Spacer()
                Button("Cancel") { setClearConfirmation(false) }
                    .keyboardShortcut(.cancelAction)
                Button("Clear", role: .destructive) {
                    store.clearTranscript()
                    setClearConfirmation(false)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
            }
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Clear local transcript confirmation")
        } else {
            HStack(spacing: 8) {
                MimiSectionLabel("Latest transcript", symbol: "text.alignleft")
                Spacer()
                Button {
                    store.copyTranscript()
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.borderless)
                .help("Copy transcript")
                .accessibilityLabel("Copy transcript")
                .disabled(store.document.renderedText.isEmpty)

                Button {
                    setClearConfirmation(true)
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless)
                .help("Clear saved transcript")
                .accessibilityLabel("Clear saved transcript")
                .disabled(store.document.renderedText.isEmpty)
            }
        }
    }

    private var footer: some View {
        HStack(spacing: 14) {
            Button {
                openWindow(id: "transcript")
            } label: {
                Label("Transcript", systemImage: "rectangle.on.rectangle")
            }

            Button(action: openMimiSettings) {
                Label("Settings", systemImage: "gearshape")
            }

            Spacer()

            Button("Quit") {
                NSApplication.shared.terminate(nil)
            }
        }
        .buttonStyle(.borderless)
        .font(.footnote)
    }

    private var rowDivider: some View {
        Divider().padding(.leading, 28)
    }

    private var needsModelSetupAction: Bool {
        !store.selectedModelReadiness.canStart || store.selectedModelSetupState != .idle
    }

    private var sourceSummary: String {
        switch store.source {
        case .microphone: "Selected input device"
        case .outputAudio: "System mix from one output"
        case .applicationAudio: "Audio from one app"
        case .systemAudio: "Audio from one display"
        }
    }

    private func setClearConfirmation(_ confirming: Bool) {
        isConfirmingClear = confirming
    }

    private func openMimiSettings() {
        SettingsWindowFocusCoordinator.shared.requestFocus()
        openSettings()
    }
}
