import AppKit
import MimiCore
import SwiftUI

struct MenuBarView: View {
    @Bindable var store: AppStore
    @Environment(\.openWindow) private var openWindow
    @Environment(\.openSettings) private var openSettings
    @State private var showingClearConfirmation = false

    init(store: AppStore) {
        self.store = store
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            controls
            transcriptPreview

            if store.translationMode == .translateFinalSegments,
               !store.document.finalizedText(for: store.sourceLanguage).isEmpty {
                Label("Translation is available in the Transcript window.", systemImage: "translate")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Divider()

            HStack(spacing: 10) {
                Button {
                    openWindow(id: "transcript")
                } label: {
                    Label("Open Transcript", systemImage: "text.alignleft")
                }
                .buttonStyle(.borderless)

                Button {
                    openMimiSettings()
                } label: {
                    Label("Settings", systemImage: "gearshape")
                }
                .buttonStyle(.borderless)

                Spacer()

                Button("Quit") {
                    NSApplication.shared.terminate(nil)
                }
                .buttonStyle(.borderless)
            }
            .font(.footnote)
        }
        .padding(16)
        .frame(width: 430)
        .confirmationDialog("Clear local transcript?", isPresented: $showingClearConfirmation, titleVisibility: .visible) {
            Button("Clear Transcript", role: .destructive) {
                store.clearTranscript()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This removes the stored transcript from this Mac. Mimi does not retain source audio after a completed session.")
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: store.menuBarSymbolName)
                .font(.title2)
                .foregroundStyle(statusColor)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 3) {
                Text("Mimi")
                    .font(.headline)
                Text(store.isRecording ? "Recording \(store.source.displayName.lowercased()) locally" : store.recordingState.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if store.isRecording {
                Text("REC")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 4)
                    .background(.red.opacity(0.12), in: Capsule())
                    .accessibilityLabel("Recording \(store.source.displayName.lowercased()) locally")
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Input", selection: $store.source) {
                ForEach(AudioSource.allCases) { source in
                    Text(source.displayName).tag(source)
                }
            }
            .pickerStyle(.menu)
            .disabled(store.controlsLocked)

            if store.source == .microphone {
                HStack(spacing: 8) {
                    Picker("Microphone", selection: $store.selectedInputDeviceID) {
                        Text("System Default").tag(UInt32?.none)
                        ForEach(store.inputDevices) { device in
                            Text(device.displayName).tag(Optional(device.id))
                        }
                    }
                    .pickerStyle(.menu)
                    .disabled(store.controlsLocked)

                    Button {
                        store.refreshInputDevices()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .buttonStyle(.borderless)
                    .help("Refresh microphone inputs")
                    .accessibilityLabel("Refresh microphone inputs")
                    .disabled(store.controlsLocked)
                }
            }

            ScreenAudioSelectionControl(store: store, compact: true)

            Picker("Language", selection: $store.sourceLanguage) {
                ForEach(SpeechLanguage.allCases) { language in
                    Text(language.nativeName).tag(language)
                }
            }
            .pickerStyle(.menu)
            .disabled(store.controlsLocked || store.isModelSetupActive)

            Picker("Model", selection: $store.engineID) {
                ForEach(TranscriptionEngineID.allCases) { engine in
                    Text(engine.displayName).tag(engine)
                }
            }
            .pickerStyle(.menu)
            .disabled(store.controlsLocked || store.isModelSetupActive)

            modelStatus

            HStack(spacing: 8) {
                Button(store.isRecording ? "Stop Recording" : "Start Recording") {
                    store.toggleRecording()
                }
                .keyboardShortcut(.return, modifiers: [])
                .buttonStyle(.borderedProminent)
                .tint(store.isRecording ? .red : .accentColor)
                .disabled(store.isRecording ? store.recordingState == .processing : !store.canStartRecording)

                if needsModelSetupAction {
                    Button("Set Up Model…") {
                        openMimiSettings()
                    }
                    .buttonStyle(.bordered)
                    .accessibilityHint("Opens the model setup window")
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var modelStatus: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let pack = store.modelPack {
                Text(pack.recommendation)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            ModelSetupStatusView(
                readiness: store.selectedModelReadiness,
                setupState: store.selectedModelSetupState,
                compact: true
            )
        }
    }

    private var transcriptPreview: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Latest transcript")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    store.copyTranscript()
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.plain)
                .help("Copy transcript")
                .accessibilityLabel("Copy transcript")
                .disabled(store.document.renderedText.isEmpty)

                Button {
                    showingClearConfirmation = true
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .help("Clear transcript")
                .accessibilityLabel("Clear transcript")
                .disabled(store.document.renderedText.isEmpty)
            }

            ScrollView {
                TranscriptContentView(
                    document: store.document,
                    emptyMessage: "Choose a ready local model, then start speaking."
                )
                .padding(10)
            }
            .frame(height: 130)
            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
    }

    private var needsModelSetupAction: Bool {
        !store.selectedModelReadiness.canStart || store.selectedModelSetupState != .idle
    }

    private var statusColor: Color {
        store.isRecording ? .red : .accentColor
    }

    private func openMimiSettings() {
        // This is the real Mimi Settings window, not a passive app
        // activation. Registering the native window lets a menu-bar utility
        // reliably bring it in front even when another app currently owns
        // focus.
        SettingsWindowFocusCoordinator.shared.requestFocus()
        openSettings()
    }
}
