import MimiCore
import SwiftUI

struct TranscriptWindow: View {
    @Bindable var store: AppStore
    @State private var isConfirmingClear = false

    init(store: AppStore, isConfirmingClear: Bool = false) {
        self.store = store
        _isConfirmingClear = State(initialValue: isConfirmingClear)
    }

    var body: some View {
        NavigationSplitView {
            Form {
                Section("Session") {
                    Picker("Input", selection: $store.source) {
                        ForEach(AudioSource.allCases) { source in
                            Text(source.displayName).tag(source)
                        }
                    }
                    .disabled(store.controlsLocked)

                    if store.source == .microphone {
                        Picker("Microphone", selection: $store.selectedInputDeviceID) {
                            Text("System Default").tag(UInt32?.none)
                            ForEach(store.inputDevices) { device in
                                Text(device.displayName).tag(Optional(device.id))
                            }
                        }
                        .disabled(store.controlsLocked)

                        Button("Refresh Input Devices") {
                            store.refreshInputDevices()
                        }
                        .disabled(store.controlsLocked)
                    } else {
                        ScreenAudioSelectionControl(store: store)
                    }
                }

                Section("Language and model") {
                    Picker("Language", selection: $store.sourceLanguage) {
                        ForEach(SpeechLanguage.allCases) { language in
                            Text(language.nativeName).tag(language)
                        }
                    }
                    .disabled(store.controlsLocked || store.isModelSetupActive)

                    Picker("Model", selection: $store.engineID) {
                        ForEach(TranscriptionEngineID.allCases) { engine in
                            Text(engine.displayName).tag(engine)
                        }
                    }
                    .disabled(store.controlsLocked || store.isModelSetupActive)

                    if let message = store.selectedModelReadiness.message {
                        Label(message, systemImage: "info.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Translation") {
                    Picker("Mode", selection: $store.translationMode) {
                        ForEach(TranslationMode.allCases) { mode in
                            Text(mode.displayName).tag(mode)
                        }
                    }
                    .disabled(store.controlsLocked)
                }
            }
            .formStyle(.grouped)
            .navigationSplitViewColumnWidth(min: 220, ideal: 250, max: 300)
            .navigationTitle("Session")
        } detail: {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Label(store.isRecording ? "Recording \(store.source.displayName.lowercased()) locally" : store.recordingState.label, systemImage: store.menuBarSymbolName)
                        .foregroundStyle(store.isRecording ? .red : .primary)
                        .accessibilityElement(children: .combine)
                    Spacer()
                    Button(store.isRecording ? "Stop Recording" : "Start Recording") {
                        store.toggleRecording()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(store.isRecording ? .red : .accentColor)
                    .disabled(store.isRecording ? store.recordingState == .processing : !store.canStartRecording)
                }

                if let message = store.lastError {
                    Label(message, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .accessibilityLabel("Recording warning: \(message)")
                }

                ScrollView {
                    TranscriptContentView(
                        document: store.document,
                        emptyMessage: "Your local transcript will appear here.",
                        font: .title3
                    )
                    .padding(.vertical, 4)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                let translationSourceText = store.document.renderedText(
                    for: store.sourceLanguage,
                    includingLiveText: true
                )
                if store.translationMode == .translateFinalSegments,
                   !translationSourceText.isEmpty {
                    InlineTranslationView(
                        sourceText: translationSourceText,
                        sourceLanguage: store.sourceLanguage,
                        isLive: store.isRecording
                    )
                }
            }
            .padding()
            .frame(minWidth: 600, minHeight: 420, alignment: .topLeading)
            .navigationTitle("Transcript")
            .toolbar {
                ToolbarItemGroup(placement: .primaryAction) {
                    if isConfirmingClear {
                        Text("Clear saved transcript?")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Button("Cancel") {
                            isConfirmingClear = false
                        }
                        .keyboardShortcut(.cancelAction)
                        .buttonStyle(.bordered)

                        Button("Clear", role: .destructive) {
                            store.clearTranscript()
                            isConfirmingClear = false
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.red)
                    } else {
                        Button {
                            store.copyTranscript()
                        } label: {
                            Label("Copy Transcript", systemImage: "doc.on.doc")
                        }
                        .keyboardShortcut("c", modifiers: [.command, .shift])
                        .disabled(store.document.renderedText.isEmpty)

                        Button(role: .destructive) {
                            isConfirmingClear = true
                        } label: {
                            Label("Clear Transcript", systemImage: "trash")
                        }
                        .accessibilityHint("Shows an inline confirmation before removing the local transcript")
                        .disabled(store.document.renderedText.isEmpty)
                    }
                }
            }
        }
    }
}
