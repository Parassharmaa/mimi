import MimiCore
import SwiftUI

struct TranscriptWindow: View {
    @Bindable var store: AppStore
    @State private var isConfirmingClear = false
    private let fixtureTranslation: String?
    private let initiallyFollowingLatest: Bool

    init(
        store: AppStore,
        isConfirmingClear: Bool = false,
        fixtureTranslation: String? = nil,
        initiallyFollowingLatest: Bool = true
    ) {
        self.store = store
        self.fixtureTranslation = fixtureTranslation
        self.initiallyFollowingLatest = initiallyFollowingLatest
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
                    } else if store.source == .outputAudio {
                        Picker("Output", selection: $store.selectedOutputDeviceID) {
                            Text("System Default").tag(UInt32?.none)
                            ForEach(store.outputDevices) { device in
                                Text(device.displayName).tag(Optional(device.id))
                            }
                        }
                        .disabled(store.controlsLocked)

                        Button("Refresh Output Devices") {
                            store.refreshOutputDevices()
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

                let followsAppleSpeech = store.isRecording && store.engineID == .appleSpeechAnalyzer
                let translationSourceText = followsAppleSpeech
                    ? store.document.realtimeTranslationContext(for: store.sourceLanguage)
                    : store.document.finalizedText(for: store.sourceLanguage)

                if store.translationMode == .translateFinalSegments {
                    GeometryReader { proxy in
                        HStack(alignment: .top, spacing: 12) {
                            TranscriptLanguagePane(
                                document: store.document,
                                language: store.sourceLanguage,
                                initiallyFollowingLatest: initiallyFollowingLatest
                            )
                            InlineTranslationView(
                                sourceText: translationSourceText,
                                sourceLanguage: store.sourceLanguage,
                                isLive: followsAppleSpeech,
                                fillsAvailableSpace: true,
                                fixtureTranslation: fixtureTranslation,
                                initiallyFollowingLatest: initiallyFollowingLatest
                            )
                        }
                        .frame(width: proxy.size.width, height: proxy.size.height)
                    }
                } else {
                    FollowLatestScrollView(
                        contentVersion: store.document.renderedText,
                        initiallyFollowing: initiallyFollowingLatest
                    ) {
                        TranscriptContentView(
                            document: store.document,
                            emptyMessage: "Your local transcript will appear here.",
                            font: .title3
                        )
                        .padding(.vertical, 4)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
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

private struct TranscriptLanguagePane: View {
    let document: TranscriptDocument
    let language: SpeechLanguage
    let initiallyFollowingLatest: Bool

    var body: some View {
        let sourceDocument = TranscriptDocument(
            segments: document.segments.filter { $0.language == language },
            liveText: document.liveText
        )
        VStack(alignment: .leading, spacing: 6) {
            Label(language.nativeName, systemImage: "waveform")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            FollowLatestScrollView(
                contentVersion: sourceDocument.renderedText,
                initiallyFollowing: initiallyFollowingLatest
            ) {
                TranscriptContentView(
                    document: sourceDocument,
                    emptyMessage: "Speech will appear here.",
                    font: .title3
                )
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
