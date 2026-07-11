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
            SessionSidebar(store: store)
                .navigationSplitViewColumnWidth(min: 220, ideal: 250, max: 310)
                .navigationTitle("Session")
        } detail: {
            VStack(spacing: 0) {
                sessionStrip

                if let message = store.lastError {
                    inlineNotice(message, symbol: "exclamationmark.triangle.fill", tint: .orange)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 10)
                }

                if isConfirmingClear {
                    clearConfirmation
                        .padding(.horizontal, 16)
                        .padding(.bottom, 10)
                }

                transcriptContent
            }
            .frame(minWidth: 560, minHeight: 440)
            .navigationTitle("Transcript")
            .toolbar { transcriptToolbar }
        }
        .navigationSplitViewStyle(.balanced)
    }

    private var sessionStrip: some View {
        HStack(spacing: 10) {
            Image(systemName: store.menuBarSymbolName)
                .foregroundStyle(store.isRecording ? .red : .accentColor)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 1) {
                Text(store.isRecording ? "Listening locally" : store.recordingState.label)
                    .font(.callout.weight(.semibold))
                Text("\(store.source.displayName) · \(store.sourceLanguage.nativeName) · \(store.engineID.displayName)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            if store.translationMode == .translateFinalSegments {
                Label("Live translation", systemImage: "translate")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 11)
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder
    private var transcriptContent: some View {
        let followsAppleSpeech = store.isRecording && store.engineID == .appleSpeechAnalyzer
        let translationSourceText = followsAppleSpeech
            ? store.document.realtimeTranslationContext(for: store.sourceLanguage)
            : store.document.finalizedText(for: store.sourceLanguage)

        if store.translationMode == .translateFinalSegments {
            HSplitView {
                TranscriptLanguagePane(
                    document: store.document,
                    language: store.sourceLanguage,
                    initiallyFollowingLatest: initiallyFollowingLatest
                )
                .frame(minWidth: 250)

                InlineTranslationView(
                    sourceText: translationSourceText,
                    sourceLanguage: store.sourceLanguage,
                    isLive: followsAppleSpeech,
                    fillsAvailableSpace: true,
                    fixtureTranslation: fixtureTranslation,
                    initiallyFollowingLatest: initiallyFollowingLatest
                )
                .frame(minWidth: 250)
            }
        } else {
            TranscriptLanguagePane(
                document: store.document,
                language: nil,
                initiallyFollowingLatest: initiallyFollowingLatest
            )
        }
    }

    private var clearConfirmation: some View {
        HStack(spacing: 10) {
            Image(systemName: "trash")
                .foregroundStyle(.red)
                .accessibilityHidden(true)
            Text("Clear the saved transcript?")
                .font(.callout.weight(.semibold))
            Text("This can’t be undone.")
                .font(.caption)
                .foregroundStyle(.secondary)
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
        .mimiCard(padding: 10)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Clear saved transcript confirmation")
    }

    @ToolbarContentBuilder
    private var transcriptToolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            Button {
                store.copyTranscript()
            } label: {
                Label("Copy Transcript", systemImage: "doc.on.doc")
            }
            .keyboardShortcut("c", modifiers: [.command, .shift])
            .disabled(store.document.renderedText.isEmpty)

            Button(role: .destructive) {
                setClearConfirmation(true)
            } label: {
                Label("Clear Transcript", systemImage: "trash")
            }
            .keyboardShortcut(.delete, modifiers: [.command, .option])
            .disabled(store.document.renderedText.isEmpty || isConfirmingClear)

            Button {
                store.toggleRecording()
            } label: {
                Label(
                    store.isRecording ? "Stop Recording" : "Start Recording",
                    systemImage: store.isRecording ? "stop.fill" : "record.circle"
                )
            }
            .buttonStyle(.borderedProminent)
            .tint(store.isRecording ? .red : .accentColor)
            .disabled(store.isRecording ? store.recordingState == .processing : !store.canStartRecording)
        }
    }

    private func inlineNotice(_ text: String, symbol: String, tint: Color) -> some View {
        Label(text, systemImage: symbol)
            .font(.caption)
            .foregroundStyle(tint)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .mimiCard(padding: 10)
            .accessibilityLabel("Recording warning: \(text)")
    }

    private func setClearConfirmation(_ confirming: Bool) {
        isConfirmingClear = confirming
    }
}

private struct SessionSidebar: View {
    @Bindable var store: AppStore

    var body: some View {
        Form {
            Section("Input") {
                Picker("Source", selection: $store.source) {
                    ForEach(AudioSource.allCases) { source in
                        Label(source.displayName, systemImage: source.symbolName).tag(source)
                    }
                }
                .disabled(store.controlsLocked)

                sourceConfiguration
            }

            Section("Transcription") {
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

                ModelSetupStatusView(
                    readiness: store.selectedModelReadiness,
                    setupState: store.selectedModelSetupState,
                    compact: true
                )
            }

            Section("Translation") {
                Picker("Mode", selection: $store.translationMode) {
                    ForEach(TranslationMode.allCases) { mode in
                        Text(mode == .off ? "Off" : "English ↔ Japanese").tag(mode)
                    }
                }
                .disabled(store.controlsLocked)
            }
        }
        .formStyle(.grouped)
    }

    @ViewBuilder
    private var sourceConfiguration: some View {
        switch store.source {
        case .microphone:
            Picker("Microphone", selection: $store.selectedInputDeviceID) {
                Text("System Default").tag(UInt32?.none)
                ForEach(store.inputDevices) { device in
                    Text(device.displayName).tag(Optional(device.id))
                }
            }
            .disabled(store.controlsLocked)
            Button("Refresh Microphones", action: store.refreshInputDevices)
                .disabled(store.controlsLocked)
        case .outputAudio:
            Picker("Output", selection: $store.selectedOutputDeviceID) {
                Text("System Default").tag(UInt32?.none)
                ForEach(store.outputDevices) { device in
                    Text(device.displayName).tag(Optional(device.id))
                }
            }
            .disabled(store.controlsLocked)
            Button("Refresh Outputs", action: store.refreshOutputDevices)
                .disabled(store.controlsLocked)
        case .applicationAudio, .systemAudio:
            ScreenAudioSelectionControl(store: store)
        }
    }
}

private struct TranscriptLanguagePane: View {
    let document: TranscriptDocument
    let language: SpeechLanguage?
    let initiallyFollowingLatest: Bool

    private var displayedDocument: TranscriptDocument {
        guard let language else { return document }
        return TranscriptDocument(
            segments: document.segments.filter { $0.language == language },
            liveText: document.liveText
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 7) {
                Label(language?.nativeName ?? "Transcript", systemImage: language?.symbolName ?? "text.alignleft")
                    .font(.callout.weight(.semibold))
                Spacer()
                if !document.liveText.isEmpty {
                    Text("Listening")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(.quaternary, in: Capsule())
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            if displayedDocument.segments.isEmpty && displayedDocument.liveText.isEmpty {
                ContentUnavailableView(
                    "No Transcript Yet",
                    systemImage: "waveform",
                    description: Text("Start recording and speech will appear here locally.")
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                FollowLatestScrollView(
                    contentVersion: displayedDocument.renderedText,
                    initiallyFollowing: initiallyFollowingLatest
                ) {
                    TranscriptContentView(
                        document: displayedDocument,
                        emptyMessage: "Speech will appear here.",
                        font: .title3
                    )
                    .padding(18)
                }
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
    }
}
