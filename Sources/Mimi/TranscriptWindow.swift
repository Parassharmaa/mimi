import MimiCore
import SwiftUI

struct TranscriptWindow: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences
    @State private var isConfirmingClear = false

    private let fixtureTranslation: String?
    private let initiallyFollowingLatest: Bool

    init(
        store: AppStore,
        preferences: UserPreferences = UserPreferences(),
        isConfirmingClear: Bool = false,
        fixtureTranslation: String? = nil,
        initiallyFollowingLatest: Bool = true
    ) {
        self.store = store
        self.preferences = preferences
        self.fixtureTranslation = fixtureTranslation
        self.initiallyFollowingLatest = initiallyFollowingLatest
        _isConfirmingClear = State(initialValue: isConfirmingClear)
    }

    var body: some View {
        NavigationSplitView {
            TranscriptHistorySidebar(store: store)
                .navigationSplitViewColumnWidth(min: 220, ideal: 250, max: 310)
                .navigationTitle(t("Sessions", "セッション"))
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
            .navigationTitle(t("Transcript", "文字起こし"))
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
        let displayedDocument = store.viewedDocument
        let translationSourceText = followsAppleSpeech && store.selectedHistoryID == nil
            ? store.document.realtimeTranslationContext(for: store.sourceLanguage)
            : displayedDocument.finalizedText(for: store.sourceLanguage)

        if store.translationMode == .translateFinalSegments {
            HSplitView {
                TranscriptLanguagePane(
                    document: displayedDocument,
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
                document: displayedDocument,
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
            Text(t("Clear the saved transcript?", "保存した文字起こしを消去しますか？"))
                .font(.callout.weight(.semibold))
            Text(t("This can’t be undone.", "この操作は取り消せません。"))
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button(t("Cancel", "キャンセル")) { setClearConfirmation(false) }
                .keyboardShortcut(.cancelAction)
            Button(t("Clear", "消去"), role: .destructive) {
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
            .disabled(store.viewedDocument.renderedText.isEmpty)

            Button(role: .destructive) {
                setClearConfirmation(true)
            } label: {
                Label("Clear Transcript", systemImage: "trash")
            }
            .keyboardShortcut(.delete, modifiers: [.command, .option])
            .disabled(store.viewedDocument.renderedText.isEmpty || isConfirmingClear)

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

    private func t(_ english: String, _ japanese: String) -> String {
        preferences.text(english, japanese)
    }
}

private struct TranscriptHistorySidebar: View {
    @Bindable var store: AppStore

    var body: some View {
        List(selection: $store.selectedHistoryID) {
            Section("Now") {
                Button {
                    store.selectCurrentSession()
                } label: {
                    Label(store.isRecording ? "Listening now" : "Current transcript", systemImage: store.isRecording ? "waveform" : "doc.text")
                }
                .buttonStyle(.plain)
            }

            if !store.historyRecords.isEmpty {
                Section("Previous sessions") {
                    ForEach(store.historyRecords) { record in
                        VStack(alignment: .leading, spacing: 3) {
                            Text(record.title).lineLimit(1)
                            Text(record.startedAt, format: .dateTime.month().day().hour().minute())
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .tag(Optional(record.id))
                    }
                }
            }

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
                Picker("Language", selection: $store.languageMode) {
                    ForEach(TranscriptionLanguageMode.allCases) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
                .disabled(store.controlsLocked || store.isModelSetupActive)

                Picker("Model", selection: $store.engineID) {
                    ForEach(TranscriptionEngineID.selectableCases) { engine in
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
        .listStyle(.sidebar)
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
