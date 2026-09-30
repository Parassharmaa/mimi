import AppKit
import MimiCore
import SwiftUI
import UniformTypeIdentifiers

struct TranscriptWindow: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences
    @State private var isConfirmingClear = false
    @State private var clearHistoryID: UUID?
    @State private var clearTitle = ""
    @State private var clearDocument: TranscriptDocument?
    @State private var searchText = ""
    @State private var isSearching = false
    @FocusState private var searchFocused: Bool
    @Environment(\.openSettings) private var openSettings

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
        _clearHistoryID = State(initialValue: store.selectedHistoryID)
        _clearDocument = State(initialValue: store.viewedDocument)
        _clearTitle = State(initialValue: preferences.text("Current transcript", "現在の文字起こし"))
    }

    var body: some View {
        NavigationSplitView {
            TranscriptHistorySidebar(store: store, preferences: preferences)
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

                transcriptContent
            }
            .frame(minWidth: 560, minHeight: 440)
            .navigationTitle(t("Transcript", "文字起こし"))
            .toolbar { transcriptToolbar }
        }
        .navigationSplitViewStyle(.balanced)
        .searchable(text: $searchText, isPresented: $isSearching, prompt: t("Find in original", "原文を検索"))
        .searchFocused($searchFocused)
        .onChange(of: store.selectedHistoryID) { searchText = "" }
        .onChange(of: store.controlsLocked) { if store.controlsLocked { isConfirmingClear = false } }
        .alert(t("Delete transcript?", "文字起こしを削除しますか？"), isPresented: $isConfirmingClear) {
            Button(t("Cancel", "キャンセル"), role: .cancel) {}
            Button(t("Delete", "削除"), role: .destructive) {
                guard !store.controlsLocked,
                      clearHistoryID != nil || store.document == clearDocument else { return }
                store.clearTranscript(historyID: clearHistoryID)
            }
        } message: {
            Text(t("\(clearTitle) will be deleted. This cannot be undone.", "「\(clearTitle)」を削除します。この操作は取り消せません。"))
        }
    }

    private var sessionStrip: some View {
        HStack(spacing: 10) {
            Image(systemName: store.menuBarSymbolName)
                .foregroundStyle(store.isRecording ? .red : .accentColor)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 1) {
                Text(sessionTitle)
                    .font(.callout.weight(.semibold))
                Text(sessionDetail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Spacer()

            if selectedRecord != nil && store.isRecording {
                Button(t("Go to Live Session", "録音中のセッションへ"), action: store.selectCurrentSession)
                    .buttonStyle(.bordered)
            }

            if store.translationMode == .translateFinalSegments {
                Label(t("Local translation", "ローカル翻訳"), systemImage: "translate")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 11)
        .accessibilityElement(children: .combine)
    }

    private var selectedRecord: TranscriptSessionRecord? {
        store.historyRecords.first { $0.id == store.selectedHistoryID }
    }

    private var sessionTitle: String {
        if let selectedRecord { return selectedRecord.title }
        return store.isRecording ? t("Listening locally", "ローカルで文字起こし中") : t("Current session", "現在のセッション")
    }

    private var sessionDetail: String {
        if let selectedRecord {
            let languages = Set(selectedRecord.document.segments.map(\.language)).map(\.nativeName).sorted().joined(separator: " / ")
            return "\(selectedRecord.startedAt.formatted(date: .abbreviated, time: .shortened)) · \(selectedRecord.source.displayName) · \(languages)"
        }
        return "\(store.recordingState.label) · \(store.source.displayName) · \(store.sourceLanguage.nativeName) · \(store.engineID.displayName)"
    }

    @ViewBuilder
    private var transcriptContent: some View {
        let fullDocument = store.viewedDocument
        let displayedDocument = searchText.isEmpty ? fullDocument : TranscriptDocument(
            segments: fullDocument.segments.filter { $0.text.localizedStandardContains(searchText) },
            liveText: fullDocument.liveText.localizedStandardContains(searchText) ? fullDocument.liveText : ""
        )

        if store.translationMode == .translateFinalSegments {
            HSplitView {
                TranscriptLanguagePane(
                    document: displayedDocument,
                    language: nil,
                    initiallyFollowingLatest: initiallyFollowingLatest && searchText.isEmpty,
                    preferences: preferences,
                    isSearching: !searchText.isEmpty
                )
                .frame(minWidth: 250)

                InlineTranslationView(
                    segments: fullDocument.segments,
                    fillsAvailableSpace: true,
                    fixtureTranslation: fixtureTranslation,
                    initiallyFollowingLatest: initiallyFollowingLatest,
                    preferences: preferences
                )
                .frame(minWidth: 250)
            }
        } else {
            TranscriptLanguagePane(
                document: displayedDocument,
                language: nil,
                initiallyFollowingLatest: initiallyFollowingLatest && searchText.isEmpty,
                preferences: preferences,
                isSearching: !searchText.isEmpty
            )
        }
    }

    @ToolbarContentBuilder
    private var transcriptToolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            Button {
                store.newSession()
            } label: {
                Label(t("New Session", "新しいセッション"), systemImage: "plus")
            }
            .keyboardShortcut("n", modifiers: .command)
            .disabled(store.controlsLocked)

            Button {
                store.copyTranscript()
            } label: {
                Label(t("Copy Transcript", "文字起こしをコピー"), systemImage: "doc.on.doc")
            }
            .keyboardShortcut("c", modifiers: [.command, .shift])
            .disabled(store.viewedDocument.renderedText.isEmpty)

            Button(action: exportTranscript) {
                Label(t("Export Transcript…", "文字起こしを書き出す…"), systemImage: "square.and.arrow.up")
            }
            .keyboardShortcut("e", modifiers: [.command, .shift])
            .disabled(store.viewedDocument.renderedText.isEmpty)

            Button(role: .destructive) {
                setClearConfirmation(true)
            } label: {
                Label(t("Delete Transcript", "文字起こしを削除"), systemImage: "trash")
            }
            .keyboardShortcut(.delete, modifiers: [.command, .option])
            .disabled(store.viewedDocument.renderedText.isEmpty || isConfirmingClear || store.controlsLocked)

            Menu {
                Button(t("Find in Original", "原文を検索")) {
                    isSearching = true
                    searchFocused = true
                }
                .keyboardShortcut("f", modifiers: .command)
                Divider()
                Button(t("Settings…", "設定…")) { showSettings() }
                Button(t("Voice Type…", "音声入力…")) { showSettings(.voiceTyping) }
                Button(t("Models and Languages…", "モデルと言語…")) { showSettings(.models) }
            } label: {
                Label(t("More", "その他"), systemImage: "ellipsis.circle")
            }

            Button {
                store.toggleRecording()
            } label: {
                Label(
                    store.isRecording ? t("Stop Recording", "録音を停止") : (selectedRecord == nil ? t("Start Recording", "録音を開始") : t("Start New Recording", "新しい録音を開始")),
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
        if confirming {
                    clearHistoryID = store.selectedHistoryID
            clearTitle = sessionTitle
            clearDocument = store.viewedDocument
        }
        isConfirmingClear = confirming
    }

    private func showSettings(_ tab: SettingsTab? = nil) {
        SettingsWindowFocusCoordinator.shared.requestFocus(tab: tab)
        openSettings()
    }

    private func exportTranscript() {
        // Snapshot before presenting the panel, so selection changes cannot
        // silently export a different session.
        let text = store.viewedDocument.renderedText
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = "Mimi transcript.txt"
        panel.title = t("Export Transcript", "文字起こしを書き出す")
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            do { try text.write(to: url, atomically: true, encoding: .utf8) }
            catch { store.reportError(error) }
        }
    }

    private func t(_ english: String, _ japanese: String) -> String {
        preferences.text(english, japanese)
    }
}

private struct TranscriptHistorySidebar: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences
    private static let currentID = UUID(uuid: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    private var selection: Binding<UUID?> {
        Binding(
            get: { store.selectedHistoryID ?? Self.currentID },
            set: { store.selectedHistoryID = $0 == Self.currentID ? nil : $0 }
        )
    }

    var body: some View {
        List(selection: selection) {
            Section(t("Now", "現在")) {
                Label(store.isRecording ? t("Listening now", "文字起こし中") : t("Current transcript", "現在の文字起こし"), systemImage: store.isRecording ? "waveform" : "doc.text")
                    .tag(Optional(Self.currentID))
            }

            if !store.historyRecords.isEmpty {
                Section(t("Previous sessions", "過去のセッション")) {
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

            Section(t("Next recording", "次の録音")) {
                Picker(t("Source", "入力"), selection: $store.source) {
                    ForEach(AudioSource.allCases) { source in
                        Label(source.displayName, systemImage: source.symbolName).tag(source)
                    }
                }
                .disabled(store.controlsLocked)
                .accessibilityLabel(t("Audio source", "音声入力"))

                sourceConfiguration
            }

            Section(t("Transcription", "文字起こし")) {
                Picker(t("Language", "言語"), selection: $store.languageMode) {
                    ForEach(store.selectableLanguageModes) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
                .disabled(store.controlsLocked || store.isModelSetupActive)
                .accessibilityLabel(t("Transcription language", "文字起こしの言語"))

                Picker(t("Model", "モデル"), selection: $store.engineID) {
                    ForEach(TranscriptionEngineID.selectableCases) { engine in
                        Text(engine.displayName).tag(engine)
                    }
                }
                .disabled(store.controlsLocked || store.isModelSetupActive)
                .accessibilityLabel(t("Speech model", "音声認識モデル"))

                ModelSetupStatusView(
                    readiness: store.selectedModelReadiness,
                    setupState: store.selectedModelSetupState,
                    compact: true
                )
            }

            Section(t("Translation", "翻訳")) {
                Picker(t("Mode", "モード"), selection: $store.translationMode) {
                    ForEach(TranslationMode.allCases) { mode in
                        Text(mode == .off ? t("Off", "オフ") : "English ↔ 日本語").tag(mode)
                    }
                }
                .disabled(store.controlsLocked)
                .accessibilityLabel(t("Translation mode", "翻訳モード"))
            }
        }
        .listStyle(.sidebar)
    }

    @ViewBuilder
    private var sourceConfiguration: some View {
        switch store.source {
        case .microphone:
            Picker(t("Microphone", "マイク"), selection: $store.selectedInputDeviceID) {
                Text(t("System Default", "システムのデフォルト")).tag(UInt32?.none)
                ForEach(store.inputDevices) { device in
                    Text(device.displayName).tag(Optional(device.id))
                }
            }
            .disabled(store.controlsLocked)
            .accessibilityLabel(t("Microphone", "マイク"))
            Button(t("Refresh Microphones", "マイクを更新"), action: store.refreshInputDevices)
                .disabled(store.controlsLocked)
        case .outputAudio:
            Picker(t("Output", "出力"), selection: $store.selectedOutputDeviceID) {
                Text(t("System Default", "システムのデフォルト")).tag(UInt32?.none)
                ForEach(store.outputDevices) { device in
                    Text(device.displayName).tag(Optional(device.id))
                }
            }
            .disabled(store.controlsLocked)
            .accessibilityLabel(t("Audio output", "音声出力"))
            Button(t("Refresh Outputs", "出力を更新"), action: store.refreshOutputDevices)
                .disabled(store.controlsLocked)
        case .applicationAudio, .systemAudio:
            ScreenAudioSelectionControl(store: store)
        }
    }

    private func t(_ english: String, _ japanese: String) -> String {
        preferences.text(english, japanese)
    }
}

private struct TranscriptLanguagePane: View {
    let document: TranscriptDocument
    let language: SpeechLanguage?
    let initiallyFollowingLatest: Bool
    let preferences: UserPreferences
    let isSearching: Bool

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
                Label(language?.nativeName ?? preferences.text("Original", "原文"), systemImage: language?.symbolName ?? "text.alignleft")
                    .font(.callout.weight(.semibold))
                Spacer()
                if !document.liveText.isEmpty {
                    Text(preferences.text("Listening", "文字起こし中"))
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
                    isSearching ? preferences.text("No Matches", "一致する結果がありません") : preferences.text("No Transcript Yet", "文字起こしはまだありません"),
                    systemImage: isSearching ? "magnifyingglass" : "waveform",
                    description: Text(isSearching ? preferences.text("Try another word or clear the search.", "別の言葉を検索するか、検索を解除してください。") : preferences.text("Choose an input and model, then start recording. Your audio stays on this Mac.", "入力とモデルを選び、録音を開始してください。音声はこのMac内で処理されます。"))
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                FollowLatestScrollView(
                    contentVersion: displayedDocument.renderedText,
                    initiallyFollowing: initiallyFollowingLatest,
                    preferences: preferences
                ) {
                    TranscriptContentView(
                        document: displayedDocument,
                        emptyMessage: "Speech will appear here.",
                        font: .title3,
                        preferences: preferences
                    )
                    .padding(18)
                }
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
    }
}
