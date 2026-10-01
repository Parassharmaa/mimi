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
                .navigationSplitViewColumnWidth(min: 210, ideal: 240, max: 300)
                .navigationTitle("Mimi")
        } detail: {
            VStack(spacing: 0) {
                sessionStrip

                if let message = store.lastError {
                    inlineNotice(message, symbol: "exclamationmark.triangle.fill", tint: .orange)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 10)
                }

                transcriptContent
                    .padding(.horizontal, 24)

                WorkspaceCaptureBar(store: store, preferences: preferences)
            }
            .frame(minWidth: 560, minHeight: 440)
            .navigationTitle("Mimi")
            .toolbar { transcriptToolbar }
        }
        .navigationSplitViewStyle(.prominentDetail)
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
            VStack(alignment: .leading, spacing: 6) {
                Text(sessionTitle)
                    .font(.title2.weight(.semibold))
                    .lineLimit(1)
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

            Picker(t("Workspace view", "表示モード"), selection: $store.translationMode) {
                Text(t("Transcript", "文字起こし")).tag(TranslationMode.off)
                Text(t("Bilingual", "原文と翻訳")).tag(TranslationMode.translateFinalSegments)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .accessibilityLabel(t("Workspace view", "表示モード"))
            .frame(width: 190)
            .disabled(store.controlsLocked)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 22)
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
        return t("A quiet place for your words and their meaning.", "言葉と、その意味を落ち着いて見渡せる場所。")
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
            .clipShape(.rect(cornerRadius: 12))
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

            Menu {
                Button(t("Find in Original", "原文を検索")) {
                    isSearching = true
                    searchFocused = true
                }
                .keyboardShortcut("f", modifiers: .command)
                Divider()
                Button(t("Delete Transcript", "文字起こしを削除"), role: .destructive) { setClearConfirmation(true) }
                    .keyboardShortcut(.delete, modifiers: [.command, .option])
                    .disabled(store.viewedDocument.renderedText.isEmpty || isConfirmingClear || store.controlsLocked)
                Divider()
                Button(t("Settings…", "設定…")) { showSettings() }
                Button(t("Voice Type…", "音声入力…")) { showSettings(.voiceTyping) }
                Button(t("Models and Languages…", "モデルと言語…")) { showSettings(.models) }
            } label: {
                Label(t("More", "その他"), systemImage: "ellipsis.circle")
            }

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
    @Environment(\.openSettings) private var openSettings
    @State private var query = ""
    private static let currentID = UUID(uuid: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    private var selection: Binding<UUID?> {
        Binding(get: { store.selectedHistoryID ?? Self.currentID },
                set: { store.selectedHistoryID = $0 == Self.currentID ? nil : $0 })
    }

    private var records: [TranscriptSessionRecord] {
        query.isEmpty ? store.historyRecords : store.historyRecords.filter {
            $0.document.renderedText.localizedStandardContains(query)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 16) {
                Button(action: store.newSession) {
                    Label(t("New session", "新しいセッション"), systemImage: "square.and.pencil")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(MimiQuietButtonStyle())
                .font(.callout.weight(.semibold))
                .keyboardShortcut("n", modifiers: .command)
                .disabled(store.controlsLocked)
                TextField(t("Search sessions", "セッションを検索"), text: $query)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel(t("Search session history", "過去のセッションを検索"))
            }
            .padding(18)

            List(selection: selection) {
                Section(t("Workspace", "ワークスペース")) {
                    Label(store.isRecording ? t("Listening now", "文字起こし中") : t("Current session", "現在のセッション"),
                          systemImage: store.isRecording ? "waveform" : "doc.text")
                        .tag(Optional(Self.currentID))
                }
                Section(t("Recent sessions", "最近のセッション")) {
                    ForEach(records) { record in
                        VStack(alignment: .leading, spacing: 5) {
                            Text(record.title).font(.callout).lineLimit(1)
                            Text(record.startedAt, format: .dateTime.month().day().hour().minute())
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 3)
                        .tag(Optional(record.id))
                    }
                }
            }
            .listStyle(.sidebar)

            VStack(spacing: 12) {
                sidebarButton(t("Voice Type", "音声入力"), symbol: "keyboard") {
                    SettingsWindowFocusCoordinator.shared.requestFocus(tab: .voiceTyping)
                    openSettings()
                }
                sidebarButton(t("Settings", "設定"), symbol: "gearshape") {
                    SettingsWindowFocusCoordinator.shared.requestFocus()
                    openSettings()
                }
            }
            .padding(18)
        }
    }

    private func sidebarButton(_ title: String, symbol: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: symbol)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .buttonStyle(MimiQuietButtonStyle())
        .foregroundStyle(.secondary)
    }

    private func t(_ english: String, _ japanese: String) -> String { preferences.text(english, japanese) }
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
            MimiPaneHeader(language?.nativeName ?? preferences.text("Original", "原文"), symbol: language?.symbolName ?? "text.alignleft") {
                if !document.liveText.isEmpty {
                    Text(preferences.text("Listening", "文字起こし中"))
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(.quaternary, in: Capsule())
                }
            }
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
