import AppKit
import MimiCore
import SwiftUI

struct MenuBarView: View {
    @Bindable var store: AppStore
    @Bindable var preferences: UserPreferences
    @Environment(\.openSettings) private var openSettings
    @State private var isConfirmingClear = false

    private let initiallyFollowingLatest: Bool

    init(
        store: AppStore,
        preferences: UserPreferences = UserPreferences(),
        isConfirmingClear: Bool = false,
        initiallyFollowingLatest: Bool = true
    ) {
        self.store = store
        self.preferences = preferences
        self.initiallyFollowingLatest = initiallyFollowingLatest
        _isConfirmingClear = State(initialValue: isConfirmingClear)
    }

    var body: some View {
        ZStack {
            Color(nsColor: .windowBackgroundColor)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: MimiMetrics.sectionSpacing) {
                MimiStatusHeader(state: store.recordingState, source: store.source)

                if !store.controlsLocked {
                    Button {
                        store.newSession()
                    } label: {
                        Label(t("New Session", "新しいセッション"), systemImage: "plus")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                }

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
        }
        .frame(width: 430)
        .containerBackground(Color(nsColor: .windowBackgroundColor), for: .window)
        .background(MenuBarWindowBackgroundConfigurator())
    }

    private var recordingButton: some View {
        Button {
            store.toggleRecording()
        } label: {
            Label(
                store.isRecording ? t("Stop Recording", "録音を停止") : t("Start Recording", "録音を開始"),
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
                t("Input", "入力"),
                detail: sourceSummary,
                symbol: store.source.symbolName
            ) {
                Picker("Input", selection: $store.source) {
                    ForEach(AudioSource.allCases) { source in
                        Text(source.displayName).tag(source)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 194)
                .disabled(store.controlsLocked)
                .accessibilityLabel("Input source")
            }

            sourceConfiguration
            rowDivider

            MimiControlRow(t("Language", "言語"), symbol: store.sourceLanguage.symbolName) {
                Picker(t("Language", "言語"), selection: $store.languageMode) {
                    ForEach(TranscriptionLanguageMode.allCases) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
                .pickerStyle(.menu)
                .disabled(store.controlsLocked || store.isModelSetupActive)
                .accessibilityLabel("Transcription language")
            }

            rowDivider

            MimiControlRow(
                t("Speech", "音声認識"),
                detail: store.selectedModelReadiness.canStart ? t("Ready", "準備完了") : t("Setup needed", "準備が必要"),
                symbol: "cpu"
            ) {
                Picker("Model", selection: $store.engineID) {
                    ForEach(TranscriptionEngineID.selectableCases) { engine in
                        Text(engine.displayName).tag(engine)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 194)
                .disabled(store.controlsLocked || store.isModelSetupActive)
                .accessibilityLabel("Transcription model")
            }

            rowDivider

            MimiControlRow(t("Translation", "翻訳"), symbol: "translate") {
                Picker("Translation", selection: $store.translationMode) {
                    ForEach(TranslationMode.allCases) { mode in
                        Text(mode == .off ? t("Off", "オフ") : t("English ↔ Japanese", "英語 ↔ 日本語")).tag(mode)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 194)
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
        MimiControlRow(title, symbol: title == "Output" ? "hifispeaker" : "mic") {
            HStack(spacing: 8) {
                Picker(title, selection: selection) {
                    Text("System Default").tag(UInt32?.none)
                    ForEach(devices, id: \.0) { id, name in
                        Text(name).tag(Optional(id))
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 194)
                .disabled(store.controlsLocked)

                Button(action: refresh) {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .help(refreshLabel)
                .accessibilityLabel(refreshLabel)
                .disabled(store.controlsLocked)
            }
        }
    }

    private var modelSetup: some View {
        VStack(alignment: .leading, spacing: 9) {
            MimiSectionLabel(t("Language setup", "言語の準備"), symbol: "arrow.down.circle")
            ModelSetupStatusView(
                readiness: store.selectedModelReadiness,
                setupState: store.selectedModelSetupState,
                compact: true
            )

            Button(t("Open Language Settings…", "言語設定を開く…"), action: openMimiSettings)
                .buttonStyle(.bordered)
                .accessibilityHint("Opens the model setup window")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .mimiCard(padding: 14)
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
                Text(t("Clear transcript?", "文字起こしを消去しますか？"))
                    .font(.caption.weight(.semibold))
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
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Clear local transcript confirmation")
        } else {
            HStack(spacing: 8) {
                MimiSectionLabel(t("Latest transcript", "最新の文字起こし"), symbol: "text.alignleft")
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
                AppWindowCoordinator.shared.showTranscript()
            } label: {
                Label(t("Transcript", "文字起こし"), systemImage: "rectangle.on.rectangle")
            }

            Button(action: openMimiSettings) {
                Label(t("Settings", "設定"), systemImage: "gearshape")
            }

            Spacer()

            Button(t("Quit", "終了")) {
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

    private func t(_ english: String, _ japanese: String) -> String {
        preferences.text(english, japanese)
    }
}

private struct MenuBarWindowBackgroundConfigurator: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        MenuBarWindowBackgroundView()
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}

private final class MenuBarWindowBackgroundView: NSView {
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        DispatchQueue.main.async { [weak self] in
            guard let window = self?.window, let contentView = window.contentView else { return }
            window.backgroundColor = .windowBackgroundColor
            contentView.wantsLayer = true
            contentView.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        }
    }
}
