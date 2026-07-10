import MimiCore
import SwiftUI

struct MenuBarView: View {
    @Bindable var store: AppStore
    @Environment(\.openWindow) private var openWindow

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
                InlineTranslationView(
                    sourceText: store.document.finalizedText(for: store.sourceLanguage),
                    sourceLanguage: store.sourceLanguage
                )
            }

            Divider()

            HStack(spacing: 10) {
                Button("Open Transcript") {
                    openWindow(id: "transcript")
                }
                .buttonStyle(.borderless)

                Spacer()

                Button("Quit Mimi") {
                    NSApplication.shared.terminate(nil)
                }
                .buttonStyle(.borderless)
            }
            .font(.footnote)
        }
        .padding(16)
        .frame(width: 420)
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: store.menuBarSymbolName)
                .font(.title2)
                .foregroundStyle(statusColor)
                .symbolEffect(.variableColor.iterative, options: .repeating, isActive: store.isRecording)

            VStack(alignment: .leading, spacing: 3) {
                Text("Mimi")
                    .font(.headline)
                Text(store.recordingState.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if store.isRecording {
                Text("LOCAL")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 4)
                    .background(.red.opacity(0.12), in: Capsule())
            }
        }
    }

    private var controls: some View {
        VStack(spacing: 8) {
            Picker("Input", selection: $store.source) {
                ForEach([AudioSource.microphone]) { source in
                    Text(source.displayName).tag(source)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(maxWidth: .infinity, alignment: .leading)
            .disabled(store.controlsLocked)

            if store.source == .microphone {
                HStack(spacing: 6) {
                    Picker("Microphone", selection: $store.selectedInputDeviceID) {
                        Text("System Default").tag(UInt32?.none)
                        ForEach(store.inputDevices) { device in
                            Text(device.displayName).tag(Optional(device.id))
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .disabled(store.controlsLocked)

                    Button {
                        store.refreshInputDevices()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .buttonStyle(.borderless)
                    .help("Refresh microphone inputs")
                    .disabled(store.controlsLocked)
                }
            }

            HStack(spacing: 8) {
                Picker("Language", selection: $store.sourceLanguage) {
                    ForEach(SpeechLanguage.allCases) { language in
                        Text(language.nativeName).tag(language)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .disabled(store.controlsLocked)

                Picker("Model", selection: $store.engineID) {
                    ForEach(TranscriptionEngineID.allCases) { engine in
                        Text(engine.displayName).tag(engine)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .disabled(store.controlsLocked)
            }

            if let pack = store.modelPack {
                Text(pack.recommendation)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack(spacing: 8) {
                Button(store.isRecording ? "Stop" : "Start") {
                    store.toggleRecording()
                }
                .keyboardShortcut(.return, modifiers: [])
                .buttonStyle(.borderedProminent)
                .tint(store.isRecording ? .red : .accentColor)
                .disabled(store.recordingState == .preparing || store.recordingState == .processing)

                Button("Download Model") {
                    store.installSelectedModel()
                }
                .buttonStyle(.bordered)
                .disabled(store.controlsLocked)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var transcriptPreview: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Live transcript")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    store.copyTranscript()
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.plain)
                .disabled(store.document.renderedText.isEmpty)
                Button {
                    store.clearTranscript()
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .disabled(store.document.renderedText.isEmpty)
            }

            ScrollView {
                Text(store.document.renderedText.isEmpty ? "Choose a local model, then start speaking." : store.document.renderedText)
                    .font(.body)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
            .frame(height: 130)
            .background(.quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
    }

    private var statusColor: Color {
        store.isRecording ? .red : .accentColor
    }
}
