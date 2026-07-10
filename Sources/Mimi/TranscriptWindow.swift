import MimiCore
import SwiftUI

struct TranscriptWindow: View {
    @Bindable var store: AppStore

    init(store: AppStore) {
        self.store = store
    }

    var body: some View {
        NavigationSplitView {
            Form {
                Picker("Input", selection: $store.source) {
                    ForEach([AudioSource.microphone]) { source in
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
                }
                Picker("Language", selection: $store.sourceLanguage) {
                    ForEach(SpeechLanguage.allCases) { language in
                        Text(language.nativeName).tag(language)
                    }
                }
                .disabled(store.controlsLocked)
                Picker("Model", selection: $store.engineID) {
                    ForEach(TranscriptionEngineID.allCases) { engine in
                        Text(engine.displayName).tag(engine)
                    }
                }
                .disabled(store.controlsLocked)
                Picker("Translation", selection: $store.translationMode) {
                    ForEach(TranslationMode.allCases) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
                .disabled(store.controlsLocked)
            }
            .formStyle(.grouped)
            .navigationTitle("Session")
        } detail: {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Label(store.recordingState.label, systemImage: store.menuBarSymbolName)
                        .foregroundStyle(store.isRecording ? .red : .primary)
                    Spacer()
                    Button(store.isRecording ? "Stop" : "Start") {
                        store.toggleRecording()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(store.isRecording ? .red : .accentColor)
                }

                ScrollView {
                    Text(store.document.renderedText.isEmpty ? "Your local transcript will appear here." : store.document.renderedText)
                        .font(.title3)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 14, style: .continuous))

                if store.translationMode == .translateFinalSegments,
                   !store.document.finalizedText(for: store.sourceLanguage).isEmpty {
                    InlineTranslationView(
                        sourceText: store.document.finalizedText(for: store.sourceLanguage),
                        sourceLanguage: store.sourceLanguage
                    )
                }
            }
            .padding()
            .navigationTitle("Transcript")
        }
    }
}
