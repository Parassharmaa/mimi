import MimiCore
import SwiftUI

struct SettingsView: View {
    @Bindable var store: AppStore

    init(store: AppStore) {
        self.store = store
    }

    var body: some View {
        TabView {
            Form {
                Section("Local models") {
                    Picker("Active model", selection: $store.engineID) {
                        ForEach(TranscriptionEngineID.allCases) { engine in
                            Text(engine.displayName).tag(engine)
                        }
                    }
                    .disabled(store.controlsLocked)
                    if let pack = store.modelPack {
                        Text(pack.recommendation)
                            .foregroundStyle(.secondary)
                        if let size = pack.estimatedDownloadMB {
                            Text("Estimated download: about \(size) MB")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else if pack.ownership == .systemManaged {
                            Text("System-managed language assets download after you choose this model.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    Button("Download Selected Model") {
                        store.installSelectedModel()
                    }
                    .disabled(store.controlsLocked)
                    if store.canRemoveSelectedModel {
                        Button("Remove Downloaded Whisper Model", role: .destructive) {
                            store.removeSelectedModel()
                        }
                        .disabled(store.controlsLocked)
                    }
                }

                Section("Privacy") {
                    Text("Mimi stores transcript text locally. Temporary microphone audio is deleted after the selected local engine finishes.")
                    Text("Apple Translation processes text on-device; macOS may collect non-content performance metadata for the API.")
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
            .padding()
            .tabItem { Label("Models", systemImage: "cpu") }

            Form {
                Section("Audio sources") {
                    Text("Microphone is ready. Selected App Audio and All System Audio use macOS Core Audio taps and remain behind a physical-Mac permission smoke test in this initial build.")
                    Text("Mimi will never hide an active recording state. Process capture will always show the app or source being transcribed.")
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
            .padding()
            .tabItem { Label("Capture", systemImage: "waveform") }
        }
        .frame(width: 560, height: 390)
    }
}
