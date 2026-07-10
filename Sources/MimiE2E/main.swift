import Foundation
import MimiCore

@main
struct MimiE2E {
    static func main() {
        var english = DeterministicTranscriptionPipeline(language: .english)
        english.start()
        english.receive(.partial("Local transcription stays"))
        english.receive(.final("Local transcription stays on this Mac."))
        english.receive(.partial("No audio leaves the device."))
        english.stop()

        precondition(english.document.segments.map(\.text) == [
            "Local transcription stays on this Mac.",
            "No audio leaves the device."
        ])
        precondition(english.state == .idle)

        var japanese = DeterministicTranscriptionPipeline(language: .japanese)
        japanese.start()
        japanese.receive(.partial("こんにちは、"))
        japanese.receive(.final("こんにちは、ローカル文字起こしです。"))
        japanese.stop()

        precondition(japanese.document.renderedText == "こんにちは、ローカル文字起こしです。")
        precondition(ModelCatalog.pack(for: .whisperKitLargeV3Turbo)?.supportedLanguages == [.english, .japanese])

        print("Mimi E2E passed: English and Japanese local-transcription pipelines are deterministic.")
    }
}
