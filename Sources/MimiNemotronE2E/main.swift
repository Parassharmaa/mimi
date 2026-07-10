import Foundation
import MLXAudioCore
import MLXAudioSTT

@main
enum MimiNemotronE2E {
    static func main() {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard arguments.count == 4 else {
            fputs("Usage: MimiNemotronE2E <model-directory> <wav-file> <en-US|ja-JP> <expected-substring>\\n", stderr)
            exit(64)
        }

        let modelDirectory = URL(fileURLWithPath: arguments[0], isDirectory: true)
        let audioURL = URL(fileURLWithPath: arguments[1])
        let language = arguments[2]
        let expectedSubstring = arguments[3]

        do {
            let model = try NemotronASRModel.fromDirectory(modelDirectory)
            let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: model.preprocessConfig.sampleRate)
            let result = model.generate(
                audio: audio,
                generationParameters: .init(language: language, chunkDuration: 120)
            )
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard text.localizedCaseInsensitiveContains(expectedSubstring) else {
                throw FixtureError.unexpectedTranscript(expected: expectedSubstring, actual: text)
            }
            print("Mimi native Nemotron E2E passed (\(language)): \(text)")
        } catch {
            fputs("Mimi native Nemotron E2E failed: \(error.localizedDescription)\\n", stderr)
            exit(1)
        }
    }
}

private enum FixtureError: LocalizedError {
    case unexpectedTranscript(expected: String, actual: String)

    var errorDescription: String? {
        switch self {
        case let .unexpectedTranscript(expected, actual):
            "Expected transcript to contain '\(expected)', got '\(actual)'."
        }
    }
}
