import Foundation
import MimiCore

@main
struct MimiSelfTest {
    static func main() {
        testRepeatedFinalTextIsPreserved()
        testStopFinalizesJapaneseVolatileResult()
        testRecommendedPacksCoverBothV1Languages()
        print("Mimi self-test passed: transcript coalescing, Japanese finalization, and model routing.")
    }

    private static func testRepeatedFinalTextIsPreserved() {
        var document = TranscriptDocument()
        let now = Date(timeIntervalSince1970: 123)

        document.apply(.partial("hello"), language: .english, now: now)
        document.apply(.final("hello world"), language: .english, now: now)
        document.apply(.final("hello world"), language: .english, now: now)

        expect(document.liveText == "", "A final result clears volatile text")
        expect(document.segments.map(\.text) == ["hello world", "hello world"], "Repeated spoken final text is preserved")
        expect(document.segments.first?.language == .english, "English final result retains its source language")
        expect(document.finalizedText(for: .japanese).isEmpty, "A translation lane does not include text from another source language")
    }

    private static func testStopFinalizesJapaneseVolatileResult() {
        var pipeline = DeterministicTranscriptionPipeline(language: .japanese)
        pipeline.start()
        pipeline.receive(.partial("日本語のテスト"), now: .distantPast)
        pipeline.stop(now: .distantPast)

        expect(pipeline.state == .idle, "Stopping returns the pipeline to idle")
        expect(pipeline.document.segments.map(\.text) == ["日本語のテスト"], "A Japanese partial becomes a final local segment at stop")
    }

    private static func testRecommendedPacksCoverBothV1Languages() {
        expect(ModelCatalog.packs.count == 1, "Mimi presents one simple Apple Speech choice")
        expect(ModelCatalog.packs[0].supportedLanguages == [.english, .japanese], "Apple Speech setup covers English and Japanese")
        expect(TranscriptionEngineID.selectableCases == [.appleSpeechAnalyzer], "Removed experimental models never appear in the UI")
    }

    private static func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
        guard condition() else {
            fatalError("Mimi self-test failure: \(message)")
        }
    }
}
