import Foundation
import MimiCore

@main
struct MimiSelfTest {
    static func main() {
        testRepeatedFinalTextIsPreserved()
        testStopFinalizesJapaneseVolatileResult()
        testInputLanguageChangeDoesNotRewriteTranscriptLanguage()
        testFloatingCaptionAlwaysUsesNewestUtterance()
        testIncrementalMixedLanguageTranslationQueue()
        testLongHorizonCaptionTranslationCoalescing()
        testLongHorizonFinalizedTranslationQueue()
        testRecommendedPacksCoverBothV1Languages()
        print("Mimi self-test passed: transcript coalescing, long-horizon translation, Japanese finalization, and model routing.")
    }

    private static func testLongHorizonCaptionTranslationCoalescing() {
        var pipeline = LiveTranslationPipeline()
        var completedRequests = 0
        var hasShownTranslation = false

        for index in 0..<10_000 {
            let language: SpeechLanguage = index.isMultiple(of: 11) ? .japanese : .english
            _ = pipeline.enqueue(text: "live phrase \(index)", language: language)

            if index.isMultiple(of: 7), let active = pipeline.activeRequest {
                _ = pipeline.complete(requestID: active.id, translation: "translated \(active.text)")
                completedRequests += 1
                hasShownTranslation = true
            }

            if hasShownTranslation {
                expect(!pipeline.displayedTranslation.isEmpty, "A newer live partial never blanks the last good translation")
            }
        }

        while let active = pipeline.activeRequest {
            _ = pipeline.complete(requestID: active.id, translation: "translated \(active.text)")
            completedRequests += 1
        }

        expect(pipeline.pendingRequest == nil, "The live translation queue drains after sustained speech")
        expect(pipeline.displayedSourceText == "live phrase 9999", "Caption translation eventually catches the newest long-session partial")
        expect(completedRequests < 2_000, "Ten thousand partials are coalesced into bounded translation work")
        expect(
            pipeline.enqueue(text: "live phrase 9999", language: .japanese) == nil,
            "An unchanged translated caption is not submitted again every sampling tick"
        )
    }

    private static func testLongHorizonFinalizedTranslationQueue() {
        let segments = (0..<2_500).map { index in
            TranscriptSegment(
                text: index.isMultiple(of: 2) ? "English sentence \(index)" : "日本語の文 \(index)",
                language: index.isMultiple(of: 2) ? .english : .japanese
            )
        }
        var queue = SegmentTranslationQueue()
        var completedIDs: Set<UUID> = []
        var translatedLanguages: [SpeechLanguage] = []

        while let segment = queue.beginNext(in: segments, completedIDs: completedIDs) {
            expect(queue.activeSegmentID == segment.id, "Exactly one finalized sentence is active at a time")
            expect(queue.finish(segment.id), "The active finalized sentence completes deterministically")
            completedIDs.insert(segment.id)
            translatedLanguages.append(segment.language)
        }

        expect(completedIDs.count == segments.count, "Every sentence translates during a long mixed-language session")
        expect(translatedLanguages == segments.map(\.language), "Long-session translations preserve transcript order and per-sentence language")
        expect(queue.activeSegmentID == nil, "The finalized translation queue never remains stuck loading after it drains")
    }

    private static func testIncrementalMixedLanguageTranslationQueue() {
        let english = TranscriptSegment(text: "Hello", language: .english)
        let japanese = TranscriptSegment(text: "こんにちは", language: .japanese)
        let document = TranscriptDocument(segments: [english, japanese])

        expect(
            document.nextSegmentNeedingTranslation(completedIDs: [])?.id == english.id,
            "Incremental translation starts with the first immutable segment"
        )
        let next = document.nextSegmentNeedingTranslation(completedIDs: [english.id])
        expect(next?.id == japanese.id, "A completed sentence is never translated again")
        expect(next?.language.translationTarget == .english, "Each Auto segment chooses the target opposite its detected language")
    }

    private static func testFloatingCaptionAlwaysUsesNewestUtterance() {
        var document = TranscriptDocument()
        document.apply(.final("older line"), language: .english)
        document.apply(.final("newest final line"), language: .english)
        expect(document.latestCaptionText == "newest final line", "A stopped caption shows only the newest final utterance")

        document.apply(.partial("current speech arriving now"), language: .english)
        expect(document.latestCaptionText == "current speech arriving now", "A live caption immediately replaces history with the current partial")
    }

    private static func testInputLanguageChangeDoesNotRewriteTranscriptLanguage() {
        var document = TranscriptDocument()
        document.apply(.final("Existing English session"), language: .english)

        expect(
            document.contentLanguage(fallback: .japanese) == .english,
            "Changing the next-input language never relabels an existing transcript"
        )
        expect(
            document.renderedText == "Existing English session",
            "Changing the next-input language never filters or rewrites existing text"
        )
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
