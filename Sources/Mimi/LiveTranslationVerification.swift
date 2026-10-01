import Foundation
import MimiCore

private actor TranslationProbe {
    var sources: [String] = []
    var inFlight = 0
    var maximumInFlight = 0
    func translate(_ text: String, language: SpeechLanguage) async throws -> String {
        sources.append(text)
        inFlight += 1
        maximumInFlight = max(maximumInFlight, inFlight)
        defer { inFlight -= 1 }
        try await Task.sleep(for: .milliseconds(35))
        if text == "bad" { throw CocoaError(.coderInvalidValue) }
        return "\(language.rawValue):\(text)"
    }
}

@MainActor
func verifyLiveTranslationStream() async -> [String: Bool] {
    var checks: [String: Bool] = [:]
    let probe = TranslationProbe()
    let stream = LocalTranslationStream { text, language in try await probe.translate(text, language: language) }
    var published: [String] = []
    stream.onPublication = { _, output, _ in published.append(output) }
    let scope = UUID()
    func input(_ text: String, language: SpeechLanguage = .english, segments: [TranscriptSegment] = [], session: UUID? = nil) -> LocalTranslationSnapshot {
        .init(segments: segments, liveText: text, language: language, scopeID: session ?? scope)
    }
    stream.update(input("first"))
    await Task.yield()
    try? await Task.sleep(for: .milliseconds(10))
    for number in 0..<100 { stream.update(input("revision \(number)")) }
    await waitUntilIdle(stream)
    let sources = await probe.sources
    checks["firstPartialPublishesDuringContinuousUpdates"] = sources.first == "first" && sources.count == 2 && published.first == "en-US:first"
    checks["newestPendingWins"] = stream.liveTranslation == "en-US:revision 99"
    checks["oneInferenceAtATime"] = await probe.maximumInFlight == 1
    stream.update(input("revision 99"))
    await waitUntilIdle(stream)
    checks["unchangedPartialIsNotRetranslated"] = await probe.sources.count == sources.count
    stream.update(input("bad"))
    await waitUntilIdle(stream)
    checks["failedPartialRetainsLastGoodTranslation"] = stream.liveTranslation == "en-US:revision 99"
    stream.update(input("corrected"))
    await waitUntilIdle(stream)
    let beforeFinal = await probe.sources.count
    let final = TranscriptSegment(text: "corrected", language: .english)
    stream.update(input("", segments: [final]))
    await waitUntilIdle(stream)
    let afterFinal = await probe.sources.count
    checks["matchingFinalPromotesWithoutInference"] = stream.translations[final.id] == "en-US:corrected" && stream.liveTranslation.isEmpty && afterFinal == beforeFinal
    let changed = TranscriptSegment(text: "final correction", language: .english)
    stream.update(input("", segments: [final, changed]))
    await waitUntilIdle(stream)
    checks["correctedFinalIsRetranslated"] = stream.translations[changed.id] == "en-US:final correction"
    stream.update(input("old session"))
    try? await Task.sleep(for: .milliseconds(10))
    stream.update(input("新しいセッション", language: .japanese, session: UUID()))
    await waitUntilIdle(stream)
    checks["sessionChangeRejectsStaleResult"] = stream.liveTranslation == "ja-JP:新しいセッション" && stream.translations.isEmpty
    stream.update(input("old direction", language: .english))
    try? await Task.sleep(for: .milliseconds(10))
    stream.update(input("日本語", language: .japanese))
    await waitUntilIdle(stream)
    checks["languageChangeRejectsStaleResult"] = stream.liveTranslation == "ja-JP:日本語"
    stream.update(input("removed"))
    try? await Task.sleep(for: .milliseconds(10))
    stream.reset()
    try? await Task.sleep(for: .milliseconds(50))
    checks["resetRejectsInFlightResult"] = stream.liveTranslation.isEmpty && !stream.isTranslating
    let badFinal = TranscriptSegment(text: "bad", language: .english)
    let goodFinal = TranscriptSegment(text: "later", language: .english)
    stream.update(input("", segments: [badFinal, goodFinal]))
    await waitUntilIdle(stream)
    checks["failedFinalDoesNotBlockLaterFinals"] = stream.failedSegmentIDs == [badFinal.id] && stream.translations[goodFinal.id] == "en-US:later"
    stream.reset()
    let history = (0..<3).map { TranscriptSegment(text: "archived \($0)", language: .english) }
    for revision in 0..<60 {
        stream.update(input("continuous \(revision)", segments: history))
        try? await Task.sleep(for: .milliseconds(5))
    }
    checks["finalHistoryProgressesDuringContinuousSpeech"] = history.allSatisfy { stream.translations[$0.id] != nil }
    await waitUntilIdle(stream)
    checks["liveSpeechProgressesAlongsideFinalHistory"] = stream.liveTranslation == "en-US:continuous 59"
    stream.reset()
    stream.update(input("finishing"))
    try? await Task.sleep(for: .milliseconds(10))
    let inFlightFinal = TranscriptSegment(text: "finishing", language: .english)
    stream.update(input("", segments: [inFlightFinal]))
    await waitUntilIdle(stream)
    checks["inFlightMatchingFinalPublishesAsFinal"] = stream.translations[inFlightFinal.id] == "en-US:finishing" && stream.liveTranslation.isEmpty
    return checks
}

@MainActor
private func waitUntilIdle(_ stream: LocalTranslationStream) async {
    await Task.yield()
    for _ in 0..<200 {
        if !stream.isTranslating { return }
        try? await Task.sleep(for: .milliseconds(5))
    }
}
