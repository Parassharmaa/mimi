import CryptoKit
import Foundation
import MimiCore

struct LiveTranslationFixture: Sendable {
    let id: String
    let language: SpeechLanguage
    let prefixes: [String]
}

let liveTranslationFixtures: [LiveTranslationFixture] = [
    .init(id: "en-meeting", language: .english, prefixes: ["Can we", "Can we move the meeting", "Can we move the meeting to tomorrow morning?"]),
    .init(id: "en-build", language: .english, prefixes: ["The new version", "The new version is ready", "The new version is ready for testing."]),
    .init(id: "en-travel", language: .english, prefixes: ["I would like", "I would like to book a room", "I would like to book a room for two nights."]),
    .init(id: "en-audio", language: .english, prefixes: ["Please check", "Please check the microphone", "Please check the microphone before the call."]),
    .init(id: "ja-meeting", language: .japanese, prefixes: ["明日の", "明日の会議は", "明日の会議は午前十時に始まります。"]),
    .init(id: "ja-build", language: .japanese, prefixes: ["新しい", "新しいバージョンを", "新しいバージョンをテストしてください。"]),
    .init(id: "ja-travel", language: .japanese, prefixes: ["東京まで", "東京までの電車は", "東京までの電車は何時に出発しますか。"]),
    .init(id: "ja-audio", language: .japanese, prefixes: ["マイクの", "マイクの音量を", "マイクの音量を確認してください。"]),
]

struct LiveTranslationAttempt: Codable {
    let fixtureID: String
    let language: SpeechLanguage
    let policy: String
    let repetition: Int
    let source: String
    let output: String
    let eventToPublicationMS: Double
    let computeMS: Double
    let failure: String?
}

// This measures publication-ready text, not compositor paint or audio-to-ASR delay.
func benchmarkLiveTranslationBaseline(modelRoot: URL, outputURL: URL) async throws {
    let configuration = ExperimentalMLXTranslationConfiguration(modelDirectory: modelRoot)
    let engine = ExperimentalMLXTranslationEngine()
    var attempts: [LiveTranslationAttempt] = []
    var cold: [String: Double] = [:]
    for language in [SpeechLanguage.english, .japanese] {
        let fixture = liveTranslationFixtures.first { $0.language == language }!
        let start = ContinuousClock.now
        _ = try await engine.translate(fixture.prefixes.last!, sourceLanguage: language, configuration: configuration)
        cold[language.rawValue] = elapsedMS(start)
    }
    // Three repeats are paired within each fixture, not independent samples.
    for repetition in 0..<3 {
        for fixture in liveTranslationFixtures {
            let policies = repetition.isMultiple(of: 2) ? ["final-only", "caption-debounce-140ms"] : ["caption-debounce-140ms", "final-only"]
            for policy in policies {
                let source = policy == "final-only" ? fixture.prefixes.last! : fixture.prefixes.first!
                let event = ContinuousClock.now
                try await Task.sleep(for: .milliseconds(policy == "final-only" ? 1_200 : 140))
                let started = ContinuousClock.now
                var output = ""
                var failure: String?
                do { output = try await engine.translate(source, sourceLanguage: fixture.language, configuration: configuration) }
                catch { failure = error.localizedDescription }
                attempts.append(.init(fixtureID: fixture.id, language: fixture.language, policy: policy, repetition: repetition, source: source, output: output, eventToPublicationMS: elapsedMS(event), computeMS: elapsedMS(started), failure: failure))
            }
        }
    }
    let report: [String: Any] = [
        "schemaVersion": 1, "phase": "baseline", "createdAt": ISO8601DateFormatter().string(from: Date()),
        "modelRoot": modelRoot.path, "modelManifestSHA256": SHA256.hash(data: try Data(contentsOf: modelRoot.appending(path: "manifest.json"))).map { String(format: "%02x", $0) }.joined(),
        "os": ProcessInfo.processInfo.operatingSystemVersionString, "hardware": benchmarkHardware(),
        "coldFirstCallMS": cold,
        "protocol": "8 owned EN/JA utterances, 3 paired repetitions. Final at 1200ms after initial partial. Baseline caption waits 140ms after a stable first partial. No retries. Failure includes empty/unsafe model output. No per-call timeout. The original run completed without an operational timeout. Latency ends at publication-ready response, excluding paint and ASR. Same engine and model for both policies.",
        "attempts": try JSONSerialization.jsonObject(with: JSONEncoder().encode(attempts))
    ]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: outputURL, options: .atomic)
}

func elapsedMS(_ start: ContinuousClock.Instant) -> Double {
    let duration = start.duration(to: .now).components
    return Double(duration.seconds) * 1_000 + Double(duration.attoseconds) / 1e15
}

func benchmarkHardware() -> String {
    var size = 0
    sysctlbyname("hw.model", nil, &size, nil, 0)
    var bytes = [CChar](repeating: 0, count: size)
    sysctlbyname("hw.model", &bytes, &size, nil, 0)
    return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

struct LiveTranslationPublication: Codable {
    let source: String
    let output: String
    let isPartial: Bool
    let eventToPublicationMS: Double
}

struct LiveTranslationReplay: Codable {
    let fixtureID: String
    let language: SpeechLanguage
    let repetition: Int
    let firstTranslationMS: Double?
    let finalMatchesBaseline: Bool
    let finalFailureMatchesBaseline: Bool
    let publications: [LiveTranslationPublication]
    let inferenceCount: Int
    let failures: [String]
}

@MainActor
func benchmarkInstantTranslation(modelRoot: URL, baselineURL: URL, outputURL: URL) async throws -> Bool {
    let configuration = ExperimentalMLXTranslationConfiguration(modelDirectory: modelRoot)
    let engine = ExperimentalMLXTranslationEngine()
    guard let baseline = try JSONSerialization.jsonObject(with: Data(contentsOf: baselineURL)) as? [String: Any],
          let baselineAttempts = baseline["attempts"] as? [[String: Any]] else {
        throw CocoaError(.fileReadCorruptFile)
    }
    var cold: [String: Double] = [:]
    for language in [SpeechLanguage.english, .japanese] {
        let fixture = liveTranslationFixtures.first { $0.language == language }!
        let start = ContinuousClock.now
        _ = try await engine.translate(fixture.prefixes.last!, sourceLanguage: language, configuration: configuration)
        cold[language.rawValue] = elapsedMS(start)
    }
    var replays: [LiveTranslationReplay] = []
    var paired: [LiveTranslationAttempt] = []
    // Counterbalance immediate vs 140ms policies on exactly the same partial input.
    for repetition in 0..<3 {
        for fixture in liveTranslationFixtures {
            let policies = repetition.isMultiple(of: 2) ? ["instant", "caption-debounce-140ms"] : ["caption-debounce-140ms", "instant"]
            for policy in policies {
                let event = ContinuousClock.now
                if policy != "instant" { try await Task.sleep(for: .milliseconds(140)) }
                let started = ContinuousClock.now
                var output = ""
                var failure: String?
                do { output = try await engine.translate(fixture.prefixes.first!, sourceLanguage: fixture.language, configuration: configuration) }
                catch { failure = error.localizedDescription }
                paired.append(.init(fixtureID: fixture.id, language: fixture.language, policy: policy, repetition: repetition, source: fixture.prefixes.first!, output: output, eventToPublicationMS: elapsedMS(event), computeMS: elapsedMS(started), failure: failure))
            }
            let scope = UUID()
            let start = ContinuousClock.now
            var eventStarts: [String: ContinuousClock.Instant] = [:]
            var publications: [LiveTranslationPublication] = []
            var firstPublicationMS: Double?
            var inferenceCount = 0
            var failures: [String] = []
            let stream = LocalTranslationStream { text, language in
                try await engine.translate(text, sourceLanguage: language, configuration: configuration)
            }
            // The same UI worker is used here; every successful publication and every failure is retained.
            stream.onPublication = { text, output, partial in
                if firstPublicationMS == nil { firstPublicationMS = elapsedMS(start) }
                publications.append(.init(source: text, output: output, isPartial: partial, eventToPublicationMS: elapsedMS(eventStarts[text] ?? start)))
            }
            stream.onAttempt = { _, error in
                inferenceCount += 1
                if let error { failures.append(error) }
            }
            for (index, prefix) in fixture.prefixes.enumerated() {
                if index > 0 { try await Task.sleep(for: .milliseconds(220)) }
                eventStarts[prefix] = .now
                stream.update(.init(segments: [], liveText: prefix, language: fixture.language, scopeID: scope))
            }
            try await Task.sleep(for: .milliseconds(760))
            let final = TranscriptSegment(text: fixture.prefixes.last!, language: fixture.language)
            eventStarts[final.text] = .now
            stream.update(.init(segments: [final], liveText: "", language: fixture.language, scopeID: scope))
            await Task.yield()
            for _ in 0..<1_000 {
                if !stream.isTranslating { break }
                try await Task.sleep(for: .milliseconds(5))
            }
            let baselineFinal = baselineAttempts.first { $0["fixtureID"] as? String == fixture.id && $0["policy"] as? String == "final-only" }
            let reference = baselineFinal?["output"] as? String
            let baselineFailure = baselineFinal?["failure"] as? String
            replays.append(.init(fixtureID: fixture.id, language: fixture.language, repetition: repetition,
                firstTranslationMS: firstPublicationMS,
                finalMatchesBaseline: stream.translations[final.id] == reference && reference?.isEmpty == false,
                finalFailureMatchesBaseline: baselineFailure != nil && stream.failedSegmentIDs.contains(final.id) && failures.contains(baselineFailure!),
                publications: publications, inferenceCount: inferenceCount, failures: failures))
        }
    }
    let manifestSHA = SHA256.hash(data: try Data(contentsOf: modelRoot.appending(path: "manifest.json"))).map { String(format: "%02x", $0) }.joined()
    guard manifestSHA == baseline["modelManifestSHA256"] as? String else { throw CocoaError(.fileReadCorruptFile) }
    let passed = replays.allSatisfy { ($0.finalMatchesBaseline || $0.finalFailureMatchesBaseline) && $0.firstTranslationMS != nil }
        && paired.allSatisfy { $0.failure == nil }
    let report: [String: Any] = [
        "schemaVersion": 1, "phase": "instant", "createdAt": ISO8601DateFormatter().string(from: Date()),
        "modelManifestSHA256": manifestSHA, "hardware": benchmarkHardware(), "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "coldFirstCallMS": cold,
        "protocol": "Same 8 owned EN/JA fixtures and 3 paired repeats as baseline. Immediate vs 140ms calls counterbalanced. Production LocalTranslationStream receives prefixes at 0/220/440ms and final at 1200ms. Retain publications, failures, and inference counts. No retries. 5s final drain cap; missing final is failure. Publication excludes compositor paint and ASR. Fixture is independent unit; repeats and prefixes are dependent. Final output exact parity with baseline is correctness gate, not translation quality assessment.",
        "pairedAttempts": try JSONSerialization.jsonObject(with: JSONEncoder().encode(paired)),
        "replays": try JSONSerialization.jsonObject(with: JSONEncoder().encode(replays)),
        "status": passed ? "passed" : "failed"
    ]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: outputURL, options: .atomic)
    return passed
}

func benchmarkTranslationPrewarm(modelRoot: URL, outputURL: URL) async throws {
    let engine = ExperimentalMLXTranslationEngine()
    let configuration = ExperimentalMLXTranslationConfiguration(modelDirectory: modelRoot)
    let started = ContinuousClock.now
    try await engine.prepareForLiveTranslation(configuration: configuration)
    let preparationMS = elapsedMS(started)
    var results: [[String: Any]] = []
    for language in [SpeechLanguage.english, .japanese] {
        let fixture = liveTranslationFixtures.first { $0.language == language }!
        let started = ContinuousClock.now
        let output = try await engine.translate(fixture.prefixes.first!, sourceLanguage: language, configuration: configuration)
        results.append(["language": language.rawValue, "source": fixture.prefixes.first!, "output": output, "firstPartialAfterPrewarmMS": elapsedMS(started)])
    }
    let report: [String: Any] = ["schemaVersion": 1, "status": "passed", "preparationMS": preparationMS, "results": results, "note": "Fresh process, filesystem and GPU driver caches may be warm. Preparation runs at normal AppStore initialization. Excludes ASR and paint."]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: outputURL, options: .atomic)
}
