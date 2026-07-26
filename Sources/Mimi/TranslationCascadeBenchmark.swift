import CryptoKit
import Darwin
import Foundation
import MimiCore

struct TranslationCascadeBenchmarkCaseResult: Codable, Sendable {
    let caseID: String
    let sourceLanguage: SpeechLanguage
    let targetLanguage: SpeechLanguage
    let domain: String
    let source: String
    let references: [String]
    let hypothesis: String
    let outputTokenIDs: [Int]?
    let selectedEngine: String
    let failureReason: String?
    let latencySeconds: Double
    let warmLatencySeconds: [Double]
    let claimEligible: Bool
}

struct TranslationCascadeBenchmarkReport: Codable, Sendable {
    let schemaVersion: Int
    let status: String
    let engine: String
    let modelRevision: String
    let createdAt: Date
    let operatingSystem: String
    let hardware: String
    let preparationSeconds: Double
    let peakResidentBytes: Int64?
    let modelBytes: Int64
    let warmRunsPerCase: Int
    let selectedEngineCounts: [String: Int]
    let failureCounts: [String: Int]
    let fallbackCaseIDs: [String]
    let results: [TranslationCascadeBenchmarkCaseResult]
}

enum TranslationCascadeBenchmarkError: LocalizedError {
    case invalidWarmRuns(String)
    case nondeterministicResult(String)

    var errorDescription: String? {
        switch self {
        case let .invalidWarmRuns(value):
            "Translation cascade warm runs must be a nonnegative integer, got: \(value)."
        case let .nondeterministicResult(caseID):
            "Translation cascade returned a non-deterministic result for \(caseID)."
        }
    }
}

func benchmarkExperimentalMLXTranslationCascade(
    modelRoot: URL,
    suiteURL: URL,
    warmRuns: Int
) async throws -> TranslationCascadeBenchmarkReport {
    let preparationStarted = DispatchTime.now().uptimeNanoseconds
    try ExperimentalMLXTranslationEngine.validateModelPack(at: modelRoot)
    let preparationSeconds = cascadeSeconds(since: preparationStarted)
    let suite = try TranslationBenchmarkCase.loadJSONL(from: suiteURL)
    let configuration = ExperimentalMLXTranslationConfiguration(modelDirectory: modelRoot)
    var results = [TranslationCascadeBenchmarkCaseResult]()
    results.reserveCapacity(suite.count)
    var selectedEngineCounts = [String: Int]()
    var failureCounts = [String: Int]()
    var fallbackCaseIDs = [String]()

    for benchmarkCase in suite {
        let started = DispatchTime.now().uptimeNanoseconds
        do {
            let first = try await ExperimentalMLXTranslationEngine.shared
                .translateWithDiagnostics(
                benchmarkCase.source,
                sourceLanguage: benchmarkCase.sourceLanguage,
                configuration: configuration
            )
            let latencySeconds = cascadeSeconds(since: started)
            let selectedEngine = cascadeSelectedEngine(first)
            var warmLatencySeconds = [Double]()
            warmLatencySeconds.reserveCapacity(warmRuns)
            for _ in 0..<warmRuns {
                let warmStarted = DispatchTime.now().uptimeNanoseconds
                let warm = try await ExperimentalMLXTranslationEngine.shared
                    .translateWithDiagnostics(
                        benchmarkCase.source,
                        sourceLanguage: benchmarkCase.sourceLanguage,
                        configuration: configuration
                    )
                warmLatencySeconds.append(cascadeSeconds(since: warmStarted))
                guard warm.output == first.output,
                      warm.outputTokenIDs == first.outputTokenIDs,
                      cascadeSelectedEngine(warm) == selectedEngine else {
                    throw TranslationCascadeBenchmarkError.nondeterministicResult(
                        benchmarkCase.id
                    )
                }
            }
            selectedEngineCounts[selectedEngine, default: 0] += 1
            if selectedEngine == "generalist" {
                fallbackCaseIDs.append(benchmarkCase.id)
            }
            results.append(.init(
                caseID: benchmarkCase.id,
                sourceLanguage: benchmarkCase.sourceLanguage,
                targetLanguage: benchmarkCase.targetLanguage,
                domain: benchmarkCase.domain,
                source: benchmarkCase.source,
                references: benchmarkCase.references,
                hypothesis: first.output,
                outputTokenIDs: first.outputTokenIDs,
                selectedEngine: selectedEngine,
                failureReason: nil,
                latencySeconds: latencySeconds,
                warmLatencySeconds: warmLatencySeconds,
                claimEligible: benchmarkCase.claimEligible
            ))
        } catch {
            let failureReason = error.localizedDescription
            selectedEngineCounts["failed", default: 0] += 1
            failureCounts[failureReason, default: 0] += 1
            results.append(.init(
                caseID: benchmarkCase.id,
                sourceLanguage: benchmarkCase.sourceLanguage,
                targetLanguage: benchmarkCase.targetLanguage,
                domain: benchmarkCase.domain,
                source: benchmarkCase.source,
                references: benchmarkCase.references,
                hypothesis: "",
                outputTokenIDs: nil,
                selectedEngine: "failed",
                failureReason: failureReason,
                latencySeconds: cascadeSeconds(since: started),
                warmLatencySeconds: [],
                claimEligible: benchmarkCase.claimEligible
            ))
        }
    }

    let rootManifest = modelRoot.appending(path: "manifest.json")
    return .init(
        schemaVersion: 1,
        status: failureCounts.isEmpty ? "passed" : "failed-runtime-safety",
        engine: "swift-mlx:guarded-expert-cascade-v19:kv-cache",
        modelRevision: "moe-manifest-sha256:\(try cascadeSHA256(rootManifest))",
        createdAt: Date(),
        operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
        hardware: cascadeHardwareModel,
        preparationSeconds: preparationSeconds,
        peakResidentBytes: cascadePeakResidentBytes,
        modelBytes: try cascadeDirectoryBytes(modelRoot),
        warmRunsPerCase: warmRuns,
        selectedEngineCounts: selectedEngineCounts,
        failureCounts: failureCounts,
        fallbackCaseIDs: fallbackCaseIDs.sorted(),
        results: results
    )
}

private func cascadeSelectedEngine(
    _ diagnostic: ExperimentalMLXTranslationDiagnostic
) -> String {
    if diagnostic.usedTranslationMemory {
        return "translation-memory"
    }
    return diagnostic.routedToExpert ? "expert" : "generalist"
}

private func cascadeSeconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
}

private func cascadeSHA256(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
        hasher.update(data: data)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

private func cascadeDirectoryBytes(_ root: URL) throws -> Int64 {
    guard let enumerator = FileManager.default.enumerator(
        at: root,
        includingPropertiesForKeys: [.fileSizeKey],
        options: [.skipsHiddenFiles]
    ) else { return 0 }
    var total: Int64 = 0
    for case let url as URL in enumerator {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        if values.isRegularFile == true {
            total += Int64(values.fileSize ?? 0)
        }
    }
    return total
}

private var cascadePeakResidentBytes: Int64? {
    var usage = rusage()
    guard getrusage(RUSAGE_SELF, &usage) == 0 else { return nil }
    return Int64(usage.ru_maxrss)
}

private var cascadeHardwareModel: String {
    var size = 0
    guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0,
          size > 0 else { return "Apple Silicon" }
    var value = [CChar](repeating: 0, count: size)
    guard sysctlbyname("machdep.cpu.brand_string", &value, &size, nil, 0) == 0 else {
        return "Apple Silicon"
    }
    let bytes = value.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
    return String(decoding: bytes, as: UTF8.self)
}
