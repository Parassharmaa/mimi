import CryptoKit
import Foundation
import MimiCore

struct TranslationParityCaseResult: Codable, Sendable {
    let caseID: String
    let direction: String
    let sourceLanguage: SpeechLanguage
    let targetLanguage: SpeechLanguage
    let domain: String
    let source: String
    let references: [String]
    let hypothesis: String
    let claimEligible: Bool
    let pythonHypothesis: String
    let swiftHypothesis: String
    let exactMatch: Bool
    let textExactMatch: Bool
    let tokenExactMatch: Bool?
    let pythonOutputTokenIDs: [Int]?
    let swiftOutputTokenIDs: [Int]
    let latencySeconds: Double
    let warmLatencySeconds: [Double]
}

struct TranslationParityVerificationReport: Codable, Sendable {
    let schemaVersion: Int
    let status: String
    let engine: String
    let modelRevision: String
    let suiteSHA256: String
    let pythonReportSHA256: String
    let pairManifestSHA256: String
    let preparationSeconds: Double
    let peakResidentBytes: Int64?
    let modelBytes: Int64
    let cases: Int
    let exactMatches: Int
    let results: [TranslationParityCaseResult]
}

private struct ParityPythonReport: Decodable {
    let schemaVersion: Int
    let engine: String
    let modelRevision: String?
    let results: [TranslationBenchmarkResult]
}

enum TranslationParityVerificationError: LocalizedError {
    case emptySuite
    case duplicateCaseID(String)
    case unsupportedPythonReport
    case modelRevisionMismatch(expected: String, actual: String?)
    case reportCoverageMismatch
    case reportCaseMismatch(String)

    var errorDescription: String? {
        switch self {
        case .emptySuite:
            "The Swift/MLX parity suite is empty."
        case let .duplicateCaseID(caseID):
            "The Swift/MLX parity suite contains a duplicate case ID: \(caseID)."
        case .unsupportedPythonReport:
            "The Swift/MLX parity input is not a supported Python MLX benchmark report."
        case let .modelRevisionMismatch(expected, actual):
            "The Python report model revision \(actual ?? "missing") does not match \(expected)."
        case .reportCoverageMismatch:
            "The Python MLX report does not cover the exact parity suite."
        case let .reportCaseMismatch(caseID):
            "The Python MLX report disagrees with the parity suite for \(caseID)."
        }
    }
}

func verifyTranslationMLXParity(
    modelRoot: URL,
    suiteURL: URL,
    pythonReportURL: URL,
    warmRuns: Int = 0,
    cachedDecoding: Bool = true
) async throws -> TranslationParityVerificationReport {
    try ExperimentalMLXTranslationEngine.validateModelPack(at: modelRoot)
    let suiteData = try Data(contentsOf: suiteURL)
    let decoder = JSONDecoder()
    let cases = try suiteData
        .split(separator: 0x0A)
        .filter { !$0.allSatisfy { byte in byte == 0x20 || byte == 0x09 || byte == 0x0D } }
        .map { try decoder.decode(TranslationBenchmarkCase.self, from: Data($0)) }
    guard !cases.isEmpty else { throw TranslationParityVerificationError.emptySuite }
    var caseIDs = Set<String>()
    for benchmarkCase in cases where !caseIDs.insert(benchmarkCase.id).inserted {
        throw TranslationParityVerificationError.duplicateCaseID(benchmarkCase.id)
    }

    let pythonReportData = try Data(contentsOf: pythonReportURL)
    let pythonReport = try decoder.decode(ParityPythonReport.self, from: pythonReportData)
    guard pythonReport.schemaVersion == 1,
          pythonReport.engine.hasPrefix("mlx:") else {
        throw TranslationParityVerificationError.unsupportedPythonReport
    }
    let rootManifestURL = modelRoot.appending(path: "manifest.json")
    let pairManifestHash = try fileSHA256(rootManifestURL)
    let expectedRevision = "pair-manifest-sha256:\(pairManifestHash)"
    guard pythonReport.modelRevision == expectedRevision else {
        throw TranslationParityVerificationError.modelRevisionMismatch(
            expected: expectedRevision,
            actual: pythonReport.modelRevision
        )
    }
    let pythonResults = Dictionary(
        pythonReport.results.map { ($0.caseID, $0) },
        uniquingKeysWith: { _, latest in latest }
    )
    guard pythonResults.count == pythonReport.results.count,
          Set(pythonResults.keys) == caseIDs else {
        throw TranslationParityVerificationError.reportCoverageMismatch
    }
    for benchmarkCase in cases {
        guard let result = pythonResults[benchmarkCase.id],
              result.sourceLanguage == benchmarkCase.sourceLanguage,
              result.targetLanguage == benchmarkCase.targetLanguage,
              result.domain == benchmarkCase.domain,
              result.source == benchmarkCase.source,
              result.references == benchmarkCase.references,
              result.claimEligible == benchmarkCase.claimEligible else {
            throw TranslationParityVerificationError.reportCaseMismatch(benchmarkCase.id)
        }
    }

    var parityResults = [TranslationParityCaseResult]()
    var preparationSeconds = 0.0
    for (sourceLanguage, direction) in [
        (SpeechLanguage.english, "en-ja"),
        (SpeechLanguage.japanese, "ja-en"),
    ] {
        let directionCases = cases.filter { $0.sourceLanguage == sourceLanguage }
        guard !directionCases.isEmpty else { continue }
        let preparationStarted = DispatchTime.now().uptimeNanoseconds
        let runtime = try await MarianMLXTranslationRuntime.load(
            directory: modelRoot.appending(path: direction, directoryHint: .isDirectory)
        )
        preparationSeconds += seconds(since: preparationStarted)
        for benchmarkCase in directionCases {
            let python = pythonResults[benchmarkCase.id]!.hypothesis
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let started = DispatchTime.now().uptimeNanoseconds
            let swiftOutputTokenIDs = runtime.translateTokenIDs(
                benchmarkCase.source,
                cachedDecoding: cachedDecoding
            )
            let swift = runtime.decode(tokens: swiftOutputTokenIDs)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let latencySeconds = seconds(since: started)
            var warmLatencySeconds = [Double]()
            warmLatencySeconds.reserveCapacity(warmRuns)
            for _ in 0..<warmRuns {
                let warmStarted = DispatchTime.now().uptimeNanoseconds
                _ = runtime.translateTokenIDs(
                    benchmarkCase.source,
                    cachedDecoding: cachedDecoding
                )
                warmLatencySeconds.append(seconds(since: warmStarted))
            }
            let pythonOutputTokenIDs = pythonResults[benchmarkCase.id]!.outputTokenIDs
            let tokenExactMatch = pythonOutputTokenIDs.map { $0 == swiftOutputTokenIDs }
            let textExactMatch = python == swift
            parityResults.append(
                .init(
                    caseID: benchmarkCase.id,
                    direction: direction,
                    sourceLanguage: benchmarkCase.sourceLanguage,
                    targetLanguage: benchmarkCase.targetLanguage,
                    domain: benchmarkCase.domain,
                    source: benchmarkCase.source,
                    references: benchmarkCase.references,
                    hypothesis: swift,
                    claimEligible: benchmarkCase.claimEligible,
                    pythonHypothesis: python,
                    swiftHypothesis: swift,
                    exactMatch: tokenExactMatch ?? textExactMatch,
                    textExactMatch: textExactMatch,
                    tokenExactMatch: tokenExactMatch,
                    pythonOutputTokenIDs: pythonOutputTokenIDs,
                    swiftOutputTokenIDs: swiftOutputTokenIDs,
                    latencySeconds: latencySeconds,
                    warmLatencySeconds: warmLatencySeconds
                )
            )
        }
    }
    parityResults.sort { $0.caseID < $1.caseID }
    let exactMatches = parityResults.count(where: \.exactMatch)
    return .init(
        schemaVersion: 1,
        status: exactMatches == cases.count ? "passed" : "failed",
        engine: cachedDecoding
            ? "swift-mlx-marian-kv-cache-exact-output-parity"
            : "swift-mlx-marian-full-prefix-exact-output-parity",
        modelRevision: expectedRevision,
        suiteSHA256: SHA256.hash(data: suiteData).hexadecimal,
        pythonReportSHA256: SHA256.hash(data: pythonReportData).hexadecimal,
        pairManifestSHA256: pairManifestHash,
        preparationSeconds: preparationSeconds,
        peakResidentBytes: nil,
        modelBytes: try directoryBytes(modelRoot),
        cases: cases.count,
        exactMatches: exactMatches,
        results: parityResults
    )
}

private func seconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
}

private func directoryBytes(_ root: URL) throws -> Int64 {
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

private func fileSHA256(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
        hasher.update(data: data)
    }
    return hasher.finalize().hexadecimal
}

private extension Sequence where Element == UInt8 {
    var hexadecimal: String {
        map { String(format: "%02x", $0) }.joined()
    }
}
