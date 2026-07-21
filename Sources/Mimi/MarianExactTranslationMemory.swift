import CryptoKit
import Foundation
import MimiCore

struct MarianExactTranslationMemory: Sendable {
    static let normalizationContract = "NFKC then Unicode-whitespace collapse"
    static let sourceLicenseContract = "PDL-1.0-compatible-CC-BY-4.0"
    static let maximumSourceCharacters = 64
    static let maximumTargetCharacters = 128

    let trainingDataSHA256: String
    let auditSHA256: String

    private let entries: [String: [String: String]]

    var entryCount: Int {
        entries.values.reduce(0) { $0 + $1.count }
    }

    var longestSource: Int {
        entries.values
            .flatMap(\.keys)
            .map { $0.unicodeScalars.count }
            .max() ?? 0
    }

    var longestTarget: Int {
        entries.values
            .flatMap(\.values)
            .map { $0.unicodeScalars.count }
            .max() ?? 0
    }

    init(contentsOf url: URL) throws {
        let payload: Payload
        do {
            payload = try JSONDecoder().decode(Payload.self, from: Data(contentsOf: url))
        } catch {
            throw MarianExactTranslationMemoryError.invalidArtifact(url, error)
        }
        guard payload.schemaVersion == 1,
              payload.normalization == Self.normalizationContract,
              payload.sourceLicense == Self.sourceLicenseContract,
              payload.doesNotAuthorizeAppIntegration,
              Self.isSHA256(payload.trainingDataSHA256),
              Self.isSHA256(payload.auditSHA256),
              Set(payload.entries.keys) == Set(["en-ja", "ja-en"]),
              payload.entries.values.allSatisfy({ !$0.isEmpty }) else {
            throw MarianExactTranslationMemoryError.unsupportedArtifact(url)
        }

        for (direction, directionEntries) in payload.entries {
            for (source, target) in directionEntries {
                guard !source.isEmpty,
                      !target.isEmpty,
                      source == Self.normalize(source),
                      source.unicodeScalars.count <= Self.maximumSourceCharacters,
                      target.unicodeScalars.count <= Self.maximumTargetCharacters,
                      ExperimentalMLXTranslationEngine.preservesCriticalTokens(
                          source: source,
                          output: target
                      ),
                      Self.hasTargetScript(target, direction: direction) else {
                    throw MarianExactTranslationMemoryError.unsafeEntry(url, direction, source)
                }
            }
        }

        trainingDataSHA256 = payload.trainingDataSHA256.lowercased()
        auditSHA256 = payload.auditSHA256.lowercased()
        entries = payload.entries
    }

    func translation(for source: String, sourceLanguage: SpeechLanguage) -> String? {
        let direction = sourceLanguage == .english ? "en-ja" : "ja-en"
        return entries[direction]?[Self.normalize(source)]
    }

    static func normalize(_ value: String) -> String {
        value.precomposedStringWithCompatibilityMapping
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value.unicodeScalars.allSatisfy {
            (0x30...0x39).contains($0.value) || (0x61...0x66).contains($0.value)
        }
    }

    private static func hasTargetScript(_ value: String, direction: String) -> Bool {
        switch direction {
        case "en-ja":
            value.unicodeScalars.contains {
                (0x3040...0x30ff).contains($0.value) || (0x3400...0x9fff).contains($0.value)
            }
        case "ja-en":
            value.unicodeScalars.contains {
                (0x41...0x5a).contains($0.value) || (0x61...0x7a).contains($0.value)
            }
        default:
            false
        }
    }

    private struct Payload: Decodable {
        let schemaVersion: Int
        let normalization: String
        let trainingDataSHA256: String
        let auditSHA256: String
        let sourceLicense: String
        let doesNotAuthorizeAppIntegration: Bool
        let entries: [String: [String: String]]
    }
}

enum MarianExactTranslationMemoryError: LocalizedError {
    case invalidArtifact(URL, Error)
    case unsupportedArtifact(URL)
    case unsafeEntry(URL, String, String)

    var errorDescription: String? {
        switch self {
        case let .invalidArtifact(url, error):
            "The exact translation memory at \(url.path) is invalid: \(error.localizedDescription)"
        case let .unsupportedArtifact(url):
            "The exact translation memory at \(url.path) has an unsupported provenance or schema contract."
        case let .unsafeEntry(url, direction, source):
            "The exact translation memory at \(url.path) contains an unsafe \(direction) entry for \(source)."
        }
    }
}

struct TranslationMemoryParityCaseResult: Codable, Sendable {
    let caseID: String
    let direction: String
    let source: String
    let pythonUsesMemory: Bool
    let swiftUsesMemory: Bool
    let decisionExactMatch: Bool
    let hypothesisExactMatch: Bool
}

struct TranslationMemoryParityReport: Codable, Sendable {
    let schemaVersion: Int
    let status: String
    let memorySHA256: String
    let pythonReportSHA256: String
    let entries: Int
    let cases: Int
    let memoryHits: Int
    let exactDecisionMatches: Int
    let exactHypothesisMatches: Int
    let results: [TranslationMemoryParityCaseResult]
}

private struct TranslationMemoryPythonReport: Decodable {
    let translationMemory: Metadata
    let results: [Result]

    struct Metadata: Decodable {
        let sha256: String
        let entries: Int
    }

    struct Result: Decodable {
        let caseID: String
        let sourceLanguage: SpeechLanguage
        let source: String
        let hypothesis: String
        let selectedEngine: String?
    }
}

func verifyTranslationMemoryParity(
    memoryURL: URL,
    pythonReportURL: URL
) throws -> TranslationMemoryParityReport {
    let memoryData = try Data(contentsOf: memoryURL)
    let reportData = try Data(contentsOf: pythonReportURL)
    let memory = try MarianExactTranslationMemory(contentsOf: memoryURL)
    let python = try JSONDecoder().decode(TranslationMemoryPythonReport.self, from: reportData)
    let memorySHA256 = SHA256.hash(data: memoryData).hexadecimalString
    guard python.translationMemory.sha256.lowercased() == memorySHA256,
          python.translationMemory.entries == memory.entryCount,
          !python.results.isEmpty else {
        throw MarianExactTranslationMemoryError.unsupportedArtifact(pythonReportURL)
    }

    var caseIDs = Set<String>()
    var results = [TranslationMemoryParityCaseResult]()
    results.reserveCapacity(python.results.count)
    for row in python.results {
        guard caseIDs.insert(row.caseID).inserted else {
            throw MarianExactTranslationMemoryError.unsupportedArtifact(pythonReportURL)
        }
        let swiftHypothesis = memory.translation(
            for: row.source,
            sourceLanguage: row.sourceLanguage
        )
        let pythonUsesMemory = row.selectedEngine == "exact-translation-memory"
        let direction = row.sourceLanguage == .english ? "en-ja" : "ja-en"
        results.append(
            .init(
                caseID: row.caseID,
                direction: direction,
                source: row.source,
                pythonUsesMemory: pythonUsesMemory,
                swiftUsesMemory: swiftHypothesis != nil,
                decisionExactMatch: pythonUsesMemory == (swiftHypothesis != nil),
                hypothesisExactMatch: !pythonUsesMemory || swiftHypothesis == row.hypothesis
            )
        )
    }
    results.sort { $0.caseID < $1.caseID }
    let exactDecisionMatches = results.count(where: \.decisionExactMatch)
    let exactHypothesisMatches = results.count(where: \.hypothesisExactMatch)
    let passed = exactDecisionMatches == results.count && exactHypothesisMatches == results.count
    return .init(
        schemaVersion: 1,
        status: passed ? "passed" : "failed",
        memorySHA256: memorySHA256,
        pythonReportSHA256: SHA256.hash(data: reportData).hexadecimalString,
        entries: memory.entryCount,
        cases: results.count,
        memoryHits: results.count(where: \.swiftUsesMemory),
        exactDecisionMatches: exactDecisionMatches,
        exactHypothesisMatches: exactHypothesisMatches,
        results: results
    )
}

private extension Sequence where Element == UInt8 {
    var hexadecimalString: String {
        map { String(format: "%02x", $0) }.joined()
    }
}
