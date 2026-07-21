import Foundation

public struct TranslationBenchmarkCase: Codable, Identifiable, Equatable, Sendable {
    public enum ReviewStatus: String, Codable, Sendable {
        case bootstrapUnreviewed = "bootstrap-unreviewed"
        case independentlyReviewed = "independently-reviewed"
        case adjudicated = "adjudicated"
    }

    public let id: String
    public let sourceLanguage: SpeechLanguage
    public let targetLanguage: SpeechLanguage
    public let domain: String
    public let source: String
    public let references: [String]
    public let split: String
    public let license: String
    public let provenance: String
    public let reviewStatus: ReviewStatus
    public let claimEligible: Bool

    public init(
        id: String,
        sourceLanguage: SpeechLanguage,
        targetLanguage: SpeechLanguage,
        domain: String,
        source: String,
        references: [String],
        split: String,
        license: String,
        provenance: String,
        reviewStatus: ReviewStatus,
        claimEligible: Bool
    ) {
        self.id = id
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
        self.domain = domain
        self.source = source
        self.references = references
        self.split = split
        self.license = license
        self.provenance = provenance
        self.reviewStatus = reviewStatus
        self.claimEligible = claimEligible
    }

    public var isStructurallyValid: Bool {
        !id.isEmpty
            && sourceLanguage != targetLanguage
            && !domain.isEmpty
            && !source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !references.isEmpty
            && references.allSatisfy { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            && !split.isEmpty
            && !license.isEmpty
            && !provenance.isEmpty
            && (!claimEligible || reviewStatus == .adjudicated)
    }

    public static func loadJSONL(from url: URL) throws -> [Self] {
        let contents = try String(contentsOf: url, encoding: .utf8)
        let decoder = JSONDecoder()
        let cases = try contents.split(whereSeparator: \.isNewline).enumerated().map { index, line in
            do {
                return try decoder.decode(Self.self, from: Data(line.utf8))
            } catch {
                throw TranslationBenchmarkDataError.invalidJSONLine(index + 1, error)
            }
        }
        guard !cases.isEmpty else { throw TranslationBenchmarkDataError.emptySuite }
        guard Set(cases.map(\.id)).count == cases.count else {
            throw TranslationBenchmarkDataError.duplicateIDs
        }
        guard cases.allSatisfy(\.isStructurallyValid) else {
            throw TranslationBenchmarkDataError.invalidCase
        }
        return cases
    }
}

public struct TranslationBenchmarkResult: Codable, Equatable, Sendable {
    public let caseID: String
    public let sourceLanguage: SpeechLanguage
    public let targetLanguage: SpeechLanguage
    public let domain: String
    public let source: String
    public let references: [String]
    public let hypothesis: String
    public let outputTokenIDs: [Int]?
    public let latencySeconds: Double
    public let warmLatencySeconds: [Double]
    public let claimEligible: Bool

    public init(
        benchmarkCase: TranslationBenchmarkCase,
        hypothesis: String,
        outputTokenIDs: [Int]? = nil,
        latencySeconds: Double,
        warmLatencySeconds: [Double] = []
    ) {
        caseID = benchmarkCase.id
        sourceLanguage = benchmarkCase.sourceLanguage
        targetLanguage = benchmarkCase.targetLanguage
        domain = benchmarkCase.domain
        source = benchmarkCase.source
        references = benchmarkCase.references
        self.hypothesis = hypothesis
        self.outputTokenIDs = outputTokenIDs
        self.latencySeconds = latencySeconds
        self.warmLatencySeconds = warmLatencySeconds
        claimEligible = benchmarkCase.claimEligible
    }
}

public struct TranslationBenchmarkReport: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let engine: String
    public let modelRevision: String?
    public let createdAt: Date
    public let operatingSystem: String
    public let hardware: String
    public let preparationSeconds: Double
    public let peakResidentBytes: Int64?
    public let modelBytes: Int64?
    public let results: [TranslationBenchmarkResult]

    public init(
        engine: String,
        modelRevision: String? = nil,
        operatingSystem: String,
        hardware: String,
        preparationSeconds: Double,
        peakResidentBytes: Int64? = nil,
        modelBytes: Int64? = nil,
        results: [TranslationBenchmarkResult],
        createdAt: Date = Date()
    ) {
        schemaVersion = 1
        self.engine = engine
        self.modelRevision = modelRevision
        self.createdAt = createdAt
        self.operatingSystem = operatingSystem
        self.hardware = hardware
        self.preparationSeconds = preparationSeconds
        self.peakResidentBytes = peakResidentBytes
        self.modelBytes = modelBytes
        self.results = results
    }

    public var claimEligibleResultCount: Int {
        results.count(where: \.claimEligible)
    }

    public func latencyPercentile(_ percentile: Double) -> Double? {
        let values = results.flatMap {
            $0.warmLatencySeconds.isEmpty ? [$0.latencySeconds] : $0.warmLatencySeconds
        }.sorted()
        guard !values.isEmpty else { return nil }
        let bounded = min(max(percentile, 0), 1)
        let index = Int((Double(values.count - 1) * bounded).rounded(.up))
        return values[index]
    }
}

public enum TranslationBenchmarkDataError: LocalizedError {
    case emptySuite
    case duplicateIDs
    case invalidCase
    case invalidJSONLine(Int, Error)

    public var errorDescription: String? {
        switch self {
        case .emptySuite:
            "The translation benchmark suite is empty."
        case .duplicateIDs:
            "The translation benchmark contains duplicate case IDs."
        case .invalidCase:
            "The translation benchmark contains an invalid or unreviewed claim-eligible case."
        case let .invalidJSONLine(line, error):
            "Translation benchmark line \(line) is invalid: \(error.localizedDescription)"
        }
    }
}
