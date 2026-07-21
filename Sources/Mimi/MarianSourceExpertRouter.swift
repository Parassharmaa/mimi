import CryptoKit
import Foundation

struct MarianSourceExpertRouter: Sendable {
    let direction: String
    let minimumSourceCharacters: Int
    let scoreThreshold: Double

    private let vocabulary: [String: Int]
    private let inverseDocumentFrequency: [Double]
    private let coefficients: [Double]
    private let intercept: Double

    init(contentsOf url: URL) throws {
        let payload = try JSONDecoder().decode(
            MarianSourceExpertRouterPayload.self,
            from: Data(contentsOf: url)
        )
        guard payload.schemaVersion == 1,
              payload.format == "mimi-source-expert-router-v1",
              ["en-ja", "ja-en"].contains(payload.direction),
              payload.features == .supported,
              payload.vocabulary.count == payload.inverseDocumentFrequency.count,
              payload.vocabulary.count == payload.ridge.coefficients.count,
              Set(payload.vocabulary.values) == Set(0..<payload.vocabulary.count),
              payload.routing.minimumSourceCharacters >= 0 else {
            throw MarianSourceExpertRouterError.unsupportedModel(url)
        }
        direction = payload.direction
        minimumSourceCharacters = payload.routing.minimumSourceCharacters
        scoreThreshold = payload.routing.scoreThreshold
        vocabulary = payload.vocabulary
        inverseDocumentFrequency = payload.inverseDocumentFrequency
        coefficients = payload.ridge.coefficients
        intercept = payload.ridge.intercept
    }

    func score(_ source: String) -> Double {
        let lengthBin = min(source.unicodeScalars.count / 20, 20)
        let featureText = "\(source)\n__MIMI_LENGTH_BIN_\(lengthBin)__"
        let normalized = Self.collapseRepeatedWhitespace(featureText.lowercased())
        let scalars = Array(normalized.unicodeScalars)
        var counts = [Int: Int]()
        var discoveryOrder = [Int]()
        for width in 2...5 where scalars.count >= width {
            for offset in 0...(scalars.count - width) {
                let term = String(String.UnicodeScalarView(scalars[offset..<(offset + width)]))
                guard let index = vocabulary[term] else { continue }
                if counts[index] == nil {
                    discoveryOrder.append(index)
                    counts[index] = 1
                } else {
                    counts[index]! += 1
                }
            }
        }
        var weighted = [Int: Double]()
        var squaredNorm = 0.0
        for index in discoveryOrder {
            let value = (1.0 + log(Double(counts[index]!)))
                * inverseDocumentFrequency[index]
            weighted[index] = value
            squaredNorm += value * value
        }
        let norm = sqrt(squaredNorm)
        guard norm != 0 else { return intercept }
        var result = intercept
        for index in discoveryOrder {
            result += coefficients[index] * weighted[index]! / norm
        }
        return result
    }

    func routesToExpert(_ source: String) -> Bool {
        source.unicodeScalars.count >= minimumSourceCharacters
            && score(source) >= scoreThreshold
    }

    private static func collapseRepeatedWhitespace(_ value: String) -> String {
        let scalars = Array(value.unicodeScalars)
        var output = ""
        output.reserveCapacity(value.utf8.count)
        var index = 0
        while index < scalars.count {
            guard scalars[index].properties.isWhitespace else {
                output.unicodeScalars.append(scalars[index])
                index += 1
                continue
            }
            let start = index
            while index < scalars.count, scalars[index].properties.isWhitespace {
                index += 1
            }
            if index - start >= 2 {
                output.append(" ")
            } else {
                output.unicodeScalars.append(scalars[start])
            }
        }
        return output
    }
}

struct MarianSourceExpertRouterParityReport: Codable, Sendable {
    let schemaVersion: Int
    let status: String
    let cases: Int
    let exactRoutes: Int
    let maximumAbsoluteScoreDelta: Double
    let tolerance: Double
    let pythonReportSHA256: String
}

private struct MarianSourceExpertRouterPayload: Decodable {
    let schemaVersion: Int
    let format: String
    let direction: String
    let features: Features
    let vocabulary: [String: Int]
    let inverseDocumentFrequency: [Double]
    let ridge: Ridge
    let routing: Routing

    struct Features: Decodable, Equatable {
        let analyzer: String
        let ngramRange: [Int]
        let lowercase: Bool
        let minimumDocumentFrequency: Int
        let sublinearTermFrequency: Bool
        let inverseDocumentFrequency: String
        let normalization: String
        let sourceLengthBin: String

        static let supported = Self(
            analyzer: "unicode-codepoint-character",
            ngramRange: [2, 5],
            lowercase: true,
            minimumDocumentFrequency: 2,
            sublinearTermFrequency: true,
            inverseDocumentFrequency: "smooth-idf",
            normalization: "l2",
            sourceLengthBin: "append newline then __MIMI_LENGTH_BIN_{min(chars//20,20)}__"
        )
    }

    struct Ridge: Decodable {
        let coefficients: [Double]
        let intercept: Double
    }

    struct Routing: Decodable {
        let minimumSourceCharacters: Int
        let scoreThreshold: Double
    }
}

private struct MarianSourceExpertRouterPythonReport: Decodable {
    let schemaVersion: Int
    let results: [Result]

    struct Result: Decodable {
        let caseID: String
        let direction: String
        let source: String
        let score: Double
        let routesToExpert: Bool
    }
}

func verifyMarianSourceExpertRouterParity(
    enJARouterURL: URL,
    jaENRouterURL: URL,
    pythonReportURL: URL,
    tolerance: Double = 1e-5
) throws -> MarianSourceExpertRouterParityReport {
    let routers = [
        "en-ja": try MarianSourceExpertRouter(contentsOf: enJARouterURL),
        "ja-en": try MarianSourceExpertRouter(contentsOf: jaENRouterURL),
    ]
    guard routers["en-ja"]?.direction == "en-ja",
          routers["ja-en"]?.direction == "ja-en" else {
        throw MarianSourceExpertRouterError.directionMismatch
    }
    let reportData = try Data(contentsOf: pythonReportURL)
    let report = try JSONDecoder().decode(
        MarianSourceExpertRouterPythonReport.self,
        from: reportData
    )
    guard report.schemaVersion == 1, !report.results.isEmpty else {
        throw MarianSourceExpertRouterError.unsupportedReport
    }
    var caseIDs = Set<String>()
    var exactRoutes = 0
    var maximumDelta = 0.0
    for result in report.results {
        guard caseIDs.insert(result.caseID).inserted,
              let router = routers[result.direction] else {
            throw MarianSourceExpertRouterError.unsupportedReport
        }
        let swiftScore = router.score(result.source)
        maximumDelta = max(maximumDelta, abs(swiftScore - result.score))
        if router.routesToExpert(result.source) == result.routesToExpert {
            exactRoutes += 1
        }
    }
    let passed = exactRoutes == report.results.count && maximumDelta <= tolerance
    return .init(
        schemaVersion: 1,
        status: passed ? "passed" : "failed",
        cases: report.results.count,
        exactRoutes: exactRoutes,
        maximumAbsoluteScoreDelta: maximumDelta,
        tolerance: tolerance,
        pythonReportSHA256: SHA256.hash(data: reportData)
            .map { String(format: "%02x", $0) }
            .joined()
    )
}

private enum MarianSourceExpertRouterError: LocalizedError {
    case unsupportedModel(URL)
    case directionMismatch
    case unsupportedReport

    var errorDescription: String? {
        switch self {
        case let .unsupportedModel(url):
            "Unsupported Marian source-expert router: \(url.path)"
        case .directionMismatch:
            "The Marian source-expert routers do not match their directions."
        case .unsupportedReport:
            "The Python Marian source-expert router report is invalid."
        }
    }
}
