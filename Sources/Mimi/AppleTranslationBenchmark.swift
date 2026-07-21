import Darwin
import Foundation
import MimiCore
import SwiftUI
@preconcurrency import Translation

@MainActor
final class AppleTranslationBenchmarkCoordinator {
    private let suite: [TranslationBenchmarkCase]
    fileprivate let warmRuns: Int
    private let completion: (Result<TranslationBenchmarkReport, Error>) -> Void
    private var directionResults: [SpeechLanguage: AppleTranslationDirectionResult] = [:]
    private var hasCompleted = false

    init(
        suite: [TranslationBenchmarkCase],
        warmRuns: Int = 3,
        completion: @escaping (Result<TranslationBenchmarkReport, Error>) -> Void
    ) {
        self.suite = suite
        self.warmRuns = warmRuns
        self.completion = completion
    }

    fileprivate func finish(
        sourceLanguage: SpeechLanguage,
        result: Result<AppleTranslationDirectionResult, Error>
    ) {
        guard !hasCompleted else { return }
        switch result {
        case let .success(directionResult):
            directionResults[sourceLanguage] = directionResult
            let requiredLanguages = Set(suite.map(\.sourceLanguage))
            guard requiredLanguages.isSubset(of: Set(directionResults.keys)) else { return }
            hasCompleted = true
            let ordered = suite.compactMap { benchmarkCase in
                directionResults[benchmarkCase.sourceLanguage]?.results.first {
                    $0.caseID == benchmarkCase.id
                }
            }
            completion(.success(TranslationBenchmarkReport(
                engine: "apple-translation-high-fidelity",
                operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                hardware: Self.hardwareModel,
                preparationSeconds: directionResults.values.map(\.preparationSeconds).reduce(0, +),
                peakResidentBytes: Self.peakResidentBytes,
                modelBytes: nil,
                results: ordered
            )))
        case let .failure(error):
            hasCompleted = true
            completion(.failure(error))
        }
    }

    private static var hardwareModel: String {
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

    private static var peakResidentBytes: Int64? {
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else { return nil }
        return Int64(usage.ru_maxrss)
    }
}

struct AppleTranslationBenchmarkView: View {
    let suite: [TranslationBenchmarkCase]
    let coordinator: AppleTranslationBenchmarkCoordinator

    var body: some View {
        HStack(spacing: 0) {
            ForEach(SpeechLanguage.allCases) { sourceLanguage in
                AppleTranslationBenchmarkDirectionView(
                    cases: suite.filter { $0.sourceLanguage == sourceLanguage },
                    sourceLanguage: sourceLanguage,
                    coordinator: coordinator
                )
            }
        }
        .frame(width: 1, height: 1)
    }
}

private struct AppleTranslationBenchmarkDirectionView: View {
    let cases: [TranslationBenchmarkCase]
    let sourceLanguage: SpeechLanguage
    let coordinator: AppleTranslationBenchmarkCoordinator

    @State private var configuration: TranslationSession.Configuration?
    @State private var hasStarted = false

    var body: some View {
        Color.clear
            .translationTask(configuration) { @MainActor session in
                guard !hasStarted else { return }
                hasStarted = true
                do {
                    let preparationStart = ContinuousClock.now
                    try await session.prepareTranslation()
                    let preparationSeconds = preparationStart.duration(to: .now).seconds
                    var results: [TranslationBenchmarkResult] = []
                    for benchmarkCase in cases {
                        let startedAt = ContinuousClock.now
                        let response = try await session.translate(benchmarkCase.source)
                        let firstPassLatency = startedAt.duration(to: .now).seconds
                        var warmLatencies: [Double] = []
                        for _ in 0..<coordinator.warmRuns {
                            let warmStartedAt = ContinuousClock.now
                            _ = try await session.translate(benchmarkCase.source)
                            warmLatencies.append(warmStartedAt.duration(to: .now).seconds)
                        }
                        results.append(TranslationBenchmarkResult(
                            benchmarkCase: benchmarkCase,
                            hypothesis: response.targetText,
                            latencySeconds: firstPassLatency,
                            warmLatencySeconds: warmLatencies
                        ))
                    }
                    coordinator.finish(
                        sourceLanguage: sourceLanguage,
                        result: .success(.init(
                            preparationSeconds: preparationSeconds,
                            results: results
                        ))
                    )
                } catch {
                    coordinator.finish(sourceLanguage: sourceLanguage, result: .failure(error))
                }
            }
            .task {
                guard !cases.isEmpty else {
                    coordinator.finish(
                        sourceLanguage: sourceLanguage,
                        result: .success(.init(preparationSeconds: 0, results: []))
                    )
                    return
                }
                if #available(macOS 26.4, *) {
                    configuration = .init(
                        source: .init(identifier: sourceLanguage.rawValue),
                        target: .init(identifier: sourceLanguage.translationTarget.rawValue),
                        preferredStrategy: .highFidelity
                    )
                } else {
                    configuration = .init(
                        source: .init(identifier: sourceLanguage.rawValue),
                        target: .init(identifier: sourceLanguage.translationTarget.rawValue)
                    )
                }
            }
    }
}

fileprivate struct AppleTranslationDirectionResult {
    let preparationSeconds: Double
    let results: [TranslationBenchmarkResult]
}
