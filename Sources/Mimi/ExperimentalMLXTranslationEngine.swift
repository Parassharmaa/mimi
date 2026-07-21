import CryptoKit
import Foundation
import MimiCore
import os

/// Resolves Mimi's bundled local translation model, with an explicit directory
/// override retained for reproducible developer benchmarks.
struct ExperimentalMLXTranslationConfiguration: Equatable, Sendable {
    static let enabledEnvironmentKey = "MIMI_EXPERIMENTAL_LOCAL_TRANSLATION"
    static let modelDirectoryEnvironmentKey = "MIMI_TRANSLATION_MODEL_DIR"

    let modelDirectory: URL

    static func resolved(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        bundle: Bundle = .main
    ) -> Self? {
        if let override = fromEnvironment(environment) {
            return override
        }
        guard environment[enabledEnvironmentKey] != "0",
              let resources = bundle.resourceURL else { return nil }
        let directory = resources.appending(
            path: "TranslationModels",
            directoryHint: .isDirectory
        )
        guard FileManager.default.fileExists(
            atPath: directory.appending(path: "manifest.json").path
        ) else { return nil }
        return .init(modelDirectory: directory)
    }

    static func fromEnvironment(_ environment: [String: String] = ProcessInfo.processInfo.environment) -> Self? {
        guard environment[enabledEnvironmentKey] == "1",
              let modelPath = environment[modelDirectoryEnvironmentKey],
              !modelPath.isEmpty else { return nil }
        return .init(
            modelDirectory: URL(filePath: modelPath, directoryHint: .isDirectory)
        )
    }
}

private struct ExperimentalMLXTranslationCacheIdentity: Equatable {
    let configuration: ExperimentalMLXTranslationConfiguration
    let rootManifestSHA256: String
}

private struct RevisionScopedDirectionalCache<Value> {
    private(set) var identity: ExperimentalMLXTranslationCacheIdentity?
    private var values: [SpeechLanguage: Value] = [:]

    var cachedLanguages: Set<SpeechLanguage> { Set(values.keys) }

    mutating func value(
        for sourceLanguage: SpeechLanguage,
        identity requestedIdentity: ExperimentalMLXTranslationCacheIdentity
    ) -> Value? {
        prepare(for: requestedIdentity)
        return values[sourceLanguage]
    }

    mutating func insert(
        _ value: Value,
        for sourceLanguage: SpeechLanguage,
        identity requestedIdentity: ExperimentalMLXTranslationCacheIdentity
    ) {
        prepare(for: requestedIdentity)
        values[sourceLanguage] = value
    }

    private mutating func prepare(for requestedIdentity: ExperimentalMLXTranslationCacheIdentity) {
        guard identity != requestedIdentity else { return }
        values.removeAll(keepingCapacity: true)
        identity = requestedIdentity
    }
}

actor ExperimentalMLXTranslationEngine {
    static let shared = ExperimentalMLXTranslationEngine()

    private let logger = Logger(subsystem: "com.paras.mimi", category: "experimental-translation")
    private var runtimeCache = RevisionScopedDirectionalCache<MarianMLXTranslationRuntime>()
    private var expertRuntimeCache = RevisionScopedDirectionalCache<MarianMLXTranslationRuntime>()
    private var routerCache = RevisionScopedDirectionalCache<MarianSourceExpertRouter>()
    private var translationMemoryCache = RevisionScopedDirectionalCache<MarianExactTranslationMemory>()

    static func validateModelPack(at directory: URL) throws {
        let root = try validatedRootManifest(at: directory)
        for direction in ["en-ja", "ja-en"] {
            try validateEngine(
                directory,
                relativePath: root.manifest.generalists[direction]!,
                direction: direction,
                root: root.manifest
            )
            if let expertPath = root.manifest.experts[direction],
               let routerPath = root.manifest.routers[direction] {
                try validateEngine(
                    directory,
                    relativePath: expertPath,
                    direction: direction,
                    root: root.manifest
                )
                let routerURL = directory.appending(path: routerPath)
                try verify(routerURL, record: root.manifest.files[routerPath])
                let router = try MarianSourceExpertRouter(contentsOf: routerURL)
                guard router.direction == direction else {
                    throw ExperimentalMLXTranslationError.unsupportedManifest(routerURL)
                }
            }
        }
        if let metadata = root.manifest.translationMemory {
            _ = try validatedTranslationMemory(
                directory: directory,
                metadata: metadata,
                root: root.manifest
            )
        }
    }

    func translate(
        _ text: String,
        sourceLanguage: SpeechLanguage,
        configuration: ExperimentalMLXTranslationConfiguration
    ) async throws -> String {
        try await translateWithDiagnostics(
            text,
            sourceLanguage: sourceLanguage,
            configuration: configuration
        ).output
    }

    func translateWithDiagnostics(
        _ text: String,
        sourceLanguage: SpeechLanguage,
        configuration: ExperimentalMLXTranslationConfiguration
    ) async throws -> ExperimentalMLXTranslationDiagnostic {
#if arch(arm64)
        if let memory = try loadTranslationMemory(
            sourceLanguage: sourceLanguage,
            configuration: configuration
        ), let output = memory.translation(for: text, sourceLanguage: sourceLanguage) {
            guard Self.preservesCriticalTokens(source: text, output: output) else {
                throw ExperimentalMLXTranslationError.criticalTokenMismatch
            }
            guard Self.isPlausible(output, source: text, sourceLanguage: sourceLanguage) else {
                throw ExperimentalMLXTranslationError.implausibleOutput
            }
            return .init(
                output: output,
                outputTokenIDs: nil,
                routedToExpert: false,
                usedTranslationMemory: true
            )
        }
        let useExpert = try shouldUseExpert(
            text,
            sourceLanguage: sourceLanguage,
            configuration: configuration
        )
        let runtime = try await loadRuntime(
            sourceLanguage: sourceLanguage,
            configuration: configuration,
            expert: useExpert
        )
        var outputTokenIDs = runtime.translateTokenIDs(text)
        var output = Self.clean(runtime.decode(tokens: outputTokenIDs))
        var selectedExpert = useExpert
        if !Self.preservesCriticalTokens(source: text, output: output) {
            if useExpert {
                let generalist = try await loadRuntime(
                    sourceLanguage: sourceLanguage,
                    configuration: configuration,
                    expert: false
                )
                let fallbackTokenIDs = generalist.translateTokenIDs(text)
                let fallback = Self.clean(generalist.decode(tokens: fallbackTokenIDs))
                guard Self.preservesCriticalTokens(source: text, output: fallback) else {
                    throw ExperimentalMLXTranslationError.criticalTokenMismatch
                }
                output = fallback
                outputTokenIDs = fallbackTokenIDs
                selectedExpert = false
            } else {
                throw ExperimentalMLXTranslationError.criticalTokenMismatch
            }
        }
        guard Self.isPlausible(output, source: text, sourceLanguage: sourceLanguage) else {
            throw ExperimentalMLXTranslationError.implausibleOutput
        }
        return .init(
            output: output,
            outputTokenIDs: outputTokenIDs,
            routedToExpert: selectedExpert,
            usedTranslationMemory: false
        )
#else
        throw ExperimentalMLXTranslationError.unsupportedArchitecture
#endif
    }

    private func loadTranslationMemory(
        sourceLanguage: SpeechLanguage,
        configuration: ExperimentalMLXTranslationConfiguration
    ) throws -> MarianExactTranslationMemory? {
        let identity = try Self.validatedCacheIdentity(for: configuration)
        if let memory = translationMemoryCache.value(
            for: sourceLanguage,
            identity: identity
        ) {
            return memory
        }
        let root = try Self.validatedRootManifest(at: configuration.modelDirectory)
        guard root.sha256 == identity.rootManifestSHA256 else {
            throw ExperimentalMLXTranslationError.packChangedDuringLoad(
                configuration.modelDirectory
            )
        }
        guard let metadata = root.manifest.translationMemory else { return nil }
        let memory = try Self.validatedTranslationMemory(
            directory: configuration.modelDirectory,
            metadata: metadata,
            root: root.manifest
        )
        let rootAfterLoad = try Self.validatedRootManifest(at: configuration.modelDirectory)
        guard rootAfterLoad.sha256 == identity.rootManifestSHA256 else {
            throw ExperimentalMLXTranslationError.packChangedDuringLoad(
                configuration.modelDirectory
            )
        }
        translationMemoryCache.insert(memory, for: sourceLanguage, identity: identity)
        return memory
    }

    private func loadRuntime(
        sourceLanguage: SpeechLanguage,
        configuration: ExperimentalMLXTranslationConfiguration,
        expert: Bool
    ) async throws -> MarianMLXTranslationRuntime {
        let identity = try Self.validatedCacheIdentity(for: configuration)
        if expert {
            if let runtime = expertRuntimeCache.value(
                for: sourceLanguage,
                identity: identity
            ) {
                return runtime
            }
        } else if let runtime = runtimeCache.value(
            for: sourceLanguage,
            identity: identity
        ) {
            return runtime
        }
        guard Self.metalLibraryURL != nil else {
            throw ExperimentalMLXTranslationError.missingMetalRuntime
        }
        let direction = sourceLanguage == .english ? "en-ja" : "ja-en"
        let root = try Self.validatedRootManifest(at: configuration.modelDirectory)
        guard root.sha256 == identity.rootManifestSHA256 else {
            throw ExperimentalMLXTranslationError.packChangedDuringLoad(
                configuration.modelDirectory
            )
        }
        let enginePath = expert
            ? root.manifest.experts[direction]
            : root.manifest.generalists[direction]
        guard let enginePath else {
            throw ExperimentalMLXTranslationError.unsupportedManifest(
                configuration.modelDirectory.appending(path: "manifest.json")
            )
        }
        try Self.validateEngine(
            configuration.modelDirectory,
            relativePath: enginePath,
            direction: direction,
            root: root.manifest
        )
        let directionDirectory = configuration.modelDirectory.appending(
            path: enginePath,
            directoryHint: .isDirectory
        )

        logger.notice("Loading Mimi's local quantized Marian translation model")
        let sharedTokenizerURL = root.manifest.sharedTokenizer.map {
            configuration.modelDirectory.appending(path: $0)
        }
        let loaded = try await MarianMLXTranslationRuntime.load(
            directory: directionDirectory,
            tokenizerDataURL: sharedTokenizerURL
        )
        let rootAfterLoad = try Self.validatedRootManifest(
            at: configuration.modelDirectory
        )
        guard rootAfterLoad.sha256 == identity.rootManifestSHA256 else {
            throw ExperimentalMLXTranslationError.packChangedDuringLoad(
                configuration.modelDirectory
            )
        }
        try Self.validateEngine(
            configuration.modelDirectory,
            relativePath: enginePath,
            direction: direction,
            root: rootAfterLoad.manifest
        )
        if expert {
            expertRuntimeCache.insert(
                loaded,
                for: sourceLanguage,
                identity: identity
            )
        } else {
            runtimeCache.insert(
                loaded,
                for: sourceLanguage,
                identity: identity
            )
        }
        return loaded
    }

    private func shouldUseExpert(
        _ source: String,
        sourceLanguage: SpeechLanguage,
        configuration: ExperimentalMLXTranslationConfiguration
    ) throws -> Bool {
        let identity = try Self.validatedCacheIdentity(for: configuration)
        let root = try Self.validatedRootManifest(at: configuration.modelDirectory)
        let direction = sourceLanguage == .english ? "en-ja" : "ja-en"
        guard let routerPath = root.manifest.routers[direction] else { return false }
        if let router = routerCache.value(for: sourceLanguage, identity: identity) {
            return router.routesToExpert(source)
        }
        let routerURL = configuration.modelDirectory.appending(path: routerPath)
        try Self.verify(routerURL, record: root.manifest.files[routerPath])
        let router = try MarianSourceExpertRouter(contentsOf: routerURL)
        guard router.direction == direction else {
            throw ExperimentalMLXTranslationError.unsupportedManifest(routerURL)
        }
        let rootAfterLoad = try Self.validatedRootManifest(at: configuration.modelDirectory)
        guard rootAfterLoad.sha256 == identity.rootManifestSHA256 else {
            throw ExperimentalMLXTranslationError.packChangedDuringLoad(
                configuration.modelDirectory
            )
        }
        routerCache.insert(router, for: sourceLanguage, identity: identity)
        return router.routesToExpert(source)
    }

    private static func validatedCacheIdentity(
        for configuration: ExperimentalMLXTranslationConfiguration
    ) throws -> ExperimentalMLXTranslationCacheIdentity {
        let root = try validatedRootManifest(at: configuration.modelDirectory)
        return .init(
            configuration: configuration,
            rootManifestSHA256: root.sha256
        )
    }

    private static func validatedRootManifest(
        at directory: URL
    ) throws -> (manifest: TranslationPackManifest, sha256: String) {
        let rootManifestURL = directory.appending(path: "manifest.json")
        guard FileManager.default.fileExists(atPath: rootManifestURL.path) else {
            throw ExperimentalMLXTranslationError.incompleteDirectory(directory, ["manifest.json"])
        }
        let rootManifestData: Data
        do {
            rootManifestData = try Data(contentsOf: rootManifestURL)
        } catch {
            throw ExperimentalMLXTranslationError.invalidManifest(rootManifestURL, error)
        }
        let decoder = JSONDecoder()
        let header: TranslationPackFormatHeader
        do {
            header = try decoder.decode(TranslationPackFormatHeader.self, from: rootManifestData)
        } catch {
            throw ExperimentalMLXTranslationError.invalidManifest(rootManifestURL, error)
        }
        let rootManifest: TranslationPackManifest
        do {
            switch header.format {
            case "mimi-mlx-marian-pair-v1":
                let payload = try decoder.decode(
                    TranslationPairPackManifestPayload.self,
                    from: rootManifestData
                )
                guard payload.interface == "bidirectional-en-ja",
                      Set(payload.engines) == Set(["en-ja", "ja-en"]),
                      [4, 6, 8].contains(payload.quantization.bits),
                      [32, 64, 128].contains(payload.quantization.groupSize) else {
                    throw ExperimentalMLXTranslationError.unsupportedManifest(rootManifestURL)
                }
                rootManifest = .init(
                    format: payload.format,
                    quantization: .init(
                        bits: payload.quantization.bits,
                        groupSize: payload.quantization.groupSize
                    ),
                    generalists: ["en-ja": "en-ja", "ja-en": "ja-en"],
                    experts: [:],
                    routers: [:],
                    translationMemory: nil,
                    sharedTokenizer: nil,
                    files: payload.files
                )
            case "mimi-mlx-marian-moe-v1", "mimi-mlx-marian-moe-v2":
                let payload = try decoder.decode(
                    TranslationMoEPackManifestPayload.self,
                    from: rootManifestData
                )
                let sharedTokenizer: String?
                if header.format == "mimi-mlx-marian-moe-v2" {
                    guard let path = payload.sharedTokenizer,
                          isSafeRelativePath(path),
                          payload.files[path] != nil else {
                        throw ExperimentalMLXTranslationError.unsupportedManifest(rootManifestURL)
                    }
                    sharedTokenizer = path
                } else {
                    guard payload.sharedTokenizer == nil else {
                        throw ExperimentalMLXTranslationError.unsupportedManifest(rootManifestURL)
                    }
                    sharedTokenizer = nil
                }
                guard payload.interface == "bidirectional-en-ja",
                      payload.quantization.bits == 4,
                      payload.quantization.groupSize == 64,
                      Set(payload.generalists.keys) == Set(["en-ja", "ja-en"]),
                      Set(payload.experts.keys) == Set(["en-ja", "ja-en"]),
                      payload.routing.inputs == "source-text-only",
                      payload.routing.defaultOnRouterFailure == "generalist" else {
                    throw ExperimentalMLXTranslationError.unsupportedManifest(rootManifestURL)
                }
                rootManifest = .init(
                    format: payload.format,
                    quantization: .init(
                        bits: payload.quantization.bits,
                        groupSize: payload.quantization.groupSize
                    ),
                    generalists: payload.generalists,
                    experts: payload.experts.mapValues(\.engine),
                    routers: payload.experts.mapValues(\.router),
                    translationMemory: payload.translationMemory,
                    sharedTokenizer: sharedTokenizer,
                    files: payload.files
                )
            default:
                throw ExperimentalMLXTranslationError.unsupportedManifest(rootManifestURL)
            }
        } catch let error as ExperimentalMLXTranslationError {
            throw error
        } catch {
            throw ExperimentalMLXTranslationError.invalidManifest(rootManifestURL, error)
        }
        guard Set(rootManifest.generalists.keys) == Set(["en-ja", "ja-en"]),
              rootManifest.generalists.values.allSatisfy({ !$0.isEmpty }) else {
            throw ExperimentalMLXTranslationError.unsupportedManifest(rootManifestURL)
        }
        let digest = SHA256.hash(data: rootManifestData)
            .map { String(format: "%02x", $0) }
            .joined()
        return (rootManifest, digest)
    }

    private static func validatedTranslationMemory(
        directory: URL,
        metadata: TranslationMemoryManifest,
        root: TranslationPackManifest
    ) throws -> MarianExactTranslationMemory {
        let rootManifestURL = directory.appending(path: "manifest.json")
        guard metadata.schemaVersion == 1,
              metadata.normalization == MarianExactTranslationMemory.normalizationContract,
              metadata.sourceLicense == MarianExactTranslationMemory.sourceLicenseContract,
              metadata.lookup == "exact normalized source before neural routing",
              metadata.entries > 0,
              metadata.maximumSourceCharacters <= MarianExactTranslationMemory.maximumSourceCharacters,
              metadata.maximumTargetCharacters <= MarianExactTranslationMemory.maximumTargetCharacters,
              isSafeRelativePath(metadata.path) else {
            throw ExperimentalMLXTranslationError.unsupportedManifest(rootManifestURL)
        }
        let memoryURL = directory.appending(path: metadata.path)
        try verify(memoryURL, record: root.files[metadata.path])
        let memory = try MarianExactTranslationMemory(contentsOf: memoryURL)
        guard memory.entryCount == metadata.entries,
              memory.longestSource == metadata.maximumSourceCharacters,
              memory.longestTarget == metadata.maximumTargetCharacters,
              memory.trainingDataSHA256 == metadata.trainingDataSHA256.lowercased(),
              memory.auditSHA256 == metadata.auditSHA256.lowercased() else {
            throw ExperimentalMLXTranslationError.unsupportedManifest(rootManifestURL)
        }
        return memory
    }

    private static func isSafeRelativePath(_ path: String) -> Bool {
        guard !path.isEmpty, !path.hasPrefix("/") else { return false }
        let components = NSString(string: path).pathComponents
        return !components.contains("..") && !components.contains(".")
    }

    private static func validateEngine(
        _ directory: URL,
        relativePath: String,
        direction: String,
        root rootManifest: TranslationPackManifest
    ) throws {
        let directionDirectory = directory.appending(
            path: relativePath,
            directoryHint: .isDirectory
        )
        var requiredFiles = ["manifest.json", "model.safetensors", "tokenizer_config.json"]
        if rootManifest.sharedTokenizer == nil {
            requiredFiles.append("tokenizer.json")
        }
        let missing = requiredFiles.filter {
            !FileManager.default.fileExists(atPath: directionDirectory.appending(path: $0).path)
        }
        guard missing.isEmpty else {
            throw ExperimentalMLXTranslationError.incompleteDirectory(directionDirectory, missing)
        }
        if let sharedTokenizer = rootManifest.sharedTokenizer {
            let sharedTokenizerURL = directory.appending(path: sharedTokenizer)
            guard FileManager.default.fileExists(atPath: sharedTokenizerURL.path) else {
                throw ExperimentalMLXTranslationError.incompleteDirectory(
                    directory,
                    [sharedTokenizer]
                )
            }
            try verify(sharedTokenizerURL, record: rootManifest.files[sharedTokenizer])
        }
        try verify(
            directionDirectory.appending(path: "manifest.json"),
            record: rootManifest.files["\(relativePath)/manifest.json"]
        )

        let directionManifestURL = directionDirectory.appending(path: "manifest.json")
        let directionManifest: TranslationDirectionManifest
        do {
            directionManifest = try JSONDecoder().decode(
                TranslationDirectionManifest.self,
                from: Data(contentsOf: directionManifestURL)
            )
        } catch {
            throw ExperimentalMLXTranslationError.invalidManifest(directionManifestURL, error)
        }
        guard directionManifest.format == "mimi-mlx-marian-v1",
              directionManifest.direction == direction,
              directionManifest.bits == rootManifest.quantization.bits,
              directionManifest.groupSize == rootManifest.quantization.groupSize else {
            throw ExperimentalMLXTranslationError.unsupportedManifest(directionManifestURL)
        }
        for name in requiredFiles where name != "manifest.json" {
            try verify(
                directionDirectory.appending(path: name),
                record: directionManifest.files[name]
            )
        }
    }

    private static func verify(_ url: URL, record: TranslationFileRecord?) throws {
        guard let record else {
            throw ExperimentalMLXTranslationError.missingChecksum(url)
        }
        let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize
        guard size == record.bytes, try sha256(url) == record.sha256.lowercased() else {
            throw ExperimentalMLXTranslationError.checksumMismatch(url)
        }
    }

    private static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static var metalLibraryURL: URL? {
        guard let executable = CommandLine.arguments.first, !executable.isEmpty else { return nil }
        let directory = URL(filePath: executable).deletingLastPathComponent()
        return [
            directory.appending(path: "mlx.metallib"),
            directory.appending(path: "Resources/mlx.metallib"),
        ].first { FileManager.default.fileExists(atPath: $0.path) }
    }

    private static func clean(_ raw: String) -> String {
        var output = raw
        if let thinkEnd = output.range(of: "</think>", options: .backwards) {
            output = String(output[thinkEnd.upperBound...])
        }
        output = output.trimmingCharacters(in: .whitespacesAndNewlines)
        if output.count >= 2, output.first == "\"", output.last == "\"" {
            output.removeFirst()
            output.removeLast()
        }
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func isPlausible(
        _ output: String,
        source: String,
        sourceLanguage: SpeechLanguage
    ) -> Bool {
        guard !output.isEmpty, output != source, output.count <= max(64, source.count * 5) else {
            return false
        }
        switch sourceLanguage {
        case .english:
            return output.unicodeScalars.contains {
                (0x3040...0x30ff).contains($0.value) || (0x3400...0x9fff).contains($0.value)
            }
        case .japanese:
            return output.unicodeScalars.contains {
                (0x41...0x5a).contains($0.value) || (0x61...0x7a).contains($0.value)
            }
        }
    }

    static func preservesCriticalTokens(source: String, output: String) -> Bool {
        if criticalTokens(source) == criticalTokens(output) {
            return true
        }
        guard let sourcePercentage = singlePercentageSignature(source),
              let outputPercentage = singlePercentageSignature(output) else {
            return false
        }
        return sourcePercentage == outputPercentage
    }

    private struct SinglePercentageSignature: Equatable {
        let protected: [String]
        let percentage: String
        let otherNumbers: [String]
    }

    private static func singlePercentageSignature(
        _ value: String
    ) -> SinglePercentageSignature? {
        let normalized = value.precomposedStringWithCompatibilityMapping
        let protectedPattern = #"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>"#
        let percentagePattern = #"(?i)%|\bpercent\b|\bper\s+cent\b|パーセント"#
        let expressionPattern = #"(?i)(?<![\d.])((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?!\d|\.\d)\s*(?:%|\bpercent\b|\bper\s+cent\b|パーセント)"#
        let numberPattern = #"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"#
        guard let protectedExpression = try? NSRegularExpression(pattern: protectedPattern),
              let percentageExpression = try? NSRegularExpression(pattern: percentagePattern),
              let expression = try? NSRegularExpression(pattern: expressionPattern),
              let numberExpression = try? NSRegularExpression(pattern: numberPattern) else {
            return nil
        }

        let fullRange = NSRange(normalized.startIndex..., in: normalized)
        let protectedMatches = protectedExpression.matches(in: normalized, range: fullRange)
        let protected = protectedMatches.compactMap { match -> String? in
            guard let range = Range(match.range, in: normalized) else { return nil }
            return String(normalized[range])
        }.sorted()
        let maskedProtected = NSMutableString(string: normalized)
        for match in protectedMatches.reversed() {
            maskedProtected.replaceCharacters(
                in: match.range,
                with: String(repeating: " ", count: match.range.length)
            )
        }
        let masked = String(maskedProtected)
        let maskedRange = NSRange(masked.startIndex..., in: masked)
        let percentageMatches = percentageExpression.matches(in: masked, range: maskedRange)
        let expressionMatches = expression.matches(in: masked, range: maskedRange)
        guard percentageMatches.count == 1,
              expressionMatches.count == 1,
              let percentageRange = Range(expressionMatches[0].range(at: 1), in: masked) else {
            return nil
        }
        let rawPercentage = masked[percentageRange].replacingOccurrences(of: ",", with: "")
        let number = NSDecimalNumber(string: rawPercentage)
        guard number != .notANumber else { return nil }

        let remainder = NSMutableString(string: masked)
        remainder.replaceCharacters(
            in: expressionMatches[0].range,
            with: String(repeating: " ", count: expressionMatches[0].range.length)
        )
        let remainderString = String(remainder)
        let remainderRange = NSRange(remainderString.startIndex..., in: remainderString)
        let otherMatches = numberExpression.matches(in: remainderString, range: remainderRange)
        let otherNumbers = otherMatches.compactMap { match -> String? in
            guard let range = Range(match.range, in: remainderString) else { return nil }
            return remainderString[range].replacingOccurrences(of: ",", with: "")
        }.sorted()
        for match in otherMatches.reversed() {
            remainder.replaceCharacters(
                in: match.range,
                with: String(repeating: " ", count: match.range.length)
            )
        }
        guard String(remainder).unicodeScalars.allSatisfy({
            !CharacterSet.decimalDigits.contains($0)
        }) else {
            return nil
        }
        return .init(
            protected: protected,
            percentage: number.stringValue,
            otherNumbers: otherNumbers
        )
    }

    private static func criticalTokens(_ value: String) -> [String] {
        let normalized = value.precomposedStringWithCompatibilityMapping
        // Translation output commonly joins Japanese text directly to a URL.
        // Restrict URL matching to RFC 3986's ASCII character repertoire so
        // adjacent kana/kanji are not mistaken for part of the protected URL.
        let pattern = #"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>|%|(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"#
        guard let expression = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(normalized.startIndex..., in: normalized)
        return expression.matches(in: normalized, range: range).compactMap { match in
            guard let swiftRange = Range(match.range, in: normalized) else { return nil }
            return normalized[swiftRange].replacingOccurrences(of: ",", with: "")
        }.sorted()
    }
}

struct ExperimentalMLXTranslationDiagnostic: Sendable {
    let output: String
    let outputTokenIDs: [Int]?
    let routedToExpert: Bool
    let usedTranslationMemory: Bool
}

struct ExperimentalMLXTranslationMoESmokeReport: Codable, Sendable {
    let schemaVersion: Int
    let status: String
    let direction: String
    let expectedEngine: String
    let selectedEngine: String
    let source: String
    let hypothesis: String
    let outputTokenIDs: [Int]?
}

func verifyExperimentalMLXTranslationMoESmoke(
    modelRoot: URL,
    source: String,
    sourceLanguage: SpeechLanguage,
    expectedEngine: String
) async throws -> ExperimentalMLXTranslationMoESmokeReport {
    let diagnostic = try await ExperimentalMLXTranslationEngine.shared
        .translateWithDiagnostics(
            source,
            sourceLanguage: sourceLanguage,
            configuration: .init(modelDirectory: modelRoot)
        )
    let selectedEngine = diagnostic.usedTranslationMemory
        ? "translation-memory"
        : diagnostic.routedToExpert ? "expert" : "generalist"
    return .init(
        schemaVersion: 1,
        status: selectedEngine == expectedEngine ? "passed" : "failed",
        direction: sourceLanguage == .english ? "en-ja" : "ja-en",
        expectedEngine: expectedEngine,
        selectedEngine: selectedEngine,
        source: source,
        hypothesis: diagnostic.output,
        outputTokenIDs: diagnostic.outputTokenIDs
    )
}

struct TranslationRuntimeCacheVerificationReport: Codable {
    let schemaVersion: Int
    let status: String
    let retainsBothDirections: Bool
    let reusesFirstDirectionAfterSwitch: Bool
    let samePathManifestDigestChangeClearsBothDirections: Bool
    let configurationChangeClearsBothDirections: Bool
    let newConfigurationCanCacheBothDirections: Bool
    let criticalTokenGuardPasses: Bool
    let translationMemoryNormalizationPasses: Bool
}

func verifyExperimentalTranslationRuntimeCacheContract() -> TranslationRuntimeCacheVerificationReport {
    let firstConfiguration = ExperimentalMLXTranslationConfiguration(
        modelDirectory: URL(filePath: "/tmp/mimi-translation-cache-a", directoryHint: .isDirectory)
    )
    let secondConfiguration = ExperimentalMLXTranslationConfiguration(
        modelDirectory: URL(filePath: "/tmp/mimi-translation-cache-b", directoryHint: .isDirectory)
    )
    let firstRevision = ExperimentalMLXTranslationCacheIdentity(
        configuration: firstConfiguration,
        rootManifestSHA256: "manifest-a-v1"
    )
    let replacedFirstRevision = ExperimentalMLXTranslationCacheIdentity(
        configuration: firstConfiguration,
        rootManifestSHA256: "manifest-a-v2"
    )
    let secondRevision = ExperimentalMLXTranslationCacheIdentity(
        configuration: secondConfiguration,
        rootManifestSHA256: "manifest-b-v1"
    )
    var cache = RevisionScopedDirectionalCache<Int>()

    cache.insert(101, for: .english, identity: firstRevision)
    cache.insert(202, for: .japanese, identity: firstRevision)
    let retainsBothDirections = cache.cachedLanguages == Set(SpeechLanguage.allCases)
    let reusesFirstDirectionAfterSwitch = cache.value(
        for: .english,
        identity: firstRevision
    ) == 101

    let oldValueAfterSamePathRevisionChange = cache.value(
        for: .japanese,
        identity: replacedFirstRevision
    )
    let samePathManifestDigestChangeClearsBothDirections = oldValueAfterSamePathRevisionChange == nil
        && cache.cachedLanguages.isEmpty

    cache.insert(303, for: .english, identity: replacedFirstRevision)
    cache.insert(404, for: .japanese, identity: replacedFirstRevision)
    let oldValueAfterConfigurationChange = cache.value(
        for: .english,
        identity: secondRevision
    )
    let configurationChangeClearsBothDirections = oldValueAfterConfigurationChange == nil
        && cache.cachedLanguages.isEmpty

    cache.insert(505, for: .english, identity: secondRevision)
    cache.insert(606, for: .japanese, identity: secondRevision)
    let newConfigurationCanCacheBothDirections = cache.value(
        for: .english,
        identity: secondRevision
    ) == 505
        && cache.value(for: .japanese, identity: secondRevision) == 606
        && cache.cachedLanguages == Set(SpeechLanguage.allCases)
    let criticalTokenGuardPasses = ExperimentalMLXTranslationEngine.preservesCriticalTokens(
        source: "Open https://example.com at 14:30 with {name}.",
        output: "{name}を使って14:30にhttps://example.comを開きます。"
    ) && ExperimentalMLXTranslationEngine.preservesCriticalTokens(
        source: "Version １２ costs 1,200 yen.",
        output: "バージョン12は1200円です。"
    ) && ExperimentalMLXTranslationEngine.preservesCriticalTokens(
        source: "Record 12. Version 1.2.3.",
        output: "記録は12。バージョン1.2.3。"
    ) && ExperimentalMLXTranslationEngine.preservesCriticalTokens(
        source: "Provide 25 percent by 2025.",
        output: "2025年までに25%を供給する。"
    ) && ExperimentalMLXTranslationEngine.preservesCriticalTokens(
        source: "費用は25パーセントです。",
        output: "The cost is 25%."
    ) && !ExperimentalMLXTranslationEngine.preservesCriticalTokens(
        source: "Provide 25 percent by 2025.",
        output: "2026年までに25%を供給する。"
    ) && !ExperimentalMLXTranslationEngine.preservesCriticalTokens(
        source: "25 percent and 30 percent",
        output: "30%と25%"
    ) && !ExperimentalMLXTranslationEngine.preservesCriticalTokens(
        source: "25 percent at https://example.com",
        output: "https://example.netで25%"
    ) && !ExperimentalMLXTranslationEngine.preservesCriticalTokens(
        source: "Values 1,2.",
        output: "値は12。"
    ) && !ExperimentalMLXTranslationEngine.preservesCriticalTokens(
        source: "Keep 25% at https://example.com.",
        output: "https://example.netで20%を維持します。"
    )
    let translationMemoryNormalizationPasses = MarianExactTranslationMemory.normalize(
        "  Ａ\t\nＢ　Ｃ  "
    ) == "A B C"

    let passed = retainsBothDirections
        && reusesFirstDirectionAfterSwitch
        && samePathManifestDigestChangeClearsBothDirections
        && configurationChangeClearsBothDirections
        && newConfigurationCanCacheBothDirections
        && criticalTokenGuardPasses
        && translationMemoryNormalizationPasses
    return .init(
        schemaVersion: 1,
        status: passed ? "passed" : "failed",
        retainsBothDirections: retainsBothDirections,
        reusesFirstDirectionAfterSwitch: reusesFirstDirectionAfterSwitch,
        samePathManifestDigestChangeClearsBothDirections:
            samePathManifestDigestChangeClearsBothDirections,
        configurationChangeClearsBothDirections: configurationChangeClearsBothDirections,
        newConfigurationCanCacheBothDirections: newConfigurationCanCacheBothDirections,
        criticalTokenGuardPasses: criticalTokenGuardPasses,
        translationMemoryNormalizationPasses: translationMemoryNormalizationPasses
    )
}

struct TranslationCriticalTokenFailureReport: Codable, Sendable {
    let schemaVersion: Int
    let status: String
    let direction: String
    let source: String
    let expectedFailure: String
    let observedFailure: String
}

func verifyExperimentalTranslationCriticalTokenFailure(
    modelRoot: URL,
    source: String,
    sourceLanguage: SpeechLanguage
) async -> TranslationCriticalTokenFailureReport {
    let expectedFailure = "critical-token-mismatch"
    do {
        let diagnostic = try await ExperimentalMLXTranslationEngine.shared
            .translateWithDiagnostics(
                source,
                sourceLanguage: sourceLanguage,
                configuration: .init(modelDirectory: modelRoot)
            )
        return .init(
            schemaVersion: 1,
            status: "failed",
            direction: sourceLanguage == .english ? "en-ja" : "ja-en",
            source: source,
            expectedFailure: expectedFailure,
            observedFailure: "unexpected-success:\(diagnostic.output)"
        )
    } catch ExperimentalMLXTranslationError.criticalTokenMismatch {
        return .init(
            schemaVersion: 1,
            status: "passed",
            direction: sourceLanguage == .english ? "en-ja" : "ja-en",
            source: source,
            expectedFailure: expectedFailure,
            observedFailure: expectedFailure
        )
    } catch {
        return .init(
            schemaVersion: 1,
            status: "failed",
            direction: sourceLanguage == .english ? "en-ja" : "ja-en",
            source: source,
            expectedFailure: expectedFailure,
            observedFailure: error.localizedDescription
        )
    }
}

private enum ExperimentalMLXTranslationError: LocalizedError {
    case unsupportedArchitecture
    case missingMetalRuntime
    case incompleteDirectory(URL, [String])
    case invalidManifest(URL, Error)
    case unsupportedManifest(URL)
    case packChangedDuringLoad(URL)
    case missingChecksum(URL)
    case checksumMismatch(URL)
    case implausibleOutput
    case criticalTokenMismatch

    var errorDescription: String? {
        switch self {
        case .unsupportedArchitecture:
            "The experimental MLX translator requires Apple silicon."
        case .missingMetalRuntime:
            "The experimental MLX translator needs Mimi's matching bundled mlx.metallib."
        case let .incompleteDirectory(directory, missing):
            "The experimental translation directory at \(directory.path) is missing: \(missing.joined(separator: ", "))."
        case let .invalidManifest(url, error):
            "The experimental translation manifest at \(url.path) is invalid: \(error.localizedDescription)"
        case let .unsupportedManifest(url):
            "The experimental translation manifest at \(url.path) has an unsupported format or quantization."
        case let .packChangedDuringLoad(url):
            "The experimental translation model pack at \(url.path) changed while it was being loaded."
        case let .missingChecksum(url):
            "The experimental translation manifest has no checksum for \(url.lastPathComponent)."
        case let .checksumMismatch(url):
            "The experimental translation file failed size or SHA-256 verification: \(url.path)"
        case .implausibleOutput:
            "The experimental translator returned an empty, untranslated, or malformed result."
        case .criticalTokenMismatch:
            "The selected local translation path changed a protected URL, placeholder, markup token, or number."
        }
    }
}

private struct TranslationFileRecord: Decodable {
    let bytes: Int
    let sha256: String
}

private struct TranslationQuantizationManifest {
    let bits: Int
    let groupSize: Int
}

private struct TranslationPackFormatHeader: Decodable {
    let format: String
}

private struct TranslationPairQuantizationPayload: Decodable {
    let bits: Int
    let groupSize: Int

    private enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }
}

private struct TranslationPairPackManifestPayload: Decodable {
    let format: String
    let interface: String
    let engines: [String]
    let quantization: TranslationPairQuantizationPayload
    let files: [String: TranslationFileRecord]
}

private struct TranslationMoEPackManifestPayload: Decodable {
    let format: String
    let interface: String
    let quantization: TranslationQuantizationManifestPayload
    let generalists: [String: String]
    let experts: [String: Expert]
    let routing: Routing
    let translationMemory: TranslationMemoryManifest?
    let sharedTokenizer: String?
    let files: [String: TranslationFileRecord]

    struct TranslationQuantizationManifestPayload: Decodable {
        let bits: Int
        let groupSize: Int
    }

    struct Expert: Decodable {
        let engine: String
        let router: String
    }

    struct Routing: Decodable {
        let inputs: String
        let defaultOnRouterFailure: String
    }
}

struct TranslationMemoryManifest: Decodable {
    let path: String
    let schemaVersion: Int
    let normalization: String
    let sourceLicense: String
    let trainingDataSHA256: String
    let auditSHA256: String
    let entries: Int
    let maximumSourceCharacters: Int
    let maximumTargetCharacters: Int
    let lookup: String
}

private struct TranslationPackManifest {
    let format: String
    let quantization: TranslationQuantizationManifest
    let generalists: [String: String]
    let experts: [String: String]
    let routers: [String: String]
    let translationMemory: TranslationMemoryManifest?
    let sharedTokenizer: String?
    let files: [String: TranslationFileRecord]
}

private struct TranslationDirectionManifest: Decodable {
    let format: String
    let direction: String
    let bits: Int
    let groupSize: Int
    let files: [String: TranslationFileRecord]

    private enum CodingKeys: String, CodingKey {
        case format, direction, bits, files
        case groupSize = "group_size"
    }
}
