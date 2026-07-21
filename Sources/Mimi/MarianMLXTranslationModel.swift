import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

private let marianDimensions = 512
private let marianHeads = 8
private let marianHeadDimensions = 64
private let marianVocabularySize = 32_001
private let marianPadTokenID = 32_000
private let marianEOSTokenID = 0

private struct MarianKeyValueCache {
    let keys: MLXArray
    let values: MLXArray
}

private struct MarianDecoderLayerCache {
    let selfAttention: MarianKeyValueCache
    let encoderAttention: MarianKeyValueCache
}

private final class MarianAttention: Module {
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    override init() {
        _keyProjection.wrappedValue = Linear(marianDimensions, marianDimensions, bias: true)
        _valueProjection.wrappedValue = Linear(marianDimensions, marianDimensions, bias: true)
        _queryProjection.wrappedValue = Linear(marianDimensions, marianDimensions, bias: true)
        _outputProjection.wrappedValue = Linear(marianDimensions, marianDimensions, bias: true)
        super.init()
    }

    private func splitHeads(_ value: MLXArray) -> MLXArray {
        let batch = value.dim(0)
        let length = value.dim(1)
        return value
            .reshaped(batch, length, marianHeads, marianHeadDimensions)
            .transposed(0, 2, 1, 3)
    }

    private func joinHeads(_ value: MLXArray) -> MLXArray {
        let batch = value.dim(0)
        let length = value.dim(2)
        return value
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, marianDimensions)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        keyValueStates: MLXArray? = nil,
        causal: Bool = false
    ) -> MLXArray {
        let source = keyValueStates ?? hiddenStates
        let queries = splitHeads(queryProjection(hiddenStates))
        let keys = splitHeads(keyProjection(source))
        let values = splitHeads(valueProjection(source))
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: Float(marianHeadDimensions).squareRoot().reciprocal,
            mask: causal ? .causal : .none
        )
        return outputProjection(joinHeads(attended))
    }

    func step(
        _ hiddenStates: MLXArray,
        keyValueStates: MLXArray? = nil,
        cache: MarianKeyValueCache? = nil
    ) -> (states: MLXArray, cache: MarianKeyValueCache) {
        let queries = splitHeads(queryProjection(hiddenStates))
        let keys: MLXArray
        let values: MLXArray

        if keyValueStates != nil, let cache {
            keys = cache.keys
            values = cache.values
        } else {
            let source = keyValueStates ?? hiddenStates
            let nextKeys = splitHeads(keyProjection(source))
            let nextValues = splitHeads(valueProjection(source))
            if keyValueStates == nil, let cache {
                keys = concatenated([cache.keys, nextKeys], axis: 2)
                values = concatenated([cache.values, nextValues], axis: 2)
            } else {
                keys = nextKeys
                values = nextValues
            }
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: Float(marianHeadDimensions).squareRoot().reciprocal,
            mask: .none
        )
        return (
            outputProjection(joinHeads(attended)),
            MarianKeyValueCache(keys: keys, values: values)
        )
    }
}

private extension Float {
    var reciprocal: Float { 1 / self }
}

private final class MarianEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MarianAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    override init() {
        _selfAttention.wrappedValue = MarianAttention()
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: marianDimensions)
        _fc1.wrappedValue = Linear(marianDimensions, 2_048, bias: true)
        _fc2.wrappedValue = Linear(2_048, marianDimensions, bias: true)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: marianDimensions)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let attended = selfAttentionLayerNorm(hiddenStates + selfAttention(hiddenStates))
        return finalLayerNorm(attended + fc2(silu(fc1(attended))))
    }
}

private final class MarianDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MarianAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var encoderAttention: MarianAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttentionLayerNorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    override init() {
        _selfAttention.wrappedValue = MarianAttention()
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: marianDimensions)
        _encoderAttention.wrappedValue = MarianAttention()
        _encoderAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: marianDimensions)
        _fc1.wrappedValue = Linear(marianDimensions, 2_048, bias: true)
        _fc2.wrappedValue = Linear(2_048, marianDimensions, bias: true)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: marianDimensions)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        encoderStates: MLXArray
    ) -> MLXArray {
        let selfAttended = selfAttentionLayerNorm(
            hiddenStates + selfAttention(hiddenStates, causal: true)
        )
        let crossAttended = encoderAttentionLayerNorm(
            selfAttended + encoderAttention(selfAttended, keyValueStates: encoderStates)
        )
        return finalLayerNorm(crossAttended + fc2(silu(fc1(crossAttended))))
    }

    func step(
        _ hiddenStates: MLXArray,
        encoderStates: MLXArray,
        cache: MarianDecoderLayerCache? = nil
    ) -> (states: MLXArray, cache: MarianDecoderLayerCache) {
        let selfAttentionResult = selfAttention.step(
            hiddenStates,
            cache: cache?.selfAttention
        )
        let selfAttended = selfAttentionLayerNorm(
            hiddenStates + selfAttentionResult.states
        )
        let encoderAttentionResult = encoderAttention.step(
            selfAttended,
            keyValueStates: encoderStates,
            cache: cache?.encoderAttention
        )
        let crossAttended = encoderAttentionLayerNorm(
            selfAttended + encoderAttentionResult.states
        )
        let output = finalLayerNorm(crossAttended + fc2(silu(fc1(crossAttended))))
        return (
            output,
            MarianDecoderLayerCache(
                selfAttention: selfAttentionResult.cache,
                encoderAttention: encoderAttentionResult.cache
            )
        )
    }
}

private final class MarianEncoder: Module {
    let layers = (0..<6).map { _ in MarianEncoderLayer() }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        layers.reduce(hiddenStates) { state, layer in layer(state) }
    }
}

private final class MarianDecoder: Module {
    let layers = (0..<6).map { _ in MarianDecoderLayer() }

    func callAsFunction(_ hiddenStates: MLXArray, encoderStates: MLXArray) -> MLXArray {
        layers.reduce(hiddenStates) { state, layer in
            layer(state, encoderStates: encoderStates)
        }
    }

    func step(
        _ hiddenStates: MLXArray,
        encoderStates: MLXArray,
        caches: [MarianDecoderLayerCache]? = nil
    ) -> (states: MLXArray, caches: [MarianDecoderLayerCache]) {
        var states = hiddenStates
        var nextCaches = [MarianDecoderLayerCache]()
        nextCaches.reserveCapacity(layers.count)
        for (index, layer) in layers.enumerated() {
            let result = layer.step(
                states,
                encoderStates: encoderStates,
                cache: caches?[index]
            )
            states = result.states
            nextCaches.append(result.cache)
        }
        return (states, nextCaches)
    }
}

private final class MarianModel: Module {
    @ModuleInfo var shared: Embedding
    let encoder = MarianEncoder()
    let decoder = MarianDecoder()
    @ModuleInfo(key: "final_logits_bias") var finalLogitsBias: MLXArray

    override init() {
        _shared.wrappedValue = Embedding(
            embeddingCount: marianVocabularySize,
            dimensions: marianDimensions
        )
        _finalLogitsBias.wrappedValue = MLXArray.zeros([1, marianVocabularySize])
        super.init()
    }

    private func positions(length: Int, offset: Int = 0) -> MLXArray {
        let position = arange(
            offset,
            offset + length,
            step: 1,
            dtype: .float32
        ).expandedDimensions(axis: 1)
        let dimensions = arange(0, marianDimensions, step: 2, dtype: .float32)
        let inverseFrequency = pow(
            MLXArray(10_000 as Float),
            -(dimensions / Float(marianDimensions))
        ).expandedDimensions(axis: 0)
        let angles = position * inverseFrequency
        return concatenated([sin(angles), cos(angles)], axis: -1)
    }

    private func embed(_ tokenIDs: MLXArray, positionOffset: Int = 0) -> MLXArray {
        let embeddings = shared(tokenIDs) * Float(marianDimensions).squareRoot()
        return embeddings + positions(
            length: tokenIDs.dim(1),
            offset: positionOffset
        ).asType(embeddings.dtype)
    }

    func encode(_ tokenIDs: MLXArray) -> MLXArray {
        encoder(embed(tokenIDs))
    }

    func decode(_ tokenIDs: MLXArray, encoderStates: MLXArray) -> MLXArray {
        let states = decoder(embed(tokenIDs), encoderStates: encoderStates)
        return shared.asLinear(states) + finalLogitsBias
    }

    func decodeStep(
        decoderID: Int,
        encoderStates: MLXArray,
        caches: [MarianDecoderLayerCache]?,
        positionOffset: Int
    ) -> (logits: MLXArray, caches: [MarianDecoderLayerCache]) {
        let decoderIDs = MLXArray([decoderID]).expandedDimensions(axis: 0)
        let result = decoder.step(
            embed(decoderIDs, positionOffset: positionOffset),
            encoderStates: encoderStates,
            caches: caches
        )
        return (shared.asLinear(result.states) + finalLogitsBias, result.caches)
    }

    func generate(inputIDs: [Int], maximumTokens: Int = 192) -> [Int] {
        let encoderInput = MLXArray(inputIDs).expandedDimensions(axis: 0)
        let encoderStates = encode(encoderInput)
        var decoderIDs = [marianPadTokenID]
        var output = [Int]()
        output.reserveCapacity(min(maximumTokens, 64))
        for _ in 0..<maximumTokens {
            let decoderInput = MLXArray(decoderIDs).expandedDimensions(axis: 0)
            let logits = decode(decoderInput, encoderStates: encoderStates)[0, -1]
            let token = logits[0..<marianPadTokenID].argMax().item(Int.self)
            if token == marianEOSTokenID { break }
            output.append(token)
            decoderIDs.append(token)
        }
        return output
    }

    func generateCached(inputIDs: [Int], maximumTokens: Int = 192) -> [Int] {
        let encoderInput = MLXArray(inputIDs).expandedDimensions(axis: 0)
        let encoderStates = encode(encoderInput)
        var decoderID = marianPadTokenID
        var caches: [MarianDecoderLayerCache]?
        var output = [Int]()
        output.reserveCapacity(min(maximumTokens, 64))
        for positionOffset in 0..<maximumTokens {
            let result = decodeStep(
                decoderID: decoderID,
                encoderStates: encoderStates,
                caches: caches,
                positionOffset: positionOffset
            )
            caches = result.caches
            let token = result.logits[0, -1][0..<marianPadTokenID].argMax().item(Int.self)
            if token == marianEOSTokenID { break }
            output.append(token)
            decoderID = token
        }
        return output
    }
}

struct MarianMLXTranslationRuntime {
    private let model: MarianModel
    private let tokenizer: any Tokenizer

    static func load(directory: URL, tokenizerDataURL: URL? = nil) async throws -> Self {
        let manifest = try JSONDecoder().decode(
            MarianRuntimeManifest.self,
            from: Data(contentsOf: directory.appending(path: "manifest.json"))
        )
        guard [4, 6, 8].contains(manifest.bits),
              [32, 64, 128].contains(manifest.groupSize) else {
            throw MarianMLXTranslationRuntimeError.unsupportedQuantization(
                bits: manifest.bits,
                groupSize: manifest.groupSize
            )
        }
        let model = MarianModel()
        quantize(model: model, groupSize: manifest.groupSize, bits: manifest.bits)
        let weights = try loadArrays(url: directory.appending(path: "model.safetensors"))
        try model.update(parameters: .unflattened(weights), verify: [.all])
        eval(model)
        let tokenizer: any Tokenizer
        if let tokenizerDataURL {
            let hub = HubApi()
            let tokenizerConfig = try hub.configuration(
                fileURL: directory.appending(path: "tokenizer_config.json")
            )
            let tokenizerData = try hub.configuration(fileURL: tokenizerDataURL)
            tokenizer = try PreTrainedTokenizer(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData
            )
        } else {
            tokenizer = try await loadTokenizer(
                configuration: .init(directory: directory),
                hub: HubApi()
            )
        }
        return .init(model: model, tokenizer: tokenizer)
    }

    func translate(_ text: String, cachedDecoding: Bool = true) -> String {
        decode(tokens: translateTokenIDs(text, cachedDecoding: cachedDecoding))
    }

    func translateTokenIDs(_ text: String, cachedDecoding: Bool = true) -> [Int] {
        let input = tokenizer.encode(text: text)
        return cachedDecoding
            ? model.generateCached(inputIDs: input)
            : model.generate(inputIDs: input)
    }

    func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
    }
}

private struct MarianRuntimeManifest: Decodable {
    let bits: Int
    let groupSize: Int

    private enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }
}

private enum MarianMLXTranslationRuntimeError: LocalizedError {
    case unsupportedQuantization(bits: Int, groupSize: Int)

    var errorDescription: String? {
        switch self {
        case let .unsupportedQuantization(bits, groupSize):
            "Unsupported Marian MLX quantization: \(bits)-bit, group size \(groupSize)."
        }
    }
}
