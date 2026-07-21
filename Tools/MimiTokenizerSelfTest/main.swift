import Foundation
import Hub
import Tokenizers

enum TokenizerSelfTestError: LocalizedError {
    case usage
    case mismatch(String)

    var errorDescription: String? {
        switch self {
        case .usage:
            "usage: MimiTokenizerSelfTest <ElanMT MLX direction directory>"
        case let .mismatch(message):
            "tokenizer mismatch: \(message)"
        }
    }
}

do {
    guard CommandLine.arguments.count == 2 else { throw TokenizerSelfTestError.usage }
    let directory = URL(
        filePath: CommandLine.arguments[1],
        directoryHint: .isDirectory
    )
    let hub = HubApi()
    let configuration = try hub.configuration(
        fileURL: directory.appending(path: "tokenizer_config.json")
    )
    let tokenizerData = try hub.configuration(
        fileURL: directory.appending(path: "tokenizer.json")
    )
    let tokenizer = try PreTrainedTokenizer(
        tokenizerConfig: configuration,
        tokenizerData: tokenizerData
    )

    let english = "The train leaves at 18:42 from platform 7 on October 3."
    let expectedEnglish = [30, 1732, 7020, 44, 349, 59, 2746, 39, 4411, 569, 35, 644, 308, 7, 0]
    guard tokenizer.encode(text: english) == expectedEnglish else {
        throw TokenizerSelfTestError.mismatch("English IDs differ from SentencePiece")
    }

    let japanese = "Mimiを再起動してから、LanguageをEnglishに戻してください。"
    let expectedJapanese = [1031, 582, 9321, 16791, 15918, 4, 602, 242, 842, 1001, 19, 29013, 21, 3283, 12028, 6, 0]
    guard tokenizer.encode(text: japanese) == expectedJapanese else {
        throw TokenizerSelfTestError.mismatch("Japanese IDs differ from SentencePiece")
    }

    let translatedIDs = [3, 2214, 12, 148, 45, 68, 73, 136, 1749, 3876, 54, 360, 454, 2746, 451, 800, 509, 6]
    let decoded = tokenizer.decode(tokens: translatedIDs, skipSpecialTokens: true)
    let normalizedDecoded = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    guard normalizedDecoded == "列車は10月3日7番ホームから18時42分発です。" else {
        throw TokenizerSelfTestError.mismatch("decoded Japanese differs: \(decoded)")
    }
    print("Mimi ElanMT tokenizer parity passed.")
} catch {
    print(error.localizedDescription)
    exit(1)
}
