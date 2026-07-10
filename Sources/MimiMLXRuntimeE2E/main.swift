import Foundation
import MLX

/// A weight-free package smoke. When copied next to Mimi's executable this
/// forces MLX to load the app-bundled `mlx.metallib` and execute one GPU
/// kernel, catching broken shader placement before release.
@main
enum MimiMLXRuntimeE2E {
    static func main() {
        do {
            var result: Float = 0
            Stream.withNewDefaultStream(device: .gpu) {
                let output = MLXArray(Float(41)) + 1
                output.eval()
                result = output.item(Float.self)
            }
            guard result == 42 else {
                throw RuntimeSmokeError.unexpectedResult(result)
            }
            print("Mimi packaged MLX runtime smoke passed.")
        } catch {
            fputs("Mimi packaged MLX runtime smoke failed: \(error.localizedDescription)\\n", stderr)
            exit(1)
        }
    }
}

private enum RuntimeSmokeError: LocalizedError {
    case unexpectedResult(Float)

    var errorDescription: String? {
        switch self {
        case let .unexpectedResult(value):
            "Expected MLX to evaluate 41 + 1 as 42, got \(value)."
        }
    }
}
