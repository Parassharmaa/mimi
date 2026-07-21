// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "Mimi",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "MimiCore", targets: ["MimiCore"]),
        .library(name: "MimiSession", targets: ["MimiSession"]),
        .executable(name: "Mimi", targets: ["Mimi"]),
        .executable(name: "MimiE2E", targets: ["MimiE2E"]),
        .executable(name: "MimiSessionE2E", targets: ["MimiSessionE2E"]),
        .executable(name: "MimiSelfTest", targets: ["MimiSelfTest"]),
        .executable(name: "MimiTokenizerSelfTest", targets: ["MimiTokenizerSelfTest"])
    ],
    dependencies: [
        // WhisperKit supplies the small, optional English/Japanese Auto router.
        .package(url: "https://github.com/argmaxinc/argmax-oss-swift.git", exact: "0.18.0"),
        // Developer-only local translation candidate. Model assets are loaded
        // from an explicit directory and Apple Translation remains the fallback.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", exact: "2.30.6"),
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.30.6"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "1.1.9")
    ],
    targets: [
        .target(
            name: "MimiCore",
            path: "Sources/MimiCore"
        ),
        .target(
            name: "MimiSession",
            dependencies: ["MimiCore"],
            path: "Sources/MimiSession"
        ),
        .executableTarget(
            name: "Mimi",
            dependencies: [
                "MimiCore",
                "MimiSession",
                .product(name: "WhisperKit", package: "argmax-oss-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ],
            exclude: [
                "NemotronMLXAccuracyEngine.swift",
                "QwenMLXLiveEngine.swift"
            ]
        ),
        .executableTarget(
            name: "MimiE2E",
            dependencies: ["MimiCore"],
            path: "Sources/MimiE2E"
        ),
        .executableTarget(
            name: "MimiSessionE2E",
            dependencies: ["MimiSession", "MimiCore"],
            path: "Sources/MimiSessionE2E"
        ),
        .executableTarget(
            name: "MimiSelfTest",
            dependencies: ["MimiCore"],
            path: "Tools/MimiSelfTest"
        ),
        .executableTarget(
            name: "MimiTokenizerSelfTest",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ],
            path: "Tools/MimiTokenizerSelfTest"
        )
    ]
)
