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
        .executable(name: "MimiNemotronE2E", targets: ["MimiNemotronE2E"]),
        .executable(name: "MimiSelfTest", targets: ["MimiSelfTest"])
    ],
    dependencies: [
        // WhisperKit is intentionally a selectable product rather than a bundled model.
        // Models download only after a person chooses the accuracy pack in Mimi.
        .package(url: "https://github.com/argmaxinc/argmax-oss-swift.git", exact: "0.18.0"),
        // Native Apple-silicon ASR implementation for the optional Nemotron
        // model. The pinned release includes its FastConformer/RNNT loader.
        .package(url: "https://github.com/Blaizzy/mlx-audio-swift.git", exact: "0.1.3"),
        // Declared directly for the no-model package smoke that proves the
        // bundled Metal shader can execute from inside Mimi.app.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.6"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", exact: "0.8.1")
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
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface")
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
            name: "MimiNemotronE2E",
            dependencies: [
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift")
            ],
            path: "Sources/MimiNemotronE2E"
        ),
        .executableTarget(
            name: "MimiMLXRuntimeE2E",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "Sources/MimiMLXRuntimeE2E"
        ),
        .executableTarget(
            name: "MimiSelfTest",
            dependencies: ["MimiCore"],
            path: "Tools/MimiSelfTest"
        )
    ]
)
