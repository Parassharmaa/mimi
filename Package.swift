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
        .executable(name: "MimiSelfTest", targets: ["MimiSelfTest"])
    ],
    dependencies: [
        // WhisperKit supplies the small, optional English/Japanese Auto router.
        .package(url: "https://github.com/argmaxinc/argmax-oss-swift.git", exact: "0.18.0")
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
                .product(name: "WhisperKit", package: "argmax-oss-swift")
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
        )
    ]
)
