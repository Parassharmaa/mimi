// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "Mimi",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "MimiCore", targets: ["MimiCore"]),
        .executable(name: "Mimi", targets: ["Mimi"]),
        .executable(name: "MimiE2E", targets: ["MimiE2E"]),
        .executable(name: "MimiSelfTest", targets: ["MimiSelfTest"])
    ],
    dependencies: [
        // WhisperKit is intentionally a selectable product rather than a bundled model.
        // Models download only after a person chooses the accuracy pack in Mimi.
        .package(url: "https://github.com/argmaxinc/argmax-oss-swift.git", exact: "0.18.0")
    ],
    targets: [
        .target(
            name: "MimiCore",
            path: "Sources/MimiCore"
        ),
        .executableTarget(
            name: "Mimi",
            dependencies: [
                "MimiCore",
                .product(name: "WhisperKit", package: "argmax-oss-swift")
            ],
            path: "Sources/Mimi"
        ),
        .executableTarget(
            name: "MimiE2E",
            dependencies: ["MimiCore"],
            path: "Sources/MimiE2E"
        ),
        .executableTarget(
            name: "MimiSelfTest",
            dependencies: ["MimiCore"],
            path: "Tools/MimiSelfTest"
        )
    ]
)
