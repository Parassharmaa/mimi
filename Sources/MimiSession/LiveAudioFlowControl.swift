import Foundation

/// The reason a bounded live-ASR window should be finalized.
public enum LiveWindowBoundary: Equatable, Sendable {
    case none
    case silence
    case maximumDuration
}

/// Tracks a live ASR window without retaining source audio outside its caller.
/// A pause is preferred for a natural segment break; the maximum protects a
/// model implementation that otherwise grows work with every input chunk.
public struct BoundedLiveWindowPolicy: Sendable {
    private let minimumWindowSampleCount: Int
    private let maximumWindowSampleCount: Int
    private let silenceBoundarySampleCount: Int
    private let silenceRMS: Float
    private var windowSampleCount = 0
    private var consecutiveSilentSampleCount = 0

    public init(
        minimumWindowSampleCount: Int = 16_000 * 3,
        maximumWindowSampleCount: Int = 16_000 * 30,
        silenceBoundarySampleCount: Int = 16_000 * 8 / 10,
        silenceRMS: Float = 0.008
    ) {
        precondition(minimumWindowSampleCount > 0)
        precondition(maximumWindowSampleCount >= minimumWindowSampleCount)
        precondition(silenceBoundarySampleCount > 0)
        self.minimumWindowSampleCount = minimumWindowSampleCount
        self.maximumWindowSampleCount = maximumWindowSampleCount
        self.silenceBoundarySampleCount = silenceBoundarySampleCount
        self.silenceRMS = silenceRMS
    }

    public mutating func append(_ samples: [Float]) -> LiveWindowBoundary {
        guard !samples.isEmpty else { return .none }
        windowSampleCount += samples.count
        if rootMeanSquare(samples) < silenceRMS {
            consecutiveSilentSampleCount += samples.count
        } else {
            consecutiveSilentSampleCount = 0
        }

        if windowSampleCount >= maximumWindowSampleCount {
            return .maximumDuration
        }
        if windowSampleCount >= minimumWindowSampleCount,
           consecutiveSilentSampleCount >= silenceBoundarySampleCount {
            return .silence
        }
        return .none
    }

    public mutating func reset() {
        windowSampleCount = 0
        consecutiveSilentSampleCount = 0
    }

    private func rootMeanSquare(_ samples: [Float]) -> Float {
        let sum = samples.reduce(Float.zero) { $0 + $1 * $1 }
        return (sum / Float(samples.count)).squareRoot()
    }
}

/// A small FIFO which drops oldest input in whole decode chunks under local
/// inference backpressure. This keeps live captions current and bounds the
/// amount Stop needs to flush.
public struct BoundedAudioSampleQueue: Sendable {
    private let maximumSampleCount: Int
    private let preferredChunkSize: Int
    private var samples: [Float] = []

    public init(maximumSampleCount: Int, preferredChunkSize: Int) {
        precondition(maximumSampleCount > 0)
        precondition(preferredChunkSize > 0)
        self.maximumSampleCount = maximumSampleCount
        self.preferredChunkSize = preferredChunkSize
    }

    public var count: Int { samples.count }
    public var isEmpty: Bool { samples.isEmpty }

    /// Returns the number of oldest samples discarded to enforce the bound.
    @discardableResult
    public mutating func append(_ newSamples: [Float]) -> Int {
        guard !newSamples.isEmpty else { return 0 }
        samples.append(contentsOf: newSamples)
        guard samples.count > maximumSampleCount else { return 0 }

        let overflow = samples.count - maximumSampleCount
        let chunksToDrop = max(1, (overflow + preferredChunkSize - 1) / preferredChunkSize)
        let droppedCount = min(samples.count, chunksToDrop * preferredChunkSize)
        samples.removeFirst(droppedCount)
        return droppedCount
    }

    public mutating func dequeue(upTo maximumCount: Int) -> [Float] {
        guard maximumCount > 0, !samples.isEmpty else { return [] }
        let count = min(maximumCount, samples.count)
        let next = Array(samples.prefix(count))
        samples.removeFirst(count)
        return next
    }

    public mutating func removeAll(keepingCapacity: Bool = false) {
        samples.removeAll(keepingCapacity: keepingCapacity)
    }
}
