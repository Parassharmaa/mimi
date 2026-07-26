import Foundation

public struct LowEnergyAudioBoundaryPartition: Sendable, Equatable {
    public let boundarySampleIndex: Int
    public let finalizedSamples: [Float]
    public let carriedSamples: [Float]

    public init(
        boundarySampleIndex: Int,
        finalizedSamples: [Float],
        carriedSamples: [Float]
    ) {
        self.boundarySampleIndex = boundarySampleIndex
        self.finalizedSamples = finalizedSamples
        self.carriedSamples = carriedSamples
    }
}

public enum LowEnergyAudioBoundarySelector {
    public static func partition(
        _ samples: [Float],
        lookbackSamples: Int,
        minimumSegmentSamples: Int,
        minimumCarrySamples: Int,
        energyWindowSamples: Int,
        searchStrideSamples: Int,
        zeroCrossingRadiusSamples: Int
    ) -> LowEnergyAudioBoundaryPartition {
        let boundary = sampleIndex(
            in: samples,
            lookbackSamples: lookbackSamples,
            minimumSegmentSamples: minimumSegmentSamples,
            minimumCarrySamples: minimumCarrySamples,
            energyWindowSamples: energyWindowSamples,
            searchStrideSamples: searchStrideSamples,
            zeroCrossingRadiusSamples: zeroCrossingRadiusSamples
        )
        return LowEnergyAudioBoundaryPartition(
            boundarySampleIndex: boundary,
            finalizedSamples: Array(samples[..<boundary]),
            carriedSamples: Array(samples[boundary...])
        )
    }

    public static func sampleIndex(
        in samples: [Float],
        lookbackSamples: Int,
        minimumSegmentSamples: Int,
        minimumCarrySamples: Int,
        energyWindowSamples: Int,
        searchStrideSamples: Int,
        zeroCrossingRadiusSamples: Int
    ) -> Int {
        guard lookbackSamples > 0,
              minimumSegmentSamples >= 0,
              minimumCarrySamples >= 0,
              energyWindowSamples > 0,
              searchStrideSamples > 0,
              zeroCrossingRadiusSamples >= 0,
              minimumCarrySamples <= samples.count else {
            return samples.count
        }
        let searchUpperBound = samples.count - minimumCarrySamples
        let lookbackLowerBound = lookbackSamples >= samples.count
            ? 0
            : samples.count - lookbackSamples
        let searchLowerBound = max(
            minimumSegmentSamples,
            lookbackLowerBound
        )
        guard searchLowerBound <= searchUpperBound,
              energyWindowSamples
                <= searchUpperBound - searchLowerBound else {
            return samples.count
        }

        var candidates: [(rms: Float, boundary: Int)] = []
        var start = searchLowerBound
        while energyWindowSamples <= searchUpperBound - start {
            let end = start + energyWindowSamples
            var energy: Float = 0
            for sample in samples[start..<end] {
                energy += sample * sample
            }
            candidates.append((
                rms: sqrt(energy / Float(energyWindowSamples)),
                boundary: start + energyWindowSamples / 2
            ))
            guard searchStrideSamples <= searchUpperBound - start else {
                break
            }
            start += searchStrideSamples
        }
        guard let minimumRMS = candidates.map(\.rms).min() else {
            return samples.count
        }
        let nearMinimumRMS = minimumRMS * 1.05 + 0.000_001
        let selected = candidates.last {
            $0.rms <= nearMinimumRMS
        }?.boundary ?? candidates.min {
            $0.rms < $1.rms
        }!.boundary
        return nearestZeroCrossing(
            to: selected,
            in: samples,
            lowerBound: searchLowerBound,
            upperBound: searchUpperBound,
            radiusSamples: zeroCrossingRadiusSamples
        )
    }

    private static func nearestZeroCrossing(
        to target: Int,
        in samples: [Float],
        lowerBound: Int,
        upperBound: Int,
        radiusSamples: Int
    ) -> Int {
        let boundedLowerRadius = min(radiusSamples, target)
        let boundedUpperRadius = min(
            radiusSamples,
            upperBound - target
        )
        let start = max(
            lowerBound + 1,
            target - boundedLowerRadius
        )
        let end = min(
            upperBound - 1,
            target + boundedUpperRadius
        )
        guard start <= end else { return target }

        var bestIndex = target
        var bestScore = Float.greatestFiniteMagnitude
        for index in start...end {
            let previous = samples[index - 1]
            let current = samples[index]
            guard (previous <= 0 && current >= 0)
                    || (previous >= 0 && current <= 0) else {
                continue
            }
            let amplitude = abs(previous) + abs(current)
            let distancePenalty = Float(abs(index - target))
                / Float(max(radiusSamples, 1))
                * 0.000_001
            let score = amplitude + distancePenalty
            if score < bestScore {
                bestScore = score
                bestIndex = index
            }
        }
        return bestIndex
    }
}
