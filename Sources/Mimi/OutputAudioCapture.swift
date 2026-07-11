@preconcurrency import AVFoundation
import CoreAudio
import Foundation
import MimiSession

/// Captures the unmuted system mix routed to a selected output device. The
/// Core Audio tap is private to Mimi, excludes no processes, and is destroyed
/// with its temporary aggregate device at every stop or failed start.
@MainActor
final class OutputAudioCapture: OutputAudioCapturing {
    private let inputCapture = MicrophoneCapture()
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateDeviceID = AudioObjectID(kAudioObjectUnknown)
    private var configuredOutputDeviceID: AudioObjectID?

    func configureInput(deviceID: UInt32?) throws -> AVAudioFormat {
        let resolvedDeviceID = try deviceID ?? AudioDeviceCatalog.defaultOutputDeviceID()
        if configuredOutputDeviceID != resolvedDeviceID || aggregateDeviceID == kAudioObjectUnknown {
            try tearDownTap()
            try createTap(for: resolvedDeviceID)
        }
        return try inputCapture.configureInput(deviceID: aggregateDeviceID)
    }

    func start(
        recordingTo url: URL?,
        deviceID: UInt32?,
        onBuffer: @escaping @Sendable (AVAudioPCMBuffer) -> Void
    ) throws {
        _ = try configureInput(deviceID: deviceID)
        do {
            try inputCapture.start(
                recordingTo: url,
                deviceID: aggregateDeviceID,
                onBuffer: onBuffer
            )
        } catch {
            try? tearDownTap()
            throw error
        }
    }

    @discardableResult
    func stop() throws -> URL? {
        let url: URL?
        do {
            url = try inputCapture.stop()
        } catch {
            try? tearDownTap()
            throw error
        }
        try tearDownTap()
        return url
    }

    private func createTap(for outputDeviceID: AudioObjectID) throws {
        let outputUID = try AudioDeviceCatalog.uid(for: outputDeviceID)
        let description = CATapDescription(monoGlobalTapButExcludeProcesses: [])
        description.name = "Mimi Output Tap \(UUID().uuidString)"
        description.isPrivate = true
        description.muteBehavior = .unmuted
        description.deviceUID = outputUID

        var newTapID = AudioObjectID(kAudioObjectUnknown)
        var status = AudioHardwareCreateProcessTap(description, &newTapID)
        guard status == noErr, newTapID != kAudioObjectUnknown else {
            throw OutputAudioCaptureError.coreAudio("create the output-audio tap", status)
        }
        tapID = newTapID

        do {
            let aggregateUID = "dev.paras.mimi.output-tap.\(UUID().uuidString)"
            let aggregateDescription: [String: Any] = [
                kAudioAggregateDeviceNameKey: "Mimi Output Capture",
                kAudioAggregateDeviceUIDKey: aggregateUID,
                kAudioAggregateDeviceIsPrivateKey: true,
                kAudioAggregateDeviceTapAutoStartKey: true
            ]
            var newAggregateID = AudioObjectID(kAudioObjectUnknown)
            status = AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &newAggregateID)
            guard status == noErr, newAggregateID != kAudioObjectUnknown else {
                throw OutputAudioCaptureError.coreAudio("create the private output-audio input", status)
            }
            aggregateDeviceID = newAggregateID

            let tapUID = try stringProperty(kAudioTapPropertyUID, on: newTapID)
            var tapList: CFArray? = [tapUID as CFString] as CFArray
            var address = AudioObjectPropertyAddress(
                mSelector: kAudioAggregateDevicePropertyTapList,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            let listSize = UInt32(MemoryLayout<CFString>.stride)
            status = withUnsafeMutablePointer(to: &tapList) {
                AudioObjectSetPropertyData(newAggregateID, &address, 0, nil, listSize, $0)
            }
            guard status == noErr else {
                throw OutputAudioCaptureError.coreAudio("attach the output tap to its private input", status)
            }
            configuredOutputDeviceID = outputDeviceID
        } catch {
            try? tearDownTap()
            throw error
        }
    }

    private func stringProperty(
        _ selector: AudioObjectPropertySelector,
        on objectID: AudioObjectID
    ) throws -> String {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var value: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.stride)
        let status = withUnsafeMutablePointer(to: &value) {
            AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, $0)
        }
        guard status == noErr, !(value as String).isEmpty else {
            throw OutputAudioCaptureError.coreAudio("read the output tap identifier", status)
        }
        return value as String
    }

    private func tearDownTap() throws {
        configuredOutputDeviceID = nil
        var firstError: OutputAudioCaptureError?
        if aggregateDeviceID != kAudioObjectUnknown {
            let status = AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
            if status != noErr {
                firstError = .coreAudio("destroy the private output-audio input", status)
            }
            aggregateDeviceID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != kAudioObjectUnknown {
            let status = AudioHardwareDestroyProcessTap(tapID)
            if status != noErr, firstError == nil {
                firstError = .coreAudio("destroy the output-audio tap", status)
            }
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
        if let firstError { throw firstError }
    }
}

enum OutputAudioCaptureError: LocalizedError {
    case coreAudio(String, OSStatus)

    var errorDescription: String? {
        switch self {
        case let .coreAudio(action, status):
            "Mimi could not \(action) (Core Audio status \(status)). Try another output device or restart audio playback."
        }
    }
}
