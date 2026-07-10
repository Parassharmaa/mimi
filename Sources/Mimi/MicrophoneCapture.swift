@preconcurrency import AVFoundation
import AudioToolbox
import Foundation

@MainActor
final class MicrophoneCapture {
    private let audioEngine = AVAudioEngine()
    private let recorder = TemporaryAudioRecorder()

    func requestPermission() async -> Bool {
        await AVCaptureDevice.requestAccess(for: .audio)
    }

    func configureInput(deviceID: AudioDeviceID?) throws -> AVAudioFormat {
        let input = audioEngine.inputNode
        if let deviceID {
            try input.auAudioUnit.setDeviceID(deviceID)
        }
        return input.outputFormat(forBus: 0)
    }

    func start(
        recordingTo url: URL?,
        deviceID: AudioDeviceID?,
        onBuffer: @escaping (AVAudioPCMBuffer) -> Void
    ) throws {
        _ = try? stop()

        let format = try configureInput(deviceID: deviceID)
        try recorder.begin(url: url, format: format)

        let input = audioEngine.inputNode

        input.installTap(onBus: 0, bufferSize: 1_024, format: format) { [weak self] buffer, _ in
            self?.recorder.write(buffer)
            onBuffer(buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()
    }

    @discardableResult
    func stop() throws -> URL? {
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        return try recorder.finish()
    }
}

private final class TemporaryAudioRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recordingFile: AVAudioFile?
    private var recordingURL: URL?
    private var writeError: Error?

    func begin(url: URL?, format: AVAudioFormat) throws {
        lock.lock()
        defer { lock.unlock() }
        recordingFile = try url.map { try AVAudioFile(forWriting: $0, settings: format.settings) }
        recordingURL = url
        writeError = nil
    }

    func write(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard let recordingFile, writeError == nil else { return }
        do {
            try recordingFile.write(from: buffer)
        } catch {
            writeError = error
        }
    }

    func finish() throws -> URL? {
        lock.lock()
        defer { lock.unlock() }
        let url = recordingURL
        let error = writeError
        recordingFile = nil
        recordingURL = nil
        writeError = nil
        if let error { throw error }
        return url
    }
}
