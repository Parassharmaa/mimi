@preconcurrency import AVFoundation
@preconcurrency import CoreMedia
@preconcurrency import ScreenCaptureKit
import AudioToolbox
import Foundation
import MimiCore
import MimiSession

/// Captures only the audio output associated with content the person selected
/// in macOS's own ScreenCaptureKit picker. It never adds a screen/video stream
/// output, so Mimi does not receive or persist screen images.
@MainActor
final class ScreenAudioCapture: NSObject, ScreenAudioCapturing {
    private let picker = SCContentSharingPicker.shared
    private let audioOutput = ScreenAudioStreamOutput()
    private var selectedFilter: SCContentFilter?
    private var selectedScreenContent: ScreenAudioSelection?
    private var activeStream: SCStream?
    private var pendingPickerSource: AudioSource?
    private var pickerContinuation: CheckedContinuation<Void, Error>?

    var selectedContent: ScreenAudioSelection? { selectedScreenContent }

    override init() {
        super.init()
        picker.add(self)
    }

    func selectContent(for source: AudioSource) async throws {
        guard source == .applicationAudio || source == .systemAudio else {
            throw ScreenAudioCaptureError.unsupportedSource(source)
        }
        guard pickerContinuation == nil else {
            throw ScreenAudioCaptureError.pickerAlreadyVisible
        }

        pendingPickerSource = source
        var configuration = SCContentSharingPickerConfiguration()
        configuration.allowedPickerModes = source == .applicationAudio ? [.singleApplication] : [.singleDisplay]
        configuration.excludedBundleIDs = [Bundle.main.bundleIdentifier].compactMap { $0 }
        configuration.allowsChangingSelectedContent = false
        picker.defaultConfiguration = configuration
        picker.isActive = true

        try await withCheckedThrowingContinuation { continuation in
            pickerContinuation = continuation
            picker.present(using: source == .applicationAudio ? .application : .display)
        }
    }

    func configureInput() throws -> AVAudioFormat {
        try makeCaptureFormat()
    }

    func start(
        recordingTo url: URL?,
        onBuffer: @escaping @Sendable (AVAudioPCMBuffer) -> Void,
        onStreamStopped: @escaping @MainActor @Sendable (String?) -> Void
    ) async throws {
        guard let filter = selectedFilter, selectedScreenContent != nil else {
            throw ScreenAudioCaptureError.selectionRequired
        }
        guard activeStream == nil else {
            throw ScreenAudioCaptureError.captureAlreadyActive
        }

        let format = try makeCaptureFormat()
        try audioOutput.prepare(
            recordingTo: url,
            format: format,
            onBuffer: onBuffer,
            onStreamStopped: { [weak self] message in
                self?.activeStream = nil
                onStreamStopped(message)
            }
        )

        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = true
        configuration.sampleRate = Int(format.sampleRate)
        configuration.channelCount = Int(format.channelCount)
        configuration.excludesCurrentProcessAudio = true

        let stream = SCStream(filter: filter, configuration: configuration, delegate: audioOutput)
        do {
            // Deliberately add only an audio output. Mimi neither requests nor
            // receives ScreenCaptureKit video frames for this capture lane.
            try stream.addStreamOutput(audioOutput, type: .audio, sampleHandlerQueue: audioOutput.queue)
            activeStream = stream
            try await stream.startCapture()
        } catch {
            activeStream = nil
            _ = try? audioOutput.finish()
            throw error
        }
    }

    func stop() async throws -> URL? {
        let stream = activeStream
        activeStream = nil
        audioOutput.suppressUnexpectedStopCallback()

        do {
            if let stream {
                try await stream.stopCapture()
            }
            return try audioOutput.finish()
        } catch {
            _ = try? audioOutput.finish()
            throw error
        }
    }

    private func makeCaptureFormat() throws -> AVAudioFormat {
        guard let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1) else {
            throw ScreenAudioCaptureError.couldNotCreateAudioFormat
        }
        return format
    }

    private func resolvePicker(_ result: Result<Void, Error>) {
        let continuation = pickerContinuation
        pickerContinuation = nil
        pendingPickerSource = nil
        picker.isActive = false

        switch result {
        case .success:
            continuation?.resume()
        case let .failure(error):
            continuation?.resume(throwing: error)
        }
    }
}

extension ScreenAudioCapture: SCContentSharingPickerObserver {
    /// ScreenCaptureKit delivers picker callbacks from its XPC connection,
    /// rather than the main actor. Do not make these witness methods
    /// `@MainActor`: Swift 6 correctly traps when the system invokes an
    /// actor-isolated Objective-C callback on that queue. Hop explicitly
    /// before touching the picker/session state instead.
    nonisolated func contentSharingPicker(
        _ picker: SCContentSharingPicker,
        didCancelFor stream: SCStream?
    ) {
        Task { @MainActor [weak self] in
            self?.resolvePicker(.failure(ScreenAudioCaptureError.pickerCancelled))
        }
    }

    nonisolated func contentSharingPicker(
        _ picker: SCContentSharingPicker,
        didUpdateWith filter: SCContentFilter,
        for stream: SCStream?
    ) {
        Task { @MainActor [weak self] in
            guard let self, let source = self.pendingPickerSource else { return }
            self.selectedFilter = filter
            self.selectedScreenContent = ScreenAudioSelection(
                source: source,
                description: self.selectionDescription(for: source)
            )
            self.resolvePicker(.success(()))
        }
    }

    nonisolated func contentSharingPickerStartDidFailWithError(_ error: Error) {
        Task { @MainActor [weak self] in
            self?.resolvePicker(.failure(error))
        }
    }
}

extension ScreenAudioCapture {
    private func selectionDescription(for source: AudioSource) -> String {
        switch source {
        case .applicationAudio:
            "App chosen with the macOS content picker"
        case .systemAudio:
            "Display chosen with the macOS content picker"
        case .microphone, .outputAudio:
            ""
        }
    }
}

/// Receives ScreenCaptureKit audio on a dedicated serial queue, copies each
/// sample into an owned PCM buffer, and releases any temporary recording as
/// soon as capture stops. The class is protected by its lock because stream
/// delegates and the main-actor stop path run on different executors.
private final class ScreenAudioStreamOutput: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    let queue = DispatchQueue(label: "dev.paras.mimi.screen-audio", qos: .userInitiated)

    private let lock = NSLock()
    private var recordingFile: AVAudioFile?
    private var recordingURL: URL?
    private var writeError: Error?
    private var onBuffer: (@Sendable (AVAudioPCMBuffer) -> Void)?
    private var onStreamStopped: (@MainActor @Sendable (String?) -> Void)?

    func prepare(
        recordingTo url: URL?,
        format: AVAudioFormat,
        onBuffer: @escaping @Sendable (AVAudioPCMBuffer) -> Void,
        onStreamStopped: @escaping @MainActor @Sendable (String?) -> Void
    ) throws {
        lock.lock()
        defer { lock.unlock() }

        recordingFile = try url.map { try AVAudioFile(forWriting: $0, settings: format.settings) }
        recordingURL = url
        writeError = nil
        self.onBuffer = onBuffer
        self.onStreamStopped = onStreamStopped
    }

    func suppressUnexpectedStopCallback() {
        lock.lock()
        onStreamStopped = nil
        lock.unlock()
    }

    func finish() throws -> URL? {
        lock.lock()
        defer { lock.unlock() }

        let url = recordingURL
        let error = writeError
        recordingFile = nil
        recordingURL = nil
        writeError = nil
        onBuffer = nil
        onStreamStopped = nil
        if let error { throw error }
        return url
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .audio, let buffer = makePCMBuffer(from: sampleBuffer) else { return }

        let callback: (@Sendable (AVAudioPCMBuffer) -> Void)?
        lock.lock()
        if let recordingFile, writeError == nil {
            do {
                try recordingFile.write(from: buffer)
            } catch {
                writeError = error
            }
        }
        callback = onBuffer
        lock.unlock()

        callback?(buffer)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        let callback: (@MainActor @Sendable (String?) -> Void)?
        let message: String

        lock.lock()
        callback = onStreamStopped
        onStreamStopped = nil
        onBuffer = nil
        recordingFile = nil
        recordingURL = nil
        message = writeError?.localizedDescription ?? error.localizedDescription
        writeError = nil
        lock.unlock()

        guard let callback else { return }
        Task { @MainActor in
            callback(message)
        }
    }

    private func makePCMBuffer(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard CMSampleBufferDataIsReady(sampleBuffer),
              let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer) else {
            return nil
        }

        let format = AVAudioFormat(cmAudioFormatDescription: formatDescription)

        let sampleCount = CMSampleBufferGetNumSamples(sampleBuffer)
        guard sampleCount > 0,
              sampleCount <= CMItemCount(Int32.max),
              let buffer = AVAudioPCMBuffer(
                  pcmFormat: format,
                  frameCapacity: AVAudioFrameCount(sampleCount)
              ) else {
            return nil
        }

        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(sampleCount),
            into: buffer.mutableAudioBufferList
        )
        guard status == noErr else { return nil }
        buffer.frameLength = AVAudioFrameCount(sampleCount)
        return buffer
    }
}

private enum ScreenAudioCaptureError: LocalizedError {
    case unsupportedSource(AudioSource)
    case pickerAlreadyVisible
    case pickerCancelled
    case selectionRequired
    case captureAlreadyActive
    case couldNotCreateAudioFormat

    var errorDescription: String? {
        switch self {
        case let .unsupportedSource(source):
            "\(source.displayName) does not use the macOS content picker."
        case .pickerAlreadyVisible:
            "The macOS content picker is already open."
        case .pickerCancelled:
            "No app or display was selected."
        case .selectionRequired:
            "Choose an app or display in the macOS content picker before recording."
        case .captureAlreadyActive:
            "The selected app or display is already being captured."
        case .couldNotCreateAudioFormat:
            "Mimi could not configure the selected app or display audio stream."
        }
    }
}
