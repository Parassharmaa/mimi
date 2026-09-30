import AVFoundation
import MimiCore

struct OnboardingRequirements: Equatable {
    let usesAppleSpeech: Bool
    let usesAppleTranslation: Bool

    init(engineID: TranscriptionEngineID, hasLocalTranslation: Bool) {
        usesAppleSpeech = engineID == .appleSpeechAnalyzer
        usesAppleTranslation = !hasLocalTranslation
    }
}

enum OnboardingMicrophonePermission: Equatable {
    case notRequired
    case granted
    case requestable
    case denied
    case unknown

    init(source: AudioSource, authorizationStatus: AVAuthorizationStatus) {
        guard source == .microphone else {
            self = .notRequired
            return
        }
        self = switch authorizationStatus {
        case .authorized: .granted
        case .notDetermined: .requestable
        case .denied, .restricted: .denied
        @unknown default: .unknown
        }
    }
}
