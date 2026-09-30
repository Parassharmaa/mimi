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

struct OnboardingVoiceTypingChoice: Equatable {
    let model: VoiceTypingModel
    let language: SpeechLanguage

    static func initialChoice(
        engineID: TranscriptionEngineID,
        language: SpeechLanguage,
        speechIsReady: Bool,
        hasConfiguredModel: Bool
    ) -> Self? {
        guard speechIsReady, !hasConfiguredModel else { return nil }
        switch engineID {
        case .appleSpeechAnalyzer:
            return .init(model: .appleSpeech, language: language)
        case .whisperKitLargeV3Turbo:
            return .init(model: .mimiWhisper, language: language)
        case .phonon2:
            return .init(model: .phonon2, language: .english)
        case .nemotronStreamingExperimental, .qwen3StreamingExperimental:
            return nil
        }
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
