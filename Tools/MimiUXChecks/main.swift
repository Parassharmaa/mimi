import AVFoundation
import MimiCore

var checks = 0
for engine in TranscriptionEngineID.allCases {
    for hasLocalTranslation in [true, false] {
        let requirements = OnboardingRequirements(
            engineID: engine,
            hasLocalTranslation: hasLocalTranslation
        )
        precondition(
            requirements.usesAppleSpeech == (engine == .appleSpeechAnalyzer),
            "Only Apple Speech should prepare Apple speech assets: \(engine)"
        )
        precondition(
            requirements.usesAppleTranslation == !hasLocalTranslation,
            "Bundled translation must not require Apple language downloads: \(engine)"
        )
        checks += 2
    }
}
let microphoneCases: [(AVAuthorizationStatus, OnboardingMicrophonePermission)] = [
    (.authorized, .granted),
    (.notDetermined, .requestable),
    (.denied, .denied),
    (.restricted, .denied)
]
for source in AudioSource.allCases {
    for (status, expectedWhenRequired) in microphoneCases {
        let expected: OnboardingMicrophonePermission = source == .microphone
            ? expectedWhenRequired : .notRequired
        precondition(
            OnboardingMicrophonePermission(source: source, authorizationStatus: status) == expected,
            "Microphone permission must be truthful for \(source) / \(status)"
        )
        checks += 1
    }
}
print("PASS: \(checks) onboarding provider and permission checks")
