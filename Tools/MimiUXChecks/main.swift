import AVFoundation
import Foundation
import MimiCore

MainActor.assumeIsolated {
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
    (.restricted, .denied),
  ]
  for source in AudioSource.allCases {
    for (status, expectedWhenRequired) in microphoneCases {
      let expected: OnboardingMicrophonePermission =
        source == .microphone
        ? expectedWhenRequired : .notRequired
      precondition(
        OnboardingMicrophonePermission(source: source, authorizationStatus: status) == expected,
        "Microphone permission must be truthful for \(source) / \(status)"
      )
      checks += 1
    }
  }
  let voiceTypingCases:
    [(TranscriptionEngineID, SpeechLanguage, VoiceTypingModel, SpeechLanguage)] = [
      (.appleSpeechAnalyzer, .english, .appleSpeech, .english),
      (.appleSpeechAnalyzer, .japanese, .appleSpeech, .japanese),
      (.whisperKitLargeV3Turbo, .english, .mimiWhisper, .english),
      (.whisperKitLargeV3Turbo, .japanese, .mimiWhisper, .japanese),
      (.phonon2, .english, .phonon2, .english),
      (.phonon2, .japanese, .phonon2, .english),
    ]
  for (engine, language, expectedModel, expectedLanguage) in voiceTypingCases {
    let choice = OnboardingVoiceTypingChoice.initialChoice(
      engineID: engine, language: language,
      speechIsReady: true, hasConfiguredModel: false
    )
    precondition(choice == .init(model: expectedModel, language: expectedLanguage))
    precondition(
      OnboardingVoiceTypingChoice.initialChoice(
        engineID: engine, language: language,
        speechIsReady: false, hasConfiguredModel: false
      ) == nil, "Unprepared models must not become Voice Type defaults")
    precondition(
      OnboardingVoiceTypingChoice.initialChoice(
        engineID: engine, language: language,
        speechIsReady: true, hasConfiguredModel: true
      ) == nil, "Explicitly saved models must never be replaced")
    checks += 3
  }
  for engine in [TranscriptionEngineID.nemotronStreamingExperimental, .qwen3StreamingExperimental] {
    precondition(
      OnboardingVoiceTypingChoice.initialChoice(
        engineID: engine, language: .english,
        speechIsReady: true, hasConfiguredModel: false
      ) == nil, "Unsupported Voice Type engines must not silently map to another model")
    checks += 1
  }

  @MainActor
  func withFreshPreferences(_ run: @MainActor (UserPreferences, UserDefaults) -> Void) {
    let suite = "dev.paras.mimi.onboarding-checks.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    run(UserPreferences(defaults: defaults), defaults)
  }

  for (engine, language, expectedModel, expectedLanguage) in voiceTypingCases {
    withFreshPreferences { preferences, defaults in
      precondition(defaults.object(forKey: "voiceTypingModel") == nil)
      preferences.configureVoiceTypingForFirstUse(
        engineID: engine, language: language, speechIsReady: true
      )
      precondition(preferences.voiceTypingModel == expectedModel)
      precondition(preferences.voiceTypingLanguage == expectedLanguage)
      precondition(defaults.string(forKey: "voiceTypingModel") == expectedModel.rawValue)
      precondition(defaults.string(forKey: "voiceTypingLanguage") == expectedLanguage.rawValue)
      checks += 5
    }
  }

  for savedModel in VoiceTypingModel.allCases {
    for savedLanguage in savedModel == .phonon2 ? [.english] : SpeechLanguage.allCases {
      for (engine, language, _, _) in voiceTypingCases {
        withFreshPreferences { preferences, defaults in
          preferences.voiceTypingModel = savedModel
          preferences.voiceTypingLanguage = savedLanguage
          preferences.configureVoiceTypingForFirstUse(
            engineID: engine, language: language, speechIsReady: true
          )
          precondition(preferences.voiceTypingModel == savedModel)
          precondition(preferences.voiceTypingLanguage == savedLanguage)
          precondition(defaults.string(forKey: "voiceTypingModel") == savedModel.rawValue)
          precondition(defaults.string(forKey: "voiceTypingLanguage") == savedLanguage.rawValue)
          checks += 4
        }
      }
    }
  }
  withFreshPreferences { preferences, defaults in
    preferences.voiceTypingLanguage = .japanese
    preferences.configureVoiceTypingForFirstUse(
      engineID: .phonon2, language: .english, speechIsReady: true
    )
    precondition(preferences.voiceTypingLanguage == .japanese)
    precondition(defaults.object(forKey: "voiceTypingModel") == nil)
    checks += 2
  }
  withFreshPreferences { preferences, defaults in
    preferences.configureVoiceTypingForFirstUse(
      engineID: .phonon2, language: .english, speechIsReady: false
    )
    precondition(defaults.object(forKey: "voiceTypingModel") == nil)
    precondition(defaults.object(forKey: "voiceTypingLanguage") == nil)
    checks += 2
  }
  print("PASS: \(checks) onboarding provider, permission and Voice Type preference checks")
}
