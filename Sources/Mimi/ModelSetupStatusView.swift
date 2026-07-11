import MimiSession
import SwiftUI

/// One truthful presentation of model readiness shared by the menu extra and
/// Settings. It never turns a platform-capable Apple Speech engine into a
/// "ready" model before the selected language asset is installed.
struct ModelSetupStatusView: View {
    let readiness: ModelReadiness
    let setupState: ModelSetupState
    var compact = false

    var body: some View {
        Group {
            switch setupState {
            case let .downloading(_, _, progress):
                progressStatus(progress)
            case .checking, .prewarming, .removing:
                activityStatus
            case .waitingForSystem:
                waitingForSystemStatus
            case .cancelled:
                Label("Download paused. Retry resumes saved data.", systemImage: "pause.circle")
                    .foregroundStyle(.secondary)
            case let .failed(_, _, message):
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            case .idle:
                idleStatus
            }
        }
        .font(compact ? .caption : .callout)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder
    private func progressStatus(_ progress: ModelDownloadProgress?) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            if let fraction = progress?.fractionCompleted {
                let percent = Int((fraction * 100).rounded())
                ProgressView(value: fraction)
                    .accessibilityValue("\(percent) percent of model files downloaded")
            } else {
                ProgressView()
                    .accessibilityValue("Model download progress unavailable")
            }
            Text(downloadMessage(progress))
                .foregroundStyle(.secondary)
        }
    }

    private var activityStatus: some View {
        HStack(spacing: 7) {
            ProgressView()
                .controlSize(.small)
            Text(readiness.message ?? "Preparing local model…")
                .foregroundStyle(.secondary)
        }
    }

    private var waitingForSystemStatus: some View {
        HStack(alignment: .top, spacing: 7) {
            ProgressView()
                .controlSize(.small)
            Text(readiness.message ?? "macOS is continuing the Apple Speech download.")
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var idleStatus: some View {
        if readiness.canStart {
            Label("Local model ready", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        } else if let message = readiness.message {
            Label(message, systemImage: readinessSymbol)
                .foregroundStyle(.secondary)
        }
    }

    private var readinessSymbol: String {
        switch readiness {
        case .unavailable:
            "exclamationmark.triangle"
        case .checking, .needsDownload, .downloading, .experimental, .ready:
            "arrow.down.circle"
        }
    }

    private func downloadMessage(_ progress: ModelDownloadProgress?) -> String {
        guard let progress else {
            return readiness.message ?? "Downloading local model…"
        }

        if let fraction = progress.fractionCompleted {
            let percent = Int((fraction * 100).rounded())
            return "Downloading Whisper Large-v3 — \(percent)% of model files"
        }
        return readiness.message ?? "Downloading local model…"
    }
}
