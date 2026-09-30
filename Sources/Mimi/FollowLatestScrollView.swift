import SwiftUI

/// A transcript-oriented scroll view that follows incoming text until the
/// person deliberately scrolls away from the bottom.
struct FollowLatestScrollView<Content: View>: View {
    let contentVersion: String
    private let content: Content
    private let preferences: UserPreferences?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.mimiAccessibilityPreview) private var accessibilityPreview
    @State private var followsLatest: Bool
    @State private var isAtBottom = true
    @State private var userIsScrolling = false
    @State private var hasUnseenContent = false

    private let bottomAnchor = "mimi-follow-latest-bottom"

    init(
        contentVersion: String,
        initiallyFollowing: Bool = true,
        preferences: UserPreferences? = nil,
        @ViewBuilder content: () -> Content
    ) {
        self.contentVersion = contentVersion
        self.preferences = preferences
        self.content = content()
        _followsLatest = State(initialValue: initiallyFollowing)
        _hasUnseenContent = State(initialValue: !initiallyFollowing)
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                content

                Color.clear
                    .frame(height: 1)
                    .id(bottomAnchor)
            }
            .onScrollGeometryChange(for: Bool.self) { geometry in
                geometry.visibleRect.maxY >= geometry.contentSize.height - 20
            } action: { _, atBottom in
                isAtBottom = atBottom
            }
            .onScrollPhaseChange { _, phase in
                switch phase {
                case .tracking, .interacting, .decelerating:
                    userIsScrolling = true
                case .idle where userIsScrolling:
                    userIsScrolling = false
                    followsLatest = isAtBottom
                    if isAtBottom {
                        hasUnseenContent = false
                    }
                case .animating, .idle:
                    break
                }
            }
            .onChange(of: contentVersion) { _, _ in
                if followsLatest {
                    scrollToLatest(using: proxy, animated: false)
                } else {
                    hasUnseenContent = true
                }
            }
            .onAppear {
                guard followsLatest else { return }
                scrollToLatest(using: proxy, animated: false)
            }
            .overlay(alignment: .bottomTrailing) {
                if !followsLatest {
                    Button {
                        followsLatest = true
                        hasUnseenContent = false
                        scrollToLatest(using: proxy, animated: true)
                    } label: {
                        Label(
                            hasUnseenContent ? t("New text", "新しいテキスト") : t("Jump to Latest", "最新のテキストへ"),
                            systemImage: "arrow.down"
                        )
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .tint(hasUnseenContent ? .accentColor : .secondary)
                    .padding(8)
                    .help(t("Return to the newest transcript text and resume automatic scrolling", "最新の文字起こしへ戻り、自動スクロールを再開します"))
                    .accessibilityHint(t("Resumes following new text automatically", "新しいテキストへの自動スクロールを再開します"))
                }
            }
        }
    }

    private func scrollToLatest(using proxy: ScrollViewProxy, animated: Bool) {
        Task { @MainActor in
            await Task.yield()
            if animated && !reduceMotion && !accessibilityPreview.contains(.reduceMotion) {
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo(bottomAnchor, anchor: .bottom)
                }
            } else {
                proxy.scrollTo(bottomAnchor, anchor: .bottom)
            }
        }
    }

    private func t(_ english: String, _ japanese: String) -> String {
        preferences?.text(english, japanese) ?? english
    }
}
