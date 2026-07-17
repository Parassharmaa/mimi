public struct LiveTextEdit: Equatable, Sendable {
    public let removalCount: Int
    public let insertion: String

    public init(previous: String, next: String) {
        let oldCharacters = Array(previous)
        let newCharacters = Array(next)
        var prefixCount = 0
        while prefixCount < min(oldCharacters.count, newCharacters.count),
              oldCharacters[prefixCount] == newCharacters[prefixCount] {
            prefixCount += 1
        }
        removalCount = oldCharacters.count - prefixCount
        insertion = String(newCharacters.dropFirst(prefixCount))
    }
}
