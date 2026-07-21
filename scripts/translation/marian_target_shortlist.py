"""Tokenizer-derived target vocabulary shortlist with source-surface expansion."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
DIGIT_RE = re.compile(r"\d")
SUPPORTED_DIRECTIONS = ("en-ja", "ja-en")
SPECIAL_TOKEN_IDS = (0, 1, 32_000)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stripped(token: str) -> str:
    return token.lstrip("▁")


def has_latin(token: str) -> bool:
    return any("LATIN" in unicodedata.name(character, "") for character in token)


def short_latin(token: str) -> bool:
    value = stripped(token)
    return bool(value) and len(value) <= 3 and all(
        "LATIN" in unicodedata.name(character, "") for character in value
    )


def symbol_only(token: str) -> bool:
    value = stripped(token)
    return bool(value) and all(not character.isalnum() for character in value)


def common_token(token: str) -> bool:
    return (
        token == "▁"
        or bool(DIGIT_RE.search(token))
        or symbol_only(token)
        or short_latin(token)
    )


def build_static_ids(vocabulary: dict[str, int], direction: str) -> tuple[int, ...]:
    if direction not in SUPPORTED_DIRECTIONS:
        raise ValueError(f"unsupported direction: {direction}")
    selected = set(SPECIAL_TOKEN_IDS)
    for token, token_id in vocabulary.items():
        target_script = (
            bool(JAPANESE_RE.search(token)) if direction == "en-ja" else has_latin(token)
        )
        if target_script or common_token(token):
            selected.add(int(token_id))
    return tuple(sorted(selected))


def artifact_payload(tokenizer_path: Path, vocabulary: dict[str, int]) -> dict:
    if sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError("tokenizer vocabulary IDs must be contiguous")
    directions = {
        direction: {"staticTokenIDs": build_static_ids(vocabulary, direction)}
        for direction in SUPPORTED_DIRECTIONS
    }
    return {
        "schemaVersion": 1,
        "format": "mimi-marian-tokenizer-target-shortlist-v1",
        "purpose": "research-only output-projection acceleration; no quality claim",
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": sha256(tokenizer_path),
            "vocabularySize": len(vocabulary),
        },
        "rules": {
            "en-ja": (
                "all Japanese-script tokens plus common numeric, symbol, SentencePiece "
                "whitespace, and <=3-codepoint Latin tokens"
            ),
            "ja-en": (
                "all Unicode-Latin tokens plus common numeric, symbol, SentencePiece "
                "whitespace, and <=3-codepoint Latin tokens"
            ),
            "specialTokenIDs": SPECIAL_TOKEN_IDS,
            "dynamicSourceExpansion": (
                "include every input token plus every vocabulary ID with the same token "
                "surface after stripping leading SentencePiece U+2581 markers"
            ),
        },
        "directions": directions,
        "dataProvenance": {
            "parallelCorpusRowsUsed": 0,
            "heldOutSourcesOrReferencesUsed": 0,
            "derivedOnlyFromAuthenticatedTokenizerVocabulary": True,
        },
        "claimEligible": False,
        "doesNotAuthorizeAppIntegration": True,
    }


@dataclass(frozen=True)
class MarianTargetShortlist:
    static_ids: dict[str, tuple[int, ...]]
    surface_ids: dict[str, tuple[int, ...]]
    id_tokens: tuple[str, ...]

    @classmethod
    def load(cls, path: Path, tokenizer_path: Path, tokenizer) -> "MarianTargetShortlist":
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocabulary = tokenizer.get_vocab()
        if (
            payload.get("schemaVersion") != 1
            or payload.get("format") != "mimi-marian-tokenizer-target-shortlist-v1"
            or payload.get("tokenizer", {}).get("sha256") != sha256(tokenizer_path)
            or payload.get("tokenizer", {}).get("vocabularySize") != len(vocabulary)
        ):
            raise ValueError("target shortlist does not authenticate the tokenizer")
        static_ids: dict[str, tuple[int, ...]] = {}
        for direction in SUPPORTED_DIRECTIONS:
            actual = tuple(payload.get("directions", {}).get(direction, {}).get("staticTokenIDs", ()))
            expected = build_static_ids(vocabulary, direction)
            if actual != expected:
                raise ValueError(f"target shortlist rule output mismatch: {direction}")
            static_ids[direction] = actual
        surfaces: dict[str, list[int]] = {}
        for token, token_id in vocabulary.items():
            surfaces.setdefault(stripped(token), []).append(int(token_id))
        by_id = [""] * len(vocabulary)
        for token, token_id in vocabulary.items():
            by_id[int(token_id)] = token
        return cls(
            static_ids=static_ids,
            surface_ids={key: tuple(sorted(value)) for key, value in surfaces.items()},
            id_tokens=tuple(by_id),
        )

    def expand(self, direction: str, input_ids: list[int]) -> tuple[int, ...]:
        selected = set(self.static_ids[direction])
        for token_id in input_ids:
            selected.add(int(token_id))
            selected.update(self.surface_ids[stripped(self.id_tokens[int(token_id)])])
        return tuple(sorted(selected))
