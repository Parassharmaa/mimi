#!/usr/bin/env python3
"""Conservative bilingual typed signatures for translation critical tokens."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation


PROTECTED_RE = re.compile(
    r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"
    r"|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>"
)
PERCENT_RE = re.compile(r"%|\bpercent\b|\bper\s+cent\b|パーセント", re.IGNORECASE)
ASCII_NUMBER_RE = re.compile(
    r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"
)
ISO_DATE_RE = re.compile(
    r"(?<!\d)(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])-"
    r"(?P<day>0[1-9]|[12]\d|3[01])(?!\d)"
)
JA_DATE_RE = re.compile(
    r"(?<!\d)(?P<year>\d{4})年(?P<month>0?[1-9]|1[0-2])月"
    r"(?P<day>0?[1-9]|[12]\d|3[01])日"
)
EN_MONTHS = {
    name.lower(): month
    for month, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}
EN_MONTH_NAME = "|".join(name.title() for name in EN_MONTHS)
EN_MONTH_DAY_YEAR_RE = re.compile(
    rf"\b(?P<month_name>{EN_MONTH_NAME})\s+"
    r"(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?(?:,)?\s+"
    r"(?P<year>\d{4})\b",
    re.IGNORECASE,
)
EN_DAY_MONTH_YEAR_RE = re.compile(
    r"\b(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+"
    rf"(?P<month_name>{EN_MONTH_NAME})(?:,)?\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
COLON_TIME_RE = re.compile(
    r"(?<![\d:])(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)(?![\d:])"
)
JA_TIME_RE = re.compile(
    r"(?<!\d)(?P<hour>[01]?\d|2[0-3])時"
    r"(?P<minute>[0-5]?\d)分"
)
AMBIGUOUS_TEMPORAL_CONTEXT_RE = re.compile(
    r"\b(?:a\.?m\.?|p\.?m\.?|UTC|GMT|BST|JST|EST|EDT|CST|CDT|MST|MDT|PST|PDT|"
    r"AWST|AEST|AEDT)\b|午前|午後|協定世界時|標準時|夏時間",
    re.IGNORECASE,
)
PERCENT_EXPRESSION_RE = re.compile(
    r"(?<![\d.])(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?!\d|\.\d)\s*(?:%|\bpercent\b|\bper\s+cent\b|パーセント)",
    re.IGNORECASE,
)
EN_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[A-Za-z]+(?:-[A-Za-z]+)*")
JA_NUMBER_RE = re.compile(r"[0-9.,〇零一二三四五六七八九十百千万億兆]+")
JA_ERA_RE = re.compile(
    r"(明治|大正|昭和|平成|令和)([0-9〇零一二三四五六七八九十百千万]+)年"
)

EN_SMALL = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
}
EN_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "twentieth": 20,
    "thirtieth": 30,
    "fortieth": 40,
    "fiftieth": 50,
    "sixtieth": 60,
    "seventieth": 70,
    "eightieth": 80,
    "ninetieth": 90,
}
EN_SCALES = {
    "hundred": 100,
    "hundredth": 100,
    "thousand": 1_000,
    "thousandth": 1_000,
    "million": 1_000_000,
    "millionth": 1_000_000,
    "billion": 1_000_000_000,
    "billionth": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "trillionth": 1_000_000_000_000,
}
JA_DIGITS = {
    "〇": 0,
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
JA_SMALL_UNITS = {"十": 10, "百": 100, "千": 1_000}
JA_LARGE_UNITS = {"万": 10_000, "億": 100_000_000, "兆": 1_000_000_000_000}
JA_COUNTER_RE = re.compile(
    r"^(?:人|年|月|日|時|分|秒|歳|才|個|台|件|回|度|本|枚|冊|巻|話|章|条|項|号|"
    r"節|款|円|ドル|キロ|km|カ国|か国|ヶ国|つ|頭|匹|機|名|社|週|ヶ月|ヵ月|"
    r"箇月|割|倍|位|番|番目|目|戦|試合|ゴール|ウィケット|ラン|マイル)"
)
ERA_BASE = {"明治": 1867, "大正": 1911, "昭和": 1925, "平成": 1988, "令和": 2018}


@dataclass(frozen=True)
class TypedCriticalSignature:
    protected: tuple[str, ...]
    percentages: int
    numbers: tuple[str, ...]
    opaque_numbers: tuple[str, ...]


@dataclass(frozen=True)
class SinglePercentageSignature:
    protected: tuple[str, ...]
    percentage: str
    other_numbers: tuple[str, ...]


@dataclass(frozen=True)
class NarrowTemporalSignature:
    protected: tuple[str, ...]
    dates: tuple[str, ...]
    times: tuple[str, ...]
    literal_percentages: int
    other_numbers: tuple[str, ...]
    ambiguous_context: bool


def normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def decimal_text(value: Decimal) -> str:
    if value == value.to_integral():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def decimal_value(token: str) -> Decimal | None:
    if not re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", token):
        return None
    try:
        return Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None


def mask_protected(value: str) -> tuple[str, tuple[str, ...]]:
    protected = tuple(sorted(PROTECTED_RE.findall(value)))
    return PROTECTED_RE.sub(lambda match: " " * len(match.group(0)), value), protected


def single_percentage_signature(value: str) -> SinglePercentageSignature | None:
    normalized = normalize(value)
    masked, protected = mask_protected(normalized)
    expressions = list(PERCENT_EXPRESSION_RE.finditer(masked))
    if len(expressions) != 1 or len(PERCENT_RE.findall(masked)) != 1:
        return None
    expression = expressions[0]
    percentage = decimal_value(expression.group("number"))
    if percentage is None:
        return None
    remainder = (
        masked[: expression.start()]
        + " " * (expression.end() - expression.start())
        + masked[expression.end() :]
    )
    number_matches = list(ASCII_NUMBER_RE.finditer(remainder))
    without_numbers = list(remainder)
    for match in number_matches:
        without_numbers[match.start() : match.end()] = " " * len(match.group(0))
    if any(character.isdigit() for character in without_numbers):
        return None
    return SinglePercentageSignature(
        protected=protected,
        percentage=decimal_text(percentage),
        other_numbers=tuple(
            sorted(match.group(0).replace(",", "") for match in number_matches)
        ),
    )


def single_percentage_preserves(source: str, output: str) -> bool:
    source_signature = single_percentage_signature(source)
    return source_signature is not None and source_signature == single_percentage_signature(
        output
    )


def _valid_date(match: re.Match[str]) -> str | None:
    year = int(match.group("year"))
    groups = match.groupdict()
    month = (
        EN_MONTHS[groups["month_name"].lower()]
        if groups.get("month_name")
        else int(match.group("month"))
    )
    day = int(match.group("day"))
    try:
        date(year, month, day)
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def narrow_temporal_signature(value: str, language: str) -> NarrowTemporalSignature:
    """Parse only unambiguous ISO/Japanese dates and 24-hour clock times.

    This intentionally excludes eras, month names, AM/PM, relative dates, word
    numbers, units, and arithmetic. It is a bounded candidate relaxation for
    reference-backed evaluation, not a general bilingual numeric parser.
    """

    if language not in {"en-US", "ja-JP"}:
        raise ValueError(f"unsupported language: {language}")
    normalized = normalize(value)
    masked, protected = mask_protected(normalized)
    characters = list(masked)
    occupied = [False] * len(masked)
    dated: list[tuple[int, str]] = []
    timed: list[tuple[int, str]] = []

    date_patterns = (
        (ISO_DATE_RE, EN_MONTH_DAY_YEAR_RE, EN_DAY_MONTH_YEAR_RE)
        if language == "en-US"
        else (ISO_DATE_RE, JA_DATE_RE)
    )
    for pattern in date_patterns:
        for match in pattern.finditer(masked):
            if any(occupied[match.start() : match.end()]):
                continue
            parsed = _valid_date(match)
            if parsed is None:
                continue
            dated.append((match.start(), parsed))
            characters[match.start() : match.end()] = " " * (match.end() - match.start())
            occupied[match.start() : match.end()] = [True] * (match.end() - match.start())

    after_dates = "".join(characters)
    time_patterns = (COLON_TIME_RE,) if language == "en-US" else (COLON_TIME_RE, JA_TIME_RE)
    for pattern in time_patterns:
        for match in pattern.finditer(after_dates):
            if any(occupied[match.start() : match.end()]):
                continue
            hour = int(match.group("hour"))
            minute = int(match.group("minute"))
            timed.append((match.start(), f"{hour:02d}:{minute:02d}"))
            characters[match.start() : match.end()] = " " * (match.end() - match.start())
            occupied[match.start() : match.end()] = [True] * (match.end() - match.start())

    remainder = "".join(characters)
    return NarrowTemporalSignature(
        protected=protected,
        dates=tuple(value for _, value in sorted(dated)),
        times=tuple(value for _, value in sorted(timed)),
        literal_percentages=remainder.count("%"),
        other_numbers=tuple(
            sorted(match.group(0).replace(",", "") for match in ASCII_NUMBER_RE.finditer(remainder))
        ),
        ambiguous_context=bool(AMBIGUOUS_TEMPORAL_CONTEXT_RE.search(masked)),
    )


def narrow_temporal_preserves(
    source: str,
    output: str,
    source_language: str,
    target_language: str,
) -> bool:
    source_signature = narrow_temporal_signature(source, source_language)
    if (
        source_signature.ambiguous_context
        or len(source_signature.dates) > 1
        or len(source_signature.times) > 1
        or (not source_signature.dates and not source_signature.times)
    ):
        return False
    output_signature = narrow_temporal_signature(output, target_language)
    return (
        not output_signature.ambiguous_context
        and len(output_signature.dates) <= 1
        and len(output_signature.times) <= 1
        and source_signature == output_signature
    )


def english_word_value(words: list[str]) -> int | None:
    current = 0
    total = 0
    saw_number = False
    for word in words:
        if word == "and":
            continue
        if word in EN_SMALL:
            current += EN_SMALL[word]
            saw_number = True
        elif word in EN_TENS:
            current += EN_TENS[word]
            saw_number = True
        elif word in {"hundred", "hundredth"}:
            current = max(current, 1) * 100
            saw_number = True
        elif word in EN_SCALES:
            total += max(current, 1) * EN_SCALES[word]
            current = 0
            saw_number = True
        else:
            return None
    return total + current if saw_number else None


def joins_number_phrase(
    value: str,
    previous: re.Match[str],
    following: re.Match[str],
) -> bool:
    return bool(re.fullmatch(r"[\s-]+", value[previous.end() : following.start()]))


def english_numbers(value: str) -> tuple[list[Decimal], list[str]]:
    matches = list(EN_TOKEN_RE.finditer(value))
    numbers: list[Decimal] = []
    opaque: list[str] = []
    index = 0
    while index < len(matches):
        match = matches[index]
        token = match.group(0).lower()
        direct = decimal_value(token) if token[0].isdigit() else None
        if direct is not None:
            if (
                index + 1 < len(matches)
                and matches[index + 1].group(0).lower() in EN_SCALES
                and joins_number_phrase(value, match, matches[index + 1])
            ):
                direct *= EN_SCALES[matches[index + 1].group(0).lower()]
                index += 1
            numbers.append(direct)
            index += 1
            continue
        if token[0].isdigit():
            opaque.append(token)
            index += 1
            continue
        components = token.split("-")
        if not all(
            component in EN_SMALL
            or component in EN_TENS
            or component in EN_SCALES
            or component == "and"
            for component in components
        ):
            index += 1
            continue
        phrase = list(components)
        cursor = index + 1
        while cursor < len(matches):
            if not joins_number_phrase(value, matches[cursor - 1], matches[cursor]):
                break
            following = matches[cursor].group(0).lower().split("-")
            if not all(
                component in EN_SMALL
                or component in EN_TENS
                or component in EN_SCALES
                or component == "and"
                for component in following
            ):
                break
            phrase.extend(following)
            cursor += 1
        parsed = english_word_value(phrase)
        if parsed is not None:
            numbers.append(Decimal(parsed))
            index = cursor
        else:
            index += 1
    return numbers, opaque


def japanese_small_number(value: str) -> Decimal | None:
    value = value.replace(",", "")
    if not value:
        return None
    if all(character.isdigit() or character == "." for character in value):
        return decimal_value(value)
    if all(character in JA_DIGITS for character in value):
        return Decimal(int("".join(str(JA_DIGITS[character]) for character in value)))
    total = 0
    pending: int | None = None
    for character in value:
        if character.isdigit():
            pending = (pending or 0) * 10 + int(character)
        elif character in JA_DIGITS:
            pending = (pending or 0) * 10 + JA_DIGITS[character]
        elif character in JA_SMALL_UNITS:
            total += (pending if pending is not None else 1) * JA_SMALL_UNITS[character]
            pending = None
        else:
            return None
    return Decimal(total + (pending or 0))


def japanese_number_value(value: str) -> Decimal | None:
    remaining = value
    total = Decimal(0)
    for unit, scale in (("兆", 10**12), ("億", 10**8), ("万", 10**4)):
        if unit not in remaining:
            continue
        left, remaining = remaining.split(unit, 1)
        parsed = japanese_small_number(left) if left else Decimal(1)
        if parsed is None:
            return None
        total += parsed * scale
    tail = japanese_small_number(remaining) if remaining else Decimal(0)
    return None if tail is None else total + tail


def is_unambiguous_japanese_number(value: str, text: str, start: int, end: int) -> bool:
    if any(character.isdigit() for character in value):
        return True
    if len(value) > 1 or any(character in JA_SMALL_UNITS or character in JA_LARGE_UNITS for character in value):
        return True
    previous = text[start - 1] if start > 0 else ""
    following = text[end:]
    return previous == "第" or bool(JA_COUNTER_RE.match(following)) or (
        start == 0 and bool(following[:1].isspace())
    )


def japanese_numbers(value: str) -> tuple[list[Decimal], list[str]]:
    working = value
    numbers: list[Decimal] = []
    opaque: list[str] = []
    era_spans = []
    for match in JA_ERA_RE.finditer(working):
        year = japanese_number_value(match.group(2))
        if year is not None:
            numbers.append(Decimal(ERA_BASE[match.group(1)]) + year)
            era_spans.append(match.span())
    if era_spans:
        characters = list(working)
        for start, end in era_spans:
            characters[start:end] = " " * (end - start)
        working = "".join(characters)
    for match in JA_NUMBER_RE.finditer(working):
        token = match.group(0)
        if not is_unambiguous_japanese_number(
            token,
            working,
            match.start(),
            match.end(),
        ):
            continue
        parsed = japanese_number_value(token)
        if parsed is not None:
            numbers.append(parsed)
        elif any(character.isdigit() for character in token):
            opaque.append(token)
    return numbers, opaque


def typed_signature(value: str, language: str) -> TypedCriticalSignature:
    normalized = normalize(value)
    masked, protected = mask_protected(normalized)
    percentages = len(PERCENT_RE.findall(masked))
    masked = PERCENT_RE.sub(" ", masked)
    if language == "en-US":
        numbers, opaque_numbers = english_numbers(masked)
    elif language == "ja-JP":
        numbers, opaque_numbers = japanese_numbers(masked)
    else:
        raise ValueError(f"unsupported language: {language}")
    return TypedCriticalSignature(
        protected=protected,
        percentages=percentages,
        numbers=tuple(sorted(decimal_text(number) for number in numbers)),
        opaque_numbers=tuple(sorted(opaque_numbers)),
    )


def typed_preserves(
    source: str,
    output: str,
    source_language: str,
    target_language: str,
) -> bool:
    return typed_signature(source, source_language) == typed_signature(
        output,
        target_language,
    )
