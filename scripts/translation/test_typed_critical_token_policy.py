#!/usr/bin/env python3
"""Adversarial contracts for the offline typed critical-token policy."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from evaluate_typed_critical_token_policy import strict_tokens
from typed_critical_token_policy import (
    narrow_temporal_preserves,
    single_percentage_preserves,
    typed_preserves,
    typed_signature,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATE = ROOT / "scripts/translation/evaluate_typed_critical_token_policy.py"


assert typed_preserves("seven people", "7人", "en-US", "ja-JP")
assert typed_preserves("twenty-five percent", "25%", "en-US", "ja-JP")
assert typed_preserves("4.5 million miles", "450万マイル", "en-US", "ja-JP")
assert typed_preserves("40 degrees", "四十度", "en-US", "ja-JP")
assert typed_preserves("平成二十七年", "2015", "ja-JP", "en-US")
assert typed_preserves("1億3,600万円", "136 million dollars", "ja-JP", "en-US")
assert not typed_preserves("Born in 626", "30年に誕生", "en-US", "ja-JP")
assert not typed_preserves("平成二十七年", "2002", "ja-JP", "en-US")
assert not typed_preserves(
    "Open https://example.com with {name}",
    "{name}でhttps://example.netを開く",
    "en-US",
    "ja-JP",
)
assert typed_signature("No numbers", "en-US").numbers == ()
assert typed_signature("one, two", "en-US").numbers == ("1", "2")
assert typed_signature("1,2", "en-US").opaque_numbers == ("1,2",)
assert not typed_preserves("one, two", "3", "en-US", "ja-JP")
assert not typed_preserves("1,2", "12", "en-US", "ja-JP")
assert strict_tokens("Record 12.") == ["12"]
assert strict_tokens("Version 1.2.3.") == ["1.2.3"]
assert strict_tokens("Values 1,2") == ["1", "2"]
assert single_percentage_preserves(
    "Provide 25 percent by 2025.",
    "2025年までに25%を供給する。",
)
assert single_percentage_preserves("費用は25パーセントです。", "The cost is 25%.")
assert not single_percentage_preserves("25 percent", "20%")
assert not single_percentage_preserves("25 percent in 2025", "25% in 2026")
assert not single_percentage_preserves("25 and 30 percent", "25%と30%")
assert not single_percentage_preserves("25 percent and 30 percent", "30%と25%")
assert not single_percentage_preserves(
    "25 percent at https://example.com",
    "https://example.netで25%",
)
assert narrow_temporal_preserves(
    "Meet on 2027-01-03 before 10:40.",
    "2027年1月3日の10時40分前に会いましょう。",
    "en-US",
    "ja-JP",
)
assert narrow_temporal_preserves(
    "2027年1月3日の10時40分前に会いましょう。",
    "Meet on January 3, 2027 before 10:40.",
    "ja-JP",
    "en-US",
)
assert narrow_temporal_preserves(
    "Meet on 3 January 2027 before 10:40.",
    "2027年1月3日の10時40分前に会いましょう。",
    "en-US",
    "ja-JP",
)
assert not narrow_temporal_preserves(
    "2027年1月3日に会いましょう。",
    "Meet on February 3, 2027.",
    "ja-JP",
    "en-US",
)
assert not narrow_temporal_preserves(
    "Meet at 4:30 PM.",
    "午前4時30分に会いましょう。",
    "en-US",
    "ja-JP",
)
assert not narrow_temporal_preserves(
    "Meet at 18:00 UTC.",
    "日本標準時18時00分に会いましょう。",
    "en-US",
    "ja-JP",
)
assert not narrow_temporal_preserves(
    "From January 3, 2027 to February 4, 2027.",
    "2027年2月4日から2027年1月3日まで。",
    "en-US",
    "ja-JP",
)
assert narrow_temporal_preserves(
    "2027年1月3日の10時40分前に会いましょう。",
    "Meet on 2027-01-03 before 10:40.",
    "ja-JP",
    "en-US",
)
assert not narrow_temporal_preserves(
    "Meet on 2027-01-03 before 10:40.",
    "2027年1月3日-3日の10時40分前に会いましょう。",
    "en-US",
    "ja-JP",
)
assert not narrow_temporal_preserves(
    "Meet on 2027-01-03 before 10:40.",
    "2027年1月4日の10時40分前に会いましょう。",
    "en-US",
    "ja-JP",
)
assert not narrow_temporal_preserves(
    "Meet on 2027-01-03 before 10:40 at https://example.com.",
    "2027年1月3日の10時40分前にhttps://example.netで会いましょう。",
    "en-US",
    "ja-JP",
)


with tempfile.TemporaryDirectory(prefix="mimi-typed-critical-") as directory:
    root = Path(directory)
    report = root / "report.json"
    output = root / "output.json"
    report.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "caseID": "safe",
                        "sourceLanguage": "en-US",
                        "targetLanguage": "ja-JP",
                        "domain": "fixture",
                        "source": "seven people",
                        "hypothesis": "7人",
                        "references": ["七人"],
                        "selectedEngine": "generalist",
                    },
                    {
                        "caseID": "unsafe",
                        "sourceLanguage": "en-US",
                        "targetLanguage": "ja-JP",
                        "domain": "fixture",
                        "source": "Born in 626",
                        "hypothesis": "30年に誕生",
                        "references": ["626年に誕生"],
                        "selectedEngine": "generalist",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        ["python3", str(EVALUATE), str(report), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["counts"] == {
        "referenceValidatedAccepted": 1,
        "strictFailures": 2,
        "typedAccepted": 1,
        "typedRejected": 1,
        "unsafeAccepted": 0,
    }, payload["counts"]
    assert payload["results"][0]["caseID"] == "safe"

print("Typed critical-token policy contracts passed.")
