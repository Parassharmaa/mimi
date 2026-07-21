#!/usr/bin/env python3
"""Create Mimi's deterministic source-only automated claim-suite draft.

The renderer is deliberately data-free and does not call a language model. It
freezes product-domain source scenarios only; references remain absent and the
rows remain non-claimable until the independent automated reference contract
passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DOMAIN_COUNTS = {
    "meeting-and-live-speech": 120,
    "everyday-conversation": 80,
    "macos-and-technical-ui": 60,
    "numbers-dates-and-entities": 60,
    "politeness-ambiguity-and-omission": 60,
    "code-switching": 20,
}


VALUES = {
    "en-US": {
        "name": ["Aiko", "Ben", "Chika", "Daniel", "Emi", "Farah", "Gen", "Hana"],
        "time": ["9:15", "10:40", "13:05", "14:30", "16:20", "18:45", "20:10"],
        "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "next Monday"],
        "topic": ["release plan", "support handoff", "budget draft", "design review", "incident follow-up", "vendor decision", "localization pass"],
        "component": ["caption panel", "audio device", "translation cache", "model bundle", "menu bar control", "export flow", "permission prompt"],
        "place": ["front desk", "side entrance", "kitchen counter", "parcel locker", "station gate", "second-floor lobby"],
        "item": ["blue umbrella", "small package", "spare key", "paper receipt", "charging cable", "train pass"],
        "app": ["Finder", "Preview", "Mail", "Safari", "System Settings", "TextEdit"],
        "menu": ["File", "Edit", "View", "Window", "Format", "Help"],
        "setting": ["Live Captions", "Input Source", "Microphone Mode", "Stage Manager", "Voice Control", "AutoFill"],
        "entity": ["Mizuho Clinic", "Northwind Studio", "Sakura Line", "Orion Hotel", "Harbor Hall", "Maple Bank"],
        "amount": ["¥1,280", "¥4,750", "$18.40", "$92.15", "3,600 yen", "12,480 yen"],
        "percent": ["8%", "12%", "25%", "40%", "75%", "90%"],
        "version": ["2.4.1", "3.0.7", "5.12", "7.2.3", "12.0.1"],
    },
    "ja-JP": {
        "name": ["愛子さん", "ベンさん", "千佳さん", "ダニエルさん", "恵美さん", "ファラさん", "源さん", "花さん"],
        "time": ["9時15分", "10時40分", "13時5分", "14時30分", "16時20分", "18時45分", "20時10分"],
        "day": ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "来週の月曜日"],
        "topic": ["リリース計画", "サポートの引き継ぎ", "予算案", "デザインレビュー", "障害のフォローアップ", "取引先の選定", "ローカライズ確認"],
        "component": ["字幕パネル", "音声デバイス", "翻訳キャッシュ", "モデルバンドル", "メニューバーの操作", "書き出し機能", "権限ダイアログ"],
        "place": ["受付", "通用口", "台所のカウンター", "宅配ロッカー", "駅の改札", "2階のロビー"],
        "item": ["青い傘", "小さな荷物", "予備の鍵", "紙の領収書", "充電ケーブル", "定期券"],
        "app": ["Finder", "プレビュー", "メール", "Safari", "システム設定", "テキストエディット"],
        "menu": ["ファイル", "編集", "表示", "ウインドウ", "フォーマット", "ヘルプ"],
        "setting": ["ライブキャプション", "入力ソース", "マイクモード", "ステージマネージャ", "音声コントロール", "自動入力"],
        "entity": ["みずほクリニック", "ノースウインド・スタジオ", "さくら線", "オリオンホテル", "ハーバーホール", "メープル銀行"],
        "amount": ["1,280円", "4,750円", "18ドル40セント", "92ドル15セント", "3,600円", "12,480円"],
        "percent": ["8%", "12%", "25%", "40%", "75%", "90%"],
        "version": ["2.4.1", "3.0.7", "5.12", "7.2.3", "12.0.1"],
    },
}


TEMPLATES = {
    "en-US": {
        "meeting-and-live-speech": [
            "Before {time}, could {name} confirm whether the {component} is ready?",
            "Let's move the {topic} discussion to {day} and keep {name} in the loop.",
            "Sorry, I missed the first part—did we agree to review the {component} before {time}?",
            "The {topic} is not final yet, so please don't announce it before {day}.",
            "Could we spend ten minutes on the {topic}, then return to the {component}?",
            "If {name} cannot join at {time}, let's record the decision and send a short summary.",
            "I thought the {component} was approved, but the notes say it still needs a review.",
            "Please pause after the {topic} update so remote participants can ask questions.",
        ],
        "everyday-conversation": [
            "Could you leave the {item} at the {place} if nobody answers?",
            "I found the {item}, but I haven't taken it to the {place} yet.",
            "The {place} closes at {time}, so we should leave a little earlier.",
            "Please remind {name} that the {item} is in the {place}, not in the car.",
            "I can pick up the {item} on {day}, unless that is too late.",
            "Do you mind waiting near the {place} while I look for the {item}?",
        ],
        "macos-and-technical-ui": [
            "In {app}, open the {menu} menu and leave {setting} turned off.",
            "After updating to version {version}, restart {app} before changing {setting}.",
            "If the {component} disappears, reopen {app}; do not delete the saved session.",
            "Hold Option while choosing {menu} in {app} to reveal the additional command.",
            "The {setting} switch is disabled until {app} has microphone permission.",
            "Export the log from {app}, but remove the account name before sharing it.",
        ],
        "numbers-dates-and-entities": [
            "The payment to {entity} is {amount}, and it is due on {day} at {time}.",
            "Version {version} reduced memory use by {percent}, not by fifty percent.",
            "Train 47 leaves at {time} from platform 6 and arrives twelve minutes later.",
            "Please send {amount} to {entity}; the reference number is 804-271-63.",
            "Only {percent} of the seats remain for the {entity} event on {day}.",
            "The backup contains 12,480 files in 37 folders and uses 8.6 GB.",
        ],
        "politeness-ambiguity-and-omission": [
            "I may have misunderstood, but did you mean the {component} or the {topic}?",
            "When you have a moment, could you check the {component}? It can wait until {day}.",
            "I don't think {name} rejected the {topic}; they asked for more details.",
            "Would it be possible to postpone the {topic} without changing the deadline?",
            "That should be fine, although I would like to confirm it with {name} first.",
            "Please don't assume the {component} is broken just because it responded slowly.",
        ],
        "code-switching": [
            "Set Language to 日本語, then leave Auto Detect enabled.",
            "Open 設定, choose English, and do not change the Microphone option.",
            "The status says 処理中, so wait before pressing Cancel.",
            "Keep the project name as Mimi-字幕 and export it from {app}.",
        ],
    },
    "ja-JP": {
        "meeting-and-live-speech": [
            "{time}までに、{name}に{component}の準備ができているか確認してもらえますか。",
            "{topic}の話し合いは{day}に移して、{name}にも共有しておきましょう。",
            "すみません、最初を聞き逃したのですが、{time}までに{component}を確認するという話でしたか。",
            "{topic}はまだ確定していないので、{day}より前には発表しないでください。",
            "{topic}に10分使ってから、{component}の話に戻ってもよいでしょうか。",
            "{name}が{time}に参加できなければ、決定を記録して短い要約を送りましょう。",
            "{component}は承認済みだと思っていましたが、議事録ではまだ確認が必要になっています。",
            "リモート参加者が質問できるよう、{topic}の説明後に少し間を置いてください。",
        ],
        "everyday-conversation": [
            "誰も出なければ、{item}を{place}に置いてもらえますか。",
            "{item}は見つかりましたが、まだ{place}には持って行っていません。",
            "{place}は{time}に閉まるので、少し早めに出たほうがよさそうです。",
            "{item}は車ではなく{place}にあると、{name}に伝えてください。",
            "遅すぎなければ、{day}に{item}を取りに行けます。",
            "私が{item}を探している間、{place}の近くで待っていてもらえますか。",
        ],
        "macos-and-technical-ui": [
            "{app}で{menu}メニューを開き、{setting}はオフのままにしてください。",
            "バージョン{version}に更新したら、{setting}を変更する前に{app}を再起動してください。",
            "{component}が消えた場合は、保存済みのセッションを削除せずに{app}を開き直してください。",
            "Optionキーを押しながら{app}の{menu}を選ぶと、追加のコマンドが表示されます。",
            "{app}にマイクの権限を与えるまで、{setting}のスイッチは無効です。",
            "{app}からログを書き出しますが、共有する前にアカウント名を削除してください。",
        ],
        "numbers-dates-and-entities": [
            "{entity}への支払額は{amount}で、期限は{day}の{time}です。",
            "バージョン{version}ではメモリ使用量が{percent}減りましたが、50%ではありません。",
            "47番列車は{time}に6番ホームを出発し、12分後に到着します。",
            "{entity}に{amount}を振り込み、参照番号804-271-63を入力してください。",
            "{day}の{entity}の催しは、残席が{percent}しかありません。",
            "バックアップには37個のフォルダに12,480個のファイルがあり、容量は8.6 GBです。",
        ],
        "politeness-ambiguity-and-omission": [
            "私の理解違いかもしれませんが、{component}と{topic}のどちらを指していますか。",
            "お時間のあるときに{component}を確認していただけますか。{day}まで待てます。",
            "{name}は{topic}を却下したのではなく、詳しい説明を求めたのだと思います。",
            "期限を変えずに{topic}を延期することは可能でしょうか。",
            "それで問題ないと思いますが、先に{name}へ確認させてください。",
            "応答が遅かったというだけで、{component}が故障したと決めつけないでください。",
        ],
        "code-switching": [
            "LanguageをEnglishに変更し、Auto Detectはオンのままにしてください。",
            "Settingsを開いて日本語を選び、Microphoneの項目は変更しないでください。",
            "状態がProcessingになっているので、キャンセルを押す前に待ってください。",
            "プロジェクト名はMimi-Captionsのままにして、{app}から書き出してください。",
        ],
    },
}


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def select(values: list[str], index: int, salt: int) -> str:
    return values[(index * (salt * 2 + 1) + salt) % len(values)]


def contextualize(language: str, domain: str, index: int, source: str) -> str:
    month = 1 + index // 28
    day = 1 + index % 28
    lowered = source if source.startswith(("I ", "I'")) else source[:1].lower() + source[1:]
    if language == "en-US":
        if domain == "meeting-and-live-speech":
            return f"For the meeting on 2027-{month:02d}-{day:02d}, {lowered}"
        if domain == "everyday-conversation":
            return f"On 2027-{month:02d}-{day:02d}, {lowered}"
        if domain == "macos-and-technical-ui":
            return f"For support case M-{4100 + index}, {lowered}"
        if domain == "numbers-dates-and-entities":
            return f"For record R-{5200 + index}, {lowered}"
        if domain == "politeness-ambiguity-and-omission":
            return f"In the follow-up dated 2027-{month:02d}-{day:02d}, {lowered}"
        return f"For profile P-{6300 + index}, {lowered}"
    if domain == "meeting-and-live-speech":
        return f"2027年{month}月{day}日の会議について、{source}"
    if domain == "everyday-conversation":
        return f"2027年{month}月{day}日の予定では、{source}"
    if domain == "macos-and-technical-ui":
        return f"サポート案件M-{4100 + index}では、{source}"
    if domain == "numbers-dates-and-entities":
        return f"記録番号R-{5200 + index}では、{source}"
    if domain == "politeness-ambiguity-and-omission":
        return f"2027年{month}月{day}日付けのフォローアップでは、{source}"
    return f"プロファイルP-{6300 + index}では、{source}"


def render(language: str, domain: str, index: int) -> tuple[str, str, dict[str, str]]:
    templates = TEMPLATES[language][domain]
    template_index = index % len(templates)
    source_values = VALUES[language]
    variables = {
        key: select(values, index, salt + template_index)
        for salt, (key, values) in enumerate(source_values.items(), start=1)
    }
    source = templates[template_index].format(**variables)
    return contextualize(language, domain, index, source), f"{language}:{domain}:t{template_index + 1}", variables


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    manifest_output = args.manifest_output or args.output.with_suffix(args.output.suffix + ".manifest.json")
    for path in (args.output, manifest_output):
        if path.exists() and path.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {path}")

    rows: list[dict] = []
    seen: set[str] = set()
    direction_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for source_language, target_language, short_direction in (
        ("en-US", "ja-JP", "en-ja"),
        ("ja-JP", "en-US", "ja-en"),
    ):
        for domain, count in DOMAIN_COUNTS.items():
            for index in range(count):
                source, template_id, variables = render(source_language, domain, index)
                key = normalized(source)
                if key in seen:
                    raise SystemExit(f"duplicate rendered source: {source}")
                seen.add(key)
                identifier = f"automated-claim-v1:{short_direction}:{domain}:{index + 1:03d}"
                rows.append(
                    {
                        "id": identifier,
                        "documentID": identifier,
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language,
                        "domain": domain,
                        "source": source,
                        "sourceTemplateID": template_id,
                        "sourceVariables": variables,
                        "references": [],
                        "acceptedReferenceCandidateIDs": [],
                        "split": "heldout-automated-source-draft",
                        "reviewStatus": "references-pending",
                        "claimEligible": False,
                        "sourceGeneratedByAI": False,
                        "referenceGeneratedByAI": None,
                        "publicBenchmarkOrigin": False,
                        "paraphraseOfExistingMaterial": False,
                        "sourceCreatedAt": "2026-07-20",
                        "license": "Project-owned",
                        "provenance": "Mimi deterministic product-scenario renderer v1; source-only preregistration",
                    }
                )
                direction_counts[f"{source_language}>{target_language}"][domain] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schemaVersion": 1,
        "status": "sources-frozen-references-pending",
        "suiteID": "mimi-en-ja-automated-heldout-v1",
        "renderer": str(Path(__file__).resolve()),
        "sourceGeneratedByAI": False,
        "reasoningTracesStored": False,
        "cases": len(rows),
        "directions": {
            direction: {"cases": sum(counts.values()), "domains": dict(sorted(counts.items()))}
            for direction, counts in sorted(direction_counts.items())
        },
        "output": {"path": str(args.output.resolve()), "sha256": sha256(args.output)},
        "nextRequiredEvidence": [
            "complete authenticated exposure manifest",
            "exact and fuzzy contamination scan",
            "semantic-neighbor contamination scan",
            "isolated reference-generator report",
            "two distinct-model-family reference-judge reports",
            "deterministic structural audit",
        ],
        "claimEligible": False,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
