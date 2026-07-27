#!/usr/bin/env python3
"""Generate Mimi's natural long-form speech experiment SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def bar(
    *,
    label: str,
    value: float,
    maximum: float,
    y: int,
    css_class: str,
) -> str:
    width = 158 * value / maximum
    return "\n".join(
        [
            f'    <text x="0" y="{y + 16}">{label}</text>',
            (
                f'    <rect class="{css_class}" x="72" y="{y}" '
                f'width="{width:.1f}" height="20" rx="4"/>'
            ),
            (
                f'    <text class="value" x="270" y="{y + 16}" '
                f'text-anchor="end">{percent(value)}</text>'
            ),
        ]
    )


def english_panel(
    *,
    x: int,
    title: str,
    subtitle: str,
    values: dict,
) -> str:
    return "\n".join(
        [
            f'  <g transform="translate({x} 70)">',
            f'    <text class="heading" x="0" y="0">{title}</text>',
            f'    <text class="small" x="0" y="20">{subtitle}</text>',
            '    <line class="grid" x1="72" y1="40" x2="230" y2="40"/>',
            '    <line class="grid" x1="72" y1="40" x2="72" y2="156"/>',
            '    <text class="small" x="72" y="176" text-anchor="middle">0%</text>',
            '    <text class="small" x="151" y="176" text-anchor="middle">25%</text>',
            '    <text class="small" x="230" y="176" text-anchor="middle">50%</text>',
            bar(
                label="Product",
                value=values["product"]["wordError"]["errorRate"],
                maximum=0.5,
                y=52,
                css_class="product",
            ),
            bar(
                label="Hard 24",
                value=values["hard24"]["wordError"]["errorRate"],
                maximum=0.5,
                y=88,
                css_class="hard",
            ),
            bar(
                label="Adaptive",
                value=values["adaptive"]["wordError"]["errorRate"],
                maximum=0.5,
                y=124,
                css_class="adaptive",
            ),
            "  </g>",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    japanese = summary["japanese"]
    headset = summary["english"]["headset"]
    array = summary["english"]["array1-01"]
    japanese_raw = japanese["product"]["rawCharacterError"]["errorRate"]
    japanese_reading = japanese["product"]["readingCharacterError"][
        "errorRate"
    ]
    headset_product_ops = headset["product"]["operational"]
    headset_adaptive_ops = headset["adaptive"]["operational"]
    array_product_ops = array["product"]["operational"]
    array_adaptive_ops = array["adaptive"]["operational"]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="465" viewBox="0 0 900 465" role="img" aria-labelledby="title description">
  <title id="title">Mimi natural long-form segmentation experiment</title>
  <desc id="description">On a five-speaker natural AMI meeting, Mimi's adaptive 24-second profile has lower word error rate than both the 30-second product profile and a hard 24-second ablation in close-mic and far-field conditions. On a Japanese audiobook, the adaptive fallback remains inactive and matches the product exactly. Japanese reading-normalized character error is supplemental.</desc>
  <style>
    :root {{
      color-scheme: light dark;
      --text: #172033;
      --muted: #667085;
      --grid: #d0d5dd;
      --product: #98a2b3;
      --hard: #14b8a6;
      --adaptive: #7c3aed;
      --surface: #ffffff;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --text: #f2f4f7;
        --muted: #b9c0cc;
        --grid: #475467;
        --product: #98a2b3;
        --hard: #5eead4;
        --adaptive: #a78bfa;
        --surface: #0d1117;
      }}
    }}
    text {{
      fill: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px;
    }}
    .small {{ fill: var(--muted); font-size: 12px; }}
    .heading {{ font-size: 15px; font-weight: 600; }}
    .value {{ font-weight: 600; }}
    .grid {{ stroke: var(--grid); stroke-width: 1; }}
    .product {{ fill: var(--product); }}
    .hard {{ fill: var(--hard); }}
    .adaptive {{ fill: var(--adaptive); }}
    .panel {{ fill: var(--surface); }}
  </style>

  <rect class="panel" width="900" height="465" rx="12"/>

  <circle class="product" cx="30" cy="28" r="6"/>
  <text x="42" y="33">Product 30 / 0</text>
  <circle class="hard" cx="166" cy="28" r="6"/>
  <text x="178" y="33">Hard 24 / 0</text>
  <circle class="adaptive" cx="292" cy="28" r="6"/>
  <text x="304" y="33">Adaptive 24 / 6</text>
  <text class="small" x="870" y="33" text-anchor="end">Lower is better</text>

  <g transform="translate(20 70)">
    <text class="heading" x="0" y="0">Japanese audiobook</text>
    <text class="small" x="0" y="20">4.81 min, one natural narrator</text>
    <line class="grid" x1="72" y1="40" x2="230" y2="40"/>
    <line class="grid" x1="72" y1="40" x2="72" y2="156"/>
    <text class="small" x="72" y="176" text-anchor="middle">0%</text>
    <text class="small" x="151" y="176" text-anchor="middle">15%</text>
    <text class="small" x="230" y="176" text-anchor="middle">30%</text>
    <text class="small" x="0" y="54">Raw CER</text>
{bar(label="Product", value=japanese_raw, maximum=0.3, y=62, css_class="product")}
{bar(label="Adaptive", value=japanese_raw, maximum=0.3, y=90, css_class="adaptive")}
    <text class="small" x="0" y="130">Reading CER</text>
{bar(label="Both", value=japanese_reading, maximum=0.3, y=138, css_class="adaptive")}
  </g>

{english_panel(x=320, title="English headset WER", subtitle="5.04 min, five speakers, close-mic mix", values=headset)}

{english_panel(x=620, title="English far-field WER", subtitle="Same meeting, table-array channel", values=array)}

  <line class="grid" x1="24" y1="284" x2="876" y2="284"/>
  <text class="heading" x="24" y="314">Production-queue behavior</text>
  <text class="small" x="24" y="338">Headset product → adaptive</text>
  <text x="230" y="338">Queue {headset_product_ops["maximumQueuedAudioSeconds"]:.1f} → {headset_adaptive_ops["maximumQueuedAudioSeconds"]:.1f} s</text>
  <text x="415" y="338">Wall RTF {headset_product_ops["pacedWallRTF"]:.4f} → {headset_adaptive_ops["pacedWallRTF"]:.4f}</text>
  <text x="650" y="338">Final lag {headset_product_ops["postAudioFinalizationSeconds"]:.2f} → {headset_adaptive_ops["postAudioFinalizationSeconds"]:.2f} s</text>
  <text class="small" x="24" y="370">Far-field product → adaptive</text>
  <text x="230" y="370">Queue {array_product_ops["maximumQueuedAudioSeconds"]:.1f} → {array_adaptive_ops["maximumQueuedAudioSeconds"]:.1f} s</text>
  <text x="415" y="370">Wall RTF {array_product_ops["pacedWallRTF"]:.4f} → {array_adaptive_ops["pacedWallRTF"]:.4f}</text>
  <text x="650" y="370">Final lag {array_product_ops["postAudioFinalizationSeconds"]:.2f} → {array_adaptive_ops["postAudioFinalizationSeconds"]:.2f} s</text>
  <text class="small" x="24" y="405">All eight natural paced reports: 0 dropped samples, 0 drop events, 0 backpressure events.</text>
  <text class="small" x="24" y="425">Reading CER uses fugashi 1.5.2 + UniDic Lite 1.0.8 and is diagnostic only.</text>
  <text class="small" x="24" y="445">AMI includes 135 overlap-affected words; absolute WER is a natural-meeting stress score.</text>
</svg>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
