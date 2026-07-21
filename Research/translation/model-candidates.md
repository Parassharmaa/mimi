# Compact model candidates

Research checked 2026-07-21. Re-check every license and pinned revision before a
release; a model card is evidence, not legal advice.

| Candidate | Approximate pack | License | Apple-Silicon path | Decision |
| --- | ---: | --- | --- | --- |
| Direction-selected ElanMT controls: conversational EN→JA + hard-reference JA→EN, derived from pinned `Mitsua/elan-mt-bt-*` students and locally converted to 4-bit MLX | 73,403,570-byte combined bundle | CC-BY-SA-4.0 | Two specialized directional students; fused MLX Python and Swift Marian ports; one Mimi bidirectional interface | Preferred DQRD starting bundle at 31.31/56.52 non-claimable canary chrF++; exact 12/12 Swift/Python output parity. |
| `mlx-community/SmolLM2-135M-Instruct` revision `422de227b90002f443a21a58b1087f6ee7632731`, locally quantized to 4-bit | 89.2 MB with KFTT LoRA | Apache-2.0 | Native MLX Swift; one physical bidirectional model | Rejected: KFTT LoRA reached only 4.79/27.52 chrF++ because Japanese embeddings/tokenizer were inadequate. |
| `mlx-community/Qwen3-0.6B-4bit` revision `73e3e38d981303bc594367cd910ea6eb48349da8` | 351 MB | Apache-2.0 | Native MLX Swift; one bidirectional model; LoRA supported | Quality/reference experiment only; rejected for the preferred size target. |
| `Helsinki-NLP/opus-mt-en-jap` | ~300 MB FP32 | Apache-2.0 model card | Marian encoder-decoder; no first-party MLX Swift loader | Reference baseline only. |
| `Helsinki-NLP/opus-mt-jap-en` | ~300 MB FP32 | Apache-2.0 model card | Marian encoder-decoder; second direction requires another pack | Reference baseline only. |
| `facebook/nllb-200-distilled-600M` | 2.48 GB FP32 | CC-BY-NC-4.0 | Encoder-decoder, no first-party MLX Swift loader | Rejected: noncommercial and too large. |
| `alirezamsh/small100` revision `8ab680e26a596d2e3d2d2d17ae0f68df1037328c` | 1,337,145,439-byte authenticated FP32 snapshot; a q4 port should fit the 500 MB ceiling but was not built | MIT | One 330M-parameter M2M100-compatible model; 12-layer encoder and shallow 3-layer decoder; custom target-language source prefix; would require a new MLX/Swift port | Rejected before porting: greedy canary 24.57/51.58 chrF++, beam-5 24.26/53.33, versus roughly 31.31/56.52 for the compact Marian line. A balanced 100-step rank-16 LoRA reached at best 24.65/50.60; a 10× learning-rate control collapsed. |
| `kuotient/Hy-MT2-1.8B-1.25Bit-MLX` revision `03d1df683157fde0a4ec80636e749867d0c13a5e` | 464,192,044-byte authenticated snapshot | Apache-2.0 label; upstream training-data license inventory is incomplete | One bidirectional sparse-ternary model; working custom MLX Python/Metal kernel, but no ready MLX Swift runtime | Rejected: 35.62/60.70 canary collapses to 22.64/44.09 on 800 stress cases, 127 critical-token failures, and 1.026/0.850 s p95. Official GGUF/runtime pairing also fails to load. |
| `NiuTrans/LMT-60-0.6B` revision `dd189845cdc73346cef33c7a94f4b8bd8efdd4eb`, locally converted to affine 4-bit/group-64 MLX | 346,929,488-byte authenticated snapshot | Apache-2.0 weights; released SFT dataset has no license tag and the 90B-token training mixture lacks row-level rights lineage | One Qwen3-based bidirectional model; native MLX architecture and plausible MLX Swift path | Rejected: 31.38/54.15 canary falls to 17.92/40.42 on 800 stress cases versus 33.23/54.00 for preferred-v2, with 141 critical-token failures and hallucinated/repeated entities. |
| `quickmt/quickmt-en-ja` revision `c09e98b8438a239a8210060114cea19c426c0559` + `quickmt/quickmt-ja-en` revision `e9ae594ff322d95254d730867c6166b25fd2c704` | 813,661,710-byte local CTranslate2 int8 pair | CC-BY-4.0 weights; 63.3M-row training dataset has no license declaration or row-level rights inventory | Encoder-heavy Marian pair: 8e/2d EN→JA and 12e/2d JA→EN; CPU CTranslate2 is fast, custom MLX/Swift port required | Rejected as weights/teacher: canary is 26.54/57.98 chrF++, EN→JA is below the incumbent, peak RSS is 1.78 GB, and training lineage mixes benchmark/train/test families plus backtranslations without licenses. Retain only the shallow-decoder architecture blueprint. |
| `Yokii2/quickmt-ja-en-v2` revision `3eeaf22dcd0a0ee7331aa4fa824d0cf33e6ea088` | about 211.3 MB for the JA→EN CTranslate2 engine and tokenizers | CC-BY-4.0 repository label, but its Patchouli training input and QuickMT base lack complete distributable lineage | JA→EN-only 12e/2d Eole model; CPU CTranslate2 works, but Mimi would need a separate MLX/Swift architecture port | Excluded: only publisher-domain self-evaluation exists, the exact 31M-row training subset is unreproducible, and Patchouli derives unknown-license CC-100 Japanese text plus an unspecified Mistral teacher. Not a shipping model or teacher. |
| `LiquidAI/LFM2-350M-ENJP-MT` revision `80367784d525777ad7565b24534ba5810eeac59f` | 381.6 MB for the published MLX 8-bit pack | LFM Open License 1.0 | One bidirectional causal LFM2 model; supported by pinned MLX Swift LM | License-screened out: commercial use is unlicensed for legal entities with annual revenue of at least USD 10M. Do not ship without a separate commercial agreement, regardless of quality. |
| `WhirlwindAI/Translate-15L` revision `ce860c33668440b031e30f50cc31377c6b6fac59` | 244,616,918-byte authenticated FP32 snapshot | Apache-2.0 | One 60.5M-parameter bidirectional T5 model; would need an MLX encoder-decoder port | Rejected before porting: greedy canary 0.00/1.37 chrF++; beam 4 reaches only 0.00/4.11, emits empty/repeated-punctuation outputs, and takes 2.60 s JA→EN p95. |
| TranslateGemma 4B | multi-GB even quantized | Gemma terms | Supported architecture, but much larger | Rejected for the very-small target; retain only as an offline teacher/reference candidate after terms review. |

Primary references:

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [Pinned MLX SmolLM2 source](https://huggingface.co/mlx-community/SmolLM2-135M-Instruct/tree/422de227b90002f443a21a58b1087f6ee7632731)
- [Qwen3-0.6B MLX 4-bit pinned tree](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit/tree/73e3e38d981303bc594367cd910ea6eb48349da8)
- [MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm)
- [OPUS English→Japanese](https://huggingface.co/Helsinki-NLP/opus-mt-en-jap)
- [OPUS Japanese→English](https://huggingface.co/Helsinki-NLP/opus-mt-jap-en)
- [ElanMT-BT English→Japanese](https://huggingface.co/Mitsua/elan-mt-bt-en-ja)
- [ElanMT-BT Japanese→English](https://huggingface.co/Mitsua/elan-mt-bt-ja-en)
- [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [SMaLL-100 pinned model](https://huggingface.co/alirezamsh/small100/tree/8ab680e26a596d2e3d2d2d17ae0f68df1037328c)
- [SMaLL-100 paper](https://aclanthology.org/2022.emnlp-main.571/)
- [Hy-MT2 1.25-bit MLX conversion](https://huggingface.co/kuotient/Hy-MT2-1.8B-1.25Bit-MLX)
- [Hy-MT2 report](https://arxiv.org/abs/2605.22064)
- [LMT-60-0.6B model card](https://huggingface.co/NiuTrans/LMT-60-0.6B)
- [LMT-60 SFT data card](https://huggingface.co/datasets/NiuTrans/LMT-60-sft-data)
- [LMT report](https://arxiv.org/abs/2511.07003)
- [QuickMT English→Japanese model](https://huggingface.co/quickmt/quickmt-en-ja)
- [QuickMT Japanese→English model](https://huggingface.co/quickmt/quickmt-ja-en)
- [QuickMT training-data card](https://huggingface.co/datasets/quickmt/quickmt-train.ja-en)
- [QuickMT JA→EN v2 pinned model](https://huggingface.co/Yokii2/quickmt-ja-en-v2/tree/3eeaf22dcd0a0ee7331aa4fa824d0cf33e6ea088)
- [Patchouli JA–EN pinned dataset](https://huggingface.co/datasets/Yokii2/patchouli-jaen/tree/e304a3b0b860745084976b2a50eef7b622742932)
- [LFM2-350M ENJP model card and LFM 1.0 license](https://huggingface.co/LiquidAI/LFM2-350M-ENJP-MT)
- [Translate-15L model card](https://huggingface.co/WhirlwindAI/Translate-15L)

## Experiment order

1. Keep the exact-parity 4-bit ElanMT pair fixed as the compact baseline.
2. Build licensed, production-like sources and generate structured GPT-5.6
   teacher candidates without ever exposing the protected suite.
3. Apply deterministic checks, a distinct optional fast judge, and two blind
   bilingual reviews; train only one adjudicated target per source.
4. Compare against the no-GPT hard-reference control. Its EN→JA child regressed,
   while JA→EN improved from 55.92 to 56.52 fused-kernel canary chrF++; this prevents
   crediting generic high-quality replay as a teacher gain.
5. Fine-tune full Marian weights, re-quantize to 4-bit, and compare every
   checkpoint to Apple. Keep Qwen3 only as a larger quality reference.
6. Do not expose a model selector to users until the held-out gate passes.
