# Fast Japanese ASR shortlist, 2026-10-02 JST

| Candidate | Native route | Relevant evidence | Next check |
| --- | --- | --- | --- |
| Moonshine Japanese Streaming Small/Tiny | Official C ABI and Swift package, CPU ONNX Runtime | 112.9M/27.0M Japanese variants; published native packs about 122/32 MB. Small batch-1 FLEURS CER 7.99%, Reazon 26.31%; these are publisher results, not Mimi results. MIT model terms explicitly include streaming Japanese. | Native candidate first; inspect tokenizer, configuration and fixed audio before inference. |
| ReazonSpeech K2 v2 INT8 | sherpa-onnx C/Swift offline transducer | Prior Mimi screen: compute RTF 0.022, CER 11.45%; no cached incremental encoder. | Recheck as rolling-window backup, keeping offline and live timing distinct. |
| Nemotron 3.5 streaming 0.6B | Official NeMo-Speech.cpp C ABI/Metal | True 80–1120ms cached streaming, Japanese supported. Prior MLX Q8 Japanese screen failed accuracy; current official Q8 GGUF 742 MB. | Alternate reference if small Japanese models fail. Do not assume improved accuracy from a different runtime. |
| PengChengStarling multilingual Zipformer | sherpa-onnx online C API | Japanese among eight languages; native INT8/FP32/INT8 pack about339 MB; requires Japanese language-tag handling. | Secondary true-transducer candidate. |

Primary references and pinned identifiers:
- Moonshine Small card: https://huggingface.co/moonshine-ai/moonshine-streaming-small-ja/tree/ac31061435fad5e523f84666035ba033efef4b41
- Moonshine Tiny card: https://huggingface.co/moonshine-ai/moonshine-streaming-tiny-ja/tree/a82871d17fede2efad4540d4a32eabb515fea5fa
- Moonshine native implementation: https://github.com/moonshine-ai/moonshine/tree/234f60faa0eb388b01cdf7e60aca232af37aefda
- Moonshine v2 paper: https://arxiv.org/abs/2602.12241
- Cache-aware Conformer paper: https://arxiv.org/abs/2312.17279
- Zipformer paper: https://arxiv.org/abs/2310.11230
- Nemotron card: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- Official native runtime: https://github.com/NVIDIA/NeMo-Speech.cpp
- Starling: https://github.com/PCL-Voice/PengChengStarling

The literature pass identified genuine native streaming paths, but none proves
Japanese Phonon-2-equivalent latency on this M3 Pro. Inference-only investigation;
no training, paid GPU jobs, or external model uploads are planned.
