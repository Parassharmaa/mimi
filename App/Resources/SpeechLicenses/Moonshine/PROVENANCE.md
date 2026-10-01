# Moonshine Japanese streaming preview

- Native Swift package revision: `45a14f9edf1f2a6913d3aff38c1fd4e72d5b7daa` (v0.1.5).
- Native XCFramework SHA-256: `6bc7fb4b6d3a470a2ae2d681299975f3ba9d710753786d1cd7e8beabaad066e8`.
- Native source revision: `234f60faa0eb388b01cdf7e60aca232af37aefda`.
- Model: Japanese Streaming Small, 112.9 million parameters, native quantized pack dated `26_08_23`.
- Publisher pack: https://download.moonshine.ai/model/small-streaming-ja/quantized_26_08_23/
- Evaluated eight-file payload: 121,803,780 bytes.
- Model and native runtime: MIT. The publisher explicitly includes streaming Japanese in the MIT terms; its legacy non-streaming Japanese models have different terms.
- Hugging Face architecture/card revision: https://huggingface.co/moonshine-ai/moonshine-streaming-small-ja/tree/ac31061435fad5e523f84666035ba033efef4b41

The native publisher's file metadata records size and CRC32C. Mimi's research
fetcher checked both and recorded SHA-256 for the exact downloaded native files.
`MimiMoonshineModel.swift` pins those sizes and hashes. The native ORT pack is a
separate artifact from the Hugging Face PyTorch weights; no byte identity between
them is claimed. Model files are optional downloads rather than bundled assets.

The adjacent notices cover native dependencies distributed in the static runtime.
The complete publisher model/runtime terms are in `../MOONSHINE-LICENSE.txt`.

Eigen source covered by MPL2 is available in the pinned Moonshine source tree:
https://github.com/moonshine-ai/moonshine/tree/234f60faa0eb388b01cdf7e60aca232af37aefda/core/third-party/Eigen
No Eigen source changes are made by Mimi.
