# Mimi local translation model

Mimi includes two quantized ElanMT translation models:

- `Mitsua/elan-mt-bt-en-ja`, revision
  `02c48e7031386cd2d41974b0ff1aaf52f010c5fa`
- `Mitsua/elan-mt-bt-ja-en`, revision
  `539f80eb05306e27a166b45e4264c7fa2eb4de97`

ElanMT was developed by the ELAN MITSUA Project / Abstract Engine. The original
model pages are:

- https://huggingface.co/Mitsua/elan-mt-bt-en-ja
- https://huggingface.co/Mitsua/elan-mt-bt-ja-en

The model files, tokenizer files, model manifests, and Mimi's quantized model
adaptations are distributed under Creative Commons Attribution-ShareAlike 4.0
International (CC BY-SA 4.0). The complete license is included in
`CC-BY-SA-4.0.txt` and is available at:
https://creativecommons.org/licenses/by-sa/4.0/

## Mimi modifications

Mimi converted the pinned upstream floating-point Marian encoder-decoder
weights to MLX safetensors using 4-bit affine quantization, group size 64, and
float16 computation. It then packaged the two directional models for offline
Apple Silicon inference. Mimi did not fine-tune these shipping weights.

The model component remains separately licensed from Mimi's application code.
Recipients may extract, study, modify, and redistribute the model component
under CC BY-SA 4.0. Mimi's direct GitHub archive does not add DRM to the model
files. No Mac App Store distribution of this model component is authorized by
Mimi's release contract.

## Disclaimer

Machine translation can be incorrect, harmful, or biased and must not be
treated as authoritative. The upstream authors disclaim warranties and
liability under CC BY-SA 4.0. Mimi's packaging and modifications do not imply
endorsement by the ELAN MITSUA Project or Abstract Engine.
