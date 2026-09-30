# Mimi Phonon 2 conversion

Phonon 2 is by Fermion Research and derives from NVIDIA Parakeet TDT 0.6B v3.
The model weights are licensed under Creative Commons Attribution 4.0.

Source: https://huggingface.co/FermionResearch/Phonon-2/tree/1c388bcec35d19740bf36b0b675718223fa7904e

Mimi converted the five-value encoder into two exact 2-bit affine MLX planes
with FP32 scales and group size 128. All 264 converted modules reproduce the
source weight values exactly. Remaining parameters use BF16 to match the
packed reference runtime. Inference runs through native MLX Swift and MLX
Audio Swift. This conversion does not add Japanese language support.

The upstream NOTICE and weight/code license texts are included alongside this
modification notice. The converted weights remain under CC BY 4.0.
