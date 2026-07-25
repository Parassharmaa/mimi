# Hugging Face open publication

Mimi publishes the development model and benchmark as open reproducibility
artifacts on Hugging Face.

- `mimi-en-ja-development-v1-model-card.md` is copied to the public model
  repository as `README.md`.
- `mimi-en-ja-development-v1-dataset-card.md` is copied to the public dataset
  repository as `README.md`.
- `MODEL-ATTRIBUTIONS.md` and `tatoeba-attributions.jsonl.gz` accompany the
  CC BY-SA 4.0 model.
- `DATASET-LICENSES.md` records the benchmark's mixed open licenses.

The dataset upload excludes Apple outputs, blinded-judge mappings and verdicts,
secrets, and the sealed promotion suite. The model weights and Mimi
modifications are released under CC BY-SA 4.0. The benchmark preserves its
per-row upstream licenses, while Mimi-authored selection metadata and
arrangement are CC BY-SA 4.0.

The exact public repository commits, manifest hashes, visibility check, and
exclusion audit are recorded in `publication-manifest.json`.
