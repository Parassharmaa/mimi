#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h}"
cd "$ROOT"

uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/test_benchmark_contract.py
python3 scripts/translation/test_automated_claim_contract.py
python3 scripts/translation/test_automated_claim_sources.py
PYTHONPATH=scripts/translation python3 \
  scripts/translation/test_automated_claim_exposure_manifest.py
PYTHONPATH=scripts/translation python3 \
  scripts/translation/test_automated_claim_semantic_scan.py
python3 scripts/translation/test_automated_claim_reference_batch.py
python3 scripts/translation/test_audit_synthetic_batch_privacy.py
PYTHONPATH=scripts/translation python3 \
  scripts/translation/test_automated_claim_reference_consensus.py
PYTHONPATH=scripts/translation python3 \
  scripts/translation/test_automated_claim_source_contamination.py
uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/test_automated_promotion_contract.py
uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/test_score_translation_contract.py
uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/test_licensed_data_pipeline.py
python3 scripts/translation/test_japanese_law_translation.py
python3 scripts/translation/test_public_stress_suite.py
python3 scripts/translation/test_legal_safety_validation.py
uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/test_marian_legal_safety_checkpoint_selection.py
PYTHONPATH=scripts/translation uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/test_structure_fallback.py
python3 scripts/translation/test_align_translation_report_intersection.py
python3 scripts/translation/test_filter_translation_report_to_suite.py
python3 scripts/translation/test_analyze_critical_token_failures.py
PYTHONPATH=scripts/translation python3 \
  scripts/translation/test_source_only_critical_failures.py
PYTHONPATH=scripts/translation uv run --python 3.12 \
  --with transformers==4.57.6 --with tokenizers \
  scripts/translation/test_marian_target_shortlist.py
python3 scripts/translation/test_compare_marian_target_shortlist.py
python3 scripts/translation/test_marian_negative_space_dataset.py
PYTHONPATH=scripts/translation python3 \
  scripts/translation/test_typed_critical_token_policy.py
uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/test_exact_translation_memory.py
uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/test_synthetic_pipeline.py
uv run --python 3.12 --with numpy \
  scripts/translation/test_dqrd_selection.py
uv run --python 3.12 --with torch --with transformers==4.57.6 \
  --with sentencepiece --with sacremoses --with sacrebleu==2.6.0 --with numpy \
  scripts/translation/test_dqrd_training_objective.py
uv run --python 3.12 --with torch --with transformers==4.57.6 \
  --with sentencepiece --with sacremoses --with sacrebleu==2.6.0 --with numpy \
  --with huggingface-hub scripts/translation/test_marian_negative_space_objective.py
uv run --python 3.12 --with torch --with safetensors --with numpy \
  scripts/translation/test_checkpoint_averaging.py
uv run --python 3.12 --with torch --with safetensors --with numpy \
  scripts/translation/test_checkpoint_interpolation.py
uv run --python 3.12 --with torch --with safetensors --with numpy \
  scripts/translation/test_marian_checkpoint_decoder_pruning.py
uv run --python 3.12 --with torch --with safetensors --with numpy \
  scripts/translation/test_bidirectional_model.py
uv run --python 3.12 --with torch --with transformers==4.57.6 \
  --with sentencepiece --with sacremoses --with sacrebleu==2.6.0 --with numpy \
  scripts/translation/test_bidirectional_training_objective.py
uv run --python 3.12 --with torch --with transformers==4.57.6 \
  --with sentencepiece --with sacremoses --with sacrebleu==2.6.0 --with numpy \
  --with mlx==0.30.6 scripts/translation/test_mlx_affine_qat.py
uv run --python 3.12 --with mlx==0.30.6 \
  scripts/translation/test_marian_mlx_block_cache.py
uv run --python 3.12 --with mlx==0.30.6 \
  scripts/translation/test_marian_mlx_position_table.py
PYTHONPATH=scripts/translation uv run --python 3.12 --with mlx==0.30.6 \
  scripts/translation/test_marian_mlx_output_shortlist.py
PYTHONPATH=scripts/translation uv run --python 3.12 --with mlx==0.30.6 \
  scripts/translation/test_marian_mlx_packed_projections.py
uv run --python 3.12 --with mlx==0.30.6 --with transformers==4.40.2 \
  --with sentencepiece --with sacremoses \
  scripts/translation/test_marian_mlx_draft_reuse.py
uv run --python 3.12 --with mlx==0.30.6 \
  scripts/translation/test_marian_mlx_decoder_pruning.py
uv run --python 3.12 --with mlx==0.30.6 --with transformers==4.40.2 \
  --with sentencepiece --with sacremoses \
  scripts/translation/test_mlx_marian_benchmark_cli.py
python3 scripts/translation/test_learned_metric_contract.py
python3 scripts/translation/test_compare_learned_metric.py
python3 scripts/translation/test_quantization_packaging.py
python3 scripts/translation/test_automated_consensus.py
python3 scripts/translation/test_cat_translate_finetune.py
python3 scripts/translation/test_local_teacher_consensus.py
python3 scripts/translation/test_local_qwen_teacher_retry.py
python3 scripts/translation/test_training_manifest_provenance.py
python3 scripts/translation/test_stage_translation_release_artifacts.py
python3 scripts/translation/test_make_portable_marian_release.py
python3 scripts/translation/test_freeze_translation_license_bundle.py
python3 scripts/translation/test_prepare_m2m100_feasibility_suite.py
python3 scripts/translation/test_compare_marian_portable_pack_reports.py
PYTHONPATH=scripts/translation python3 \
  scripts/translation/test_marian_release_contract.py
PYTHONPATH=scripts/translation python3 \
  scripts/translation/test_deduplicate_marian_moe_tokenizer.py
PYTHONPATH=scripts/translation python3 \
  scripts/translation/test_compare_marian_moe_pack_smokes.py
python3 scripts/translation/test_repair_local_reference_training_manifests.py
uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/test_local_reference_teacher.py
uv run --python 3.12 --with sacrebleu==2.6.0 --with numpy \
  scripts/translation/test_balanced_reference_teacher_seeds.py
python3 scripts/translation/test_balanced_human_reference_ablation.py
python3 scripts/translation/test_shallow_student_dataset.py
uv run --python 3.12 --with scikit-learn --with sacrebleu==2.6.0 --with numpy \
  scripts/translation/test_expert_router.py
PYTHONPATH=scripts/translation uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/test_roundtrip_expert_reranker.py
PYTHONPATH=scripts/translation uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/test_self_likelihood_expert_reranker.py
PYTHONPATH=scripts/translation python3 scripts/translation/test_marian_sequence_targets.py
uv run --python 3.12 --with torch --with transformers==4.57.6 \
  --with sentencepiece --with sacremoses --with sacrebleu==2.6.0 --with numpy \
  scripts/translation/test_dqo_pipeline.py

python3 -m py_compile \
  scripts/translation/build_dqo_preferences.py \
  scripts/translation/build_marian_negative_space_dataset.py \
  scripts/translation/train_marian_negative_space.py \
  scripts/translation/build_bidirectional_dataset.py \
  scripts/translation/build_shallow_student_dataset.py \
  scripts/translation/download_japanese_law_translation.py \
  scripts/translation/prepare_japanese_law_translation.py \
  scripts/translation/distill_marian_sequence_targets.py \
  scripts/translation/merge_directional_marian.py \
  scripts/translation/prepare_public_stress_suite.py \
  scripts/translation/prepare_legal_safety_validation.py \
  scripts/translation/select_marian_legal_safety_checkpoint.py \
  scripts/translation/evaluate_structure_fallback.py \
  scripts/translation/train_bidirectional_marian.py \
  scripts/translation/validate_automated_benchmark_suite.py \
  scripts/translation/prepare_automated_claim_sources.py \
  scripts/translation/build_automated_claim_exposure_manifest.py \
  scripts/translation/scan_automated_claim_semantic_contamination.py \
  scripts/translation/prepare_automated_claim_reference_batch.py \
  scripts/translation/collect_automated_claim_reference_candidates.py \
  scripts/translation/prepare_automated_claim_reference_judge_batch.py \
  scripts/translation/collect_automated_claim_reference_judgments.py \
  scripts/translation/assemble_automated_claim_reference_suite.py \
  scripts/translation/audit_automated_claim_reference_structures.py \
  scripts/translation/audit_automated_claim_source_contamination.py \
  scripts/translation/evaluate_automated_translation_promotion.py \
  scripts/translation/evaluate_supervised_win.py \
  scripts/translation/train_marian_dqo.py \
  scripts/translation/run_mlx_marian_benchmark.py \
  scripts/translation/run_mlx_benchmark.py \
  scripts/translation/run_ct2_sentencepiece_benchmark.py \
  scripts/translation/run_t5_benchmark.py \
  scripts/translation/run_small100_benchmark.py \
  scripts/translation/run_m2m100_benchmark.py \
  scripts/translation/run_mlx_marian_moe_benchmark.py \
  scripts/translation/apply_symmetric_critical_fallback.py \
  scripts/translation/compare_source_only_moe_runtime.py \
  scripts/translation/compare_source_only_moe_candidates.py \
  scripts/translation/build_critical_preservation_curriculum.py \
  scripts/translation/train_small100_lora.py \
  scripts/translation/deduplicate_marian_moe_tokenizer.py \
  scripts/translation/compare_marian_moe_pack_smokes.py \
  scripts/translation/test_compare_marian_moe_pack_smokes.py \
  scripts/translation/benchmark_marian_partial_retranslation.py \
  scripts/translation/benchmark_marian_model_draft.py \
  scripts/translation/benchmark_marian_compiled_blocks.py \
  scripts/translation/benchmark_marian_packed_projections.py \
  scripts/translation/benchmark_marian_ssru_proxy.py \
  scripts/translation/evaluate_expert_router.py \
  scripts/translation/apply_expert_routing.py \
  scripts/translation/prepare_roundtrip_expert_reranking_suites.py \
  scripts/translation/evaluate_roundtrip_expert_reranker.py \
  scripts/translation/test_roundtrip_expert_reranker.py \
  scripts/translation/score_mlx_marian_self_likelihood.py \
  scripts/translation/evaluate_self_likelihood_expert_reranker.py \
  scripts/translation/test_self_likelihood_expert_reranker.py \
  scripts/translation/evaluate_weighted_expert_router_classifier.py \
  scripts/translation/align_translation_report_intersection.py \
  scripts/translation/filter_translation_report_to_suite.py \
  scripts/translation/analyze_critical_token_failures.py \
  scripts/translation/analyze_source_only_critical_failures.py \
  scripts/translation/marian_target_shortlist.py \
  scripts/translation/build_marian_target_shortlist.py \
  scripts/translation/compare_marian_target_shortlist.py \
  scripts/translation/test_compare_marian_target_shortlist.py \
  scripts/translation/test_marian_mlx_packed_projections.py \
  scripts/translation/typed_critical_token_policy.py \
  scripts/translation/evaluate_typed_critical_token_policy.py \
  scripts/translation/evaluate_marian_typed_nbest_rescue.py \
  scripts/translation/build_exact_translation_memory.py \
  scripts/translation/prepare_exact_translation_memory_suite.py \
  scripts/translation/apply_exact_translation_memory.py \
  scripts/translation/package_marian_translation_memory.py \
  scripts/translation/audit_translation_structures.py \
  scripts/translation/export_expert_router_predictions.py \
  scripts/translation/prepare_protected_token_suite.py \
  scripts/translation/restore_protected_token_report.py \
  scripts/translation/source_expert_router.py \
  scripts/translation/package_elanmt_mlx_experts.py \
  scripts/translation/benchmark_marian_moe_residency.py \
  scripts/translation/build_marian_release_contract.py \
  scripts/translation/test_marian_release_contract.py \
  scripts/translation/prepare_cat_translate_finetune.py \
  scripts/translation/prepare_local_teacher_suite.py \
  scripts/translation/prepare_local_teacher_backtranslation_suite.py \
  scripts/translation/prepare_local_teacher_roundtrip_subset.py \
  scripts/translation/build_local_teacher_consensus.py \
  scripts/translation/score_roundtrip_nli.py \
  scripts/translation/run_local_bilingual_judge.py \
  scripts/translation/filter_local_bilingual_judge.py \
  scripts/translation/build_local_teacher_ablation.py \
  scripts/translation/prepare_local_reference_teacher_suite.py \
  scripts/translation/prepare_balanced_reference_teacher_seeds.py \
  scripts/translation/build_balanced_human_reference_ablation.py \
  scripts/translation/run_local_qwen_teacher.py \
  scripts/translation/select_local_teacher_candidates.py \
  scripts/translation/test_local_qwen_teacher_retry.py \
  scripts/translation/filter_local_reference_teacher.py \
  scripts/translation/build_reference_teacher_ablation.py \
  scripts/translation/training_manifest_provenance.py \
  scripts/translation/repair_local_reference_training_manifests.py \
  scripts/translation/repair_training_manifest_provenance.py \
  scripts/translation/train_marian_distillation.py \
  scripts/translation/marian_mlx.py \
  scripts/translation/interpolate_marian_checkpoints.py \
  scripts/translation/verify_translation_distribution.py \
  scripts/translation/stage_translation_release_artifacts.py \
  scripts/translation/test_stage_translation_release_artifacts.py \
  scripts/translation/make_portable_marian_release.py \
  scripts/translation/test_make_portable_marian_release.py \
  scripts/translation/freeze_translation_license_bundle.py \
  scripts/translation/test_freeze_translation_license_bundle.py \
  scripts/translation/prepare_m2m100_feasibility_suite.py \
  scripts/translation/test_prepare_m2m100_feasibility_suite.py \
  scripts/translation/compare_marian_portable_pack_reports.py \
  scripts/translation/test_compare_marian_portable_pack_reports.py \
  scripts/translation/compare_learned_metric.py

if [[ "${MIMI_TRANSLATION_TRAINING_SMOKE:-0}" == "1" ]]; then
  uv run --python 3.12 --with torch --with transformers==4.57.6 \
    --with sentencepiece --with sacremoses --with sacrebleu==2.6.0 --with numpy \
    scripts/translation/smoke_train_marian_distillation.py
  uv run --python 3.12 --with torch --with transformers==4.57.6 \
    --with sentencepiece --with sacremoses --with numpy \
    scripts/translation/smoke_train_marian_dqo.py
fi

echo "Mimi translation research contracts passed."
