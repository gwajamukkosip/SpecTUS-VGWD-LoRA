# Preregistered validation-only screen: rsLoRA r32 all-attention projections

Protocol frozen: 2026-08-30 KST, before any r32 all-attention training result was generated.

## Purpose

Determine whether the r16 all-attention validation gain transfers to the currently adopted rank/scaling setting. This comparison changes target modules only; rank, alpha, seed, training budget, checkpoint, and decoding are matched.

## Conditions

| condition | rank | alpha | target modules |
|---|---:|---:|---|
| baseline | 32 | 64 | `q_proj`, `v_proj` |
| all-attention | 32 | 64 | `q_proj`, `k_proj`, `v_proj`, `out_proj` |

SpecTUS uses BART attention modules, whose output projection is named `out_proj`. Targets apply to encoder self-attention, decoder self-attention, and decoder cross-attention wherever these projection names occur.

All other training fields are inherited byte-for-byte from the same-seed r32/alpha64 q/v configuration. Inference uses checkpoint 3500, deterministic Beam 10, and 10 returned sequences.

## Seeds and data

- Seeds fixed before execution: 7, 123, 2026.
- Training source: `data/vgwd_clean/train.jsonl` with the existing preprocessing.
- Selection/evaluation labels: `data/vgwd_clean/valid_filtered_mz500.jsonl`, 664 rows.
- The inference config reads raw validation and the established `m/z <= 500` preprocessing yields those same 664 rows.
- The 675-row VGWD test is prohibited. No test prediction may be generated.
- No seed may be removed or replaced based on its result. Hardware interruption is resumed with the same seed.

The q/v checkpoints already exist and are not retrained. They are evaluated at checkpoint 3500 with the same decoding as the all-attention checkpoints.

## Outcomes and locked decision rule

- Primary endpoint: strict exact Top-1 validation accuracy.
- Secondary endpoint: strict exact Top-10 validation accuracy.
- Report every seed, paired gain, mean, sample SD, range, sign consistency, and exploratory t-based 95% CI of mean seed gain.
- Within each seed, report exact McNemar and canonical-structure cluster-bootstrap 95% CI. These are validation sample/structure analyses, not seed-level significance tests.
- Select r32 all-attention as the candidate for a future locked/external evaluation only if mean Top-1 gain is positive and at least 2 of 3 seeds have a positive Top-1 gain. Otherwise retain r32 q/v.
- Top-10 is supportive and is not the gate. Passing does not authorize evaluation on the repeatedly exposed VGWD test.

## Paths

- Config generator: `analysis/generate_rslora_r32_all_attention_configs.py`
- Runner: `config_runners/run_vgwd_clean_rslora_r32_all_attention_validation.sh`
- Summarizer: `analysis/summarize_rslora_r32_all_attention_validation.py`
- Results: `analysis/rslora_r32_all_attention_validation/`
