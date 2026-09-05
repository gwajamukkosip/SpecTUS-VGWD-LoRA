# VGWD clean rsLoRA r32 all-attention — frozen seed 123 candidate

## Status

Frozen on 2026-09-01 as a **validation-selected research candidate**. This is not an externally validated or deployment-certified model.

The artifact contains only the LoRA adapter and tokenizer metadata. It requires the original SpecTUS base checkpoint at `checkpoints/SpecTUS_pretrained`.

## Configuration

- Adapter: rsLoRA
- Rank: 32
- Alpha: 64
- Dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `out_proj`
- Training seed: 123
- Checkpoint: 3500
- Inference: deterministic Beam 10, return 10, maximum length 200
- Trainable parameters during fine-tuning: 9,437,184

Seed 123 was frozen because it had the highest strict Top-1 on the fixed 664-row validation set among the three prespecified all-attention runs. This is a transparent validation-based choice and may be optimistic. No VGWD test result was used to select or freeze this artifact.

## Validation result

- Strict Top-1: 277/664 (41.7169%)
- Strict Top-10: 581/664 (87.5000%)

Across seeds 7, 123, and 2026, all-attention improved over the matched q/v model by a mean +6.8775 percentage points for strict Top-1 and +5.4217 points for strict Top-10. These results share the validation split used for method selection and are not independent confirmation.

## Inference

The packaged entry point validates the input and keeps the frozen Beam 10 settings:

```bash
bash deployment/r32_all_attention/run_inference.sh \
  /absolute/path/input.jsonl \
  predictions/frozen_r32_all_attention_seed123
```

Installation, pinned dependencies, base-model retrieval, input schema, integrity checks,
and online/offline release bundle creation are documented in
`deployment/r32_all_attention/README_KO.md`. Keep generation settings fixed for
confirmatory use.

## Integrity verification

Run:

```bash
(cd models/vgwd_clean_rslora_r32_all_attention_seed123_frozen && sha256sum -c SHA256SUMS)
```

`manifest.json` records the adapter, base-model and tokenizer hashes, all three seed adapter hashes, fixed settings, validation metrics, and source paths.

## Intended use and limitations

- Intended for research on EI mass-spectrum-to-structure candidate generation within the represented VGWD/SpecTUS domain.
- Predictions are hypotheses, not compound identifications. Confirm candidates using reference standards and independent analytical evidence.
- Do not use this model alone for safety, exposure, attribution, clinical, regulatory, or operational CBRN decisions.
- Performance on external instruments, unseen scaffolds, new laboratories, field samples, mixtures, low-quality spectra, or different distributions has not been established.
- The validation set was used for model selection, and no untouched external dataset was available at freeze time.
