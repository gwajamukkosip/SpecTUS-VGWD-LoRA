# r32 all-attention validation diagnostics protocol

Frozen before generating the diagnostic outputs on 2026-09-01.

## Scope

- Data: `data/vgwd_clean/valid_filtered_mz500.jsonl`, exactly 664 rows.
- Conditions: rsLoRA r32/alpha64 q/v and q/k/v/out all-attention.
- Seeds: 7, 123, 2026.
- Checkpoint: 3500.
- Inference: deterministic Beam 10.
- The VGWD test is excluded and paths containing `test` are rejected.
- This is descriptive error/robustness analysis. It will not be used to tune or select another architecture.

## Prespecified diagnostics

1. Strict and connectivity Top-1/Top-10 accuracy.
2. Strict Top-1 misses recovered within Top-10.
3. Connectivity-correct but stereochemistry-strict-wrong Top-1 cases.
4. Invalid SMILES among Top-10 candidates.
5. Paired q/v to all-attention Top-1 transitions for each seed.
6. Mean seed-wise performance by RDKit molecular-weight bin, maximum observed m/z bin, peak-count bin, stereochemistry annotation, and train-scaffold status.
7. Per-sample cross-seed consistency and samples that remain hard across all seeds.

## Fixed subgroup definitions

- Molecular weight: `<150`, `150-249`, `250-349`, `>=350` Da using RDKit `MolWt`.
- Maximum observed m/z: `<100`, `100-199`, `200-299`, `300-399`, `400-500`.
- Peak count: `<25`, `25-49`, `50-99`, `>=100`.
- Scaffold status: `train_seen`, `train_unseen`, or `acyclic`, using Bemis-Murcko scaffolds.
- Stereochemistry: whether canonical isomeric and non-isomeric SMILES differ.

Subgroups with fewer than 20 validation rows remain in CSV outputs but are omitted from the concise report table. No subgroup is used to change the frozen model configuration.
