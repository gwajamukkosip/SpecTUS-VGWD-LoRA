# r32 q/v vs all-attention main-model grouped 5-fold CV protocol

Status: preregistered locally on 2026-09-04; locked holdout must not be evaluated.

## Purpose

Test whether the validation-selected r32/alpha64 all-attention adapter improves
over the r32/alpha64 q/v adapter when the SpecTUS main model itself is retrained
under structure-grouped out-of-fold evaluation.

## Split and leakage controls

- Source pool: the existing cleaned VGWD train, validation, and test JSONL files.
- Eligibility is fixed to the model preprocessing limits: valid canonical SMILES,
  connectivity SMILES length <=100, maximum m/z <=500, and <=300 peaks.
- Group: RDKit canonical connectivity SMILES after stereochemistry removal.
- A deterministic 20% row-targeted group holdout is selected with seed 20260904.
- The remaining development groups are assigned to five row-balanced outer folds.
- Every spectrum of one connectivity structure stays in exactly one partition and
  one OOF fold. Exact spectrum hashes are also recorded.
- The runner contains no holdout command and rejects paths containing `holdout`.
- `data/vgwd_main_grouped_cv_locked/holdout_LOCKED.jsonl` may be evaluated only
  after the CV, calibration, abstention, and Top-k rules are frozen.

This is a prospectively locked **internal re-split**, not a never-observed external
dataset. The source splits had prior experimental roles, so a later holdout result
must be reported with that limitation. Repartitioning cannot erase prior analyst
exposure.

## Fixed paired model comparison

- Base checkpoint: `checkpoints/SpecTUS_pretrained`.
- Method A: rsLoRA r32, alpha64, q_proj/v_proj.
- Method B: rsLoRA r32, alpha64, q_proj/k_proj/v_proj/out_proj.
- Training seed and data seed: 123 for the first complete 5-fold comparison.
- Learning rate: 1e-4; dropout: 0.05; effective batch size: 4.
- Fixed training endpoint: 3,500 optimizer steps.
- An operational checkpoint is saved every 500 steps and only the latest is kept,
  so an interruption can resume without selecting a checkpoint by OOF performance.
- No outer-fold metric is used for checkpoint selection or early stopping.
- Prediction: deterministic Beam 10, ten returned candidates.
- Both methods use identical fold membership, seed, preprocessing, steps, and beam.

## Endpoints and decision rule

- Primary endpoint: strict canonical-isomeric Top-1 exact accuracy pooled over all
  OOF predictions.
- Secondary endpoints: strict Top-3, Top-5, and Top-10 accuracy; invalid Top-10
  rate; connectivity-only results; risk-coverage and calibration analyses.
- Primary uncertainty: paired canonical-connectivity-group bootstrap, 5,000 draws.
- Primary paired test: exact McNemar test over OOF rows, with a group-bootstrap
  interval reported because multiple spectra can belong to one compound.
- The all-attention method advances only if pooled OOF Top-1 gain is positive, its
  uncertainty is reported, and no prespecified major subgroup shows a material
  safety or data-quality failure. The locked holdout is not used to revise this
  rule.

## Execution stages

1. Create and hash the split once; subsequently use verify-only mode.
2. Run two-step q/v and all-attention pipeline smoke tests on fold 0.
3. Run all 5 folds x 2 methods on the new computer, or resume completed folds.
4. Build pooled OOF predictions and freeze calibration/abstention/Top-k rules.
5. Evaluate the locked holdout exactly once with the frozen analysis.

No conclusion about the grouped-CV result or locked holdout is recorded until the
corresponding predictions actually exist.
