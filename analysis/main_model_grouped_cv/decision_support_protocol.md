# Frozen OOF decision-support analysis protocol

Status: frozen before inspecting calibration, abstention, or risk-coverage results on
2026-09-05. The locked holdout must remain unevaluated.

## Scope

This analysis uses only the completed q/v and all-attention grouped 5-fold OOF
predictions. The validation-selected all-attention model is the primary operational
candidate; q/v is a secondary comparator. No model is retrained and no locked
holdout row is read.

## Outcome and confidence features

- Outcome: strict canonical-isomeric Top-1 exact correctness from the existing
  probability-ranked audit.
- Candidate scores: sequence scores already stored in each `predictions.jsonl`.
- The scores are treated as ranking scores, not as calibrated probabilities.
- Fixed calibration features: log top-1 score, log top-1/top-2 score ratio,
  normalized entropy of the Beam-10 score shares, and valid candidate count.
- Scores use an epsilon of `1e-300`; a missing second candidate uses the top-1
  score as the denominator and is flagged through candidate count.
- Data-quality amendment recorded before any holdout access: the OOF audit found
  one all-attention row with zero valid candidates. A row with no candidate has no
  defined score feature, is assigned confidence zero, is always abstained, and is
  excluded from calibrator fitting while remaining in every reported denominator.
  This missing-output rule does not change the model, feature set, or thresholds.

## Cross-fitted calibration

- Five-fold cross-fitting follows the already locked OOF fold IDs.
- For each held-out fold, a standard-scaled logistic regression (`C=1`, L2,
  `max_iter=2000`, deterministic solver) is fitted on the other four folds only.
- Every reported calibrated probability is therefore produced by a calibrator
  that did not fit that row or its connectivity group.
- Calibration metrics: Brier score, binary log loss, and 10-bin equal-frequency
  expected calibration error (ECE). The uncalibrated comparator is the top-1
  Beam-10 score share.

## Abstention and risk-coverage

- Primary frozen policy: emit a Top-1 answer only when cross-fitted calibrated
  `P(strict Top-1 correct) >= 0.80`; otherwise abstain.
- Secondary thresholds: 0.50, 0.60, 0.70, and 0.90.
- Report coverage, selective accuracy, selective risk, accepted row count, and
  abstained row count.
- The risk-coverage curve is evaluated at descending calibrated-confidence
  prefixes from 5% through 100% coverage in 5 percentage-point increments.
- The primary 0.80 policy receives a 95% canonical-connectivity cluster-bootstrap
  interval for coverage and selective accuracy using 5,000 draws and seed
  20260905.
- This is decision support, not a safety guarantee. Abstention cannot turn the
  model into a definitive chemical identifier.

## Top-k quantitative utility

- Fixed K values: 1, 3, 5, and 10.
- Report strict accuracy, total recovered correct rows, incremental recovery over
  Top-1, mean reciprocal rank through 10, and extra candidate slots per additional
  correct recovery relative to Top-1.
- These are quantitative workload proxies. They must not be described as an
  actual researcher user study or proof of field utility.

## Interpretation and freezing

- Results apply to this prospectively locked internal re-split and training/data
  seed 123. The five OOF folds are not five independent training seeds.
- Thresholds or methods are not changed after viewing results. If the primary
  threshold has negligible coverage, that limitation is reported rather than
  replacing it post hoc.
- After outputs and hashes are generated, the method, confidence features,
  primary threshold, and analysis script are frozen before any holdout evaluation.
