# Main-model grouped-CV OOF summary

- Completed paired folds: [0, 1, 2, 3, 4] (5/5)
- Holdout state: `LOCKED_NOT_EVALUATED`
- Partial-fold values are operational checks, not the final 5-fold conclusion.

```json
{
  "scope": "complete_5fold_oof",
  "completed_folds": [
    0,
    1,
    2,
    3,
    4
  ],
  "n": 5228,
  "qv_top1": 0.32115531752104054,
  "allattn_top1": 0.3817903596021423,
  "gain_pp": 6.063504208110176,
  "cluster_bootstrap_95ci_pp": [
    4.586981347544727,
    7.514907433766317
  ],
  "qv_wrong_allattn_right": 851,
  "qv_right_allattn_wrong": 534,
  "mcnemar_exact_p": 1.4878574978433783e-17
}
```
