# OOF calibration, abstention, risk-coverage, and Top-k utility

- Scope: completed grouped 5-fold OOF only; locked holdout untouched.
- Primary candidate: r32/alpha64 all-attention.
- Calibration: five-fold cross-fitted logistic model; each fold calibrated from the other four.
- All-attention raw/calibrated Brier: 0.275246 / 0.203792.
- All-attention raw/calibrated ECE: 0.252551 / 0.017805.
- Frozen abstention threshold: calibrated P(correct) >= 0.80.
- Primary coverage: 0.8416% (44/5,228).
- Primary selective accuracy: 90.9091%.

## All-attention Top-k quantitative workload proxies

| K | Strict accuracy | Hits | Added hits vs Top-1 | Extra slots per added hit |
|---:|---:|---:|---:|---:|
| 1 | 38.1790% | 1,996 | 0 | — |
| 3 | 65.1301% | 3,405 | 1,409 | 7.42 |
| 5 | 76.4728% | 3,998 | 2,002 | 10.45 |
| 10 | 85.3481% | 4,462 | 2,466 | 19.08 |

These workload figures are quantitative proxies, not an actual researcher user study.
The threshold is frozen for a future one-time holdout evaluation and was not changed after viewing results.
