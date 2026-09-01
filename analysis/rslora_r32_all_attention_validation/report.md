# rsLoRA r32 all-attention validation-only screen

Seeds 7, 123, 2026; checkpoint-3500; deterministic Beam 10; 664 validation rows.
The VGWD test was not used or generated for this screen.

## Individual validation runs

| condition     |   seed |   n |   strict_top1_correct |   strict_top1_percent |   strict_top10_correct |   strict_top10_percent |
|:--------------|-------:|----:|----------------------:|----------------------:|-----------------------:|-----------------------:|
| all_attention |      7 | 664 |                   261 |               39.3072 |                    581 |                87.5    |
| all_attention |    123 | 664 |                   277 |               41.7169 |                    581 |                87.5    |
| all_attention |   2026 | 664 |                   250 |               37.6506 |                    582 |                87.6506 |
| qv            |      7 | 664 |                   227 |               34.1867 |                    551 |                82.9819 |
| qv            |    123 | 664 |                   213 |               32.0783 |                    547 |                82.3795 |
| qv            |   2026 | 664 |                   211 |               31.7771 |                    538 |                81.0241 |

## Paired all-attention minus q/v effects

|   seed | metric   |   qv_percent |   all_attention_percent |   gain_pp |   promoted |   demoted |   mcnemar_exact_p |   structure_cluster_bootstrap_95ci_low_pp |   structure_cluster_bootstrap_95ci_high_pp |
|-------:|:---------|-------------:|------------------------:|----------:|-----------:|----------:|------------------:|------------------------------------------:|-------------------------------------------:|
|      7 | top1     |      34.1867 |                 39.3072 |   5.12048 |        119 |        85 |       0.0206353   |                                  0.470758 |                                    9.6678  |
|    123 | top1     |      32.0783 |                 41.7169 |   9.63855 |        121 |        57 |       1.8294e-06  |                                  5.46282  |                                   13.6924  |
|   2026 | top1     |      31.7771 |                 37.6506 |   5.87349 |        115 |        76 |       0.00582338  |                                  1.37615  |                                   10.4234  |
|      7 | top10    |      82.9819 |                 87.5    |   4.51807 |         46 |        16 |       0.000176327 |                                  2.17729  |                                    6.92308 |
|    123 | top10    |      82.3795 |                 87.5    |   5.12048 |         53 |        19 |       7.55581e-05 |                                  2.59542  |                                    7.65857 |
|   2026 | top10    |      81.0241 |                 87.6506 |   6.62651 |         57 |        13 |       1.02899e-07 |                                  4.0995   |                                    9.231   |

## Seed aggregate

| metric   |   number_of_seeds |   mean_gain_pp |   sd_gain_pp |   min_gain_pp |   max_gain_pp |   positive_seed_count |   seed_t_95ci_low_pp |   seed_t_95ci_high_pp |
|:---------|------------------:|---------------:|-------------:|--------------:|--------------:|----------------------:|---------------------:|----------------------:|
| top1     |                 3 |        6.87751 |      2.4206  |       5.12048 |       9.63855 |                     3 |             0.864419 |              12.8906  |
| top10    |                 3 |        5.42169 |      1.08601 |       4.51807 |       6.62651 |                     3 |             2.72389  |               8.11948 |

## Preregistered decision

Select r32 all-attention for a future locked/external evaluation: **YES**.
The gate uses strict Top-1 only: mean gain > 0 and positive gain in at least 2/3 seeds.
Top-10 is secondary. n=3 intervals are exploratory and do not establish external generalization.
