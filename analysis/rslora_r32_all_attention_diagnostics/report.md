# r32 all-attention validation diagnostics

Descriptive analysis of 664 filtered validation rows, seeds 7/123/2026, checkpoint-3500 and Beam 10.
The VGWD test was not used. These diagnostics do not select another architecture.

## Overall diagnostics

| condition     |   seed |   n |   strict_top1_percent |   strict_top10_percent |   connectivity_top1_percent |   connectivity_top10_percent |   stereo_gap_top1_count |   stereo_gap_top1_percent |   top10_rescue_count |   top10_rescue_percent |   invalid_top1_count |   invalid_candidates_in_top10 |
|:--------------|-------:|----:|----------------------:|-----------------------:|----------------------------:|-----------------------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|---------------------:|------------------------------:|
| all_attention |      7 | 664 |               39.3072 |                87.5    |                     39.3072 |                      88.8554 |                       0 |                  0        |                  320 |                48.1928 |                    0 |                             0 |
| all_attention |    123 | 664 |               41.7169 |                87.5    |                     41.8675 |                      88.7048 |                       1 |                  0.150602 |                  304 |                45.7831 |                    0 |                             0 |
| all_attention |   2026 | 664 |               37.6506 |                87.6506 |                     37.6506 |                      88.8554 |                       0 |                  0        |                  332 |                50      |                    0 |                             0 |
| qv            |      7 | 664 |               34.1867 |                82.9819 |                     34.6386 |                      84.1867 |                       3 |                  0.451807 |                  324 |                48.7952 |                    0 |                             0 |
| qv            |    123 | 664 |               32.0783 |                82.3795 |                     32.5301 |                      83.5843 |                       3 |                  0.451807 |                  334 |                50.3012 |                    0 |                             0 |
| qv            |   2026 | 664 |               31.7771 |                81.0241 |                     31.7771 |                      82.2289 |                       0 |                  0        |                  327 |                49.247  |                    0 |                             0 |

## Paired strict Top-1 transitions

|   seed |   both_correct |   qv_only_demoted |   all_attention_only_promoted |   both_wrong |   net_gain |
|-------:|---------------:|------------------:|------------------------------:|-------------:|-----------:|
|      7 |            142 |                85 |                           119 |          318 |         34 |
|    123 |            156 |                57 |                           121 |          330 |         64 |
|   2026 |            135 |                76 |                           115 |          338 |         39 |

## Prespecified subgroup summary

Rows below have at least 20 validation samples. Percentages are means of the three seed-specific accuracies.

| dimension            | group      |   n |   mean_qv_top1_percent |   mean_all_attention_top1_percent |   mean_top1_gain_pp |   positive_top1_seeds |   mean_qv_top10_percent |   mean_all_attention_top10_percent |   mean_top10_gain_pp |   positive_top10_seeds |
|:---------------------|:-----------|----:|-----------------------:|----------------------------------:|--------------------:|----------------------:|------------------------:|-----------------------------------:|---------------------:|-----------------------:|
| max_mz_bin           | 100-199    | 313 |                34.2918 |                           39.7231 |             5.43131 |                     3 |                 84.8775 |                            89.5634 |              4.68584 |                      3 |
| max_mz_bin           | 200-299    | 299 |                32.9989 |                           41.5831 |             8.58417 |                     3 |                 82.6087 |                            88.2943 |              5.68562 |                      3 |
| max_mz_bin           | 300-399    |  50 |                22      |                           28      |             6       |                     3 |                 65.3333 |                            74      |              8.66667 |                      3 |
| molecular_weight_bin | 150-249    | 343 |                38.5811 |                           44.8008 |             6.21963 |                     3 |                 88.4354 |                            91.0593 |              2.62391 |                      3 |
| molecular_weight_bin | 250-349    | 255 |                28.2353 |                           35.2941 |             7.05882 |                     3 |                 80.6536 |                            88.6275 |              7.97386 |                      3 |
| molecular_weight_bin | <150       |  31 |                17.2043 |                           25.8065 |             8.60215 |                     3 |                 51.6129 |                            63.4409 |             11.828   |                      3 |
| molecular_weight_bin | >=350      |  35 |                20.9524 |                           31.4286 |            10.4762  |                     3 |                 58.0952 |                            66.6667 |              8.57143 |                      3 |
| peak_count_bin       | 25-49      | 376 |                35.1064 |                           42.9965 |             7.89007 |                     3 |                 84.1312 |                            89.3617 |              5.2305  |                      3 |
| peak_count_bin       | 50-99      | 220 |                30.6061 |                           36.0606 |             5.45455 |                     3 |                 81.6667 |                            87.2727 |              5.60606 |                      3 |
| peak_count_bin       | >=100      |  51 |                31.3725 |                           37.9085 |             6.53595 |                     3 |                 80.3922 |                            84.9673 |              4.57516 |                      2 |
| scaffold_status      | acyclic    | 518 |                33.5907 |                           39.704  |             6.11326 |                     3 |                 84.2342 |                            88.417  |              4.18275 |                      3 |
| scaffold_status      | train_seen | 141 |                29.7872 |                           39.9527 |            10.1655  |                     3 |                 75.6501 |                            85.5792 |              9.92908 |                      3 |
| stereo_annotated     | False      | 651 |                33.3333 |                           40.3482 |             7.01485 |                     3 |                 83.7686 |                            89.2985 |              5.52995 |                      3 |

## Cross-seed consistency

|   qv_correct_seeds |   all_attention_correct_seeds |   sample_count |
|-------------------:|------------------------------:|---------------:|
|                  0 |                             0 |            188 |
|                  0 |                             1 |             84 |
|                  0 |                             2 |             27 |
|                  0 |                             3 |              8 |
|                  1 |                             0 |             36 |
|                  1 |                             1 |             52 |
|                  1 |                             2 |             46 |
|                  1 |                             3 |             20 |
|                  2 |                             0 |             14 |
|                  2 |                             1 |             20 |
|                  2 |                             2 |             43 |
|                  2 |                             3 |             35 |
|                  3 |                             0 |              5 |
|                  3 |                             1 |             15 |
|                  3 |                             2 |             17 |
|                  3 |                             3 |             54 |

- Both methods wrong in all three seeds: 188 samples.
- q/v wrong in all seeds but all-attention correct in at least two: 35 samples.
- q/v correct in at least two seeds but all-attention wrong in all seeds: 19 samples.

## Interpretation boundary

Subgroup results are descriptive and share the validation data used for method selection. They are not independent confirmation, are not multiplicity-adjusted hypothesis tests, and must not be presented as external generalization.
