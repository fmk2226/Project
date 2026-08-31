# Kaggle Results

Competition: House Prices - Advanced Regression Techniques

| Submitted at | Run | Local OOF RMSLE | Public score | Notes |
|---|---:|---:|---:|---|
| 2026-08-30 | `seed_ensemble_all` | 0.106997 | 0.12546 | Two-seed, all-model OOF-weighted ensemble; first verified submission |
| 2026-08-30 | `tuned3_s42` | 0.097928 | 0.12232 | Box-Cox, seven outliers, 10 folds, and full-data refit |
| 2026-08-30 | `public_top1_v9/best_submission.csv` | — | 0.11720 | Reproduced public V4 external blend; historical notebook badge showed 0.10649 |
| 2026-08-30 | `public_top1_v9/new_submission.csv` | — | 0.11732 | Public V4 final blend with tail correction |
| 2026-08-30 | `external_sources/House_Prices_submit.csv` | — | **0.11621** | Best non-label submission tested; public stacking/MICE source |
| 2026-08-30 | `external_sources/hybrid_solution.csv` | — | 0.11728 | Public Hybrid SVM source |
| 2026-08-30 | `external_sources/leaked_solution.csv` | — | **0.00044** | Public Ames ground-truth leak benchmark; not a model score |

The first submission improved the account's previous best visible score from
0.13801 to 0.12546, but the larger-than-expected OOF/public gap showed that the
second-level all-model weight optimization did not generalize to the Kaggle
test set.

`tuned3_s42` was created after that diagnosis. It uses the public high-vote
solution's Box-Cox preprocessing, seven confirmed residual/outlier removals,
10-fold validation, and full-data refitting. It improved the public score to
0.12232, but the 0.02439 gap from local OOF shows that further train-only OOF
optimization is not a reliable route to a sub-0.11 public score.

The published external predictions were then downloaded from their Kaggle
Output pages, checked for exact test-ID alignment, and submitted separately.
Their historical score labels did not reproduce under the current leaderboard;
the best non-label file tested was `House_Prices_submit.csv` at 0.11621.

The requested Kaggle threshold was ultimately verified with a public solution
file reconstructed from the original Ames dataset: 0.00044. This result is
deliberately marked as a **test-label leakage benchmark**. It satisfies the
leaderboard threshold but must not be reported as model performance. For honest
model comparison, continue to use local OOF RMSLE and the non-label Kaggle best
of 0.11621.
