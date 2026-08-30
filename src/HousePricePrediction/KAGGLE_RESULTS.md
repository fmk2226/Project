# Kaggle Results

Competition: House Prices - Advanced Regression Techniques

| Submitted at | Run | Local OOF RMSLE | Public score | Notes |
|---|---:|---:|---:|---|
| 2026-08-30 | `seed_ensemble_all` | 0.106997 | 0.12546 | Two-seed, all-model OOF-weighted ensemble; first verified submission |
| 2026-08-30 | `tuned3_s42` | 0.097928 | 0.12232 | Box-Cox, seven outliers, 10 folds, and full-data refit; current best |

The first submission improved the account's previous best visible score from
0.13801 to 0.12546, but the larger-than-expected OOF/public gap showed that the
second-level all-model weight optimization did not generalize to the Kaggle
test set.

`tuned3_s42` was created after that diagnosis. It uses the public high-vote
solution's Box-Cox preprocessing, seven confirmed residual/outlier removals,
10-fold validation, and full-data refitting. It improved the public score to
0.12232, but the 0.02439 gap from local OOF shows that further train-only OOF
optimization is not a reliable route to a sub-0.1 public score.
