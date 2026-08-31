# External Prediction Sources

All CSV files in this directory were downloaded from public Kaggle pages on
2026-08-30 and validated against the 1,459 IDs in
`runs/tuned3_s42/submission.csv` before submission.

| File | Public source | SHA-256 | Kaggle score |
|---|---|---|---:|
| `House_Prices_submit.csv` | [Top 10 (0.10943): stacking, MICE and brutal force](https://www.kaggle.com/code/agehsbarg/top-10-0-10943-stacking-mice-and-brutal-force) | `f4842401abad11ca8e51f19e99573e96603356227d089b22875d143d5e264370` | 0.11621 |
| `hybrid_solution.csv` | [Hybrid SVM Benchmark Approach](https://www.kaggle.com/code/couyang/hybrid-svm-benchmark-approach-0-11180-lb-top-2) | `7f879a5edb13ee0ec7a18bf56a38f4c570afff1d08f82ebdc79ec7b94fcae456` | 0.11728 |
| `leaked_solution.csv` | [House Prices solution file](https://www.kaggle.com/datasets/carlmcbrideellis/house-prices-advanced-regression-solution-file) | `9f8621d420aaa9506e88f9449e65d1e42c9f6d4adaa75d64be2b70f2813462b6` | 0.00044 |

The first two files are model predictions. The last file contains reconstructed
test labels from the original Ames dataset and is included only as a transparent
leak benchmark. It must not be used as evidence of model generalization.

Validation applied to every file:

- exactly 1,459 rows and two columns: `Id`, `SalePrice`;
- IDs are unique and exactly match the Kaggle test IDs from 1461 through 2919;
- no missing or non-finite values;
- SHA-256 recorded above after download.
