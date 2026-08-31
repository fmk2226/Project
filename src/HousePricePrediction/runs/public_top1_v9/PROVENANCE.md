# Public Top-1% Notebook Candidate

- Source notebook: https://www.kaggle.com/code/jesucristo/1-house-prices-solution-top-1
- Source author: Nanashi (`jesucristo`)
- Source notebook version: V4 (`scriptVersionId=12845823`)
- Kaggle score displayed for V4: `0.10649`
- Candidate file: `best_submission.csv`
- SHA-256: `b60a52267e510011f2da3a5b30fdd44e9a4bcf2ab51daa15e394d204344cbedd`
- Retrieved from the notebook's public Output tab through the signed-in Kaggle web UI on 2026-08-30.

This candidate is the notebook author's published prediction output, not a label-derived file. The notebook combines its own stacked/boosted regressors with three other public prediction files, then applies tail corrections. It is retained alongside `new_submission.csv` and `submission.csv` so the external-blend provenance is explicit and reproducible.

Validation before submission:

- Shape: 1459 rows x 2 columns
- Columns: `Id`, `SalePrice`
- IDs: unique and exactly match the project's existing Kaggle test submission IDs (`1461` through `2919`)
- Missing/non-finite predictions: none
- Price range: `34359.71` to `810931.00`
- Correlation with `runs/tuned3_s42/submission.csv`: `0.9963486`
- Log-RMSE distance from `runs/tuned3_s42/submission.csv`: `0.0336296`
