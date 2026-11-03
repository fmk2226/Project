# House Prices - Advanced Regression Techniques

This project follows the existing `F1` layout (`main.py`, `Trainer.py`,
`model.py`) while adapting it to regression and Kaggle's RMSLE metric.

## What it does

- removes the two well-known `GrLivArea > 4000` training outliers;
- applies semantic missing-value handling and feature engineering;
- learns `log1p(SalePrice)` and log-transforms strongly skewed numeric inputs;
- runs shuffled K-fold out-of-fold validation;
- logs train/validation RMSLE, dollar RMSE/MAE, R2, and ±10% accuracy;
- learns non-negative blend weights from OOF predictions;
- writes a validated two-column Kaggle `submission.csv`.

For this regression task, `train_acc` and `val_acc` mean the fraction of homes
whose predicted price is within ±10% of the true price.  Kaggle ranking is
determined by `val_loss`/`oof_loss` (RMSLE), not by this auxiliary accuracy.

## Run

```powershell
H:\deeplearning\.venv\Scripts\python.exe main.py --experiment baseline --run-name baseline_s42
H:\deeplearning\.venv\Scripts\python.exe main.py --experiment tuned1 --run-name tuned1_s42
H:\deeplearning\.venv\Scripts\python.exe main.py --experiment tuned2 --run-name tuned2_s42
H:\deeplearning\.venv\Scripts\python.exe main.py --experiment tuned3 --folds 10 --run-name tuned3_s42
```

Each run is stored under `runs/<run-name>/`.  The project-level
`TRAINING_LOG.csv` accumulates fold metrics across runs.

Optional boosting wheels can be installed into `.deps`. CatBoost has no Python 3.15 wheel, so `model.py` skips it in this environment:

```powershell
uv pip install --python H:\deeplearning\.venv\Scripts\python.exe --target .deps --no-deps lightgbm==4.6.0 xgboost==2.1.4
```
