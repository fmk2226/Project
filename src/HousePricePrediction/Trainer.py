"""Cross-validation trainer and out-of-fold blender."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def regression_metrics(y_log: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    """Metrics in Kaggle log space and in original dollar space."""
    pred_log = np.asarray(pred_log, dtype=np.float64)
    y_log = np.asarray(y_log, dtype=np.float64)
    y_price = np.expm1(y_log)
    pred_price = np.maximum(0.0, np.expm1(pred_log))
    relative_error = np.abs(pred_price - y_price) / np.maximum(y_price, 1.0)
    return {
        "loss": float(np.sqrt(mean_squared_error(y_log, pred_log))),
        "rmse": float(np.sqrt(mean_squared_error(y_price, pred_price))),
        "mae": float(mean_absolute_error(y_price, pred_price)),
        "r2": float(r2_score(y_price, pred_price)),
        "acc": float(np.mean(relative_error <= 0.10)),
    }


@dataclass
class TrainingResult:
    oof_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    log: pd.DataFrame
    summary: pd.DataFrame
    weights: pd.DataFrame
    submission: pd.DataFrame


class Trainer:
    """Train each regressor on K folds and blend its OOF predictions."""

    def __init__(self, models, n_splits: int = 5, random_state: int = 42):
        self.models = models
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _epoch_hint(estimator) -> int:
        params = estimator.get_params(deep=False)
        return int(params.get("n_estimators", params.get("iterations", 1)))

    @staticmethod
    def _optimized_weights(y_log: np.ndarray, prediction_matrix: np.ndarray) -> np.ndarray:
        n_models = prediction_matrix.shape[1]
        initial = np.full(n_models, 1.0 / n_models)

        def objective(weights):
            prediction = prediction_matrix @ weights
            return np.sqrt(np.mean(np.square(y_log - prediction)))

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        return result.x if result.success else initial

    def run(
        self,
        X: np.ndarray,
        y_log: np.ndarray,
        X_test: np.ndarray,
        test_ids: np.ndarray,
        run_id: str,
        experiment: str,
    ) -> TrainingResult:
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        log_rows: list[dict] = []
        oof = {}
        test = {}

        for model_index, (model_name, template) in enumerate(self.models.items(), start=1):
            print(f"[{model_index}/{len(self.models)}] {model_name}")
            model_oof = np.zeros(len(X), dtype=np.float64)
            fold_test = np.zeros((len(X_test), self.n_splits), dtype=np.float64)

            for fold, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
                estimator = clone(template)
                started = time.perf_counter()
                estimator.fit(X[train_idx], y_log[train_idx])
                elapsed = time.perf_counter() - started

                train_pred = estimator.predict(X[train_idx])
                val_pred = estimator.predict(X[val_idx])
                model_oof[val_idx] = val_pred
                fold_test[:, fold - 1] = estimator.predict(X_test)
                train_metrics = regression_metrics(y_log[train_idx], train_pred)
                val_metrics = regression_metrics(y_log[val_idx], val_pred)
                row = {
                    "run_id": run_id,
                    "experiment": experiment,
                    "model": model_name,
                    "fold": fold,
                    "epoch": self._epoch_hint(estimator),
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                    "val_rmse_dollars": val_metrics["rmse"],
                    "val_mae_dollars": val_metrics["mae"],
                    "val_r2": val_metrics["r2"],
                    "seconds": elapsed,
                }
                log_rows.append(row)
                print(
                    f"  fold {fold}: train={train_metrics['loss']:.5f} "
                    f"val={val_metrics['loss']:.5f} val_acc={val_metrics['acc']:.3f} "
                    f"time={elapsed:.1f}s"
                )

            oof[model_name] = model_oof
            test[f"{model_name}_cv"] = fold_test.mean(axis=1)
            full_estimator = clone(template)
            full_estimator.fit(X, y_log)
            test[model_name] = full_estimator.predict(X_test)

        oof_df = pd.DataFrame(oof)
        test_df = pd.DataFrame(test)
        model_names = list(oof_df.columns)
        prediction_matrix = oof_df.to_numpy()
        weights = self._optimized_weights(y_log, prediction_matrix)
        weight_df = pd.DataFrame({"model": model_names, "weight": weights})

        oof_df["blend_equal"] = prediction_matrix.mean(axis=1)
        test_df["blend_equal"] = test_df[model_names].to_numpy().mean(axis=1)
        oof_df["blend_optimized"] = prediction_matrix @ weights
        test_df["blend_optimized"] = test_df[model_names].to_numpy() @ weights

        summary_rows = []
        for name in oof_df.columns:
            metrics = regression_metrics(y_log, oof_df[name].to_numpy())
            summary_rows.append(
                {
                    "run_id": run_id,
                    "experiment": experiment,
                    "model": name,
                    "oof_loss": metrics["loss"],
                    "oof_acc": metrics["acc"],
                    "oof_rmse_dollars": metrics["rmse"],
                    "oof_mae_dollars": metrics["mae"],
                    "oof_r2": metrics["r2"],
                }
            )
        summary_df = pd.DataFrame(summary_rows).sort_values("oof_loss")
        best_prediction = summary_df.iloc[0]["model"]
        submission = pd.DataFrame(
            {
                "Id": test_ids.astype(int),
                "SalePrice": np.maximum(0.0, np.expm1(test_df[best_prediction].to_numpy())),
            }
        )
        return TrainingResult(oof_df, test_df, pd.DataFrame(log_rows), summary_df, weight_df, submission)


def save_result(
    result: TrainingResult,
    output_dir: Path,
    train_ids: np.ndarray,
    config: dict,
    project_log_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.log.to_csv(output_dir / "training_log.csv", index=False)
    result.summary.to_csv(output_dir / "summary.csv", index=False)
    result.weights.to_csv(output_dir / "blend_weights.csv", index=False)
    result.submission.to_csv(output_dir / "submission.csv", index=False)
    result.oof_predictions.assign(Id=train_ids).to_csv(
        output_dir / "oof_predictions.csv", index=False
    )
    result.test_predictions.assign(Id=result.submission["Id"].to_numpy()).to_csv(
        output_dir / "test_predictions_log.csv", index=False
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if project_log_path.exists():
        old = pd.read_csv(project_log_path)
        combined = pd.concat([old, result.log], ignore_index=True)
    else:
        combined = result.log
    combined.to_csv(project_log_path, index=False)
