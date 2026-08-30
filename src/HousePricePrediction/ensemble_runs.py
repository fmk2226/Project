"""Blend complete CV runs made with different fold seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from Trainer import regression_metrics


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parents[1] / "data" / "house_price"


def optimize_weights(y_log: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    initial = np.full(predictions.shape[1], 1.0 / predictions.shape[1])
    result = minimize(
        lambda w: np.sqrt(np.mean(np.square(y_log - predictions @ w))),
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * predictions.shape[1],
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run directory names under runs/")
    parser.add_argument("--output", default="seed_ensemble")
    parser.add_argument(
        "--mode",
        choices=["run_blends", "all_models"],
        default="run_blends",
        help="Blend each run's optimized prediction or all base-model predictions.",
    )
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["Id", "SalePrice"])
    test_ids = pd.read_csv(DATA_DIR / "test.csv", usecols=["Id"])["Id"].to_numpy()
    y_by_id = train.set_index("Id")["SalePrice"]
    oof_parts = []
    test_parts = []

    for run_name in args.runs:
        run_dir = PROJECT_DIR / "runs" / run_name
        oof = pd.read_csv(run_dir / "oof_predictions.csv").set_index("Id")
        test = pd.read_csv(run_dir / "test_predictions_log.csv")
        if "Id" in test:
            test = test.set_index("Id").loc[test_ids]
        else:
            test.index = test_ids

        columns = ["blend_optimized"] if args.mode == "run_blends" else [
            column for column in oof.columns if not column.startswith("blend_")
        ]
        oof_parts.append(oof[columns].rename(columns=lambda c: f"{run_name}::{c}"))
        test_parts.append(test[columns].rename(columns=lambda c: f"{run_name}::{c}"))

    oof_all = pd.concat(oof_parts, axis=1, join="inner").sort_index()
    test_all = pd.concat(test_parts, axis=1, join="inner").loc[test_ids]
    y_log = np.log1p(y_by_id.loc[oof_all.index].to_numpy(dtype=np.float64))
    weights = optimize_weights(y_log, oof_all.to_numpy())
    oof_prediction = oof_all.to_numpy() @ weights
    test_prediction = test_all.to_numpy() @ weights
    metrics = regression_metrics(y_log, oof_prediction)

    output_dir = PROJECT_DIR / "runs" / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source": oof_all.columns, "weight": weights}).to_csv(
        output_dir / "blend_weights.csv", index=False
    )
    pd.DataFrame(
        {"Id": oof_all.index, "prediction_log": oof_prediction}
    ).to_csv(output_dir / "oof_predictions.csv", index=False)
    submission = pd.DataFrame(
        {"Id": test_ids, "SalePrice": np.maximum(0.0, np.expm1(test_prediction))}
    )
    submission.to_csv(output_dir / "submission.csv", index=False)
    summary = pd.DataFrame(
        [{
            "run_id": args.output,
            "experiment": f"seed_ensemble_{args.mode}",
            "model": "blend_optimized",
            "oof_loss": metrics["loss"],
            "oof_acc": metrics["acc"],
            "oof_rmse_dollars": metrics["rmse"],
            "oof_mae_dollars": metrics["mae"],
            "oof_r2": metrics["r2"],
        }]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Submission: {output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
