"""Train, validate, blend and export a Kaggle House Prices submission."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax, skew

import model
from Trainer import Trainer, save_result


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parents[1]
DEFAULT_DATA_DIR = REPO_DIR / "data" / "house_price"


def random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _safe_sum(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    return frame[available].fillna(0).sum(axis=1)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["TotalSF"] = _safe_sum(frame, ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"])
    frame["TotalBathrooms"] = (
        _safe_sum(frame, ["FullBath", "BsmtFullBath"])
        + 0.5 * _safe_sum(frame, ["HalfBath", "BsmtHalfBath"])
    )
    frame["TotalPorchSF"] = _safe_sum(
        frame, ["OpenPorchSF", "3SsnPorch", "EnclosedPorch", "ScreenPorch", "WoodDeckSF"]
    )
    frame["TotalSqrFootage"] = _safe_sum(
        frame, ["BsmtFinSF1", "BsmtFinSF2", "1stFlrSF", "2ndFlrSF"]
    )
    frame["AgeAtSale"] = (frame["YrSold"] - frame["YearBuilt"]).clip(lower=0)
    frame["AgeRemodAtSale"] = (frame["YrSold"] - frame["YearRemodAdd"]).clip(lower=0)
    frame["GarageAge"] = (frame["YrSold"] - frame["GarageYrBlt"]).clip(lower=0)
    frame["YrBltAndRemod"] = frame["YearBuilt"] + frame["YearRemodAdd"]
    for source, target in {
        "PoolArea": "HasPool",
        "2ndFlrSF": "Has2ndFloor",
        "GarageArea": "HasGarage",
        "TotalBsmtSF": "HasBsmt",
        "Fireplaces": "HasFireplace",
        "MasVnrArea": "HasMasVnr",
    }.items():
        frame[target] = (frame[source].fillna(0) > 0).astype(int)
    frame["OverallQual_GrLivArea"] = frame["OverallQual"] * frame["GrLivArea"]
    frame["GarageScore"] = frame["GarageCars"].fillna(0) * frame["GarageArea"].fillna(0)
    return frame


def prepare_data(train: pd.DataFrame, test: pd.DataFrame):
    """Competition-aware cleaning followed by deterministic joint encoding."""
    train = train.copy()
    test = test.copy()
    # The public high-vote solution removes only the two large-area/low-price
    # anomalies, not every legitimate luxury home above 4,000 square feet.
    known_residual_outliers = {31, 89, 463, 633, 1325}
    outlier_mask = (train["GrLivArea"] > 4500) | train["Id"].isin(known_residual_outliers)
    removed_outliers = train.loc[outlier_mask, "Id"].astype(int).tolist()
    train = train.loc[~outlier_mask].reset_index(drop=True)

    train_ids = train.pop("Id").to_numpy()
    test_ids = test.pop("Id").to_numpy()
    y_price = train.pop("SalePrice").to_numpy(dtype=np.float64)
    y_log = np.log1p(y_price)
    n_train = len(train)
    all_features = pd.concat([train, test], axis=0, ignore_index=True)

    none_columns = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType",
        "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond",
        "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType",
    ]
    zero_columns = [
        "GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2",
        "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
    ]
    for column in none_columns:
        if column in all_features:
            all_features[column] = all_features[column].fillna("None")
    for column in zero_columns:
        if column in all_features:
            all_features[column] = all_features[column].fillna(0)

    if "LotFrontage" in all_features:
        neighborhood_median = all_features.groupby("Neighborhood")["LotFrontage"].transform("median")
        all_features["LotFrontage"] = all_features["LotFrontage"].fillna(neighborhood_median)

    if "MSZoning" in all_features:
        grouped_mode = all_features.groupby("MSSubClass")["MSZoning"].transform(
            lambda values: values.fillna(values.mode().iloc[0] if not values.mode().empty else "RL")
        )
        all_features["MSZoning"] = all_features["MSZoning"].fillna(grouped_mode)

    all_features = engineer_features(all_features)
    all_features = all_features.drop(columns=["Utilities", "Street", "PoolQC"], errors="ignore")

    for column in ["MSSubClass", "YrSold", "MoSold"]:
        all_features[column] = all_features[column].fillna(0).astype(int).astype(str)

    categorical = all_features.select_dtypes(include=["object", "category"]).columns
    numeric = all_features.select_dtypes(exclude=["object", "category"]).columns
    for column in categorical:
        mode = all_features[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Missing"
        all_features[column] = all_features[column].fillna(fill_value).astype(str)
    for column in numeric:
        all_features[column] = all_features[column].replace([np.inf, -np.inf], np.nan)
        all_features[column] = all_features[column].fillna(0)

    skewed = []
    for column in numeric:
        values = all_features[column]
        if values.min() >= 0 and values.nunique() > 2 and skew(values) > 0.50:
            lam = boxcox_normmax(values.to_numpy(dtype=np.float64) + 1.0)
            all_features[column] = boxcox1p(values, lam)
            skewed.append(column)

    all_features = pd.get_dummies(all_features, dummy_na=False, dtype=np.float32)
    frequency = all_features.apply(
        lambda column: column.value_counts(dropna=False).iloc[0] / len(column)
    )
    all_features = all_features.loc[:, frequency <= 0.9994]
    matrix = all_features.to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite values remain after preprocessing")

    metadata = {
        "removed_outlier_ids": removed_outliers,
        "n_train": n_train,
        "n_test": len(test),
        "n_features": matrix.shape[1],
        "log_transformed_numeric_features": skewed,
    }
    return matrix[:n_train], y_log, matrix[n_train:], train_ids, test_ids, metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--experiment",
        choices=["baseline", "tuned1", "tuned2", "tuned3", "full"],
        default="tuned1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random_seed(args.seed)
    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X, y_log, X_test, train_ids, test_ids, metadata = prepare_data(train, test)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_name or f"{args.experiment}_s{args.seed}_{timestamp}"
    output_dir = PROJECT_DIR / "runs" / run_id
    models = model.build_models(args.experiment, args.seed)
    print(
        f"run={run_id} train={X.shape[0]} test={X_test.shape[0]} "
        f"features={X.shape[1]} models={list(models)}"
    )
    trainer = Trainer(models, n_splits=args.folds, random_state=args.seed)
    result = trainer.run(X, y_log, X_test, test_ids, run_id, args.experiment)
    config = {
        "run_id": run_id,
        "experiment": args.experiment,
        "folds": args.folds,
        "seed": args.seed,
        "data_dir": str(args.data_dir.resolve()),
        "models": list(models),
        "metadata": metadata,
    }
    save_result(result, output_dir, train_ids, config, PROJECT_DIR / "TRAINING_LOG.csv")
    print("\nOOF summary")
    print(result.summary.to_string(index=False))
    print(f"\nSubmission: {output_dir / 'submission.csv'}")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
