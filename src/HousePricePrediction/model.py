"""Model definitions for the House Prices regression project.

Every model predicts ``log1p(SalePrice)``.  The optional boosting libraries are
loaded from ``.deps`` when they are not installed in the active interpreter.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR


LOCAL_DEPS = Path(__file__).resolve().parent / ".deps"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))


def _optional_boosters(random_state: int, profile: str):
    """Return installed third-party boosters without making them mandatory."""
    models = OrderedDict()

    try:
        from lightgbm import LGBMRegressor

        if profile == "tuned2":
            params = dict(
                n_estimators=4200,
                learning_rate=0.008,
                num_leaves=16,
                max_depth=-1,
                min_child_samples=18,
                max_bin=63,
                subsample=0.80,
                colsample_bytree=0.78,
                reg_alpha=0.0008,
                reg_lambda=0.08,
            )
        elif profile == "tuned3":
            params = dict(
                n_estimators=5000,
                learning_rate=0.010,
                num_leaves=4,
                max_depth=-1,
                min_child_samples=20,
                max_bin=200,
                subsample=0.75,
                subsample_freq=5,
                colsample_bytree=0.20,
                reg_alpha=0.0,
                reg_lambda=0.0,
            )
        else:
            params = dict(
                n_estimators=3000,
                learning_rate=0.010,
                num_leaves=20,
                max_depth=-1,
                min_child_samples=20,
                max_bin=63,
                subsample=0.82,
                colsample_bytree=0.80,
                reg_alpha=0.0005,
                reg_lambda=0.05,
            )
        models["lightgbm"] = LGBMRegressor(
            objective="regression",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBRegressor

        if profile == "tuned2":
            params = dict(
                n_estimators=4200,
                learning_rate=0.010,
                max_depth=2,
                min_child_weight=1,
                subsample=0.82,
                colsample_bytree=0.82,
                reg_alpha=0.002,
                reg_lambda=1.15,
            )
        elif profile == "tuned3":
            params = dict(
                n_estimators=3460,
                learning_rate=0.010,
                max_depth=3,
                min_child_weight=0,
                gamma=0.0,
                subsample=0.70,
                colsample_bytree=0.70,
                reg_alpha=0.00006,
                reg_lambda=1.0,
            )
        else:
            params = dict(
                n_estimators=3200,
                learning_rate=0.012,
                max_depth=3,
                min_child_weight=1,
                subsample=0.82,
                colsample_bytree=0.80,
                reg_alpha=0.001,
                reg_lambda=1.0,
            )
        models["xgboost"] = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor

        if profile == "tuned2":
            params = dict(iterations=3200, learning_rate=0.020, depth=5, l2_leaf_reg=5.0)
        else:
            params = dict(iterations=2400, learning_rate=0.025, depth=4, l2_leaf_reg=4.0)
        models["catboost"] = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            **params,
        )
    except ImportError:
        pass

    return models


def build_models(profile: str = "tuned1", random_state: int = 42):
    """Build a deliberately diverse model collection for OOF blending."""
    if profile not in {"baseline", "tuned1", "tuned2", "tuned3", "full"}:
        raise ValueError(f"Unknown profile: {profile}")

    models = OrderedDict(
        ridge=make_pipeline(RobustScaler(), Ridge(alpha=12.0)),
        elasticnet=make_pipeline(
            RobustScaler(),
            ElasticNet(
                alpha=0.00045,
                l1_ratio=0.90,
                max_iter=30000,
                random_state=random_state,
            ),
        ),
    )

    if profile in {"tuned2", "tuned3"}:
        # These three smooth models are intentionally different from boosted
        # trees and are strong members of the classic high-vote Kaggle stack.
        models["lasso"] = make_pipeline(
            RobustScaler(),
            Lasso(alpha=0.00050, max_iter=30000, random_state=random_state),
        )
        models["kernel_ridge"] = make_pipeline(
            RobustScaler(),
            KernelRidge(alpha=0.60, kernel="polynomial", degree=2, coef0=2.5),
        )
        models["svr"] = make_pipeline(
            RobustScaler(),
            SVR(C=20.0, epsilon=0.008, gamma=0.0003),
        )

    if profile == "baseline":
        models["gradient_boosting"] = GradientBoostingRegressor(
            n_estimators=1200,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=12,
            min_samples_split=10,
            loss="huber",
            random_state=random_state,
        )
        models["random_forest"] = RandomForestRegressor(
            n_estimators=700,
            max_features=0.75,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
        return models

    if profile == "tuned2":
        gbr_params = dict(
            n_estimators=3000,
            learning_rate=0.022,
            max_depth=4,
            min_samples_leaf=15,
            min_samples_split=12,
            max_features="sqrt",
        )
    elif profile == "tuned3":
        gbr_params = dict(
            n_estimators=3000,
            learning_rate=0.050,
            max_depth=4,
            min_samples_leaf=15,
            min_samples_split=10,
            max_features="sqrt",
        )
    else:
        gbr_params = dict(
            n_estimators=2200,
            learning_rate=0.025,
            max_depth=3,
            min_samples_leaf=10,
            min_samples_split=10,
            max_features=None,
        )

    models["gradient_boosting"] = GradientBoostingRegressor(
        loss="huber", random_state=random_state, **gbr_params
    )
    booster_profile = profile if profile in {"tuned2", "tuned3"} else "tuned1"
    models.update(_optional_boosters(random_state, booster_profile))
    return models
