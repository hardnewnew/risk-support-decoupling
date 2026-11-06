"""Reference implementation following the method specification of the accompanying paper; see README for scope."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config_loader import latency_months, validate_feature_lists
from .evidence_rates import as_month


def _fallback(
    scoring_rows: pd.DataFrame,
    config: dict[str, Any],
    mode: str,
    observed_rows: int,
) -> pd.DataFrame:
    """Return scores from the configured cold-start field."""

    column = str(config.get("fallback_score_column", "historical_smoothed_rate"))
    if column not in scoring_rows.columns:
        raise ValueError(f"Fallback score column is missing: {column}")
    result = scoring_rows.copy()
    result["raw_score"] = pd.to_numeric(result[column], errors="raise").astype(float)
    result["model_mode"] = mode
    result["observed_training_rows"] = int(observed_rows)
    return result


def fit_or_fallback(
    training_rows: pd.DataFrame,
    scoring_rows: pd.DataFrame,
    prediction_month: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Fit the configured monthly learner or use the cold-start score."""

    validate_feature_lists(config)
    learner = config.get("learner", "system_lightgbm")
    if learner not in {"system_lightgbm", "gbm9_sklearn"}:
        raise ValueError("Unknown learner.")
    prediction = as_month(prediction_month)
    cutoff = prediction - latency_months(config)
    required = {"decision_month", "target_month", "target_observed", "target"}
    train = training_rows.copy()
    if train.empty:
        return _fallback(scoring_rows, config, "fallback_insufficient_rows", 0)
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"Training rows are missing columns: {missing}")
    decisions = train["decision_month"].map(as_month)
    targets = train["target_month"].map(as_month)
    if ((decisions + 1) > cutoff).any() or (targets > cutoff).any():
        raise ValueError("Training rows cross the forward-safe prediction boundary.")
    if (targets != decisions + 1).any():
        raise ValueError("Training labels must come from the next month.")
    eligible = decisions >= as_month(config.get("training_start_month", "2024-07"))
    observed = train["target_observed"].eq(True) & train["target"].isin([0, 1])
    train = train.loc[eligible & observed].copy()
    count = len(train)
    minimum = int(config.get("min_training_rows", 1))
    if count == 0 or count < minimum:
        return _fallback(scoring_rows, config, "fallback_insufficient_rows", count)
    labels = train["target"].astype(int)
    if labels.nunique() < 2:
        return _fallback(scoring_rows, config, "fallback_single_class", count)
    numeric = list(config["numeric_features"])
    categorical = list(config["categorical_features"])
    features = numeric + categorical
    missing_train = sorted(set(features).difference(train.columns))
    missing_score = sorted(set(features).difference(scoring_rows.columns))
    if missing_train or missing_score:
        raise ValueError(f"Configured features missing; training={missing_train}, scoring={missing_score}")
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=str(config["missing_category_token"]))),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop")
    if learner == "system_lightgbm":
        from lightgbm import LGBMClassifier

        params = dict(
            objective="binary", n_estimators=200, num_leaves=7, max_depth=3,
            learning_rate=0.03, min_child_samples=40, reg_lambda=2.0, n_jobs=4,
            deterministic=True, force_row_wise=True, random_state=int(config["random_state"]),
            verbose=-1,
        )
        params.update(config.get("learner_params", {}))
        classifier = LGBMClassifier(**params)
    else:
        params = dict(
            n_estimators=80, max_depth=2, learning_rate=0.05,
            random_state=int(config["random_state"]),
        )
        params.update(config.get("learner_params", {}))
        classifier = GradientBoostingClassifier(**params)
    pipeline = Pipeline([("preprocess", preprocess), ("classifier", classifier)])
    pipeline.fit(train[features], labels)
    probabilities = pipeline.predict_proba(scoring_rows[features])
    positive = int(np.flatnonzero(pipeline.named_steps["classifier"].classes_ == 1)[0])
    result = scoring_rows.copy()
    result["raw_score"] = probabilities[:, positive].astype(float)
    result["model_mode"] = "gradient_boosting"
    result["observed_training_rows"] = count
    return result

