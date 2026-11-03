"""Candidate reference implementation; not yet verified as identical to the implementation used for the reported results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .evidence_rates import as_month


def _fallback(
    scoring_rows: pd.DataFrame,
    config: dict[str, Any],
    mode: str,
    observed_rows: int,
) -> pd.DataFrame:
    """Return the documented adaptive-hierarchy fallback scores."""

    score_column = str(config["fallback_score_column"])
    if score_column not in scoring_rows.columns:
        raise ValueError(f"Fallback score column is missing: {score_column}")
    result = scoring_rows.copy()
    result["raw_score"] = pd.to_numeric(result[score_column], errors="raise").astype(float)
    result["model_mode"] = mode
    result["observed_training_rows"] = int(observed_rows)
    return result


def fit_or_fallback(
    training_rows: pd.DataFrame,
    scoring_rows: pd.DataFrame,
    prediction_month: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Fit the month-specific lean GBM or apply the deterministic V4.3 fallback."""

    prediction = as_month(prediction_month)
    required_training = {"decision_month", "target_month", "target_observed", "target"}
    missing_training = sorted(required_training.difference(training_rows.columns))
    if missing_training:
        raise ValueError(f"Training rows are missing columns: {missing_training}")
    train = training_rows.copy()
    decision_periods = train["decision_month"].map(as_month)
    target_periods = train["target_month"].map(as_month)
    if (decision_periods >= prediction).any() or (target_periods > prediction).any():
        raise ValueError("Training rows cross the forward-safe prediction boundary.")
    train = train.loc[train["target_observed"].astype(bool) & train["target"].notna()].copy()
    observed_rows = len(train)
    minimum = int(config["min_training_rows"])
    if observed_rows < minimum:
        return _fallback(scoring_rows, config, "fallback_insufficient_rows", observed_rows)
    labels = train["target"].astype(int)
    if labels.nunique(dropna=True) < 2:
        return _fallback(scoring_rows, config, "fallback_single_class", observed_rows)

    numeric = list(config["numeric_features"])
    categorical = list(config["categorical_features"])
    feature_columns = numeric + categorical
    missing_features = sorted(set(feature_columns).difference(train.columns))
    missing_scoring = sorted(set(feature_columns).difference(scoring_rows.columns))
    if missing_features or missing_scoring:
        raise ValueError(
            f"Configured features missing; training={missing_features}, scoring={missing_scoring}"
        )
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0))])
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=str(config["missing_category_token"])),
            ),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)],
        remainder="drop",
    )
    classifier = GradientBoostingClassifier(
        n_estimators=int(config["n_estimators"]),
        learning_rate=float(config["learning_rate"]),
        max_depth=int(config["max_depth"]),
        random_state=int(config["random_state"]),
    )
    pipeline = Pipeline([("preprocess", preprocess), ("classifier", classifier)])
    pipeline.fit(train[feature_columns], labels)
    probabilities = pipeline.predict_proba(scoring_rows[feature_columns])
    positive_index = int(np.flatnonzero(pipeline.named_steps["classifier"].classes_ == 1)[0])
    result = scoring_rows.copy()
    result["raw_score"] = probabilities[:, positive_index].astype(float)
    result["model_mode"] = "gradient_boosting"
    result["observed_training_rows"] = observed_rows
    return result

