"""Reference implementation following the method specification of the accompanying paper; see README for scope."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import yaml

NUMERIC_FEATURES = [
    "historical_smoothed_rate",
    "rolling_two_month_smoothed_rate",
    "rolling_quarter_smoothed_rate",
    "log_saturated_support",
    "cumulative_sample_count",
    "rolling_quarter_sample_count",
    "month_of_year",
]
CATEGORICAL_FEATURES = ["broad_category", "sampling_stage"]
EXTRA_FEATURES = ["level_0_rate", "adaptive_hierarchy_score"]
UNIT_COLUMNS = ["broad_category", "category", "sampling_stage", "tested_item"]
HIERARCHY_LEVELS = [
    ["broad_category", "sampling_stage", "tested_item"],
    ["broad_category", "category", "sampling_stage"],
    ["broad_category", "sampling_stage"],
    ["broad_category"],
]
REQUIRED_KEYS = {
    "prior_strength_a", "support_saturation_n_max", "prior_start_month",
    "local_history_start_month", "training_start_month", "latency_months",
    "default_prior", "learner", "random_state", "min_training_rows",
    "fallback_score_column", "numeric_features", "categorical_features",
    "score_smoothing", "tie_breaking", "missing_category_token",
    "hierarchy_levels", "rolling_two_months", "rolling_quarter_months",
}


def validate_feature_lists(config: dict[str, Any]) -> None:
    """Allow only Table 2 model inputs and the two specified ablation fields."""

    numeric = list(config["numeric_features"])
    categorical = list(config["categorical_features"])
    invalid = set(numeric).difference(NUMERIC_FEATURES + EXTRA_FEATURES)
    invalid.update(set(categorical).difference(CATEGORICAL_FEATURES))
    if invalid:
        raise ValueError(f"Fields are not permitted as model inputs: {sorted(invalid)}")
    if len(numeric + categorical) != len(set(numeric + categorical)):
        raise ValueError("Model feature lists must not contain duplicates.")


def latency_months(config: dict[str, Any]) -> int:
    """Return the publication-latency setting shared by all decision cutoffs."""

    value = config.get("latency_months", 0)
    if isinstance(value, bool) or value not in (0, 1):
        raise ValueError("latency_months must be 0 or 1.")
    return int(value)


def load_reference_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the centralized method configuration."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    missing = sorted(REQUIRED_KEYS.difference(config))
    if missing:
        raise ValueError(f"Configuration is missing required keys: {missing}")
    if config.get("method_status") != "reference_implementation_following_paper_specification":
        raise ValueError("Unexpected method status in reference configuration.")
    latency_months(config)
    for key in ("prior_strength_a", "support_saturation_n_max"):
        value = float(config[key])
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite.")
    if config["learner"] not in {"system_lightgbm", "gbm9_sklearn"}:
        raise ValueError("Unknown learner.")
    validate_feature_lists(config)
    return config

