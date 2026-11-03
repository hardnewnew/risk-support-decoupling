"""Candidate reference implementation; not yet verified as identical to the implementation used for the reported results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = {
    "m0",
    "kappa",
    "tau",
    "alpha",
    "min_training_rows",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "random_state",
    "hierarchy_levels",
    "numeric_features",
    "categorical_features",
    "tie_breaking",
}


def load_reference_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the centralized candidate YAML configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    missing = sorted(REQUIRED_KEYS.difference(config))
    if missing:
        raise ValueError(f"Configuration is missing required keys: {missing}")
    if config.get("method_status") != "candidate_reference_implementation_pending_author_confirmation":
        raise ValueError("Unexpected method status in candidate configuration.")
    return config

