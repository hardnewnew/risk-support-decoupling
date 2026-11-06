"""Reference implementation following the method specification of the accompanying paper; see README for scope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config_loader import HIERARCHY_LEVELS, UNIT_COLUMNS
from .evidence_rates import aggregate_counts, as_month, beta_rate, empirical_prior, restrict_to_cutoff


@dataclass(frozen=True)
class HierarchyScore:
    """Selected source and its count, window, and prior provenance."""

    score: float
    selected_level: str
    selected_level_index: int | None
    selected_window_months: int | None
    support_count: float
    failure_count: float
    prior: float


def _matching_history(
    history: pd.DataFrame,
    candidate: dict[str, Any],
    columns: list[str],
    missing_token: str,
) -> pd.DataFrame:
    selected = history
    for column in columns:
        if column not in selected.columns:
            raise ValueError(f"Missing hierarchy key: {column}")
        value = candidate.get(column, missing_token)
        value = missing_token if pd.isna(value) else str(value)
        values = selected[column].where(selected[column].notna(), missing_token).astype(str)
        selected = selected.loc[values == value]
    return selected


def adaptive_hierarchy_score(
    data: pd.DataFrame,
    candidate: dict[str, Any],
    cutoff_month: Any,
    config: dict[str, Any],
) -> HierarchyScore:
    """Select the first eligible source in the Appendix E.1 backoff order.

    cutoff_month is the already latency-adjusted history cutoff.
    """

    cutoff = as_month(cutoff_month)
    history = restrict_to_cutoff(data, cutoff)
    prior = empirical_prior(
        history, cutoff, float(config["default_prior"]), config.get("prior_start_month", "2024-01")
    )
    start = as_month(config.get("local_history_start_month", "2024-07"))
    local = history.loc[history["month"] >= start]
    levels = [list(level) for level in config.get("hierarchy_levels", HIERARCHY_LEVELS)]
    if levels != HIERARCHY_LEVELS:
        raise ValueError("hierarchy_levels must follow the specified Appendix E.1 group order.")
    token = str(config["missing_category_token"])
    unit = _matching_history(local, candidate, UNIT_COLUMNS, token)
    group_window = config.get("group_backoff_window_months")
    sources = [("risk_unit_id", unit, window) for window in (1, 2, 3)]
    sources += [
        ("+".join(columns), _matching_history(local, candidate, columns, token), group_window)
        for columns in levels
    ]
    for index, (name, source, window) in enumerate(sources):
        n, f = aggregate_counts(source, cutoff, start_month=start, window_months=window)
        eligible = (n >= 30 or (n >= 10 and f >= 1)) if index < 2 else (
            n >= 50 or (n >= 20 and f >= 1)
        )
        if eligible:
            return HierarchyScore(
                score=beta_rate(f, n, prior, float(config["prior_strength_a"])),
                selected_level=name,
                selected_level_index=index,
                selected_window_months=window,
                support_count=n,
                failure_count=f,
                prior=prior,
            )
    return HierarchyScore(prior, "global_prior", None, None, 0.0, 0.0, prior)

