"""Candidate reference implementation; not yet verified as identical to the implementation used for the reported results."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .evidence_rates import as_month


def smooth_scores(
    current_scores: pd.DataFrame,
    config: dict[str, Any],
    *,
    previous_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply backward-only smoothing and initialize first appearances to current scores."""

    required = {"risk_unit_id", "decision_month", "raw_score"}
    missing = sorted(required.difference(current_scores.columns))
    if missing:
        raise ValueError(f"Current scores are missing columns: {missing}")
    result = current_scores.copy()
    if result["risk_unit_id"].isna().any() or result["risk_unit_id"].duplicated().any():
        raise ValueError("Current scores require one nonmissing row per risk_unit_id.")
    current_months = result["decision_month"].map(as_month)
    if current_months.nunique() != 1:
        raise ValueError("Current scores must contain exactly one decision month.")
    current_month = current_months.iloc[0]
    alpha = float(config["alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")

    previous_map: dict[str, float] = {}
    if previous_scores is not None and not previous_scores.empty:
        previous_required = {"risk_unit_id", "decision_month", "smoothed_score"}
        missing_previous = sorted(previous_required.difference(previous_scores.columns))
        if missing_previous:
            raise ValueError(f"Previous scores are missing columns: {missing_previous}")
        previous = previous_scores.copy()
        previous["_period"] = previous["decision_month"].map(as_month)
        if (previous["_period"] >= current_month).any():
            raise ValueError("Previous scores must be strictly earlier than the current month.")
        previous["risk_unit_id"] = previous["risk_unit_id"].astype(str)
        previous = previous.sort_values(["_period", "risk_unit_id"], kind="mergesort")
        latest = previous.drop_duplicates("risk_unit_id", keep="last")
        previous_map = dict(zip(latest["risk_unit_id"], latest["smoothed_score"], strict=True))

    smoothed: list[float] = []
    initialization: list[str] = []
    for unit_id, current in zip(result["risk_unit_id"], result["raw_score"], strict=True):
        previous_value = previous_map.get(str(unit_id))
        if previous_value is None or pd.isna(previous_value):
            smoothed.append(float(current))
            initialization.append("current_score_initialization")
        else:
            smoothed.append(alpha * float(current) + (1.0 - alpha) * float(previous_value))
            initialization.append("previous_score_update")
    result["smoothed_score"] = smoothed
    result["smoothing_mode"] = initialization
    return result

