"""Candidate reference implementation; not yet verified as identical to the implementation used for the reported results."""

from __future__ import annotations

from typing import Any

import pandas as pd


def rank_and_select(
    scores: pd.DataFrame,
    capacity: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply configured stable tie-breaking and fixed integer capacity selection."""

    if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 0:
        raise ValueError("capacity must be a nonnegative integer.")
    if "risk_unit_id" not in scores.columns:
        raise ValueError("Scores require risk_unit_id.")
    if scores["risk_unit_id"].isna().any() or scores["risk_unit_id"].duplicated().any():
        raise ValueError("Scores require one nonmissing row per risk_unit_id.")
    tie_rules = list(config["tie_breaking"])
    columns = [str(rule["column"]) for rule in tie_rules]
    ascending = [bool(rule["ascending"]) for rule in tie_rules]
    missing = sorted(set(columns).difference(scores.columns))
    if missing:
        raise ValueError(f"Tie-breaking columns are missing: {missing}")
    result = scores.copy()
    result["risk_unit_id"] = result["risk_unit_id"].astype(str)
    result = result.sort_values(columns, ascending=ascending, kind="mergesort").reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    result["selected"] = result["rank"] <= min(int(capacity), len(result))
    return result
