"""Reference implementation following the method specification of the accompanying paper; see README for scope."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .config_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES, UNIT_COLUMNS, latency_months
from .evidence_rates import (
    aggregate_counts, as_month, beta_rate, empirical_prior,
    log_saturated_support, restrict_to_cutoff, validate_aggregates,
)
from .hierarchy import adaptive_hierarchy_score


def _normalize_input(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Require the monthly schema and one string identifier per four-field unit."""

    required = {"risk_unit_id", *UNIT_COLUMNS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Aggregate data are missing required columns: {missing}")
    normalized = validate_aggregates(data)
    if normalized["risk_unit_id"].isna().any():
        raise ValueError("risk_unit_id cannot be missing.")
    normalized["risk_unit_id"] = normalized["risk_unit_id"].astype(str)
    token = str(config["missing_category_token"])
    for column in UNIT_COLUMNS:
        normalized[column] = normalized[column].where(normalized[column].notna(), token).astype(str)
    units = normalized[["risk_unit_id", *UNIT_COLUMNS]].drop_duplicates()
    if units["risk_unit_id"].duplicated().any() or units.duplicated(UNIT_COLUMNS).any():
        raise ValueError("risk_unit_id must map one-to-one to the four-field risk unit.")
    return normalized


def _candidate_table(
    history: pd.DataFrame,
    config: dict[str, Any],
    candidates: pd.DataFrame | None,
) -> pd.DataFrame:
    """Retain one cutoff-available row per unit; explicit pools remain supported."""

    columns = ["risk_unit_id", *UNIT_COLUMNS]
    if candidates is None:
        ordered = history.sort_values(["month", "risk_unit_id"], kind="mergesort")
        return ordered.drop_duplicates("risk_unit_id", keep="last")[columns].reset_index(drop=True)
    if "risk_unit_id" not in candidates.columns or candidates["risk_unit_id"].isna().any():
        raise ValueError("Explicit candidates require nonmissing risk_unit_id.")
    result = candidates.copy()
    result["risk_unit_id"] = result["risk_unit_id"].astype(str)
    if result["risk_unit_id"].duplicated().any():
        raise ValueError("Explicit candidates must be unique by risk_unit_id.")
    token = str(config["missing_category_token"])
    for column in UNIT_COLUMNS:
        if column not in result.columns:
            result[column] = token
        result[column] = result[column].where(result[column].notna(), token).astype(str)
    return result[columns].reset_index(drop=True)


def construct_features_for_month(
    data: pd.DataFrame,
    decision_month: Any,
    config: dict[str, Any],
    *,
    candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct the nine Table 2 fields for one decision month."""

    normalized = _normalize_input(data, config)
    decision = as_month(decision_month)
    cutoff = decision - latency_months(config)
    history = restrict_to_cutoff(normalized, cutoff)
    candidate_table = _candidate_table(history, config, candidates)
    prior = empirical_prior(
        normalized, cutoff, float(config["default_prior"]), config.get("prior_start_month", "2024-01")
    )
    start = as_month(config.get("local_history_start_month", "2024-07"))
    strength = float(config["prior_strength_a"])
    extra = bool(config.get("include_extra_fields", False))
    audit_columns = ["risk_unit_id", "empirical_prior", "decision_month", "history_cutoff"]
    extra_columns = [
        "level_0_rate", "adaptive_hierarchy_score", "selected_hierarchy_level",
        "selected_hierarchy_level_index", "selected_window_months",
        "selected_support_count", "selected_failure_count",
    ] if extra else []
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + extra_columns + audit_columns
    rows: list[dict[str, Any]] = []
    for candidate in candidate_table.to_dict(orient="records"):
        unit_id = str(candidate["risk_unit_id"])
        unit_history = history.loc[history["risk_unit_id"] == unit_id]
        n, f = aggregate_counts(unit_history, cutoff, start_month=start)
        two_n, two_f = aggregate_counts(
            unit_history, cutoff, start_month=start, window_months=int(config["rolling_two_months"])
        )
        quarter_n, quarter_f = aggregate_counts(
            unit_history, cutoff, start_month=start, window_months=int(config["rolling_quarter_months"])
        )
        row = {
            "historical_smoothed_rate": beta_rate(f, n, prior, strength),
            "rolling_two_month_smoothed_rate": beta_rate(two_f, two_n, prior, strength),
            "rolling_quarter_smoothed_rate": beta_rate(quarter_f, quarter_n, prior, strength),
            "log_saturated_support": log_saturated_support(n, float(config["support_saturation_n_max"])),
            "cumulative_sample_count": n,
            "rolling_quarter_sample_count": quarter_n,
            "month_of_year": int(decision.month),
            "broad_category": candidate["broad_category"],
            "sampling_stage": candidate["sampling_stage"],
            "risk_unit_id": unit_id,
            "empirical_prior": prior,
            "decision_month": str(decision),
            "history_cutoff": str(cutoff),
        }
        if extra:
            hierarchy = adaptive_hierarchy_score(history, candidate, cutoff, config)
            row.update({
                "level_0_rate": f / n if n > 0 else prior,
                "adaptive_hierarchy_score": hierarchy.score,
                "selected_hierarchy_level": hierarchy.selected_level,
                "selected_hierarchy_level_index": hierarchy.selected_level_index,
                "selected_window_months": hierarchy.selected_window_months,
                "selected_support_count": hierarchy.support_count,
                "selected_failure_count": hierarchy.failure_count,
            })
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_supervised_rows(
    data: pd.DataFrame,
    decision_months: Iterable[Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Rebuild each earlier cutoff and attach only observed next-month labels."""

    normalized = _normalize_input(data, config)
    start = as_month(config.get("training_start_month", "2024-07"))
    panels: list[pd.DataFrame] = []
    for value in decision_months:
        decision = as_month(value)
        if decision < start:
            continue
        features = construct_features_for_month(normalized, decision, config)
        if features.empty:
            continue
        target_month = decision + 1
        target = normalized.loc[normalized["month"] == target_month]
        counts = target.groupby("risk_unit_id")[["sample_count", "failure_count"]].sum()
        counts = counts.reindex(features["risk_unit_id"])
        observed = counts["sample_count"].fillna(0).gt(0).to_numpy()
        features["target_month"] = str(target_month)
        features["target_observed"] = observed
        features["target"] = counts["failure_count"].gt(0).astype(float).where(observed).to_numpy()
        panels.append(features)
    if not panels:
        empty = construct_features_for_month(normalized.iloc[:0], start, config)
        return empty.assign(
            target_month=pd.Series(dtype=str),
            target_observed=pd.Series(dtype=bool),
            target=pd.Series(dtype=float),
        )
    return pd.concat(panels, ignore_index=True)

